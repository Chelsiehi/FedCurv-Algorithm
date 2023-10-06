# coding: utf-8
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class BaseLocal(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter_ in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


def register_buffer(module: torch.nn.Module, name: str, value: torch.Tensor):
    """Add a buffer to module.

    Args:
        module: Module
        name: Buffer name. Supports complex module names like 'model.conv1.bias'.
        value: Buffer value
    """
    module_path, _, name = name.rpartition('.')
    mod = module.get_submodule(module_path)
    mod.register_buffer(name, value)


def get_buffer(module, target):
    """Get module buffer.

    Remove after pinning to a version
    where https://github.com/pytorch/pytorch/pull/61429 is included.
    Use module.get_buffer() instead.
    """
    module_path, _, buffer_name = target.rpartition('.')

    mod: torch.nn.Module = module.get_submodule(module_path)
    if not hasattr(mod, buffer_name):
        raise AttributeError(f'{mod._get_name()} has no attribute `{buffer_name}`')

    buffer: torch.Tensor = getattr(mod, buffer_name)

    if buffer_name not in mod._buffers:
        raise AttributeError('`' + buffer_name + '` is not a buffer')

    return buffer


class FedCurv:
    """Federated Curvature class.

    Requires torch>=1.9.0.
    """

    def __init__(self, model: torch.nn.Module, importance: float):
        """Initialize.

        Args:
            model: Base model. Parameters of it are used in loss penalty calculation.
            importance: Lambda coefficient of FedCurv algorithm.
        """
        self.importance = importance
        self._params = {}
        self._register_fisher_parameters(model)

    def _register_fisher_parameters(self, model):
        params = list(model.named_parameters())
        for n, p in params:
            u = torch.zeros_like(p, requires_grad=False)
            v = torch.zeros_like(p, requires_grad=False)
            w = torch.zeros_like(p, requires_grad=False)

            # Add buffers to model for aggregation
            register_buffer(model, f'{n}_u', u)
            register_buffer(model, f'{n}_v', v)
            register_buffer(model, f'{n}_w', w)

            # Store buffers locally for subtraction in loss function
            setattr(self, f'{n}_u', u)
            setattr(self, f'{n}_v', v)
            setattr(self, f'{n}_w', w)

    def _update_params(self, model):
        self._params = deepcopy({n: p for n, p in model.named_parameters() if p.requires_grad})

    def _diag_fisher(self, model, data_loader, device):
        precision_matrices = {}
        for n, p in self._params.items():
            p.data.zero_()
            precision_matrices[n] = p.data.to(device)

        model.eval()
        model.to(device)
        for sample, target in data_loader:
            model.zero_grad()
            sample = sample.to(device)
            target = target.to(device)
            output = model(sample)
            loss = F.nll_loss(F.log_softmax(output, dim=1), target)
            loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad:
                    precision_matrices[n].data = p.grad.data ** 2 / len(data_loader)

        return precision_matrices

    def get_penalty(self, model):
        """Calculate the penalty term for the loss function.

        Args:
            model(torch.nn.Module): Model that stores global u_t and v_t values as buffers.

        Returns:
            float: Penalty term.
        """
        penalty = 0
        if not self._params:
            return penalty
        for name, param in model.named_parameters():
            if param.requires_grad:
                u_global, v_global, w_global = (
                    get_buffer(model, target).detach()
                    for target in (f'{name}_u', f'{name}_v', f'{name}_w')
                )
                u_local, v_local, w_local = (
                    getattr(self, name).detach()
                    for name in (f'{name}_u', f'{name}_v', f'{name}_w')
                )
                u = u_global - u_local
                v = v_global - v_local
                w = w_global - w_local
                _penalty = param ** 2 * u - 2 * param * v + w
                penalty += _penalty.sum()
        penalty = self.importance * penalty
        return penalty.float()

    def on_train_begin(self, model):
        """Pre-train steps.

        Args:
            model(torch.nn.Module): model for training.
        """
        self._update_params(model)

    def on_train_end(self, model: torch.nn.Module, data_loader, device):
        """Post-train steps.

        Args:
            model(torch.nn.Module): Trained model.
            data_loader(Iterable): Train dataset iterator.
            device(str): Model device.
            loss_fn(Callable): Train loss function.
        """
        precision_matrices = self._diag_fisher(model, data_loader, device)
        for n, m in precision_matrices.items():
            u = m.data.to(device)
            v = m.data * model.get_parameter(n)
            v = v.to(device)
            w = m.data * model.get_parameter(n) ** 2
            w = w.to(device)
            register_buffer(model, f'{n}_u', u.clone().detach())
            register_buffer(model, f'{n}_v', v.clone().detach())
            register_buffer(model, f'{n}_w', w.clone().detach())
            setattr(self, f'{n}_u', u.clone().detach())
            setattr(self, f'{n}_v', v.clone().detach())
            setattr(self, f'{n}_w', w.clone().detach())


class FedCurvLocal(BaseLocal):
    def __init__(self, args, dataset, idx):
        super(FedCurvLocal, self).__init__(args, dataset, idx)

    def train(self, net, fedcurv=None):
        fedcurv.on_train_begin(net)
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter_ in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels) + fedcurv.get_penalty(net)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        fedcurv.on_train_end(net, self.ldr_train, self.args.device)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
