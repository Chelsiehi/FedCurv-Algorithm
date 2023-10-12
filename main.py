if __name__ == '__main__':
    args = args_parser()

    log_root = "./log"
    os.makedirs(log_root, exist_ok=True)
    log_path = os.path.join(log_root, f"{args.alg}-{args.model}-{args.dataset}-B={args.local_bs}-E={args.local_ep}-{'iid' if args.iid else 'noniid'}")
    os.makedirs(log_path, exist_ok=True)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
    else:
        exit('Error: unrecognized dataset')

    if args.iid:
        dict_users = build_iid_data(dataset_train, args.num_users)
    else:
        if args.dir:
            print("dir noniid")
            dict_users = build_dir_data(dataset_train, args.num_users, args.dir_alpha)
        else:
            print("noniid")
            dict_users = build_noniid_data(dataset_train, args.num_users)

    draw_data_distribution(dict_users, dataset_train, args.num_classes, fig_path=log_path)

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == "resnet":
        net_glob = ResNet18(num_classes=10, in_channels=1)
        # 如果是mnist数据集，需要修改通道数
        #if args.dataset == "mnist":
        #    net_glob.prep[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif args.model == "lenet":
        if args.dataset == "mnist":
            net_glob = LeNetMnist(channels=args.channels, num_classes=args.num_classes).to(args.device)
        elif args.dataset == "cifar":
            net_glob = LeNetCifar(channels=args.channels, num_classes=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    net_glob.train()
    fedcurv = FedCurv(net_glob, args.importance)

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    acc_file = os.path.join(log_path, "accuracy.dat")
    acc_f = open(acc_file, "w", encoding="utf-8")

    loss_file = os.path.join(log_path, "loss.dat")
    loss_f = open(loss_file, "w", encoding="utf-8")

    client_loss_file = os.path.join(log_path, "client_avg_loss.dat")
    client_loss_f = open(client_loss_file, "w", encoding="utf-8")

    for iter_ in range(args.epochs):
        print(f"---Round {iter_}---")
        loss_locals = []
        w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            if args.alg == "fedavg":
                local = BaseLocal(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            elif args.alg == "fedcurv":
                local = FedCurvLocal(args, dataset_train, dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device), fedcurv=fedcurv)
            else:
                exit("not supported alg")
            print(f"client {loss}")
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        client_avg_loss = sum(loss_locals) / len(loss_locals)
        print(f"Client Train Avg Loss {client_avg_loss:.2f}")
        client_loss_f.write(str(client_avg_loss))
        client_loss_f.flush()

        # update global weights
        # w_locals = [w1, w2, w3...]
        if args.alg == "fedavg":
            w_glob = FedAvgAgg(w_locals)
        elif args.alg == "fedcurv":
            w_glob = FedCurvAgg(w_locals)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        net_glob.eval()

        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        acc_f.write(str(acc_test) + "\n")
        acc_f.flush()
        loss_f.write(str(loss_test) + "\n")
        loss_f.flush()

        print("Global Testing accuracy: {:.2f}".format(acc_test))
