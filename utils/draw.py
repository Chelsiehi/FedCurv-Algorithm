# coding: utf-8
import matplotlib.pyplot as plt
import os

def read_data(log_path, data_type="acc"):
    # 读取每个实验的实验数据
    all_acc_list = []
    for exp in os.listdir(log_path):
        acc_list = read_exp_data(os.path.join(log_path, exp), data_type)
        all_acc_list.append((exp, acc_list))

    plt.figure(figsize=(10, 6))
    for exp, acc_list in all_acc_list:
        plt.plot(acc_list, label=exp)  # 绘制每一个实验的精确度曲线

    plt.xlabel("Round")
    plt.ylabel('Accuracy' if data_type == 'acc' else 'Loss')
    plt.title(f"Experiment {'Accuracy' if data_type == 'acc' else 'Loss'} Comparison")
    plt.legend()
    plt.savefig(f"{data_type}.pdf")
    plt.show()


def read_exp_data(exp_path, data_type="acc"):
    data_file = "accuracy.dat" if data_type == "acc" else "loss.dat"
    with open(os.path.join(exp_path, data_file)) as f:
        acc_list = [float(i) for i in f.read().split()]
    return acc_list


if __name__ == '__main__':
    read_data("../log/log1", "acc")
    read_data("../log/log1", "loss")

