import os
import pickle
import numpy as np


def load_local_cifar100(cifar_dir='./cifar-100-python'):
    """从本地加载CIFAR-100数据集并显示类别信息"""

    # 加载meta文件获取类别信息
    with open(os.path.join(cifar_dir, 'meta'), 'rb') as f:
        meta = pickle.load(f, encoding='bytes')

    # 获取类别名称和超类别名称
    fine_label_names = [label.decode('utf-8') for label in meta[b'fine_label_names']]
    coarse_label_names = [label.decode('utf-8') for label in meta[b'coarse_label_names']]

    # 打印类别名称和编号的对应关系
    print("\n===== CIFAR-100 类别与编号对应表 =====\n")
    print(f"{'编号':<5} {'类别名称':<20} {'超类别'}")
    print("-" * 50)

    # 创建一个从细粒度类别到超类别的映射
    with open(os.path.join(cifar_dir, 'train'), 'rb') as f:
        train_data = pickle.load(f, encoding='bytes')

    # 获取细粒度标签和粗粒度标签的对应关系
    fine_labels = train_data[b'fine_labels']
    coarse_labels = train_data[b'coarse_labels']

    # 创建细粒度标签到粗粒度标签的映射
    fine_to_coarse = {}
    for fine, coarse in zip(fine_labels, coarse_labels):
        fine_to_coarse[fine] = coarse

    # 打印每个类别及其超类别
    for i, class_name in enumerate(fine_label_names):
        coarse_idx = fine_to_coarse.get(i, i // 5)  # 如果映射不存在，使用整除计算超类别
        superclass = coarse_label_names[coarse_idx]
        print(f"{i:<5} {class_name:<20} {superclass}")

    # 打印超类别信息
    print("\n===== CIFAR-100 超类别信息 =====\n")
    print(f"{'编号':<5} {'超类别名称':<30}")
    print("-" * 40)

    for i, superclass in enumerate(coarse_label_names):
        print(f"{i:<5} {superclass:<30}")

    # 返回各种标签信息，以便进一步使用
    return {
        'fine_label_names': fine_label_names,
        'coarse_label_names': coarse_label_names,
        'fine_to_coarse': fine_to_coarse
    }


if __name__ == "__main__":
    # 参数是您本地CIFAR-100数据集的路径，根据需要修改
    cifar_dir = '/data/cifar-100-python'  # 默认路径，根据实际情况调整
    label_info = load_local_cifar100(cifar_dir)
