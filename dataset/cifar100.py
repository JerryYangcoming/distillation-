from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

"""
mean = {
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar100': (0.2675, 0.2565, 0.2761),
}
"""

def get_data_folder():
    """
    根据主机名返回用于存储数据的路径。如果路径不存在，则创建路径。
    """
    hostname = socket.gethostname()  # 获取当前主机的名称
    data_folder = './data/'  # 默认情况下使用当前目录下的 data 文件夹
    if not os.path.isdir(data_folder):  # 如果目录不存在
        os.makedirs(data_folder)  # 创建目录

    return data_folder  # 返回数据存储路径

class CIFAR100Instance(datasets.CIFAR100):
    """
    扩展 `datasets.CIFAR100` 类，新增功能：
    - 在返回图像和标签时，额外返回图像的索引。
    """
    def __getitem__(self, index):
        """
        重写 `__getitem__` 方法。
        参数:
            index (int): 图像的索引。
        返回:
            tuple: 图像、标签、索引。
        """
        img, target = super().__getitem__(index)  # 调用父类方法获取图像和标签
        return img, target, index  # 增加返回索引

def get_cifar100_dataloaders(batch_size=128, num_workers=8, is_instance=True):
    """
    返回 CIFAR-100 数据集的训练和测试加载器。
    参数:
        batch_size (int): 每个批次的数据量。
        num_workers (int): 数据加载时的并行线程数。
        is_instance (bool): 是否返回索引。
    返回:
        train_loader, test_loader: 数据加载器。
        如果 `is_instance=True`，还会返回训练集的样本数量。
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    if is_instance:
        train_set = CIFAR100Instance(root=data_folder, download=True, train=True, transform=train_transform)
        n_data = len(train_set)  # 获取训练样本的数量
    else:
        train_set = datasets.CIFAR100(root=data_folder, download=True, train=True, transform=train_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_set = datasets.CIFAR100(root=data_folder, download=True, train=False, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=int(batch_size/2), shuffle=False, num_workers=int(num_workers/2))

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader

class CIFAR100InstanceSample(datasets.CIFAR100):
    """
    CIFAR100Instance+Sample Dataset，支持对比学习采样，用于 CRD。
    """
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 100
        num_samples = len(self.data)  # 使用 self.data
        label = self.targets  # 使用 self.targets

        # 构建正样本索引
        self.cls_positive = [[] for _ in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        # 构建负样本索引
        self.cls_negative = [[] for _ in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]  # 使用 self.data 和 self.targets

        # 将 numpy 数组转换为 PIL 图像
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            # 直接返回图像、标签和索引
            return img, target, index
        else:
            # 为对比学习采样正负样本
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx

def get_cifar100_dataloaders_sample(batch_size=128, num_workers=8, k=4096, mode='exact',
                                    is_sample=True, percent=1.0):
    """
    返回支持 CRD 的 CIFAR-100 数据加载器。
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_set = CIFAR100InstanceSample(root=data_folder,
                                       download=True,
                                       train=True,
                                       transform=train_transform,
                                       k=k,
                                       mode=mode,
                                       is_sample=is_sample,
                                       percent=percent)
    n_data = len(train_set)  # 直接使用 len(train_set)，依赖父类的 __len__
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    return train_loader, test_loader, n_data
