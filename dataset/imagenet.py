from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


def get_data_folder():
    """
    返回用于存储数据的服务器依赖的路径
    根据不同的主机名来选择数据存储路径。
    """
    hostname = socket.gethostname()  # 获取当前机器的主机名
    if hostname.startswith('visiongpu'):
        # 如果主机名以 "visiongpu" 开头，返回指定的路径
        data_folder = '/data/vision/phillipi/rep-learn/datasets/imagenet'
    elif hostname.startswith('yonglong-home'):
        # 如果主机名以 "yonglong-home" 开头，返回家用路径
        data_folder = '/home/yonglong/Data/data/imagenet'
    else:
        # 默认的路径
        data_folder = './data/imagenet'

    # 如果数据文件夹不存在，则创建文件夹
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


class ImageFolderInstance(datasets.ImageFolder):
    """
    扩展 ImageFolder 类，返回图像的索引以及目标类别索引。
    """
    def __getitem__(self, index):
        """
        获取指定索引的图像和标签，并返回图像、标签和索引。

        参数:
            index (int): 图像的索引
        返回:
            tuple: (image, target, index) 其中 target 是目标类别的索引
        """
        img, target = super().__getitem__(index)  # 获取图像和标签
        return img, target, index


class ImageFolderSample(datasets.ImageFolder):
    """
    扩展 ImageFolder 类，返回图像、标签、索引以及对比学习用的负样本索引。
    """
    def __init__(self, root, transform=None, target_transform=None,
                 is_sample=False, k=4096):
        """
        初始化数据集并根据需要生成对比样本。

        参数:
            root: 数据集根目录
            transform: 图像转换函数
            target_transform: 标签转换函数
            is_sample: 是否生成对比学习样本
            k: 每个正样本对应的负样本数量
        """
        super().__init__(root=root, transform=transform, target_transform=target_transform)

        self.k = k  # 每个正样本对应的负样本数量
        self.is_sample = is_sample  # 是否使用对比样本
        print('stage1 finished!')

        if self.is_sample:
            # 生成每个类别的正负样本索引
            num_classes = len(self.classes)
            num_samples = len(self.samples)
            label = np.zeros(num_samples, dtype=np.int32)
            for i in range(num_samples):
                path, target = self.imgs[i]
                label[i] = target

            self.cls_positive = [[] for i in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for i in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]

        print('dataset initialized!')

    def __getitem__(self, index):
        """
        获取指定索引的图像、标签，并返回图像、标签、索引及对比样本的索引。

        参数:
            index (int): 图像的索引
        返回:
            tuple: (image, target, index, contrast_index)
        """
        path, target = self.imgs[index]  # 获取图像路径和标签
        img = self.loader(path)  # 加载图像
        if self.transform is not None:
            img = self.transform(img)  # 应用转换函数
        if self.target_transform is not None:
            target = self.target_transform(target)  # 应用标签转换函数

        if self.is_sample:
            # 如果是对比学习模式，生成正负样本对
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))  # 合并正负样本索引
            return img, target, index, sample_idx
        else:
            return img, target, index


def get_test_loader(dataset='imagenet', batch_size=128, num_workers=8):
    """
    获取测试数据的 DataLoader

    参数:
        dataset: 数据集名称
        batch_size: 每批次的样本数量
        num_workers: 使用的线程数
    返回:
        test_loader: 测试集的 DataLoader
    """
    if dataset == 'imagenet':
        data_folder = get_data_folder()
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    # 图像的归一化处理
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_folder = os.path.join(data_folder, 'val')  # 测试集文件夹路径
    test_set = datasets.ImageFolder(test_folder, transform=test_transform)  # 加载测试集
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)  # 创建 DataLoader

    return test_loader


def get_dataloader_sample(dataset='imagenet', batch_size=128, num_workers=8, is_sample=False, k=4096):
    """
    获取带有对比样本的训练和测试数据加载器

    参数:
        dataset: 数据集名称
        batch_size: 每批次的样本数量
        num_workers: 使用的线程数
        is_sample: 是否启用对比学习
        k: 每个样本的负样本数量
    返回:
        train_loader: 训练集的 DataLoader
        test_loader: 测试集的 DataLoader
        num_samples: 训练集样本数量
        num_classes: 类别数量
    """
    if dataset == 'imagenet':
        data_folder = get_data_folder()
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    # 图像转换（包括随机裁剪、水平翻转等）
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    train_folder = os.path.join(data_folder, 'train')
    test_folder = os.path.join(data_folder, 'val')

    # 根据是否启用对比学习选择数据集类型
    train_set = ImageFolderSample(train_folder, transform=train_transform, is_sample=is_sample, k=k)
    test_set = datasets.ImageFolder(test_folder, transform=test_transform)

    # 创建训练集和测试集的 DataLoader
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)

    print('num_samples', len(train_set.samples))
    print('num_class', len(train_set.classes))

    return train_loader, test_loader, len(train_set), len(train_set.classes)


def get_imagenet_dataloader(dataset='imagenet', batch_size=128, num_workers=16, is_instance=False):
    """
    获取 ImageNet 数据集的训练和测试加载器

    参数:
        dataset: 数据集名称
        batch_size: 每批次的样本数量
        num_workers: 使用的线程数
        is_instance: 是否启用图像实例级别数据集
    返回:
        train_loader: 训练集的 DataLoader
        test_loader: 测试集的 DataLoader
    """
    if dataset == 'imagenet':
        data_folder = get_data_folder()
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    # 图像归一化处理
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_folder = os.path.join(data_folder, 'train')
    test_folder = os.path.join(data_folder, 'val')

    if is_instance:
        # 如果是实例级数据集，则使用自定义的 ImageFolderInstance
        train_set = ImageFolderInstance(train_folder, transform=train_transform)
        n_data = len(train_set)
    else:
        # 否则使用默认的 ImageFolder 数据集
        train_set = datasets.ImageFolder(train_folder, transform=train_transform)

    test_set = datasets.ImageFolder(test_folder, transform=test_transform)

    # 创建训练集和测试集的 DataLoader
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)

    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers//2,
                             pin_memory=True)

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader
