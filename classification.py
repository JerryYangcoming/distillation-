from __future__ import print_function

import os
import argparse
import socket
import threading
import time
import pynvml
import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown, nvmlDeviceGetName

# 假设这些模块已在您的项目中定义
from models import model_dict
from helper.util import adjust_learning_rate, accuracy, AverageMeter
from helper.loops import train_vanilla as train, validate

# CIFAR-100 的均值和标准差
MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


def get_data_folder():
    """返回数据存储路径，若不存在则创建"""
    hostname = socket.gethostname()
    data_folder = './data/'
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    return data_folder


def get_cifar100_group_dataloaders(batch_size=128, num_workers=8):
    """
    返回 CIFAR-100 的训练和测试数据加载器，将 100 个类别映射到 10 个组。
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_set = datasets.CIFAR100(root=data_folder, download=True, train=True, transform=train_transform)
    test_set = datasets.CIFAR100(root=data_folder, download=True, train=False, transform=test_transform)

    train_set.targets = [label // 10 for label in train_set.targets]
    test_set.targets = [label // 10 for label in test_set.targets]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=int(batch_size / 2), shuffle=False, num_workers=int(num_workers / 2))

    return train_loader, test_loader


def parse_option():
    """解析命令行参数"""
    parser = argparse.ArgumentParser('分类器训练参数')

    parser.add_argument('--print_freq', type=int, default=100, help='打印频率')
    parser.add_argument('--tb_freq', type=int, default=500, help='TensorBoard 日志频率')
    parser.add_argument('--save_freq', type=int, default=40, help='模型保存频率')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=16, help='数据加载线程数')
    parser.add_argument('--epochs', type=int, default=240, help='训练轮数')

    parser.add_argument('--learning_rate', type=float, default=0.05, help='学习率')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='学习率衰减的轮次')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='学习率衰减率')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减')
    parser.add_argument('--momentum', type=float, default=0.9, help='动量')

    parser.add_argument('--model', type=str, default='ShuffleV2',
                        choices=['resnet8', 'resnet14', 'resnet18', 'resnet20', 'resnet32', 'resnet44', 'resnet56',
                                 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'],
                        help='分类器模型架构')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='数据集')
    parser.add_argument('-t', '--trial', type=int, default=1, help='实验 ID')

    opt = parser.parse_args()

    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    opt.model_path = './save/classifier_models'
    opt.tb_path = './save/classifier_tensorboard'
    opt.lr_decay_epochs = [int(it) for it in opt.lr_decay_epochs.split(',')]
    opt.model_name = f'classifier_{opt.model}_{opt.dataset}_lr_{opt.learning_rate}_decay_{opt.weight_decay}_trial_{opt.trial}'
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    os.makedirs(opt.tb_folder, exist_ok=True)
    os.makedirs(opt.save_folder, exist_ok=True)

    return opt


def monitor_gpu(interval=30):
    """在单独线程中监控 GPU 使用情况"""
    nvmlInit()
    gpu_handle = nvmlDeviceGetHandleByIndex(0)
    gpu_name = nvmlDeviceGetName(gpu_handle)  # 移除 .decode('utf-8')，因为返回值已是字符串
    print(f"[GPU Info] GPU Name: {gpu_name}")

    try:
        while True:
            memory_info = nvmlDeviceGetMemoryInfo(gpu_handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
            print(
                f"[GPU Monitor] Memory: {memory_info.used / 1024 ** 2:.2f} MB / {memory_info.total / 1024 ** 2:.2f} MB, "
                f"Utilization: {utilization.gpu}%")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("[GPU Monitor] Stopped.")
    finally:
        nvmlShutdown()


def main():
    best_acc = 0
    opt = parse_option()

    monitor_thread = threading.Thread(target=monitor_gpu, args=(30,), daemon=True)
    monitor_thread.start()

    train_loader, val_loader = get_cifar100_group_dataloaders(opt.batch_size, opt.num_workers)
    n_cls = 10  # 10 个组

    model = model_dict[opt.model](num_classes=n_cls)
    optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)
    if torch.cuda.is_available():
        cudnn.benchmark = True

    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(epoch, opt, optimizer)
        print(f"==> 训练第 {epoch}/{opt.epochs} 轮")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()
        print(f"第 {epoch} 轮, 时间: {time2 - time1:.2f}s, 训练准确率: {train_acc:.2f}%, 训练损失: {train_loss:.4f}")

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        # 修改后的调用，添加 device 参数
        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt, device)
        print(f"测试准确率: {test_acc:.2f}%, 测试损失: {test_loss:.4f}")

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, f'{opt.model}_best.pth')
            torch.save(state, save_file)
            print(f'保存最佳模型，准确率: {best_acc:.2f}%')

        if epoch % opt.save_freq == 0:
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, f'ckpt_epoch_{epoch}.pth')
            torch.save(state, save_file)
            print(f'保存检查点: {save_file}')

    print(f'最佳准确率: {best_acc:.2f}%')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, f'{opt.model}_last.pth')
    torch.save(state, save_file)
    print(f'保存最终模型: {save_file}')


if __name__ == '__main__':
    main()
