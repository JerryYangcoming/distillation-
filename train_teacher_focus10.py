from __future__ import print_function

from torch.cuda.amp import autocast, GradScaler

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
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

from models import model_dict

from helper.util import adjust_learning_rate, accuracy, AverageMeter
from helper.loops import train_vanilla as train, validate

def parse_option():
    """解析命令行参数，新增目标类别参数"""
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    parser.add_argument('--model', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'],
                        help='choose the model architecture')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')
    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    parser.add_argument('--target_classes', type=str, default='0,1,2,3,4,5,6,7,8,9',
                        help='comma-separated list of 10 target class indices (0-99)')

    opt = parser.parse_args()

    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    opt.model_path = './save/models'
    opt.tb_path = './save/tensorboard'

    opt.lr_decay_epochs = [int(it) for it in opt.lr_decay_epochs.split(',')]

    opt.target_classes = [int(cls) for cls in opt.target_classes.split(',')]
    if len(opt.target_classes) != 10:
        raise ValueError("Must specify exactly 10 target classes.")
    if any(cls < 0 or cls > 99 for cls in opt.target_classes):
        raise ValueError("Target classes must be between 0 and 99.")

    opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}_target_{}'.format(
        opt.model, opt.dataset, opt.learning_rate, opt.weight_decay, opt.trial,
        '-'.join(map(str, opt.target_classes))
    )

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

def monitor_gpu(interval=30):
    """实时打印GPU型号及其使用情况"""
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(gpu_handle)
    print(f"[GPU Info] GPU Name: {gpu_name}")

    try:
        while True:
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
            print(f"[GPU Monitor] Memory: {memory_info.used / 1024 ** 2:.2f} MB / {memory_info.total / 1024 ** 2:.2f} MB, "
                  f"Utilization: {utilization.gpu}%")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("[GPU Monitor] Stopped.")
    finally:
        pynvml.nvmlShutdown()

class FilteredCIFAR100(Dataset):
    """自定义数据集类，用于过滤CIFAR-100中的特定类别并重新映射标签"""
    def __init__(self, dataset, target_classes, label_map):
        self.dataset = dataset
        self.target_classes = target_classes
        self.label_map = label_map
        self.indices = [i for i, (_, label) in enumerate(dataset) if label in target_classes]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index = self.indices[index]
        img, label = self.dataset[original_index]
        new_label = self.label_map[label]
        return img, new_label

def get_filtered_cifar100_dataloaders(batch_size, num_workers, target_classes):
    """获取过滤后的CIFAR-100数据加载器"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    label_map = {old_label: new_label for new_label, old_label in enumerate(target_classes)}

    filtered_train_dataset = FilteredCIFAR100(train_dataset, target_classes, label_map)
    filtered_val_dataset = FilteredCIFAR100(val_dataset, target_classes, label_map)

    train_loader = DataLoader(filtered_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(filtered_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Filtered train dataset size: {len(filtered_train_dataset)}, val dataset size: {len(filtered_val_dataset)}")
    return train_loader, val_loader

def main():
    best_acc = 0
    opt = parse_option()

    monitor_thread = threading.Thread(target=monitor_gpu, args=(30,), daemon=True)
    monitor_thread.start()

    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_filtered_cifar100_dataloaders(
            batch_size=opt.batch_size, num_workers=opt.num_workers, target_classes=opt.target_classes
        )
        n_cls = 10
    else:
        raise NotImplementedError(opt.dataset)

    model = model_dict[opt.model](num_classes=n_cls)

    optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    scaler = GradScaler('cuda')
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()
        print(f'epoch {epoch}, total time {time2 - time1:.2f}')

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)
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
            print('saving the best model!')
            torch.save(state, save_file)

        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, f'ckpt_epoch_{epoch}.pth')
            torch.save(state, save_file)

    print('best accuracy:', best_acc)

    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, f'{opt.model}_last.pth')
    torch.save(state, save_file)

if __name__ == '__main__':
    main()
