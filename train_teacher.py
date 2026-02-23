from __future__ import print_function

from torch.cuda.amp import autocast, GradScaler

import os
import argparse
import socket
import subprocess
import threading
import time

import pynvml
import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown, nvmlDeviceGetName

from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders

from helper.util import adjust_learning_rate, accuracy, AverageMeter
from helper.loops import train_vanilla as train, validate


def parse_option():
    # 解析命令行传入的训练配置参数，并进行一定的逻辑处理（如调整学习率、设置路径等），最终返回一个包含所有这些配置的对象。

    # 获取当前主机的名称。通常用于根据主机名来配置不同的路径等。
    # hostname = socket.gethostname()

    # 创建一个 ArgumentParser 对象实例，用于解析命令行参数
    parser = argparse.ArgumentParser('argument for training')

    # 定义训练过程中的一些超参数
    # print_freq: 每多少步打印一次训练信息
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')

    # tb_freq: 每多少步记录一次 TensorBoard 数据
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')

    # save_freq: 每多少步保存一次模型
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')

    # batch_size: 每批次处理多少样本
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')

    # num_workers: 数据加载时使用多少个工作线程
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')

    # epochs: 训练的总轮数
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # 优化器相关的参数
    # learning_rate: 初始学习率
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')

    # lr_decay_epochs: 学习率衰减的epoch列表（例如：150,180,210）
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')

    # lr_decay_rate: 学习率衰减的倍率
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')

    # weight_decay: 权重衰减，用于L2正则化，限制权重的大小
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')

    # momentum: 动量，用于优化器的更新
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # 数据集和模型相关的参数
    # model: 使用的模型名称，支持多种模型架构
    parser.add_argument('--model', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'],
                        help='choose the model architecture')

    # dataset: 使用的数据集，这里支持的是CIFAR-100
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    # trial: 用于标记不同实验的编号
    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    # 解析命令行参数，返回包含所有参数的对象,解析命令行参数，
    # 并将解析结果存储到 opt 这个对象中。之后，可以通过访问 opt 中的属性来获取命令行传递的所有参数。
    opt = parser.parse_args()

    # 对某些模型设置不同的学习率
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01  # 这些模型使用较小的学习率

    # 选择模型和日志路径
    opt.model_path = './save/models'
    opt.tb_path = './save/tensorboard'

    # 将学习率衰减的epoch列表从字符串转化为整数列表,iterations:迭代
    iterations = opt.lr_decay_epochs.split(',')  # 按逗号分隔
    opt.lr_decay_epochs = list([])  # 初始化空列表
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))  # 转换为整数并添加到列表中

    # 根据选定的模型、数据集等信息生成模型名称
    opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                            opt.weight_decay, opt.trial)

    # 创建TensorBoard记录文件夹路径，并检查是否存在，若不存在则创建
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    # 创建模型保存文件夹路径，并检查是否存在，若不存在则创建
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    # 该函数解析命令行传入的训练配置参数，并进行一定的逻辑处理（如调整特定模型学习率、设置好路径等），
    # 最终返回一个包含所有这些配置的对象。
    return opt

def monitor_gpu(interval=30):
    """实时打印 GPU 型号及其使用情况."""
    pynvml.nvmlInit()
    gpu_handle = nvmlDeviceGetHandleByIndex(0)  # 假设使用第一个 GPU
    gpu_name = nvmlDeviceGetName(gpu_handle)  # 获取 GPU 型号（直接为字符串）

    print(f"[GPU Info] GPU Name: {gpu_name}")  # 打印 GPU 型号

    try:
        while True:
            memory_info = nvmlDeviceGetMemoryInfo(gpu_handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
            print(f"[GPU Monitor] Memory: {memory_info.used / 1024**2:.2f} MB / {memory_info.total / 1024**2:.2f} MB, "
                  f"Utilization: {utilization.gpu}%")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("[GPU Monitor] Stopped.")
    finally:
        pynvml.nvmlShutdown()


def main():

    best_acc = 0  # 初始化最佳准确率为0，用于保存最优模型的准确率

    opt = parse_option()  # 解析命令行参数（超参数、模型选择等）

    # 启动 GPU 监控线程,指定线程要执行的目标函数是 monitor_gpu
    monitor_thread = threading.Thread(target=monitor_gpu, args=(30,), daemon=True)
    monitor_thread.start()

    # 数据加载器
    if opt.dataset == 'cifar100':  # 如果选择的是CIFAR100数据集
        # 获取CIFAR100的训练和验证数据加载器
        # train_loader：训练集, val_loader：测试集
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 100  # CIFAR100有100个类
    else:
        raise NotImplementedError(opt.dataset)  # 如果选择了未实现的数据集，抛出异常

    # 根据配置选择models中对应的模型，并指定类别数（n_cls）
    # 未细读resnet
    model = model_dict[opt.model](num_classes=n_cls)

    # 优化器
    optimizer = optim.SGD(model.parameters(),  # 使用SGD优化器
                          lr=opt.learning_rate,  # 学习率
                          momentum=opt.momentum,  # 动量
                          weight_decay=opt.weight_decay)  # 权重衰减

    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数（适用于分类任务）

    if torch.cuda.is_available():  # 如果有可用的GPU
        model = model.cuda()  # 将模型转移到GPU
        criterion = criterion.cuda()  # 将损失函数转移到GPU
        cudnn.benchmark = True  # 优化CuDNN性能（如果输入数据的大小固定）

    scaler = GradScaler()

    # TensorBoard日志记录器
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)  # 设置TensorBoard日志路径，并指定刷新频率

    # 训练与验证的主循环
    for epoch in range(1, opt.epochs + 1):  # 遍历所有的训练周期（epochs）

        adjust_learning_rate(epoch, opt, optimizer)  # 根据当前的epoch调整学习率
        print("==> training...")

        time1 = time.time()  # 记录训练开始时间
        train_acc, train_loss = tra0 in(epoch, train_loader, model, criterion, optimizer, opt)  # 训练模型
        time2 = time.time()  # 记录训练结束时间
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))  # 输出每个epoch的训练时间

        # 记录训练精度和损失到TensorBoard
        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        # 验证模型
        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)

        # 记录测试集的精度和损失到TensorBoard
        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        # 保存最佳模型
        if test_acc > best_acc:  # 如果当前测试集准确率大于最佳准确率
            best_acc = test_acc  # 更新最佳准确率
            state = {  # 保存当前epoch的模型状态
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))  # 保存路径
            print('saving the best model!')
            torch.save(state, save_file)  # 保存模型状态

        # 定期保存模型（按照设定的保存频率）
        if epoch % opt.save_freq == 0:  # 每隔opt.save_freq个epoch保存一次
            print('==> Saving...')
            state = {  # 保存当前epoch的模型状态
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))  # 保存路径
            torch.save(state, save_file)  # 保存模型状态

    # 输出最终的最佳精度（用于打印目的）
    print('best accuracy:', best_acc)

    # 保存最终模型
    state = {  # 保存最终模型的配置和状态
        'opt': opt,  # 配置（包括参数）
        'model': model.state_dict(),  # 模型的状态字典
        'optimizer': optimizer.state_dict(),  # 优化器的状态字典
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))  # 保存路径
    torch.save(state, save_file)  # 保存模型


if __name__ == '__main__':
    main()