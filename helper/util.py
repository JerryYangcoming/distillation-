from __future__ import print_function  # 导入未来版本的print函数，确保兼容性

import torch
import numpy as np


def adjust_learning_rate_new(epoch, optimizer, LUT):
    """
    根据RotNet的方法调整学习率。

    参数:
    - epoch (int): 当前的epoch数
    - optimizer (torch.optim.Optimizer): 用于优化的优化器
    - LUT (list): 学习率查找表，包含(max_epoch, lr)元组，按照max_epoch升序排列

    返回:
    - None
    """
    # 根据LUT中的最大epoch来选择对应的学习率
    lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
    # 将优化器的学习率更新为新的学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(epoch, opt, optimizer):
    """
    设置学习率，每经过预定的衰减周期（lr_decay_epochs），按lr_decay_rate衰减学习率。

    参数:
    - epoch (int): 当前的epoch数
    - opt (argparse.Namespace): 命令行参数对象
    - optimizer (torch.optim.Optimizer): 用于优化的优化器

    返回:
    - None
    """
    # 计算当前epoch经过了多少个衰减周期
    # np.asarray(opt.lr_decay_epochs) 将衰减周期列表转换为 NumPy 数组，方便进行向量化比较。
    # epoch > np.asarray(opt.lr_decay_epochs) 会生成一个布尔数组，如果当前 epoch 大于某个阈值，则对应位置为 True。
    # np.sum(...) 对布尔数组求和，计算出当前 epoch 已经超过了多少个衰减周期。这相当于统计已经衰减了几次。
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        # 如果有衰减周期，计算新的学习率
        # 计算方法：
        # opt.learning_rate 是初始学习率。
        # opt.lr_decay_rate 是每个衰减周期的衰减倍率（通常小于 1，比如 0.1）。
        # ** steps 表示衰减的次数。例如，如果 steps 为 2，则新的学习率为初始学习率乘以衰减率的平方。
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        # 更新优化器中的学习率

        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """
    计算并存储当前值和平均值的类
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        重置所有计数器
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        更新计数器并计算平均值

        参数:
        - val (float): 当前的数值
        - n (int): 该数值的计数，默认为1

        返回:
        - None
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    计算给定k个top预测值的准确率。

    参数:
    - output (torch.Tensor): 模型输出的预测值
    - target (torch.Tensor): 真实标签
    - topk (tuple): 需要计算准确率的topk值，默认为(1,)

    返回:
    - res (list): 包含每个topk值准确率的列表
    """
    with torch.no_grad():  # 禁用梯度计算
        maxk = max(topk)  # 获取最大的topk值
        batch_size = target.size(0)  # 获取batch的大小

        # 获取前k个预测结果的索引
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()  # 转置预测结果
        # 判断预测结果是否正确
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        # 计算每个topk的准确率
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))  # 转换为百分比准确率
        return res


if __name__ == '__main__':
    pass  # 这里是程序入口，当前不做任何操作
