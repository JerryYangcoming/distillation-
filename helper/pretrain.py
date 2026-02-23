from __future__ import print_function, division  # 导入未来版本的print函数和除法操作符，确保兼容性

import time
import sys
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from .util import AverageMeter  # 从自定义的util模块中导入AverageMeter类


def init(model_s, model_t, init_modules, criterion, train_loader, logger, opt):
    """
    初始训练过程，用于初始化模型参数。

    Args:
    - model_s (torch.nn.Module): 学生模型
    - model_t (torch.nn.Module): 教师模型
    - init_modules (torch.nn.Module): 初始模块
    - criterion (torch.nn.Module): 损失函数
    - train_loader (torch.utils.data.DataLoader): 训练数据加载器
    - logger: 日志记录器
    - opt (argparse.Namespace): 命令行参数选项

    Returns:
    - None
    """
    model_t.eval()  # 设置教师模型为评估模式
    model_s.eval()  # 设置学生模型为评估模式
    init_modules.train()  # 设置初始模块为训练模式

    if torch.cuda.is_available():  # 如果GPU可用，将模型和数据移动到GPU上
        model_s.cuda()
        model_t.cuda()
        init_modules.cuda()
        cudnn.benchmark = True  # 提高训练速度的标志位

    # 根据模型选择和蒸馏方式调整学习率
    if opt.model_s in ['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                       'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2'] and \
            opt.distill == 'factor':
        lr = 0.01  # 对于特定的ResNet和蒸馏方式为factor的情况，设定学习率为0.01
    else:
        lr = opt.learning_rate  # 其他情况下使用命令行中指定的学习率
    optimizer = optim.SGD(init_modules.parameters(),  # 使用随机梯度下降优化器
                          lr=lr,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # 初始化计量器
    batch_time = AverageMeter()  # 批次时间
    data_time = AverageMeter()  # 数据加载时间
    losses = AverageMeter()  # 损失

    for epoch in range(1, opt.init_epochs + 1):  # 遍历初始训练的epoch
        batch_time.reset()  # 重置批次时间计量器
        data_time.reset()  # 重置数据加载时间计量器
        losses.reset()  # 重置损失计量器
        end = time.time()  # 记录开始时间

        for idx, data in enumerate(train_loader):  # 遍历训练数据
            if opt.distill in ['crd']:  # 如果是CRD蒸馏，则获取对比样本索引
                input, target, index, contrast_idx = data
            else:
                input, target, index = data
            data_time.update(time.time() - end)  # 更新数据加载时间

            input = input.float()  # 将输入转换为浮点型
            if torch.cuda.is_available():  # 如果GPU可用，将数据移动到GPU上
                input = input.cuda()
                target = target.cuda()
                index = index.cuda()
                if opt.distill in ['crd']:  # 如果是CRD蒸馏，还需要将对比样本索引移到GPU
                    contrast_idx = contrast_idx.cuda()

            # ============= 前向传播 ==============
            preact = (opt.distill == 'abound')  # 是否预激活
            feat_s, _ = model_s(input, is_feat=True, preact=preact)  # 获取学生模型的特征
            with torch.no_grad():  # 教师模型不需要计算梯度
                feat_t, _ = model_t(input, is_feat=True, preact=preact)  # 获取教师模型的特征
                feat_t = [f.detach() for f in feat_t]  # 将教师模型的特征从计算图中分离

            if opt.distill == 'abound':  # 如果是abound蒸馏方式
                g_s = init_modules[0](feat_s[1:-1])  # 计算学生模型的中间特征
                g_t = feat_t[1:-1]  # 教师模型的中间特征
                loss_group = criterion(g_s, g_t)  # 计算损失组
                loss = sum(loss_group)  # 总损失为损失组的总和
            elif opt.distill == 'factor':  # 如果是factor蒸馏方式
                f_t = feat_t[-2]  # 获取教师模型倒数第二层的特征
                _, f_t_rec = init_modules[0](f_t)  # 重构教师模型特征
                loss = criterion(f_t_rec, f_t)  # 计算损失
            elif opt.distill == 'fsp':  # 如果是fsp蒸馏方式
                loss_group = criterion(feat_s[:-1], feat_t[:-1])  # 计算特征空间损失组
                loss = sum(loss_group)  # 总损失为损失组的总和
            else:
                raise NotImplementedError('Not supported in init training: {}'.format(opt.distill))

            losses.update(loss.item(), input.size(0))  # 更新损失

            # =============== 反向传播 ================
            optimizer.zero_grad()  # 清除优化器中的梯度
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数

            batch_time.update(time.time() - end)  # 更新批次时间
            end = time.time()

        # 每个epoch结束后记录损失到日志中，并打印当前epoch的训练信息
        logger.log_value('init_train_loss', losses.avg, epoch)
        print('Epoch: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'losses: {losses.val:.3f} ({losses.avg:.3f})'.format(
               epoch, opt.init_epochs, batch_time=batch_time, losses=losses))
        sys.stdout.flush()
