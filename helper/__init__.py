from __future__ import print_function, division  # 引入未来版本的print函数和除法操作符，确保兼容性

import sys
import time
import torch

from .util import AverageMeter, accuracy  # 从 util 模块导入 AverageMeter 和 accuracy 函数


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """传统训练过程（不含蒸馏）"""
    model.train()  # 设置模型为训练模式

    # 初始化多个计量器，用于统计训练过程中各类指标
    batch_time = AverageMeter()  # 记录每个批次的时间
    data_time = AverageMeter()  # 记录数据加载时间
    losses = AverageMeter()  # 记录损失
    top1 = AverageMeter()  # 记录准确率 top-1
    top5 = AverageMeter()  # 记录准确率 top-5

    end = time.time()  # 记录开始时间
    for idx, (input, target) in enumerate(train_loader):  # 遍历训练数据
        data_time.update(time.time() - end)  # 更新数据加载时间

        input = input.float()  # 将输入转换为float类型
        if torch.cuda.is_available():  # 如果有GPU，则将数据移动到GPU上
            input = input.cuda()
            target = target.cuda()

        # ===================前向传播=====================
        output = model(input)  # 将输入传入模型，获得预测输出
        loss = criterion(output, target)  # 计算损失

        # 计算准确率 top-1 和 top-5
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))  # 更新损失
        top1.update(acc1[0], input.size(0))  # 更新 top-1 准确率
        top5.update(acc5[0], input.size(0))  # 更新 top-5 准确率

        # ===================反向传播=====================
        optimizer.zero_grad()  # 清除优化器中的梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数

        # ===================统计信息=====================
        batch_time.update(time.time() - end)  # 更新批次时间
        end = time.time()  # 重新记录时间

        # 输出训练进度信息
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'  # 打印每个批次的时间
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'  # 打印数据加载时间
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'  # 打印损失
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'  # 打印 top-1 准确率
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()  # 刷新输出

    # 打印每个epoch的最终 top-1 和 top-5 准确率
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg  # 返回 top-1 准确率和损失的平均值


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """一个epoch的蒸馏训练"""
    # 设置学生模型为训练模式，教师模型为评估模式
    for module in module_list:
        module.train()  # 设置学生模型为训练模式
    module_list[-1].eval()  # 设置教师模型为评估模式

    if opt.distill == 'abound':
        module_list[1].eval()  # 在abound蒸馏中，第二个模块是教师模型
    elif opt.distill == 'factor':
        module_list[2].eval()  # 在factor蒸馏中，第三个模块是教师模型

    # 获取各类损失函数
    criterion_cls = criterion_list[0]  # 分类损失函数
    criterion_div = criterion_list[1]  # 分布损失函数
    criterion_kd = criterion_list[2]  # 知识蒸馏损失函数

    model_s = module_list[0]  # 学生模型
    model_t = module_list[-1]  # 教师模型

    # 初始化多个计量器，用于统计训练过程中的各类指标
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()  # 记录开始时间
    for idx, data in enumerate(train_loader):  # 遍历训练数据
        if opt.distill in ['crd']:  # 如果是 CRD 蒸馏，获取对比样本索引
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)  # 更新数据加载时间

        input = input.float()  # 将输入转换为float类型
        if torch.cuda.is_available():  # 如果有GPU，则将数据移动到GPU
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            if opt.distill in ['crd']:  # 如果是CRD蒸馏，还需要将对比样本索引移到GPU
                contrast_idx = contrast_idx.cuda()

        # ===================前向传播=====================
        preact = False  # 初始化 preact 标志
        if opt.distill in ['abound']:  # 如果是 'abound' 蒸馏，则需要预激活
            preact = True
        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)  # 获取学生模型的特征和预测结果
        with torch.no_grad():  # 教师模型不需要计算梯度
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact)  # 获取教师模型的特征和预测结果
            feat_t = [f.detach() for f in feat_t]  # 将教师模型的特征从计算图中分离

        # 分类损失和分布损失
        loss_cls = criterion_cls(logit_s, target)  # 分类损失
        loss_div = criterion_div(logit_s, logit_t)  # 分布损失

        # 根据不同的蒸馏方式计算不同的知识蒸馏损失
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'factor':
            factor_s = module_list[1](feat_s[-2])
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(opt.distill)

        # 总损失为分类损失、分布损失和知识蒸馏损失的加权和
        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        # 计算准确率 top-1 和 top-5
        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))  # 更新损失
        top1.update(acc1[0], input.size(0))  # 更新 top-1 准确率
        top5.update(acc5[0], input.size(0))  # 更新 top-5 准确率

        # ===================反向传播=====================
        optimizer.zero_grad()  # 清除优化器中的梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数

        # ===================统计信息=====================
        batch_time.update(time.time() - end)  # 更新批次时间
        end = time.time()  # 重新记录时间

        # 输出训练进度信息
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    # 打印每个epoch的最终 top-1 和 top-5 准确率
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg  # 返回 top-1 准确率和损失的平均值


def validate(val_loader, model, criterion, opt):
    """验证过程"""
    batch_time = AverageMeter()  # 记录每个批次的时间
    losses = AverageMeter()  # 记录损失
    top1 = AverageMeter()  # 记录 top-1 准确率
    top5 = AverageMeter()  # 记录 top-5 准确率

    model.eval()  # 将模型设置为评估模式

    with torch.no_grad():  # 在评估模式下，不需要计算梯度
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):
            input = input.float()  # 转换输入数据为浮点型
            if torch.cuda.is_available():  # 如果有GPU，则将数据移动到GPU
                input = input.cuda()
                target = target.cuda()

            # 前向传播，计算输出
            output = model(input)
            loss = criterion(output, target)  # 计算损失

            # 计算准确率和记录损失
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # 记录耗时
            batch_time.update(time.time() - end)
            end = time.time()

            # 输出验证过程中的信息
            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        # 打印最终准确率
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg  # 返回验证集的 top-1、top-5 准确率和损失
