from __future__ import print_function, division

import sys
import time
import torch
import torch.nn as nn
import tensorboard_logger as tb_logger
from helper.util import AverageMeter, accuracy


def get_temperature(loss, min_temp=1.0, max_temp=10.0, loss_threshold=1.0):
    """根据损失动态计算温度"""
    if loss > loss_threshold:
        return max_temp
    else:
        return min_temp + (max_temp - min_temp) * (loss / loss_threshold)


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """传统训练过程（不含蒸馏）"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % opt.print_freq == 0:
            print(f'Epoch: [{epoch}][{idx}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')
            sys.stdout.flush()

    print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
    return top1.avg, losses.avg


def train_with_accuracy_objective(epoch, train_loader, model, optimizer, opt, device):
    """
    使用准确率为直接优化目标的训练过程

    这实现了一个基于REINFORCE算法的变体，用于优化不可微的准确率目标
    """
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    accuracies = AverageMeter()
    losses = AverageMeter()  # 仅用于记录，不用于优化
    rewards = AverageMeter()

    # 温度参数，控制softmax输出的锐度
    temperature = opt.temperature if hasattr(opt, 'temperature') else 1.0
    # 熵正则化权重，避免策略退化
    entropy_weight = opt.entropy_weight if hasattr(opt, 'entropy_weight') else 0.01
    # 基线值，用于减少REINFORCE方差
    baseline = 0
    baseline_decay = opt.baseline_decay if hasattr(opt, 'baseline_decay') else 0.9

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float().to(device)
        target = target.to(device)
        batch_size = input.size(0)

        # 前向传播
        logits = model(input)

        # 计算交叉熵损失（仅用于记录，不用于优化）
        ce_loss = nn.CrossEntropyLoss()(logits, target)
        losses.update(ce_loss.item(), batch_size)

        # 计算当前准确率和奖励
        probs = torch.softmax(logits / temperature, dim=1)
        pred = torch.argmax(probs, dim=1)
        correct = (pred == target).float()
        acc = correct.mean() * 100.0
        accuracies.update(acc.item(), batch_size)

        # 样本级别奖励（1表示分类正确，0表示错误）
        rewards_tensor = correct.detach()
        batch_reward = rewards_tensor.mean().item()
        rewards.update(batch_reward, batch_size)

        # 更新基线值（滑动平均）
        baseline = baseline_decay * baseline + (1 - baseline_decay) * batch_reward

        # 准备优化器
        optimizer.zero_grad()

        # 使用REINFORCE算法优化准确率（不可微目标）
        log_probs = torch.log(probs + 1e-10)

        # 计算策略梯度，使用基线减少方差
        policy_loss = 0
        for i in range(batch_size):
            # 只给出预测类别的策略梯度，并减去基线
            policy_loss -= log_probs[i, pred[i]] * (rewards_tensor[i] - baseline)
        policy_loss = policy_loss / batch_size

        # 添加熵正则化以防止过早收敛到次优策略
        entropy = -torch.sum(probs * log_probs, dim=1).mean()
        loss = policy_loss - entropy_weight * entropy

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % opt.print_freq == 0:
            print(f'Epoch: [{epoch}][{idx}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Accuracy {accuracies.val:.3f} ({accuracies.avg:.3f})\t'
                  f'Reward {rewards.val:.4f} ({rewards.avg:.4f})\t'
                  f'Baseline {baseline:.4f}\t'
                  f'CE Loss {losses.val:.4f} ({losses.avg:.4f})')
            sys.stdout.flush()

    print(f' * Train Acc@1 {accuracies.avg:.3f} Reward {rewards.avg:.4f}')
    return accuracies.avg, losses.avg


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt, logger, min_temp, max_temp):
    """一个epoch的蒸馏训练，支持动态温度范围"""
    for module in module_list:
        module.train()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.distill == 'crd':
            input, target, index, contrast_idx = data
            if torch.cuda.is_available():
                index = index.cuda()
                contrast_idx = contrast_idx.cuda()
        else:
            input, target, index = data

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()

        feat_s, logit_s = model_s(input, is_feat=True)
        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True)
            feat_t = [f.detach() for f in feat_t]

        loss_cls = criterion_cls(logit_s, target)

        current_temp = get_temperature(loss_cls.item(), min_temp=min_temp, max_temp=max_temp,
                                       loss_threshold=opt.temp_loss_threshold)
        current_temp = torch.tensor(current_temp, device=logit_s.device)

        loss_div = criterion_div(logit_s, logit_t, T=current_temp)

        # 修改部分：添加 pkt 支持
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'crd':
            f_s, f_t = feat_s[-1], feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'pkt':
            f_s, f_t = feat_s[-1], feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        else:
            raise NotImplementedError(f'Unsupported distillation method: {opt.distill}')

        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        # 后续代码保持不变
        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % opt.print_freq == 0:
            logger.log_value('batch_temperature', current_temp.item(), epoch * len(train_loader) + idx)
            logger.log_value('batch_loss', loss.item(), epoch * len(train_loader) + idx)
            logger.log_value('batch_acc1', acc1.item(), epoch * len(train_loader) + idx)
            print(f'Epoch: [{epoch}][{idx}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Temperature {current_temp.item():.3f}\t'
                  f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')
            sys.stdout.flush()

    print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
    return top1.avg, losses.avg


def accuracy_target_classes(output, target, target_classes, topk=(1,)):
    """计算指定类别的 top-k 准确率"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    mask = torch.zeros_like(target, dtype=torch.bool)
    for cls in target_classes:
        mask = mask | (target == cls)
    correct = correct[:, mask]

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        total_valid = mask.sum().clamp(min=1)
        res.append(correct_k.mul_(100.0 / total_valid))
    return res


def validate(val_loader, model, criterion, opt, device, target_classes=None):
    """验证过程"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):
            input = input.float().to(device)
            target = target.to(device)

            output = model(input)
            loss = criterion(output, target)

            if target_classes is not None:
                acc1, acc5 = accuracy_target_classes(output, target, target_classes, topk=(1, 5))
            else:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print(f'Test: [{idx}/{len(val_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')

        print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
    return top1.avg, top5.avg, losses.avg
