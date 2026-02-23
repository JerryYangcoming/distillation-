from __future__ import print_function, division

import os
import argparse
import socket
import time
import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt  # Added for visualization

from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed, Connector, Translator, Paraphraser
from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from helper.util import adjust_learning_rate
from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss

try:
    from crd.criterion import CRDLoss
except ImportError:
    CRDLoss = None
    print("警告: 未找到或无法导入 CRDLoss。")

from helper.loops import validate
from helper.pretrain import init

# 动态蒸馏训练模块
class DynamicDistillationTrainer(nn.Module):
    def __init__(self, teacher, student, num_tasks=20):
        super(DynamicDistillationTrainer, self).__init__()
        self.teacher = teacher
        self.student = student
        self.num_tasks = num_tasks

        # 动态温度参数
        self.base_temp = 4.0
        self.task_weights = nn.Parameter(torch.ones(num_tasks))

        # 优化器
        self.optimizer = optim.AdamW([
            {'params': student.parameters()},
            {'params': [self.task_weights]}
        ], lr=1e-3)

    def dynamic_temperature(self, task_id):
        return self.base_temp * (1 + F.softmax(self.task_weights, dim=0)[task_id])

    def compute_loss(self, x, labels, task_id):
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        student_logits = self.student(x)
        ce_loss = F.cross_entropy(student_logits, labels)
        T = self.dynamic_temperature(task_id)
        soft_teacher = F.softmax(teacher_logits / T, dim=1)
        soft_student = F.log_softmax(student_logits / T, dim=1)
        kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T ** 2)
        total_loss = 0.7 * kd_loss + 0.3 * ce_loss
        return total_loss

    def train_epoch(self, dataloader, epoch, logger, opt, task_id):
        self.student.train()
        self.teacher.eval()
        total_loss = 0
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end = time.time()
        for idx, (input, target, _) in enumerate(dataloader):
            input = input.float().to(next(self.student.parameters()).device)
            target = target.to(next(self.student.parameters()).device)

            self.optimizer.zero_grad()
            T = self.dynamic_temperature(task_id)
            loss = self.compute_loss(input, target, task_id)
            loss.backward()
            nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            nn.utils.clip_grad_norm_([self.task_weights], 1.0)
            self.optimizer.step()
            total_loss += loss.item()

            output = self.student(input)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                logger.log_value('batch_temperature', T.item(), epoch * len(dataloader) + idx)
                logger.log_value('batch_loss', loss.item(), epoch * len(dataloader) + idx)
                logger.log_value('batch_acc1', acc1.item(), epoch * len(dataloader) + idx)
                print(f'Epoch: [{epoch}][{idx}/{len(dataloader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Temperature {T.item():.3f}\t'
                      f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')

        avg_loss = total_loss / len(dataloader)
        print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
        return top1.avg, avg_loss

# GradCAM 类
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature = None
        self.gradient = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.feature = output

        def backward_hook(module, grad_in, grad_out):
            self.gradient = grad_out[0]

        layer = dict(self.model.named_modules())[self.target_layer]
        self.hook_handles.append(layer.register_forward_hook(forward_hook))
        self.hook_handles.append(layer.register_backward_hook(backward_hook))

    def __call__(self, input, index=None):
        self.model.eval()
        output = self.model(input)
        if index is None:
            index = output.argmax(dim=1)

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, index.view(-1, 1), 1)
        output.backward(gradient=one_hot, retain_graph=True)

        weights = self.gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.feature).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)  # 避免除以零
        return cam

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

# 保存 Grad-CAM 图像
def save_gradcam(image, cam, save_path):
    image = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    image = (image - image.min()) / (image.max() - image.min())  # 归一化到 [0,1]
    cam = F.interpolate(cam, size=(32, 32), mode='bilinear', align_corners=False).squeeze().cpu().numpy()
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.imshow(cam, alpha=0.5, cmap='jet')
    ax.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# 获取关键层
def get_key_layer(model_name):
    if 'resnet' in model_name:
        depth = int(model_name.replace('resnet', '').split('x')[0])
        n = (depth - 2) // 6
        return f'layer3.{n-1}.conv2'
    else:
        raise ValueError(f"不支持的模型: {model_name}")

# 辅助类和函数
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=100, help='打印频率')
    parser.add_argument('--tb_freq', type=int, default=500, help='TensorBoard 记录频率')
    parser.add_argument('--save_freq', type=int, default=40, help='模型保存频率 (epoch)')
    parser.add_argument('--batch_size', type=int, default=256, help='批量大小')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--epochs', type=int, default=240, help='训练总轮数')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='初始学习率')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='学习率衰减的 epoch 列表')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='学习率衰减率')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD 动量')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='使用的数据集')
    parser.add_argument('--model_s', type=str, default='resnet20',
                        choices=['resnet8', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'],
                        help='学生模型架构')
    parser.add_argument('--path_t', type=str, default='save/models/resnet56_vanilla/ckpt_epoch_240.pth',
                        help='教师模型预训练权重的路径')
    parser.add_argument('--distill', type=str, default='kd',
                        choices=['kd', 'hint', 'attention', 'nst', 'similarity', 'rkd', 'pkt', 'kdsvd',
                                 'correlation', 'vid', 'abound', 'factor', 'fsp', 'crd'],
                        help='蒸馏方法')
    parser.add_argument('-r', '--gamma', type=float, default=1.0, help='分类损失权重')
    parser.add_argument('-a', '--alpha', type=float, default=0.9, help='知识蒸馏散度损失权重')
    parser.add_argument('-b', '--beta', type=float, default=0.0, help='其他蒸馏损失权重')

    opt = parser.parse_args()
    if not os.path.exists(opt.path_t):
        raise ValueError(f"教师模型路径必须存在: {opt.path_t}")
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01
    opt.model_path = './save/student_model'
    opt.tb_path = './save/student_tensorboards'
    opt.lr_decay_epochs = [int(it) for it in opt.lr_decay_epochs.split(',')]
    opt.model_t = get_teacher_name(opt.path_t)
    return opt

def get_teacher_name(model_path):
    try:
        model_folder_name = os.path.basename(os.path.dirname(model_path))
        parts = model_folder_name.split('_')
        if parts[0] == 'wrn':
            return '_'.join(parts[:3])
        elif parts[0] in model_dict:
            return parts[0]
        else:
            print(f"警告: 无法解析教师模型名称，使用 '{model_folder_name}'")
            return model_folder_name
    except Exception as e:
        print(f"解析教师名称出错: {e}，使用默认值 'teacher'")
        return 'teacher'

def load_teacher(model_path, n_cls, device):
    print('==> 正在加载教师模型...')
    model_t_name = get_teacher_name(model_path)
    if model_t_name not in model_dict:
        raise KeyError(f"教师模型 '{model_t_name}' 未在 model_dict 中找到")
    model = model_dict[model_t_name](num_classes=n_cls)
    state_dict = torch.load(model_path, map_location='cpu')['model']
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    print(f"    教师模型 '{model_t_name}' 已加载")
    model = model.to(device)
    model.eval()
    return model

def get_superclass_to_classes():
    superclass_to_classes = {
        'aquatic_mammals': [4, 30, 55, 72, 95],
        'fish': [1, 32, 67, 73, 91],
        'flowers': [54, 62, 70, 82, 92],
        'food_containers': [9, 10, 16, 28, 61],
        'fruit_and_vegetables': [0, 51, 53, 57, 83],
        'household_electrical_devices': [22, 39, 40, 86, 87],
        'household_furniture': [5, 20, 25, 84, 94],
        'insects': [6, 7, 14, 18, 24],
        'large_carnivores': [3, 42, 43, 88, 97],
        'large_man-made_outdoor_things': [12, 17, 37, 68, 76],
        'large_natural_outdoor_scenes': [23, 33, 49, 60, 71],
        'large_omnivores_and_herbivores': [15, 19, 21, 31, 38],
        'medium_mammals': [34, 63, 64, 66, 75],
        'non-insect_invertebrates': [26, 45, 77, 79, 99],
        'people': [2, 11, 35, 46, 98],
        'reptiles': [27, 29, 44, 78, 93],
        'small_mammals': [36, 50, 65, 74, 80],
        'trees': [47, 52, 56, 59, 96],
        'vehicles_1': [8, 13, 48, 58, 90],
        'vehicles_2': [41, 69, 81, 85, 89],
    }
    superclass_order = [
        'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
        'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
        'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
        'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals', 'trees',
        'vehicles_1', 'vehicles_2'
    ]
    ordered_target_class_groups = [superclass_to_classes[superclass] for superclass in superclass_order]
    return ordered_target_class_groups, superclass_order

def main():
    opt = parse_option()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    target_class_groups, superclass_names = get_superclass_to_classes()
    print("根据CIFAR-100超类组织的20个模型训练计划:")
    for i, (classes, superclass) in enumerate(zip(target_class_groups, superclass_names)):
        print(f"模型 {i + 1}: 超类 '{superclass}' - 类别 {classes}")

    for model_index, target_classes in enumerate(target_class_groups):
        superclass_name = superclass_names[model_index]
        print(f"\n===== 训练模型 {model_index + 1}，聚焦于超类 '{superclass_name}'，类别 {target_classes} =====")

        model_name_parts = [
            f'S_{opt.model_s}',
            f'T_{opt.model_t}',
            opt.dataset,
            opt.distill,
            f'SC{model_index}',
            f'r_{opt.gamma}',
            f'a_{opt.alpha}',
        ]
        if opt.beta > 0:
            model_name_parts.append(f'b_{opt.beta}')
        model_name_parts.append(f'Dynamic_model_{model_index + 1}')
        opt.model_name = '_'.join(model_name_parts)

        opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
        opt.save_folder = os.path.join(opt.model_path, opt.model_name)
        os.makedirs(opt.tb_folder, exist_ok=True)
        os.makedirs(opt.save_folder, exist_ok=True)

        logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
        print(f"TensorBoard 日志保存至: {opt.tb_folder}")
        print(f"模型检查点保存至: {opt.save_folder}")

        n_cls = 100
        train_loader, val_loader, _ = get_cifar100_dataloaders(
            batch_size=opt.batch_size, num_workers=opt.num_workers, is_instance=True
        )

        model_t = load_teacher(opt.path_t, n_cls, device)
        model_s = model_dict[opt.model_s](num_classes=n_cls).to(device)

        try:
            teacher_dict = model_t.state_dict()
            student_dict = model_s.state_dict()
            for name, param_t in teacher_dict.items():
                if name in student_dict and param_t.shape == student_dict[name].shape:
                    student_dict[name].copy_(param_t)
        except Exception as e:
            print(f"迁移学习失败: {e}")

        trainer = DynamicDistillationTrainer(model_t, model_s, num_tasks=20).to(device)

        class_weights = torch.ones(n_cls, device=device)
        for cls in target_classes:
            class_weights[cls] = 10000.0
        criterion_cls = nn.CrossEntropyLoss(weight=class_weights).to(device)

        if device.type == 'cuda':
            cudnn.benchmark = True

        teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt, device=device, target_classes=target_classes)
        print(f'教师模型在超类 "{superclass_name}" (类别 {target_classes}) 上的准确率: {teacher_acc:.2f}%')
        logger.log_value('teacher_acc_target', teacher_acc, 0)

        best_acc = 0
        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(epoch, opt, trainer.optimizer)
            current_lr = trainer.optimizer.param_groups[0]['lr']
            print(f"==> Epoch {epoch}/{opt.epochs}, 当前学习率: {current_lr:.6f}")

            train_acc, train_loss = trainer.train_epoch(train_loader, epoch, logger, opt, task_id=model_index)
            test_acc, test_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt, device=device, target_classes=target_classes)

            print(f"    当前 test_acc: {test_acc:.2f}%，best_acc: {best_acc:.2f}%")

            logger.log_value('train_acc', train_acc, epoch)
            logger.log_value('train_loss', train_loss, epoch)
            logger.log_value('test_acc_target', test_acc, epoch)
            logger.log_value('test_loss', test_loss, epoch)
            logger.log_value('test_acc_top5_target', test_acc_top5, epoch)
            logger.log_value('learning_rate', current_lr, epoch)

            if test_acc >= best_acc:
                best_acc = test_acc
                state = {
                    'epoch': epoch,
                    'model': model_s.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': trainer.optimizer.state_dict(),
                    'trainer': trainer.state_dict()
                }
                save_file = os.path.join(opt.save_folder, f'{opt.model_s}_best_SC{model_index}.pth')
                try:
                    torch.save(state, save_file)
                    print(f'    新的最佳模型已保存，准确率: {best_acc:.2f}%，路径: {save_file}')
                except Exception as e:
                    print(f'    保存最佳模型时出错: {e}')

            if epoch % opt.save_freq == 0:
                state = {
                    'epoch': epoch,
                    'model': model_s.state_dict(),
                    'accuracy': test_acc,
                    'optimizer': trainer.optimizer.state_dict(),
                    'trainer': trainer.state_dict()
                }
                save_file = os.path.join(opt.save_folder, f'ckpt_epoch_{epoch}_SC{model_index}.pth')
                try:
                    torch.save(state, save_file)
                    print(f'    检查点已保存至 {save_file}')
                except Exception as e:
                    print(f'    保存检查点时出错: {e}')

        state = {
            'epoch': opt.epochs,
            'model': model_s.state_dict(),
            'accuracy': test_acc,
            'optimizer': trainer.optimizer.state_dict(),
            'trainer': trainer.state_dict(),
            'opt': opt
        }
        save_file = os.path.join(opt.save_folder, f'{opt.model_s}_last_SC{model_index}.pth')
        try:
            torch.save(state, save_file)
            print(f'    最后一个 epoch 的模型状态已保存至 {save_file}')
        except Exception as e:
            print(f'    保存最后一个模型时出错: {e}')

        print(f'==> 超类 "{superclass_name}" 模型训练完成。最佳验证准确率: {best_acc:.2f}%')

        accuracy_file = os.path.join(opt.save_folder, f'accuracy_SC{model_index}.txt')
        with open(accuracy_file, 'w') as f:
            f.write(f"超类: {superclass_name}\n")
            f.write(f"类别: {target_classes}\n")
            f.write(f"Best accuracy: {best_acc:.2f}%\n")
        print(f'    最佳准确率已记录至 {accuracy_file}')

        # Grad-CAM 可视化（已修复）
        if 'resnet' in opt.model_s:
            print("==> 生成 Grad-CAM 可视化...")
            target_layer = get_key_layer(opt.model_s)
            print(f"    模型 {opt.model_s} 的关键层: {target_layer}")
            gradcam = GradCAM(model_s, target_layer)

            for i, (inputs, targets) in enumerate(val_loader):
                if i >= 10:  # 只处理前 10 张图像
                    break
                input = inputs[0:1].to(device)
                target = targets[0:1].to(device)
                cam = gradcam(input, target)
                save_path = os.path.join(opt.save_folder, f'gradcam_{i}_SC{model_index}.png')
                save_gradcam(input, cam, save_path)
                print(f'    Grad-CAM 图像已保存至 {save_path}')

            gradcam.remove_hooks()
        else:
            print(f"    跳过非 ResNet 模型 {opt.model_s} 的 Grad-CAM 可视化")

if __name__ == '__main__':
    main()
