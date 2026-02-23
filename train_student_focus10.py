from __future__ import print_function

import os
import argparse
import socket
import time
import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

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

from helper.loops import train_distill as train, validate
from helper.pretrain import init

def parse_option():
    """解析命令行参数"""
    hostname = socket.gethostname()
    parser = argparse.ArgumentParser('argument for training')

    # 基本参数
    parser.add_argument('--print_freq', type=int, default=100, help='打印频率')
    parser.add_argument('--tb_freq', type=int, default=500, help='TensorBoard 记录频率')
    parser.add_argument('--save_freq', type=int, default=40, help='模型保存频率 (epoch)')
    parser.add_argument('--batch_size', type=int, default=256, help='批量大小')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--epochs', type=int, default=240, help='训练总轮数')

    # 优化器参数
    parser.add_argument('--learning_rate', type=float, default=0.05, help='初始学习率')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='学习率衰减的 epoch 列表')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='学习率衰减率')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='权重衰减')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD 动量')

    # 数据集和模型参数
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='使用的数据集')
    parser.add_argument('--model_s', type=str, default='wrn_40_1',
                        choices=['resnet8', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'],
                        help='学生模型架构')
    parser.add_argument('--path_t', type=str, default='save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth', help='教师模型预训练权重的路径')

    # 蒸馏参数
    parser.add_argument('--distill', type=str, default='kd',
                        choices=['kd', 'hint', 'attention', 'nst', 'similarity', 'rkd', 'pkt', 'kdsvd',
                                 'correlation', 'vid', 'abound', 'factor', 'fsp', 'crd'],
                        help='蒸馏方法')
    parser.add_argument('-r', '--gamma', type=float, default=1.0, help='分类损失权重')
    parser.add_argument('-a', '--alpha', type=float, default=0.9, help='知识蒸馏散度损失权重')
    parser.add_argument('-b', '--beta', type=float, default=0.0, help='其他蒸馏损失权重')

    # 温度参数
    parser.add_argument('--kd_T_default', type=float, default=4.0, help='默认温度')
    parser.add_argument('--use_dynamic_temp', action='store_true', help='启用动态温度')
    parser.add_argument('--min_temp', type=float, default=1.0, help='动态温度最小值初始值')
    parser.add_argument('--max_temp', type=float, default=5.0, help='动态温度最大值初始值')
    parser.add_argument('--temp_loss_threshold', type=float, default=0.5, help='动态温度的损失阈值')

    # 温度调整参数
    parser.add_argument('--temp_adjust_factor', type=float, default=0.95, help='温度范围调整因子')
    parser.add_argument('--min_temp_min', type=float, default=1.0, help='min_temp 的最小允许值')
    parser.add_argument('--min_temp_max', type=float, default=5.0, help='min_temp 的最大允许值')
    parser.add_argument('--max_temp_min', type=float, default=4.0, help='max_temp 的最小允许值')
    parser.add_argument('--max_temp_max', type=float, default=20.0, help='max_temp 的最大允许值')

    # Hint 蒸馏参数
    parser.add_argument('--hint_layer', type=int, default=2, choices=[0, 1, 2, 3, 4], help='Hint 层选择')

    # CRD 蒸馏参数
    parser.add_argument('--feat_dim', default=128, type=int, help='特征维度')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'], help='NCE 模式')
    parser.add_argument('--nce_k', default=16384, type=int, help='NCE 负样本数量')
    parser.add_argument('--nce_t', default=0.07, type=float, help='Softmax 温度')
    parser.add_argument('--nce_m', default=0.5, type=float, help='非参数更新动量')

    # VID 参数
    parser.add_argument('--vid_mid_channel', type=int, default=128, help='VID 中间通道数')

    opt = parser.parse_args()

    # 校验教师模型路径
    if not os.path.exists(opt.path_t):
        raise ValueError(f"教师模型路径必须存在: {opt.path_t}")

    # 为特定模型调整学习率
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # 设置路径
    opt.model_path = './save/student_model'
    opt.tb_path = './save/student_tensorboards'
    opt.lr_decay_epochs = [int(it) for it in opt.lr_decay_epochs.split(',')]
    opt.model_t = get_teacher_name(opt.path_t)

    # 生成 trial 字符串
    trial_parts = [
        str(opt.min_temp),
        str(opt.max_temp),
        str(opt.temp_loss_threshold),
        str(opt.temp_adjust_factor),
        str(opt.min_temp_min),
        str(opt.min_temp_max),
        str(opt.max_temp_min),
        str(opt.max_temp_max)
    ]
    opt.trial = ','.join(trial_parts)

    return opt

def get_teacher_name(model_path):
    """从教师模型路径提取模型名称"""
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
    """加载教师模型"""
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

def main():
    opt = parse_option()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 定义10个目标类别组
    target_class_groups = [
        list(range(0, 10)),    # 0-9
        list(range(10, 20)),   # 10-19
        list(range(20, 30)),   # 20-29
        list(range(30, 40)),   # 30-39
        list(range(40, 50)),   # 40-49
        list(range(50, 60)),   # 50-59
        list(range(60, 70)),   # 60-69
        list(range(70, 80)),   # 70-79
        list(range(80, 90)),   # 80-89
        list(range(90, 100)),  # 90-99
    ]

    # 循环训练10个模型
    for model_index, target_classes in enumerate(target_class_groups):
        print(f"\n===== 训练模型 {model_index+1}，聚焦于类别 {target_classes} =====")

        # 设置模型名称和路径
        model_name_parts = [
            f'S_{opt.model_s}',
            f'T_{opt.model_t}',
            opt.dataset,
            opt.distill,
            f'r_{opt.gamma}',
            f'a_{opt.alpha}',
        ]
        if opt.beta > 0:
            model_name_parts.append(f'b_{opt.beta}')
        if opt.use_dynamic_temp:
            temp_info = f'DynT_{opt.min_temp}-{opt.max_temp}_Thres{opt.temp_loss_threshold}'
        else:
            temp_info = f'FixedT_{opt.kd_T_default}'
        model_name_parts.extend([temp_info, f'trial_{opt.trial}', f'model_{model_index+1}'])
        opt.model_name = '_'.join(model_name_parts)

        opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
        opt.save_folder = os.path.join(opt.model_path, opt.model_name)
        os.makedirs(opt.tb_folder, exist_ok=True)
        os.makedirs(opt.save_folder, exist_ok=True)

        logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
        print(f"TensorBoard 日志保存至: {opt.tb_folder}")
        print(f"模型检查点保存至: {opt.save_folder}")

        # 数据加载
        n_cls = 100
        if opt.dataset == 'cifar100':
            if opt.distill == 'crd':
                if CRDLoss is None:
                    raise ImportError("CRD 需要 crd 模块")
                train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(
                    batch_size=opt.batch_size, num_workers=opt.num_workers, k=opt.nce_k, mode=opt.mode
                )
            else:
                train_loader, val_loader, n_data = get_cifar100_dataloaders(
                    batch_size=opt.batch_size, num_workers=opt.num_workers, is_instance=True
                )
        else:
            raise NotImplementedError(f"数据集 '{opt.dataset}' 不支持")

        # 模型加载
        model_t = load_teacher(opt.path_t, n_cls, device)
        model_s = model_dict[opt.model_s](num_classes=n_cls).to(device)

        # 知识迁移初始化
        try:
            teacher_dict = model_t.state_dict()
            student_dict = model_s.state_dict()
            for name, param_t in teacher_dict.items():
                if name in student_dict and param_t.shape == student_dict[name].shape:
                    student_dict[name].copy_(param_t)
        except Exception as e:
            print(f"迁移学习失败: {e}")

        # 测试模型结构
        data = torch.randn(2, 3, 32, 32).to(device)
        model_t.eval()
        model_s.eval()
        feat_t, _ = model_t(data, is_feat=True)
        feat_s, _ = model_s(data, is_feat=True)

        # 初始化模块列表
        module_list = nn.ModuleList([model_s])
        trainable_list = nn.ModuleList([model_s])

        # 创建加权损失函数
        class_weights = torch.ones(n_cls, device=device)
        for cls in target_classes:
            class_weights[cls] = 10.0  # 目标类别权重设为10
        criterion_cls = nn.CrossEntropyLoss(weight=class_weights).to(device)
        criterion_div = DistillKL(opt.kd_T_default).to(device)
        criterion_kd = None

        # 设置蒸馏损失
        if opt.distill == 'kd':
            criterion_kd = DistillKL(opt.kd_T_default).to(device)
        elif opt.distill == 'hint':
            criterion_kd = HintLoss().to(device)
            regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape).to(device)
            module_list.append(regress_s)
            trainable_list.append(regress_s)
        elif opt.distill == 'crd':
            if CRDLoss is None:
                raise ImportError("CRD 需要 crd 模块")
            opt.s_dim = feat_s[-1].shape[1]
            opt.t_dim = feat_t[-1].shape[1]
            opt.n_data = n_data
            criterion_kd = CRDLoss(opt).to(device)
            module_list.append(criterion_kd.embed_s)
            module_list.append(criterion_kd.embed_t)
            trainable_list.append(criterion_kd.embed_s)
            trainable_list.append(criterion_kd.embed_t)
        elif opt.distill == 'attention':
            criterion_kd = Attention().to(device)
        elif opt.distill == 'nst':
            criterion_kd = NSTLoss().to(device)
        elif opt.distill == 'similarity':
            criterion_kd = Similarity().to(device)
        elif opt.distill == 'rkd':
            criterion_kd = RKDLoss().to(device)
        elif opt.distill == 'pkt':
            criterion_kd = PKT().to(device)
        elif opt.distill == 'kdsvd':
            criterion_kd = KDSVD().to(device)
        elif opt.distill == 'correlation':
            criterion_kd = Correlation().to(device)
            embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim).to(device)
            embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim).to(device)
            module_list.append(embed_s)
            module_list.append(embed_t)
            trainable_list.append(embed_s)
            trainable_list.append(embed_t)
        elif opt.distill == 'vid':
            s_n = [f.shape[1] for f in feat_s[1:-1]]
            t_n = [f.shape[1] for f in feat_t[1:-1]]
            criterion_kd = nn.ModuleList([VIDLoss(s, opt.vid_mid_channel, t).to(device) for s, t in zip(s_n, t_n)])
            trainable_list.append(criterion_kd)
        elif opt.distill == 'abound':
            s_shapes = [f.shape for f in feat_s[1:-1]]
            t_shapes = [f.shape for f in feat_t[1:-1]]
            connector = Connector(s_shapes, t_shapes).to(device)
            init_trainable_list = nn.ModuleList([connector, model_s.get_feat_modules()])
            criterion_kd = ABLoss(len(feat_s[1:-1])).to(device)
            init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
            module_list.append(connector)
        elif opt.distill == 'factor':
            s_shape = feat_s[-2].shape
            t_shape = feat_t[-2].shape
            paraphraser = Paraphraser(t_shape).to(device)
            translator = Translator(s_shape, t_shape).to(device)
            init_trainable_list = nn.ModuleList([paraphraser])
            criterion_init = nn.MSELoss().to(device)
            init(model_s, model_t, init_trainable_list, criterion_init, train_loader, logger, opt)
            criterion_kd = FactorTransfer().to(device)
            module_list.append(translator)
            module_list.append(paraphraser)
            trainable_list.append(translator)
        elif opt.distill == 'fsp':
            s_shapes = [s.shape for s in feat_s[:-1]]
            t_shapes = [t.shape for t in feat_t[:-1]]
            criterion_kd = FSP(s_shapes, t_shapes).to(device)
            init_trainable_list = nn.ModuleList([model_s.get_feat_modules()])
            init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        else:
            raise NotImplementedError(f"蒸馏方法 '{opt.distill}' 不支持")

        criterion_list = nn.ModuleList([criterion_cls, criterion_div, criterion_kd]).to(device)

        # 优化器
        optimizer = optim.SGD(trainable_list.parameters(),
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)

        # 添加教师模型
        module_list.append(model_t)

        if device.type == 'cuda':
            module_list.cuda()
            criterion_list.cuda()
            cudnn.benchmark = True

        # 验证教师模型（仅关注目标类别）
        teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt, device=device, target_classes=target_classes)
        print(f'教师模型在目标类别上的准确率: {teacher_acc:.2f}%')
        logger.log_value('teacher_acc_target', teacher_acc, 0)

        # 初始化温度范围
        min_temp = opt.min_temp
        max_temp = opt.max_temp
        best_acc = 0

        # 训练循环
        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(epoch, opt, optimizer)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"==> Epoch {epoch}/{opt.epochs}, 当前学习率: {current_lr:.6f}")

            train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt, logger, min_temp, max_temp)
            test_acc, test_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt, device=device, target_classes=target_classes)

            print(f"    当前 test_acc: {test_acc:.2f}%，best_acc: {best_acc:.2f}%")

            logger.log_value('train_acc', train_acc, epoch)
            logger.log_value('train_loss', train_loss, epoch)
            logger.log_value('test_acc_target', test_acc, epoch)
            logger.log_value('test_loss', test_loss, epoch)
            logger.log_value('test_acc_top5_target', test_acc_top5, epoch)
            logger.log_value('learning_rate', current_lr, epoch)
            logger.log_value('min_temp', min_temp, epoch)
            logger.log_value('max_temp', max_temp, epoch)

            # 调整温度范围
            if test_acc > best_acc:
                best_acc = test_acc
                new_min_temp = min_temp * opt.temp_adjust_factor
                new_max_temp = max_temp * opt.temp_adjust_factor
                min_temp = max(new_min_temp, opt.min_temp_min)
                max_temp = max(new_max_temp, opt.max_temp_min)
            else:
                new_min_temp = min_temp / opt.temp_adjust_factor
                new_max_temp = max_temp / opt.temp_adjust_factor
                min_temp = min(new_min_temp, opt.min_temp_max)
                max_temp = min(new_max_temp, opt.max_temp_max)

            # 保存最佳模型
            if test_acc >= best_acc:
                best_acc = test_acc
                state = {'epoch': epoch, 'model': model_s.state_dict(), 'best_acc': best_acc, 'optimizer': optimizer.state_dict()}
                save_file = os.path.join(opt.save_folder, f'{opt.model_s}_best_model_{model_index+1}.pth')
                try:
                    torch.save(state, save_file)
                    print(f'    新的最佳模型已保存，准确率: {best_acc:.2f}%，路径: {save_file}')
                except Exception as e:
                    print(f'    保存最佳模型时出错: {e}')

            # 定期保存检查点
            if epoch % opt.save_freq == 0:
                state = {'epoch': epoch, 'model': model_s.state_dict(), 'accuracy': test_acc, 'optimizer': optimizer.state_dict()}
                save_file = os.path.join(opt.save_folder, f'ckpt_epoch_{epoch}_model_{model_index+1}.pth')
                try:
                    torch.save(state, save_file)
                    print(f'    检查点已保存至 {save_file}')
                except Exception as e:
                    print(f'    保存检查点时出错: {e}')

        # 保存最后一个 epoch 的模型
        state = {'epoch': opt.epochs, 'model': model_s.state_dict(), 'accuracy': test_acc, 'optimizer': optimizer.state_dict(), 'opt': opt}
        save_file = os.path.join(opt.save_folder, f'{opt.model_s}_last_model_{model_index+1}.pth')
        try:
            torch.save(state, save_file)
            print(f'    最后一个 epoch 的模型状态已保存至 {save_file}')
        except Exception as e:
            print(f'    保存最后一个模型时出错: {e}')

        print(f'==> 模型 {model_index+1} 训练完成。最佳验证准确率: {best_acc:.2f}%')

        # 将最佳准确率记录到TXT文件
        accuracy_file = os.path.join(opt.save_folder, f'accuracy_model_{model_index+1}.txt')
        with open(accuracy_file, 'w') as f:
            f.write(f"Best accuracy on target classes: {best_acc:.2f}%\n")
        print(f'    最佳准确率已记录至 {accuracy_file}')

if __name__ == '__main__':
    main()
