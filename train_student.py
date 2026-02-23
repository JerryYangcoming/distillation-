"""
the general training framework
"""

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
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample

from helper.util import adjust_learning_rate

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss
from crd.criterion import CRDLoss

from helper.loops import train_distill as train, validate
from helper.pretrain import init

# 定义命令行参数的解析函数
def parse_option():
    """
    解析命令行传入的参数，设置默认值，并返回配置选项。
    """
    hostname = socket.gethostname()  # 获取当前主机的名称

    parser = argparse.ArgumentParser('argument for training')  # 创建解析器

    # 基本训练参数
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')  # 打印频率
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')  # TensorBoard记录频率
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')  # 模型保存频率
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')  # 批量大小
    parser.add_argument('--num_workers', type=int, default=1, help='num of workers to use')  # 数据加载线程数
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')  # 训练轮数
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')  # 初始训练轮数

    # 优化参数
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')  # 学习率
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')  # 学习率衰减点
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')  # 学习率衰减比率
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')  # 权重衰减（正则化）
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')  # 动量

    # 数据集相关
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')  # 数据集名称

    # 模型相关
    parser.add_argument('--model_s', type=str, default='resnet32', choices=[...], help='student model')  # 学生模型
    parser.add_argument('--path_t', type=str, default='save/models/resnet110_vanilla/ckpt_epoch_240.pth', help='teacher model snapshot')  # 教师模型路径

    # 蒸馏方法相关
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', ...], help='distillation method')
    parser.add_argument('--trial', type=str, default='1', help='trial id')  # 实验编号
    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')  # 分类损失权重
    parser.add_argument('-a', '--alpha', type=float, default=0.9, help='weight balance for KD')  # 蒸馏平衡参数
    parser.add_argument('-b', '--beta', type=float, default=0.1, help='weight balance for other losses')  # 其他损失的平衡参数

    # KL蒸馏参数
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')  # 蒸馏温度

    # CRD蒸馏参数
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')  # 特征维度
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])  # NCE模式
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')  # 负样本数量
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')  # Softmax温度
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')  # 动量

    # Hint蒸馏参数
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])  # Hint层选择

    # 解析参数
    opt = parser.parse_args()

    # 根据模型类型设置默认学习率
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # 设置路径
    opt.model_path = './save/student_model'
    opt.tb_path = './save/student_tensorboards'

    # 解析学习率衰减点
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = [int(it) for it in iterations]

    # 根据教师模型路径提取模型名称
    opt.model_t = get_teacher_name(opt.path_t)

    # 构建实验保存文件夹路径
    opt.model_name = 'S:{}_T:{}_{}_{}_r:{}_a:{}_b:{}_{}'.format(
        opt.model_s,
        opt.model_t,
        opt.dataset,
        opt.distill,
        opt.gamma,
        opt.alpha if opt.alpha is not None else 0,
        opt.beta if opt.beta is not None else 0,
        opt.trial
    )

    # 防止非法字符
    opt.model_name = opt.model_name.replace(":", "_")

    # TensorBoard 文件夹路径
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder, exist_ok=True)

    # 模型保存路径
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder, exist_ok=True)

    os.makedirs(opt.tb_folder, exist_ok=True)  # 确保TensorBoard目录存在
    os.makedirs(opt.save_folder, exist_ok=True)  # 确保模型保存目录存在

    return opt  # 返回解析后的参数对象


def get_teacher_name(model_path):
    """
    从教师模型的文件路径中提取模型的名称。

    参数:
        model_path (str): 教师模型的文件路径。

    返回:
        str: 教师模型的名称。
    """
    # 分割路径，提取模型目录名，并进一步分割目录名以获取模型名称部分
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':  # 如果模型名称不是以 "wrn" (Wide ResNet) 开头
        return segments[0]  # 返回第一个分割部分作为模型名称
    else:  # 如果模型名称是 Wide ResNet
        return segments[0] + '_' + segments[1] + '_' + segments[2]  # 拼接多个部分以获得完整模型名称


def load_teacher(model_path, n_cls):
    """
    加载教师模型的权重。

    参数:
        model_path (str): 教师模型的文件路径。
        n_cls (int): 数据集中类别的数量（用于初始化模型的输出层）。

    返回:
        nn.Module: 加载了预训练权重的教师模型。
    """
    print('==> loading teacher model')  # 打印提示信息，表示开始加载教师模型
    model_t = get_teacher_name(model_path)  # 根据模型路径解析出模型名称

    # model_dict[model_t](num_classes=n_cls) 相当于调用 ResNet110(num_classes=n_cls)，
    # 从而返回一个 PyTorch nn.Module 对象（即真正的模型实例）。
    # 这一行代码最终构建出一个**指定类别数（n_cls）**的神经网络模型，并将其赋给变量 model。
    model = model_dict[model_t](num_classes=n_cls)

    # eg:model_path:'save/models/resnet110_vanilla/ckpt_epoch_240.pth'
    # torch.load(model_path) 用于从文件 model_path 中加载保存的模型数据。
    # 这通常返回一个字典，其中包含多个键值对（例如 'model'、'epoch'、'best_acc' 等）。
    # model.load_state_dict(...) 是 PyTorch 中 nn.Module 的方法，
    # 用于将一个 state_dict（一个字典，其中键对应模型参数名，值对应模型参数张量）加载到模型实例 model 中。
    # 从加载的数据中取出实际的模型参数（通过 ['model'] 索引），
    # 然后使用 model.load_state_dict(...) 方法将这些参数赋值到当前构造的教师模型实例中。
    model.load_state_dict(torch.load(model_path)['model'])

    print('==> done')  # 打印提示信息，表示模型加载完成
    return model  # 返回加载了权重的教师模型



def main():
    best_acc = 0  # 初始化最佳准确率为0，用于保存模型过程中跟踪最高准确率

    opt = parse_option()  # 解析命令行参数，获取配置项

    # 初始化 TensorBoard 日志记录器,（每 2 秒）将训练过程中产生的日志数据保存到 opt.tb_folder 指定的目录中
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # 数据加载器
    if opt.dataset == 'cifar100':  # 如果数据集是 CIFAR-100
        if opt.distill in ['crd']:  # 如果使用对比学习蒸馏（CRD）
            # 使用支持对比学习的特殊数据加载器
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode,
                                                                               pin_memory=True,)
        else:
            # 使用普通的 CIFAR-100 数据加载器，已经默认开启instance
            train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
        n_cls = 100  # CIFAR-100 数据集有 100 个类别
    else:
        raise NotImplementedError(opt.dataset)  # 如果数据集未实现，抛出异常

    # 模型加载
    model_t = load_teacher(opt.path_t, n_cls)  # 加载教师模型
    model_s = model_dict[opt.model_s](num_classes=n_cls)  # 初始化学生模型

    # 测试模型结构是否一致
    # 生成一个形状为 (2, 3, 32, 32) 的随机数据，用于模拟图像数据
    # （例如2张 CIFAR-100 数据集的图像尺寸是 32x32，且有 3 个颜色通道）。
    data = torch.randn(2, 3, 32, 32)  # 创建随机输入数据
    # 将教师模型和学生模型设置为评估模式，这样它们就不会执行诸如 dropout 等训练时特有的操作。
    model_t.eval()  # 设置教师模型为评估模式
    model_s.eval()  # 设置学生模型为评估模式
    # 这种写法利用了 Python 的序列拆包（tuple unpacking）语法，
    # 将返回值的第一项赋给 feat_t（或 feat_s），而第二项不关心直接用 _ 丢弃。
    feat_t, _ = model_t(data, is_feat=True)  # 获取教师模型的中间特征
    feat_s, _ = model_s(data, is_feat=True)  # 获取学生模型的中间特征

    # 初始化模块列表
    module_list = nn.ModuleList([])  # 保存所有模块的列表
    module_list.append(model_s)  # 将学生模型加入模块列表

    trainable_list = nn.ModuleList([])  # 保存可训练模块的列表
    trainable_list.append(model_s)  # 将学生模型加入可训练列表

    # 初始化损失函数
    criterion_cls = nn.CrossEntropyLoss()  # 分类损失函数
    criterion_div = DistillKL(opt.kd_T)  # KL散度损失函数

    # 根据蒸馏方法选择适当的损失函数
    if opt.distill == 'kd':  # 基础知识蒸馏,T->temperature
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint':  # Hint 蒸馏
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)  # 回归模块
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'crd':  # 对比蒸馏
        opt.s_dim = feat_s[-1].shape[1]  # 学生最后一层特征维度
        opt.t_dim = feat_t[-1].shape[1]  # 教师最后一层特征维度
        opt.n_data = n_data  # 样本总数
        criterion_kd = CRDLoss(opt)  # 对比损失函数
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'attention':  # Attention 蒸馏
        criterion_kd = Attention()
    elif opt.distill == 'nst':  # NST 蒸馏
        criterion_kd = NSTLoss()
    elif opt.distill == 'similarity':  # 相似性蒸馏
        criterion_kd = Similarity()
    elif opt.distill == 'rkd':  # RKD 蒸馏
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt':  # PKT 蒸馏
        criterion_kd = PKT()
    elif opt.distill == 'kdsvd':  # KDSVD 蒸馏
        criterion_kd = KDSVD()
    elif opt.distill == 'correlation':  # 相关性蒸馏
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)  # 学生特征嵌入
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)  # 教师特征嵌入
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'vid':  # VID 蒸馏
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        trainable_list.append(criterion_kd)  # 加入可训练列表
    elif opt.distill == 'abound':  # ABound 蒸馏
        s_shapes = [f.shape for f in feat_s[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes)  # 连接模块
        # 初始化阶段训练
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(connector)
        init_trainable_list.append(model_s.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        module_list.append(connector)
    elif opt.distill == 'factor':  # Factor 蒸馏
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)  # 转换模块
        translator = Translator(s_shape, t_shape)
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, logger, opt)
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    elif opt.distill == 'fsp':  # FSP 蒸馏
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
    elif opt.distill == 'crd_kd':  # 同时使用 CRD 和 KD
        # 设置 CRD 所需的参数
        opt.s_dim = feat_s[-1].shape[1]  # 学生最后一层特征维度
        opt.t_dim = feat_t[-1].shape[1]  # 教师最后一层特征维度
        opt.n_data = n_data  # 数据样本总数
        # 构造 CRD 损失
        criterion_kd = CRDLoss(opt)
        # 添加 CRD 中需要训练的嵌入层到模块列表和训练列表
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    else:
        raise NotImplementedError(opt.distill)

    # 损失函数列表
    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # 分类损失
    criterion_list.append(criterion_div)  # KL散度损失
    criterion_list.append(criterion_kd)  # 蒸馏损失

    # 优化器
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # 最后添加教师模型，方便后续查找
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # 验证教师模型的准确率
    # 注：validate未细读
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)

    # 训练与验证循环
    for epoch in range(1, opt.epochs + 1):
        # 动态调整学习率
        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()  # 记录训练开始时间
        # train未细读
        train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()  # 记录训练结束时间
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # 记录训练指标到 TensorBoard
        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        # 验证学生模型
        test_acc, test_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        # 记录验证指标到 TensorBoard
        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        # 定期保存模型
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # 打印最终最佳准确率
    print('best accuracy:', best_acc)

    # 保存最后的学生模型
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
