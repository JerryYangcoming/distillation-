from __future__ import absolute_import

'''
ResNet for CIFAR dataset.
代码移植自 Facebook 的 fb.resnet.torch 和 PyTorch 官方 ResNet 实现，作者为 YANG, Wei。
'''
import torch.nn as nn
import torch.nn.functional as F
import math

# 定义可以导出的模块列表
__all__ = ['resnet']


def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 卷积层，带有 padding，创建一个 3x3 的二维卷积层
    参数:
        in_planes (int): 输入通道数
        out_planes (int): 输出通道数
        stride (int): 卷积步幅
    返回:
        nn.Conv2d: 带有 3x3 卷积核的卷积层
    """
    return nn.Conv2d(
        in_planes,  # 输入通道数，RGB 图像的通道数为 3
        out_planes,  # 输出通道数，等同于卷积核的数量
        kernel_size=3,  # 卷积核大小为 3x3
        stride=stride,  # 步幅大小，默认值为 1
        padding=1,  # 填充大小，保证输出特征图的空间尺寸与输入相同
        bias=False  # 不使用偏置项
    )


class BasicBlock(nn.Module):
    """
    ResNet 的基础模块（BasicBlock）。
    包含两个 3x3 卷积层，支持降采样操作，并添加残差连接。
    """
    expansion = 1  # 通道数扩展比例，BasicBlock 不改变通道数

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False):
        """
        初始化 BasicBlock。
        参数:
            inplanes (int): 输入通道数
            planes (int): 输出通道数
            stride (int): 步幅，用于降采样
            downsample (nn.Module): 降采样模块
            is_last (bool): 是否为最后一个 block
        """
        super(BasicBlock, self).__init__()
        self.is_last = is_last

        # 第一个卷积核从 3 通道映射到 planes 通道。
        self.conv1 = conv3x3(inplanes, planes, stride)  # 第一个卷积层
        self.bn1 = nn.BatchNorm2d(planes)  # 批归一化层
        self.relu = nn.ReLU(inplace=True)  # ReLU 激活

        self.conv2 = conv3x3(planes, planes)  # 第二个卷积层
        self.bn2 = nn.BatchNorm2d(planes)
        #如果输入的尺寸或通道数与输出不匹配，就需要使用 downsample 模块对输入进行调整，使得它可以与主分支的输出相加。
        self.downsample = downsample  # 残差连接中的降采样
        self.stride = stride

    def forward(self, x):
        """
        前向传播。
        参数:
            x (Tensor): 输入张量
        返回:
            Tensor: 输出张量
        """
        residual = x  # 保存输入作为残差
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:  # 如果有降采样操作
            residual = self.downsample(x)  # 对残差进行降采样

        out += residual  # 残差连接
        preact = out  # 激活前的值
        out = F.relu(out)  # 激活后的值
        if self.is_last:
            return out, preact  # 如果是最后一个 block，返回激活前和激活后的值
        else:
            # 不是最后一个只返回激活后的值
            return out


class Bottleneck(nn.Module):
    """
    ResNet 的瓶颈模块（Bottleneck）。
    使用 1x1 和 3x3 卷积进行降维和升维。
    """
    expansion = 4  # 通道数扩展比例

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False):
        """
        初始化 Bottleneck。
        参数与 BasicBlock 类似。
        通过 1x1 卷积降低特征图的通道数，减少计算量；
        利用 3x3 卷积提取空间特征；
        再通过 1x1 卷积恢复通道数，实现通道的扩展（通道数通常扩展 4 倍）
        """
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  # 1x1 卷积降维
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)  # 3x3 卷积
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)  # 1x1 卷积升维
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        前向传播。
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    """
    ResNet 主体结构，用于动态生成不同深度的网络。
    """
    def __init__(self, depth, num_filters, block_name='BasicBlock', num_classes=10):
        """
        初始化 ResNet。
        参数:
            depth (int): 网络深度
            num_filters (list): 每个阶段的通道数
            block_name (str): 模块类型（BasicBlock 或 Bottleneck）
            num_classes (int): 分类类别数
        """
        super(ResNet, self).__init__()
        # 根据 block_name 确定使用的模块类型和深度约束
        if block_name.lower() == 'basicblock':
            # 如果不符合，程序会抛出 AssertionError
            assert (depth - 2) % 6 == 0, 'BasicBlock 的深度应为 6n+2，例如 20, 32, 44 等'
            # 这个 n 用来控制每个阶段的残差块数目。具体地，n 决定每个阶段堆叠多少个 BasicBlock。
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'Bottleneck 的深度应为 9n+2，例如 29, 47, 56 等'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name 应为 BasicBlock 或 Bottleneck')

        self.inplanes = num_filters[0]  # 输入通道数
        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, padding=1, bias=False)  # 初始卷积层
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        # 创建三个阶段的卷积层
        self.layer1 = self._make_layer(block, num_filters[1], n, stride=1)
        self.layer2 = self._make_layer(block, num_filters[2], n, stride=2)
        self.layer3 = self._make_layer(block, num_filters[3], n, stride=2)
        self.avgpool = nn.AvgPool2d(8)  # 全局平均池化
        self.fc = nn.Linear(num_filters[3] * block.expansion, num_classes)  # 全连接层

        # 初始化权重
        # 遍历所有子模块，对卷积层使用 Kaiming 正态初始化，对归一化层则设置常数值
        for m in self.modules():
            # 当前模块 m 是否是二维卷积层（nn.Conv2d）。
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # 检查当前模块 m 是否是批归一化层（nn.BatchNorm2d）或分组归一化层（nn.GroupNorm）。
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        构建 ResNet 的一个阶段。
        参数:
            block (class): 模块类型（`BasicBlock` 或 `Bottleneck`）
            planes (int): 当前阶段每个残差块输出的通道数。
            blocks (int): 当前阶段的残差块数量。
            stride (int): 步幅，用于控制空间下采样。
        返回:
            nn.Sequential: 一个由多个残差块组成的阶段。
        """
        downsample = None  # 用于降采样的模块，默认为 None

        # 检查是否需要降采样
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 如果步幅不为 1 或通道数不匹配，则需要降采样。
            # nn.Sequential 是 PyTorch 中的一个容器模块，
            # 它按顺序将多个子模块（如卷积层、激活函数等）串联起来，使得每个输入都按顺序通过各个子模块。
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,  # 输入通道数
                    planes * block.expansion,  # 输出通道数（由 block 的扩展因子决定）
                    kernel_size=1,  # 1x1 卷积只改变通道数，不改变空间尺寸
                    stride=stride,  # 使用步幅进行下采样
                    bias=False  # 不使用偏置
                ),
                nn.BatchNorm2d(planes * block.expansion)  # 批归一化
            )

        layers = []  # 用于保存当前阶段的所有残差块
        # 添加第一个残差块，可能需要降采样
        layers.append(block(self.inplanes, planes, stride, downsample, is_last=(blocks == 1)))

        # 更新当前输入通道数
        self.inplanes = planes * block.expansion

        # 添加后续的残差块（没有降采样）
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, is_last=(i == blocks - 1)))

        # 返回由多个残差块组成的 nn.Sequential 对象
        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False, preact=False):
        """
        前向传播。
        参数:
            x (Tensor): 输入数据
            is_feat (bool): 是否返回中间特征
            preact (bool): 是否返回激活前的值
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f0 = x

        x, f1_pre = self.layer1(x)
        f1 = x
        x, f2_pre = self.layer2(x)
        f2 = x
        x, f3_pre = self.layer3(x)
        f3 = x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        f4 = x
        x = self.fc(x)

        if is_feat:
            if preact:
                return [f0, f1_pre, f2_pre, f3_pre, f4], x
            else:
                return [f0, f1, f2, f3, f4], x
        else:
            return x

# 定义不同深度的 ResNet 函数，方便调用
def resnet8(**kwargs):
    # 输入图片经过初始卷积层，输出尺寸仍为 32x32，通道数为 16。
    # 第一阶段（layer1）：输出尺寸为 32x32，通道数为 16。
    # 第二阶段（layer2）：输出尺寸为 16x16，通道数为 32。
    # 第三阶段（layer3）：输出尺寸为 8x8，通道数为 64。
    # 全局平均池化：输出向量大小为 64。
    # 全连接层：将向量映射为 100 个类别的 logits。
    return ResNet(8, [16, 16, 32, 64], 'basicblock', **kwargs)


# 定义不同深度的 ResNet 函数，方便调用

def resnet14(**kwargs):
    """
    创建一个深度为 14 的 ResNet。
    参数:
        **kwargs: 其他传递给 ResNet 的参数
    返回:
        ResNet: 深度为 14 的 ResNet 模型
    """
    return ResNet(14, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet20(**kwargs):
    """
    创建一个深度为 20 的 ResNet。
    参数:
        **kwargs: 其他传递给 ResNet 的参数
    返回:
        ResNet: 深度为 20 的 ResNet 模型
    """
    return ResNet(20, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet32(**kwargs):
    """
    创建一个深度为 32 的 ResNet。
    参数:
        **kwargs: 其他传递给 ResNet 的参数
    返回:
        ResNet: 深度为 32 的 ResNet 模型
    """
    return ResNet(32, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet44(**kwargs):
    """
    创建一个深度为 44 的 ResNet。
    参数:
        **kwargs: 其他传递给 ResNet 的参数
    返回:
        ResNet: 深度为 44 的 ResNet 模型
    """
    return ResNet(44, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet56(**kwargs):
    """
    创建一个深度为 56 的 ResNet。
    参数:
        **kwargs: 其他传递给 ResNet 的参数
    返回:
        ResNet: 深度为 56 的 ResNet 模型
    """
    return ResNet(56, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet110(**kwargs):
    """
    创建一个深度为 110 的 ResNet。
    参数:
        **kwargs: 其他传递给 ResNet 的参数
    返回:
        ResNet: 深度为 110 的 ResNet 模型
    """
    return ResNet(110, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet8x4(**kwargs):
    """
    创建一个深度为 8，通道数更宽（x4）的 ResNet。
    参数:
        **kwargs: 其他传递给 ResNet 的参数
    返回:
        ResNet: 深度为 8，宽度为 4 倍的 ResNet 模型
    """
    return ResNet(8, [32, 64, 128, 256], 'basicblock', **kwargs)


def resnet32x4(**kwargs):
    """
    创建一个深度为 32，通道数更宽（x4）的 ResNet。
    参数:
        **kwargs: 其他传递给 ResNet 的参数
    返回:
        ResNet: 深度为 32，宽度为 4 倍的 ResNet 模型
    """
    return ResNet(32, [32, 64, 128, 256], 'basicblock', **kwargs)


if __name__ == '__main__':
    """
    测试定义的 ResNet 模型。
    """
    import torch

    # 创建一个随机输入张量，形状为 (2, 3, 32, 32)，表示 2 张 RGB 图片，大小为 32x32。
    x = torch.randn(2, 3, 32, 32)

    # 创建一个 ResNet8x4 模型，指定分类类别数为 20。
    net = resnet8x4(num_classes=20)

    # 获取模型的特征和输出（logits），同时返回激活前的值。
    feats, logit = net(x, is_feat=True, preact=True)

    # 打印每个特征层的形状和最小值。
    for f in feats:
        print(f.shape, f.min().item())
    # 打印最终输出的形状（logits）。
    print(logit.shape)

    # 测试获取 ReLU 激活之前的批归一化层。
    for m in net.get_bn_before_relu():
        if isinstance(m, nn.BatchNorm2d):
            print('pass')  # 如果是批归一化层，打印 pass
        else:
            print('warning')  # 否则打印警告

