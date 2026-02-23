from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['resnet']


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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
    def __init__(self, depth, num_filters, block_name='BasicBlock', num_classes=10):
        super(ResNet, self).__init__()
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'BasicBlock 的深度应为 6n+2'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'Bottleneck 的深度应为 9n+2'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name 应为 BasicBlock 或 Bottleneck')

        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, num_filters[1], n, stride=1)
        self.layer2 = self._make_layer(block, num_filters[2], n, stride=2)
        self.layer3 = self._make_layer(block, num_filters[3], n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(num_filters[3] * block.expansion, num_classes)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, is_last=(blocks == 1)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, is_last=(i == blocks - 1)))
        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False, preact=False):
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


def resnet8(**kwargs):
    return ResNet(8, [16, 16, 32, 64], 'basicblock', **kwargs)

def resnet110(**kwargs):
    return ResNet(110, [16, 16, 32, 64], 'basicblock', **kwargs)


if __name__ == '__main__':
    import torch

    # ------------------------------------------------------------
    # 1) 先实例化一个ResNet110并加载预训练权重：resnet110_best.pth
    # ------------------------------------------------------------
    teacher = resnet110(num_classes=10)
    teacher_ckpt = torch.load('./save/models/resnet110_vanilla/ckpt_epoch_240.pth', map_location='cpu')
    # 如果你的文件就是纯state_dict，可以直接用下行:
    teacher.load_state_dict(teacher_ckpt)

    # 如果有额外键值，比如 {"model_state_dict":..., "epoch":...}，
    # 则需使用 teacher.load_state_dict(teacher_ckpt["model_state_dict"])

    # ------------------------------------------------------------
    # 2) 新建一个ResNet8，准备用ResNet110部分权重来初始化
    # ------------------------------------------------------------
    student = resnet8(num_classes=100)

    teacher_dict = teacher.state_dict()
    student_dict = student.state_dict()

    transferred_count = 0
    for name, param in teacher_dict.items():
        # 如果student也有同名参数，并且形状一致，就拷贝
        if name in student_dict and student_dict[name].shape == param.shape:
            student_dict[name] = param
            transferred_count += 1

    print(f"成功从ResNet110拷贝 {transferred_count} 个匹配参数到ResNet8.")
    student.load_state_dict(student_dict)

    # ------------------------------------------------------------
    # 3) 随意测试一下，验证forward是否能正常运行
    # ------------------------------------------------------------
    x = torch.randn(2, 3, 32, 32)
    feats, logit = student(x, is_feat=True, preact=True)
    print("输出 logits 的形状:", logit.shape)
    for idx, f in enumerate(feats):
        print(f"特征 {idx} 的形状: {f.shape}")
