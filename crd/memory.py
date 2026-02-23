import torch
from torch import nn
import math


class ContrastMemory(nn.Module):
    """
    对比记忆模块，用于存储大量负样本的内存缓冲区，
    为对比学习（contrastive learning）提供负样本支持。
    """

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5):
        """
        参数：
            inputSize: 输入特征的维度
            outputSize: 内存中存储的样本数量（通常等于数据集总样本数）
            K: 每个样本用于对比的负样本数量
            T: 温度参数，用于缩放相似度得分
            momentum: 动量系数，用于更新内存中的特征表示
        """
        super(ContrastMemory, self).__init__()
        # 内存中存储样本的数量
        self.nLem = outputSize
        # 初始化一个全1向量，作为每个样本的初始权重
        self.unigrams = torch.ones(self.nLem)
        # 使用 AliasMethod 进行高效采样，构造多项式分布采样器
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        # 保存负样本数量
        self.K = K

        # 将一些参数注册为 buffer，确保它们会随模型保存，但不会被当作参数训练
        # params 中存储：K, T, Z_v1, Z_v2, momentum，其中 Z_v1, Z_v2 初始设为 -1 表示未设置
        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        # 根据输入特征维度计算初始化范围
        stdv = 1. / math.sqrt(inputSize / 3)
        # 初始化 memory_v1 和 memory_v2，用于存储两组不同视角下的特征表示
        # 这里随机初始化，并将值映射到 [-stdv, stdv] 范围内
        self.register_buffer('memory_v1', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_v2', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, v1, v2, y, idx=None):
        """
        前向传播：
            v1: 第一组输入特征（通常来自学生网络），尺寸 [batch_size, inputSize]
            v2: 第二组输入特征（通常来自教师网络），尺寸 [batch_size, inputSize]
            y: 当前批次样本在整个数据集中的索引，尺寸 [batch_size]
            idx: 可选，预采样的负样本索引（如果不提供则内部进行采样）
        返回：
            out_v1, out_v2: 分别对应于两组特征的对比得分（经过归一化）
        """
        # 从注册的参数中取出 K, T, 归一化常数 Z_v1, Z_v2 和 momentum
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_v1 = self.params[2].item()
        Z_v2 = self.params[3].item()
        momentum = self.params[4].item()
        batchSize = v1.size(0)
        outputSize = self.memory_v1.size(0)
        inputSize = self.memory_v1.size(1)

        # 如果未提供负样本索引，则通过 AliasMethod 采样获取
        if idx is None:
            # 总共采样 batchSize*(K+1) 个索引，第一列保留正样本索引
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            # 将每行的第一个位置设置为真实的样本索引 y
            idx.select(1, 0).copy_(y.data)

        # 根据采样得到的索引，从 memory_v1 中选取对应的特征作为负样本
        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batchSize, K + 1, inputSize)
        # 对 v2 进行批量矩阵乘法，计算内积得分，并除以温度 T 后取指数
        out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1))
        out_v2 = torch.exp(torch.div(out_v2, T))

        # 同样的操作，对 memory_v2 和 v1 进行处理
        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + 1, inputSize)
        out_v1 = torch.bmm(weight_v2, v1.view(batchSize, inputSize, 1))
        out_v1 = torch.exp(torch.div(out_v1, T))

        # 如果归一化常数 Z 尚未设置（初始为负），则根据当前得分设置归一化常数
        if Z_v1 < 0:
            # 取 out_v1 的均值乘以输出样本总数，作为归一化常数
            self.params[2] = out_v1.mean() * outputSize
            Z_v1 = self.params[2].clone().detach().item()
            print("normalization constant Z_v1 is set to {:.1f}".format(Z_v1))
        if Z_v2 < 0:
            self.params[3] = out_v2.mean() * outputSize
            Z_v2 = self.params[3].clone().detach().item()
            print("normalization constant Z_v2 is set to {:.1f}".format(Z_v2))

        # 对得分进行归一化
        out_v1 = torch.div(out_v1, Z_v1).contiguous()
        out_v2 = torch.div(out_v2, Z_v2).contiguous()

        # 利用动量更新内存缓冲区中的特征表示
        with torch.no_grad():
            # 针对 memory_v1 的更新
            l_pos = torch.index_select(self.memory_v1, 0, y.view(-1))
            l_pos.mul_(momentum)  # 先乘以动量因子
            l_pos.add_(torch.mul(v1, 1 - momentum))  # 再加上当前特征的加权部分
            # 对更新后的特征进行归一化
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm)
            # 将更新后的特征写回内存中对应位置
            self.memory_v1.index_copy_(0, y, updated_v1)

            # 针对 memory_v2 的更新（与上面类似）
            ab_pos = torch.index_select(self.memory_v2, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(v2, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, y, updated_v2)

        # 返回归一化后的得分，用于后续的对比损失计算
        return out_v1, out_v2


class AliasMethod(object):
    """
    Alias方法用于高效采样多项分布中的样本。
    参考：https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """

    def __init__(self, probs):
        """
        参数：
            probs: 一个概率分布向量（未归一化或归一化均可）
        """
        # 如果概率和大于1，则归一化概率向量
        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        # 初始化存储各个元素调整后的概率和别名
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0] * K)

        # 将元素根据概率与均值 1/K 进行划分，分为小概率和大概率两组
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K * prob  # 将概率放大到 K 倍
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # 构造 alias 表，进行二元混合分布分配
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            # 调整 large 对应的概率值
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            # 根据调整后的概率重新分组
            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        # 对剩余元素将概率设为1
        for last_one in smaller + larger:
            self.prob[last_one] = 1

    def cuda(self):
        """
        将采样器的内部数据移到 GPU 上
        """
        self.prob = self.prob.cuda()
        self.alias = self.alias.cuda()

    def draw(self, N):
        """
        从多项分布中采样 N 个样本
        参数：
            N: 需要采样的数量
        返回：
            采样结果张量，尺寸为 [N]
        """
        K = self.alias.size(0)
        # 随机生成 N 个范围在 [0, K) 内的整数
        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        # 根据生成的索引获取对应的概率和别名
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # 对每个采样位置，根据概率决定是否使用别名
        b = torch.bernoulli(prob)
        # 如果 b 为1，则保留原来的索引，否则使用别名索引
        oq = kk.mul(b.long())
        oj = alias.mul((1 - b).long())

        return oq + oj
