import torch
from torch import nn
from .memory import ContrastMemory  # 从同级目录的 memory 模块中导入 ContrastMemory 类

# 一个很小的数值，防止在计算中出现除零错误
eps = 1e-7


class CRDLoss(nn.Module):
    """
    CRD Loss 损失函数，用于知识蒸馏中的对比学习。
    包含两个对称部分：
      (a) 以教师网络为锚点，在学生网络的特征中选择正样本和负样本
      (b) 以学生网络为锚点，在教师网络的特征中选择正样本和负样本

    参数说明（从 opt 中读取）:
        opt.s_dim: 学生网络特征的维度
        opt.t_dim: 教师网络特征的维度
        opt.feat_dim: 投影空间的维度（即经过嵌入层后的特征维度）
        opt.nce_k: 每个正样本所搭配的负样本数量
        opt.nce_t: 温度参数，用于缩放对比相似度
        opt.nce_m: 更新内存缓冲区时的动量
        opt.n_data: 训练集样本数量，因此内存缓冲区大小为 opt.n_data x opt.feat_dim
    """
    def __init__(self, opt):
        super(CRDLoss, self).__init__()
        # 定义用于将学生特征映射到投影空间的嵌入层
        self.embed_s = Embed(opt.s_dim, opt.feat_dim)
        # 定义用于将教师特征映射到投影空间的嵌入层
        self.embed_t = Embed(opt.t_dim, opt.feat_dim)
        # 初始化对比记忆模块，用于存储和对比特征
        self.contrast = ContrastMemory(opt.feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m)
        # 定义两个对比损失，分别用于教师和学生的视角
        self.criterion_t = ContrastLoss(opt.n_data)
        self.criterion_s = ContrastLoss(opt.n_data)

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """
        前向传播计算对比损失

        参数:
            f_s: 学生网络输出的原始特征，尺寸为 [batch_size, s_dim]
            f_t: 教师网络输出的原始特征，尺寸为 [batch_size, t_dim]
            idx: 当前批次中样本在数据集中的索引，尺寸为 [batch_size]
            contrast_idx: （可选）负样本的索引，尺寸为 [batch_size, nce_k]

        返回:
            计算得到的对比损失（一个标量）
        """
        # 通过嵌入层将学生和教师的特征映射到同一投影空间
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        # 利用对比记忆模块获得学生和教师对应的对比输出
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
        # 分别计算学生和教师方向的对比损失
        s_loss = self.criterion_s(out_s)
        t_loss = self.criterion_t(out_t)
        # 最终损失为两个方向损失的和
        loss = s_loss + t_loss
        return loss


class ContrastLoss(nn.Module):
    """
    对比损失函数，对应论文中的公式 (18)
    计算正样本和负样本的对比损失
    """
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data  # 数据集中样本的总数，用于计算噪声分布

    def forward(self, x):
        # x 的尺寸为 [batch_size, nce_k+1]，第一列为正样本得分，其余为负样本得分
        bsz = x.shape[0]            # 当前批次的样本数量
        m = x.size(1) - 1           # 负样本数量

        # 计算噪声分布，每个样本的噪声概率均等
        Pn = 1 / float(self.n_data)

        # 处理正样本得分，取出第一列
        P_pos = x.select(1, 0)
        # 计算正样本的对比概率并取对数
        # torch.div(P_pos, P_pos.add(m * Pn + eps)) 实现 P/(P + m*Pn)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # 处理负样本得分，取出其余部分
        P_neg = x.narrow(1, 1, m)
        # 对于负样本，先将其填充为 m * Pn，再计算对比概率并取对数
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        # 最终损失为正负样本对数概率的和，取负值后求平均
        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class Embed(nn.Module):
    """嵌入模块，用于将输入特征映射到投影空间，并进行 L2 归一化"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        # 线性映射，将输入特征从 dim_in 映射到 dim_out
        self.linear = nn.Linear(dim_in, dim_out)
        # L2 归一化层，确保输出特征具有单位长度
        self.l2norm = Normalize(2)

    def forward(self, x):
        # 将输入展平为二维张量 [batch_size, -1]
        x = x.view(x.shape[0], -1)
        # 进行线性变换
        x = self.linear(x)
        # 进行 L2 归一化
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """归一化层，实现 Lp 范数归一化（默认 L2 归一化）"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power  # 指定范数类型，比如2表示 L2 归一化

    def forward(self, x):
        # 计算 x 每一行的 Lp 范数
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        # 将 x 除以范数进行归一化
        out = x.div(norm)
        return out
