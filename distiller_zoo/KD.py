from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillKL(nn.Module):
    """支持动态温度的KL散度损失"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.default_T = T

    def forward(self, y_s, y_t, T=None):
        """计算KL散度损失，支持动态温度"""
        T = self.default_T if T is None else T
        p_s = F.log_softmax(y_s / T, dim=1)
        p_t = F.softmax(y_t / T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (T ** 2)
        return loss
