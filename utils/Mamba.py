import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class S6(nn.Module):
    def __init__(self, seq_len, d_model, state_size):
        super(S6, self).__init__()
        # 一系列线性变换
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, state_size)
        self.fc3 = nn.Linear(d_model, state_size)

        # 设定一些超参数
        self.seq_len = seq_len
        self.d_model = d_model
        self.state_size = state_size

        # 参数初始化
        self.A = nn.Parameter(F.normalize(torch.ones(d_model, state_size), p=2, dim=-1))
        nn.init.xavier_uniform_(self.A)
        
        # 定义内部状态张量
        self.initialize_states()

        # 定义LeakyReLU激活函数
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def initialize_states(self, batch_size=None):
        """初始化状态张量"""
        if batch_size is None:
            batch_size = 1  # 默认批大小设置为1
        self.B = torch.zeros(batch_size, self.seq_len, self.state_size, device=self.A.device)
        self.C = torch.zeros(batch_size, self.seq_len, self.state_size, device=self.A.device)
        self.delta = torch.zeros(batch_size, self.seq_len, self.d_model, device=self.A.device)
        self.dA = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=self.A.device)
        self.dB = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=self.A.device)
        self.h = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=self.A.device)
        self.y = torch.zeros(batch_size, self.seq_len, self.d_model, device=self.A.device)

    def discretization(self):
        """离散化函数定义介绍在Mamba论文中的第28页"""
        
        self.dB = torch.einsum("bld,bln->bldn", self.delta, self.B)
        self.dA = torch.exp(torch.einsum("bld,dn->bldn", self.delta, self.A))
        return self.dA, self.dB

    def forward(self, x):
        """前向传播参考Mamba论文中算法2"""
        batch_size = x.size(0)
        if self.h.shape[0] != batch_size:
            self.initialize_states(batch_size) # 重新 initialize
        
        self.B = self.leaky_relu(self.fc2(x))
        self.C = self.leaky_relu(self.fc3(x))
        self.delta = self.leaky_relu(self.fc1(x))
        
        # 离散化
        self.discretization()

        h_new = torch.einsum('bldn,bldn->bldn', self.dA, self.h) + rearrange(x, "b l d -> b l d 1") * self.dB
        self.y = torch.einsum('bln,bldn->bld', self.C, h_new)

        # 更新h的信息
        self.h = h_new.detach().clone() if not self.h.requires_grad else h_new.clone()
        return self.y
