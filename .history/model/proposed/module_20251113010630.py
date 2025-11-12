import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    default_act = nn.ReLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class GhostConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)
    
class FiLMBlock(nn.Module):
    def __init__(self):
        super(FiLMBlock, self).__init__()
        
    def forward(self, x, gamma, beta):
        beta = beta.view(x.size(0), x.size(1), 1, 1)
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)
        
        x = gamma * x + beta
        
        return x

# Source code from: https://github.com/windofshadow/THAT/blob/main/TransCNN.py
class Gaussian_Position(nn.Module):
    def __init__(self, d_model, total_size, K=10):
        super(Gaussian_Position, self).__init__()
        self.embedding = nn.Parameter(torch.zeros([K, d_model], dtype=torch.float), requires_grad=True)
        nn.init.xavier_uniform_(self.embedding, gain=1)

        # Assume total_size corresponds to the sequence length in x
        self.total_size = total_size

        # Setup Gaussian distribution parameters
        positions = torch.arange(total_size).unsqueeze(1).repeat(1, K)
        self.register_buffer('positions', positions)
        
        s = 0.0
        interval = total_size / K
        mu = []
        for _ in range(K):
            mu.append(s)
            s += interval
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float).unsqueeze(0), requires_grad=True)
        self.sigma = nn.Parameter(torch.ones([1, K], dtype=torch.float) * 50.0, requires_grad=True)

    def forward(self, x):
        # Ensure input x has shape [batch_size, seq_length, d_model]
        batch_size, seq_length, d_model = x.shape

        assert self.total_size == seq_length, "total_size must match seq_length of input x"

        M = normal_pdf(self.positions, self.mu, self.sigma)  # Assuming this function is defined correctly
        pos_enc = torch.matmul(M, self.embedding)  # [seq_length, d_model]
        pos_enc = pos_enc.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, seq_length, d_model]

        return x + pos_enc
    
def normal_pdf(pos, mu, sigma): 
    a = pos - mu
    log_p = -1*torch.mul(a, a)/(2*sigma) - torch.log(sigma)/2
    return F.softmax(log_p, dim=1)

class simam_module(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()
        
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)