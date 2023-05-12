"""
VICReg implementation
Author: Facebook Research Centre (github/facebookresearch/vicreg)
License: MIT
"""

import torch
import torch.nn.functional as F
from torch import nn, optim
import torch.distributed as dist
import torchvision.datasets as datasets
import torch.distributed as dist


import backbones.utae

#-- VICReg implementation for SITS
class VICReg(nn.Module):
    def __init__(
            self, 
            encoder1,
            encoder2,
            projector_dim=[32, 64, 128, 256] #-- first dimension = embedding of the encoders
            ):

        super().__init__()
        self.num_features = projector_dim[-1]
        self.backbone_s1 = encoder1
        self.backbone_s2 = encoder2

        #-- projector / expander
        # layers = []
        # for i in range(len(projector_dim) - 2):
        #     layers.append(nn.Linear(projector_dim[i], projector_dim[i + 1]))
        #     layers.append(nn.BatchNorm1d(projector_dim[i + 1]))
        #     layers.append(nn.ReLU(True))
        # layers.append(nn.Linear(projector_dim[-2], projector_dim[-1], bias=False))
        layers = []
        for i in range(len(projector_dim) - 2):
            layers.append(nn.Conv2d(projector_dim[i], projector_dim[i + 1], kernel_size=3, padding='same'))
            layers.append(nn.BatchNorm2d(projector_dim[i + 1]))
            layers.append(nn.ReLU(True))
            layers.append(nn.MaxPool2d(4))
        layers.append(nn.Conv2d(projector_dim[-2], projector_dim[-1], kernel_size=3, padding='same'))
        self.projector = nn.Sequential(*layers)

    def forward(self, s1, d1, s2, d2):
        _, maps1 = self.backbone_s1(s1, d1) #len(maps) = no. level -- maps[i] feature maps at level i - maps[-1.shape]=torch.Size([1, 32, 128, 128])        
        _, maps2 = self.backbone_s2(s2, d2)
        x = self.projector(maps1[-1])
        x = x.mean(dim=(-2, -1)) # GAP
        y = self.projector(maps2[-1])
        y = y.mean(dim=(-2, -1)) # GAP
        return x, y

class VICLoss(nn.Module):
    def __init__(self, sim_coeff=25, std_coeff=25, cov_coeff=1, weight=None, size_average=True):
        super(VICLoss, self).__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, x, y, smooth=1):
        assert x.shape[0]==y.shape[0] #-- batch_size
        assert x.shape[1]==y.shape[1] #-- num_features
        batch_size = x.shape[0]
        num_features = x.shape[1]

        repr_loss = F.mse_loss(x, y)
        
        #-- Parallel process not supported here
        #-- x = torch.cat(FullGatherLayer.apply(x), dim=0)
        #-- y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        if batch_size>1:
            std_x = torch.sqrt(x.var(dim=0) + 0.0001)
            std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        else:
            std_x = torch.sqrt(x.var(dim=0, correction=0) + 0.0001)
            std_y = torch.sqrt(y.var(dim=0, correction=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        if batch_size>1:
            cov_x = (x.T @ x) / (batch_size - 1)
            cov_y = (y.T @ y) / (batch_size- 1)
        else:
            cov_x = x.T @ x
            cov_y = y.T @ y

        cov_loss = self._off_diagonal(cov_x).pow_(2).sum().div(
            num_features
        ) + self._off_diagonal(cov_y).pow_(2).sum().div(num_features)

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss, (repr_loss, std_loss, cov_loss)

    def _off_diagonal(self,x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()