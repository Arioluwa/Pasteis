import torch
from torch import nn
import torch.nn.functional as F
from backbones.utae import UTAE



class VICProjectorConv(nn.Module):
    def __init__(
                self,
                projector_dim=[32, 64, 128, 256]
            ):
        
        super().__init__()
        layers = []
        for i in range(len(projector_dim) - 2):
            layers.append(nn.Conv2d(projector_dim[i], projector_dim[i + 1], kernel_size=3, padding='same'))
            layers.append(nn.BatchNorm2d(projector_dim[i + 1]))
            layers.append(nn.ReLU(True))
            layers.append(nn.MaxPool2d(4))
        layers.append(nn.Conv2d(projector_dim[-2], projector_dim[-1], kernel_size=3, padding='same'))
        self.projector = nn.Sequential(*layers)
    
    def forward(self, x):
        z = self.projector(x)
        z = z.mean(dim=(-2, -1)) # GAP
        return z


class VICReg(nn.Module):
    def __init__(
            self, 
            projector_dim= None #-- first dimension = embedding of the encoders
        ):

        super().__init__()
        if projector_dim is None:
            projector_dim = [32, 64, 128, 256]

        self.num_features = projector_dim[-1]
        self.backbone1 = UTAE()
        self.backbone2 = UTAE()

        self.projector_s1 = VICProjectorConv(projector_dim)
        self.projector_s2 = VICProjectorConv(projector_dim)
        

    def forward(self, sits_1, dates_1, sits_2, dates_2):
        _, maps1 = self.backbone1(sits_1, dates_1)         
        _, maps2 = self.backbone2(sits_2, dates_2)
        x1 = self.projector_s1(maps1[-1])
        x2 = self.projector_s2(maps2[-1])

        loss, losses = VICLoss(x1, x2)
        return loss , losses


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

        # invariance loss
        repr_loss = F.mse_loss(x, y)
        
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        if batch_size>1:
            std_x = torch.sqrt(x.var(dim=0) + 0.0001)
            std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        else:
            std_x = torch.sqrt(x.var(dim=0, correction=0) + 0.0001)
            std_y = torch.sqrt(y.var(dim=0, correction=0) + 0.0001)
        # variance 
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        if batch_size>1:
            cov_x = (x.T @ x) / (batch_size - 1)
            cov_y = (y.T @ y) / (batch_size- 1)
        else:
            cov_x = x.T @ x
            cov_y = y.T @ y

        # covariance loss
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