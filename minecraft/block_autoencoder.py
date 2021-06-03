import torch
import torch.nn as nn
import torch.nn.functional as F
from models.conv_block import ConvBlock


class Encoder(nn.Module):
    def __init__(self, opt, is_train=False, latent_size=10):
        super().__init__()
        self.is_cuda = torch.cuda.is_available()
        self.is_train = is_train
        self.latent_size = latent_size
        N = int(opt.nfc)
        dim = len(opt.level_shape)
        kernel = tuple(opt.ker_size for _ in range(dim))

        self.head = ConvBlock(opt.nc_current, N, kernel, 0, 1, dim)
        self.tail = ConvBlock(N, self.latent_size, kernel, 0, 1, dim)

    def forward(self, x):
        x = nn.ReplicationPad3d(2)(x)
        x = self.head(x)
        x = self.tail(x)
        z = x
        mean = torch.mean(z, 0, keepdim=True)
        logvar = torch.var(z, 0, keepdim=True, unbiased=False)
        """
        mean = x[:, 0:self.latent_size]
        logvar = x[:, self.latent_size:self.latent_size*2]
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        if self.is_train:
            z = eps * std + mean
        else:
            # z = eps * std + mean
            z = mean
        # out = torch.normal(mean, std)
        """
        return z, mean, logvar  # .unsqueeze(dim=1)


class Decoder(nn.Module):
    def __init__(self, opt, latent_size=10):
        super().__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        dim = len(opt.level_shape)
        kernel = tuple(opt.ker_size for _ in range(dim))

        self.head = ConvBlock(latent_size, N, kernel, 0, 1, dim)
        self.tail = ConvBlock(N, opt.nc_current, kernel, 0, 1, dim)

    def forward(self, x):
        x = nn.ReplicationPad3d(2)(x)
        x = self.head(x)
        x = self.tail(x)
        x = F.softmax(x, dim=1)
        return x
