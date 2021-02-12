import torch
import torch.nn as nn
import torch.nn.functional as F
from models.conv_block import ConvBlock


class Encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        dim = len(opt.level_shape)
        kernel = tuple(opt.ker_size for _ in range(dim))

        self.head = ConvBlock(opt.nc_current, N, kernel, 0, 1, dim)

    def forward(self, x):
        x = nn.ReplicationPad3d(1)(x)
        x = self.head(x)
        return x


class Decoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        dim = len(opt.level_shape)
        kernel = tuple(opt.ker_size for _ in range(dim))

        self.tail = ConvBlock(N, opt.nc_current, kernel, 0, 1, dim)

    def forward(self, x):
        x = nn.ReplicationPad3d(1)(x)
        x = self.tail(x)
        x = F.softmax(x, dim=1)
        return x
