# Code based on https://github.com/tamarott/SinGAN
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_block import ConvBlock


class Level_GeneratorConcatSkip2CleanAdd(nn.Module):
    """ Patch based Generator. Uses namespace opt. """
    def __init__(self, opt):
        super().__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_current, N, (opt.ker_size, opt.ker_size), 0, 1)  # Padding is done externally
        self.body = nn.Sequential()

        for i in range(opt.num_layer - 2):
            block = ConvBlock(N, N, (opt.ker_size, opt.ker_size), 0, 1)
            self.body.add_module("block%d" % (i + 1), block)

        block = ConvBlock(N, N, (opt.ker_size, opt.ker_size), 0, 1)
        self.body.add_module("block%d" % (opt.num_layer - 2), block)

        self.tail = nn.Sequential(nn.Conv2d(N, opt.nc_current, kernel_size=(opt.ker_size, opt.ker_size),
                                            stride=1, padding=0))

    def forward(self, x, y, temperature=1):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x = F.softmax(x * temperature, dim=1)  # Softmax is added here to allow for the temperature parameter
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        return x + y

