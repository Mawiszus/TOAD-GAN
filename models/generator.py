# Code based on https://github.com/tamarott/SinGAN
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_block import ConvBlock


class Level_GeneratorConcatSkip2CleanAdd(nn.Module):
    """ Patch based Generator. Uses namespace opt. """
    def __init__(self, opt, use_softmax=True):
        super().__init__()
        self.is_cuda = torch.cuda.is_available()
        self.use_softmax = use_softmax
        N = int(opt.nfc)
        dim = len(opt.level_shape)
        kernel = tuple(opt.ker_size for _ in range(dim))
        self.head = ConvBlock(opt.nc_current, N, kernel, 0, 1, dim)  # Padding is done externally
        self.body = nn.Sequential()

        for i in range(opt.num_layer - 2):
            block = ConvBlock(N, N, kernel, 0, 1, dim)
            self.body.add_module("block%d" % (i + 1), block)

        block = ConvBlock(N, N, kernel, 0, 1, dim)
        self.body.add_module("block%d" % (opt.num_layer - 2), block)

        if dim == 2:
            if use_softmax:
                self.tail = nn.Sequential(nn.Conv2d(N, opt.nc_current, kernel_size=kernel,
                                                    stride=1, padding=0))
            else:
                self.tail = nn.Sequential(
                    nn.Conv2d(N, opt.nc_current, kernel_size=kernel, stride=1, padding=0),
                    # nn.ReLU()
                )
        elif dim == 3:
            if use_softmax:
                self.tail = nn.Sequential(nn.Conv3d(N, opt.nc_current, kernel_size=kernel,
                                                    stride=1, padding=0))
            else:
                self.tail = nn.Sequential(
                    nn.Conv3d(N, opt.nc_current, kernel_size=kernel, stride=1, padding=0),
                    # nn.ReLU()
                )
        else:
            raise NotImplementedError("Can only make 2D or 3D Conv Layers.")

    def forward(self, x, y, temperature=1):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        if self.use_softmax:
            x = F.softmax(x * temperature, dim=1)  # Softmax is added here to allow for the temperature parameter
        ind = int((y.shape[2] - x.shape[2]) / 2)
        if len(y.shape) == 4:
            y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
        elif len(y.shape) == 5:
            y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind), ind:(y.shape[4] - ind)]
        else:
            raise NotImplementedError("only supports 4D or 5D tensors")

        return x + y

