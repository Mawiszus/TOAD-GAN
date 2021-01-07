# Code based on https://github.com/tamarott/SinGAN
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    """ Conv block containing Conv2d, BatchNorm2d and LeakyReLU Layers. """
    def __init__(self, in_channel, out_channel, ker_size, padd, stride, dim=2):
        super().__init__()
        if dim == 2:
            self.add_module("conv", nn.Conv2d(in_channel, out_channel, kernel_size=ker_size,
                                              stride=stride, padding=padd),)
            self.add_module("norm", nn.BatchNorm2d(out_channel))
        elif dim == 3:
            self.add_module("conv", nn.Conv3d(in_channel, out_channel, kernel_size=ker_size,
                                              stride=stride, padding=padd),)
            self.add_module("norm", nn.BatchNorm3d(out_channel))
        else:
            raise NotImplementedError("Can only make 2D or 3D Conv Layers.")

        self.add_module("LeakyRelu", nn.LeakyReLU(0.2, inplace=True))

