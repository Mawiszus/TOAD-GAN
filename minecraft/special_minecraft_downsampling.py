import torch
from torch.nn.functional import interpolate, grid_sample
from torch.nn import Softmax
import numpy as np
from utils import interpolate3D


def special_minecraft_downsampling(num_scales, scales, data, token_list):
    """
    Special Downsampling Method designed for minecraft block data.

    num_scales : number of scales the data is scaled down to.
    scales : downsampling scales. Should be an array tuples (scale_x, scale_y, scale_z) of length num_scales.
    image : Original data to be scaled down. Expects a torch tensor.
    token_list : list of IDs appearing in the image in order of channels from data.
    """

    # for now, normal bilinear downsampling

    scaled_list = []
    for sc in range(num_scales):
        scale_x = scales[sc][0]
        scale_y = scales[sc][1]
        scale_z = scales[sc][2]

        # Initial downscaling of one-hot level tensor is normal bilinear scaling
        # bil_scaled = interpolate(data, (int(data.shape[-3] * scale_y), int(data.shape[-2] * scale_z),
        #                                 int(data.shape[-1] * scale_x)),
        #                          mode='bilinear', align_corners=False)
        shape = (int(data.shape[-3] * scale_y), int(data.shape[-2] * scale_z), int(data.shape[-1] * scale_x))
        scaled = interpolate3D(data, shape)

        scaled_list.append(scaled)

    scaled_list.reverse()
    return scaled_list

