import torch
from torch.nn.functional import interpolate, grid_sample
from torch.nn import Softmax
import numpy as np
from utils import interpolate3D, load_pkl, save_pkl


def bf_icf(block_idx, chunk, freq_doc, token_list):
    blocktype = token_list[block_idx]
    bf = chunk[:, block_idx].sum().item()
    cf = freq_doc[blocktype]
    icf = np.log(1/cf)
    return bf * icf


def special_minecraft_downsampling(num_scales, scales, data, token_list, use_hierarchy=False):
    """
    Special Downsampling Method designed for minecraft block data.

    num_scales : number of scales the data is scaled down to.
    scales : downsampling scales. Should be an array tuples (scale_x, scale_y, scale_z) of length num_scales.
    image : Original data to be scaled down. Expects a torch tensor.
    token_list : list of IDs appearing in the image in order of channels from data.
    """

    if use_hierarchy:
        # cf_doc = load_pkl('primordial_counts', prepath='minecraft/chunk_frequencies/')
        bf_icf_list = torch.zeros((data.shape[1],), device=data.device)
        n_blocks = data.shape[2] * data.shape[3] * data.shape[4]
        for i in range(data.shape[1]):
            bf_icf_list[i] = n_blocks / data[0, i].sum()
        # for i, token in enumerate(token_list):
        #     bf_icf_list[i] = bf_icf(i, data, cf_doc, token_list)

    # for now, normal bilinear downsampling

    scaled_list = []
    for sc in range(num_scales):
        scale_1 = scales[sc][0]
        scale_2 = scales[sc][1]
        scale_3 = scales[sc][2]

        # Initial downscaling of one-hot level tensor is normal bilinear scaling
        # bil_scaled = interpolate(data, (int(data.shape[-3] * scale_y), int(data.shape[-2] * scale_z),
        #                                 int(data.shape[-1] * scale_x)),
        #                          mode='bilinear', align_corners=True)
        shape = (int(data.shape[-3] * scale_1), int(data.shape[-2] * scale_2), int(data.shape[-1] * scale_3))

        if use_hierarchy:
            scaled = interpolate3D(data, shape, align_corners=True)
            level_scaled = torch.zeros_like(scaled)
            for j in range(scaled.shape[-3]):
                for k in range(scaled.shape[-2]):
                    for l in range(scaled.shape[-1]):
                        blocklist = scaled[0, :, j, k, l] > 0
                        probs = blocklist * bf_icf_list
                        if probs.sum() == 0:
                            level_scaled[0, scaled[0, :, j, k, l].argmax(), j, k, l] = 1
                        else:
                            level_scaled[0, probs.argmax(), j, k, l] = 1
        else:
            level_scaled = interpolate3D(data, shape, align_corners=True)

        scaled_list.append(level_scaled)
    scaled_list.reverse()
    return scaled_list

