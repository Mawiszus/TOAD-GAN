import subprocess
from typing import Optional

import yaml
import math
import numpy as np
from config import Config
import os
from shutil import copyfile
from argparse import Namespace
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt

# sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))  # uncomment if opening form other dir

from minecraft.special_minecraft_downsampling import special_minecraft_downsampling
from minecraft.level_utils import one_hot_to_blockdata_level, save_level_to_world, clear_empty_world
from minecraft.level_utils import read_level as mc_read_level
from minecraft.level_renderer import render_minecraft
from generate_noise import generate_spatial_noise
from models import load_trained_pyramid
from utils import interpolate3D


class GenerateSamplesConfig(Config):
    out_: Optional[str] = None  # folder containing generator files
    scale_v: float = 1.0  # vertical scale factor
    scale_h: float = 1.0  # horizontal scale factor
    scale_d: float = 1.0  # horizontal scale factor
    gen_start_scale: int = 0  # scale to start generating in
    num_samples: int = 10  # number of samples to be generated
    save_tensors: bool = False  # save pytorch .pt tensors?
    not_cuda: bool = False  # disables cuda
    generators_dir: Optional[str] = None

    def process_args(self):
        super().process_args()
        self.seed_road: Optional[torch.Tensor] = None
        if not self.out_:
            raise Exception(
                '--out_ is required')


def generate_samples(generators, noise_maps, reals, noise_amplitudes, opt: GenerateSamplesConfig, in_s=None, scale_v=1.0, scale_h=1.0, scale_d=1.0,
                     current_scale=0, gen_start_scale=0, num_samples=50, render_images=True, save_tensors=False,
                     save_dir="random_samples"):
    """
    Generate samples given a pretrained TOAD-GAN (generators, noise_maps, reals, noise_amplitudes).
    Uses namespace "opt" that needs to be parsed.
    "in_s" can be used as a starting image in any scale set with "current_scale".
    "gen_start_scale" sets the scale generation is to be started in.
    "num_samples" is the number of different samples to be generated.
    "render_images" defines if images are to be rendered (takes space and time if many samples are generated).
    "save_tensors" defines if tensors are to be saved (can be needed for token insertion *experimental*).
    "save_dir" is the path the samples are saved in.
    """

    # Holds images generated in current scale
    images_cur = []

    # Minecraft has 3 dims
    dim = 3

    # Main sampling loop
    for sc, (G, Z_opt, noise_amp) in enumerate(zip(generators, noise_maps, noise_amplitudes)):

        # Make directories
        dir2save = opt.out_ + '/' + save_dir
        try:
            os.makedirs(dir2save, exist_ok=True)
            if save_tensors:
                os.makedirs("%s/torch" % dir2save, exist_ok=True)
            if dim == 2:
                os.makedirs("%s/txt" % dir2save, exist_ok=True)
        except OSError:
            pass

        if current_scale >= len(generators):
            break  # if we do not start at current_scale=0 we need this
        elif sc < current_scale:
            continue

        logger.info("Generating samples at scale {}", current_scale)

        # Padding (should be chosen according to what was trained with)
        n_pad = int(1*opt.num_layer)
        if not opt.pad_with_noise:
            # m = nn.ConstantPad3d(int(n_pad), 0)  # pad with zeros
            m = nn.ReplicationPad3d(int(n_pad))  # pad with reflected noise
        else:
            m = nn.ReplicationPad3d(int(n_pad))  # pad with reflected noise

        # Calculate shapes to generate
        if 0 < gen_start_scale <= current_scale:  # Special case! Can have a wildly different shape through in_s
            nz = []
            scale_v = in_s.shape[-1] / (noise_maps[gen_start_scale - 1].shape[-1] - n_pad * 2)
            scale_h = in_s.shape[-3] / (noise_maps[gen_start_scale - 1].shape[-3] - n_pad * 2)
            scale_d = in_s.shape[-2] / (noise_maps[gen_start_scale - 1].shape[-2] - n_pad * 2)
            nz.append(int(round(((Z_opt.shape[-3] - n_pad * 2) * scale_h))))
            nz.append(int(round(((Z_opt.shape[-2] - n_pad * 2) * scale_d))))
            nz.append(int(round(((Z_opt.shape[-1] - n_pad * 2) * scale_v))))  # mc ordering is y, z, x
        else:
            nz = []
            nz.append(int(round(((Z_opt.shape[-3] - n_pad * 2) * scale_h))))
            nz.append(int(round(((Z_opt.shape[-2] - n_pad * 2) * scale_d))))
            nz.append(int(round(((Z_opt.shape[-1] - n_pad * 2) * scale_v))))  # mc ordering is y, z, x

        # Save list of images of previous scale and clear current images
        images_prev = images_cur
        images_cur = []

        channels = reals[0].shape[1]

        # If in_s is none or filled with zeros reshape to correct size with channels
        if in_s is None:
            in_s = torch.zeros(reals[0].shape[0], channels,
                               *reals[0].shape[2:]).to(opt.device)
        elif in_s.sum() == 0:
            in_s = torch.zeros(1, channels, *in_s.shape[2:]).to(opt.device)

        # Generate num_samples samples in current scale
        for n in tqdm(range(0, num_samples, 1)):

            # Get noise image
            z_curr = generate_spatial_noise((1, channels,) + tuple(nz), device=opt.device)
            z_curr = m(z_curr)

            # Set up previous image I_prev
            if (not images_prev) or current_scale == 0:  # if there is no "previous" image
                I_prev = in_s
            else:
                I_prev = images_prev[n]

            I_prev = interpolate3D(I_prev, nz, mode='bilinear', align_corners=True)
            I_prev = m(I_prev)

            # We take the optimized noise map Z_opt as an input if we start generating on later scales
            if current_scale < gen_start_scale:
                z_curr = Z_opt

            # Define correct token list
            # if we have a different block2repr than during training, we need to update the token_list
            if opt.repr_type is not None:
                if opt.token_list == list(opt.block2repr.keys()):
                    token_list = opt.token_list
                    props = opt.props
                else:
                    if opt.repr_type == "block2vec":
                        token_list = list(opt.block2repr.keys())
                    else:
                        # token_list = opt.token_list
                        token_list = torch.load('input/minecraft/simple_autoencoder_token_list.pt')
                    props = []
                    for tok in token_list:
                        if tok == 'minecraft:cut_red_sandstone_slab' or tok == "minecraft:cobblestone_slab":
                            props.append({'waterlogged': 'false', 'type': 'bottom'})
                        elif tok == "minecraft:smooth_red_sandstone_stairs":
                            props.append({'half': 'bottom', 'waterlogged': 'false', 'shape': 'straight', 'facing': 'south'})
                        else:
                            props.append({})

                    # props = [{'waterlogged': 'false', 'half': 'lower', 'type': 'bottom'} for _ in range(len(token_list))]
                    # props = opt.props  # should work if you don't sub weird things?
                    # TODO: how to deal with props in transfer experiment?
            else:
                token_list = opt.token_list
                props = opt.props

            ###########
            # Generate!
            ###########
            z_in = noise_amp * z_curr + I_prev
            I_curr = G(z_in.detach(), I_prev, temperature=1)

            # Save all scales
            # if True:
            # Save scale 0 and last scale
            # if current_scale == 0 or current_scale == len(reals) - 1:
            # Save only last scale
            if current_scale == len(reals) - 1:

                # Convert to level
                to_level = one_hot_to_blockdata_level

                # Save level
                # Minecraft
                if n == 0:  # in first step make folder and save real blockdata
                    bdata_pth = "%s/torch_blockdata" % dir2save
                    os.makedirs(bdata_pth, exist_ok=True)
                    real_level = to_level(reals[current_scale], token_list, opt.block2repr, opt.repr_type)
                    torch.save(real_level, "%s/real_bdata.pt" % dir2save)
                    torch.save(token_list, "%s/token_list.pt" % dir2save)
                    if render_images:
                        real_pth = "%s/reals" % dir2save
                        os.makedirs(real_pth, exist_ok=True)
                        save_level_to_world(opt.output_dir, opt.output_name, (0, 0, 0), real_level, token_list, props)
                        curr_coords = [[0, real_level.shape[0]],
                                       [0, real_level.shape[1]],
                                       [0, real_level.shape[2]]]
                        render_minecraft(opt.output_name, curr_coords, real_pth, "%d_real" % current_scale)

                level = to_level(I_curr.detach(), token_list, opt.block2repr, opt.repr_type)
                torch.save(level, "%s/torch_blockdata/%d_sc%d.pt" % (dir2save, n, current_scale))
                # save_path = "%s/txt/%d_sc%d.schem" % (dir2save, n, current_scale)
                # new_schem = NanoMCSchematic(save_path, level.shape[:3])
                # new_schem.set_blockdata(level)
                # new_schem.saveToFile()
                if render_images:
                    obj_pth = "%s/objects/%d" % (dir2save, current_scale)
                    os.makedirs(obj_pth, exist_ok=True)
                    try:
                        subprocess.call(["wine", '--version'])
                        # Minecraft World
                        len_n = math.ceil(math.sqrt(num_samples))  # we arrange our samples in a square in the world
                        x, z = np.unravel_index(n, [len_n, len_n])  # get x, z pos according to index n
                        posx = x * (level.shape[0] + 5)
                        posz = z * (level.shape[2] + 5)
                        save_level_to_world(opt.output_dir, opt.output_name, (posx, 0, posz), level, token_list, props)
                        # save_oh_to_wrld_directly(opt.output_dir, opt.output_name, (posx, 0, posz), I_curr.detach(),
                        #                          opt.block2repr, opt.repr_type)
                        curr_coords = [[posx, posx + level.shape[0]],
                                        [0, level.shape[1]],
                                        [posz, posz + level.shape[2]]]
                        render_minecraft(opt.output_name, curr_coords, obj_pth, "%d" % n)
                    except OSError:
                        pass

                # Save torch tensor
                if save_tensors:
                    torch.save(I_curr, "%s/torch/%d_sc%d.pt" %
                               (dir2save, n, current_scale))

            # Append current image
            images_cur.append(I_curr)

        # Go to next scale
        current_scale += 1

    return I_curr.detach()  # return last generated image (usually unused)


if __name__ == '__main__':
    # NOTICE: The "output" dir is where the generator is located as with main.py, even though it is the "input" here

    opt = GenerateSamplesConfig().parse_args()

    # Init game specific inputs
    opt.ImgGen = None
    clear_empty_world(opt.output_dir, opt.output_name)
    downsample = special_minecraft_downsampling

    # Read level according to input arguments
    real = mc_read_level(opt)

    if opt.use_multiple_inputs:  # multi-input still doesn't work
        real = real[0].to(opt.device)
        opt.level_shape = real[0].shape[2:]
    else:
        real = real.to(opt.device)
        opt.level_shape = real.shape[2:]

    # Load Generator
    generators, noise_maps, reals, noise_amplitudes = load_trained_pyramid(opt)

    if opt.use_multiple_inputs:
        noise_maps = [m[0] for m in noise_maps]
        reals = reals[0]

    cur_scale = 0  # make editable from outside?

    # Get input shape for in_s
    real_down = downsample(1, [[opt.scale_v, opt.scale_h, opt.scale_d]], real, opt.token_list)
    real_down = real_down[0]
    in_s = torch.zeros_like(real_down, device=opt.device)
    prefix = "arbitrary"

    # Directory name
    s_dir_name = "%s_random_samples_v%.5f_h%.5f_st%d" % (
        prefix, opt.scale_v, opt.scale_h, opt.gen_start_scale)

    generate_samples(generators, noise_maps, reals, noise_amplitudes, opt, in_s=in_s, save_tensors=opt.save_tensors,
                     scale_v=opt.scale_v, scale_h=opt.scale_h, scale_d=opt.scale_d, save_dir=s_dir_name,
                     num_samples=opt.num_samples, current_scale=cur_scale)
