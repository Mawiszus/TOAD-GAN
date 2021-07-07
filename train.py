# Code inspired by https://github.com/tamarott/SinGAN
import subprocess
from config import Config
import os

import torch
import wandb
from tqdm import tqdm
import math

from minecraft.special_minecraft_downsampling import special_minecraft_downsampling
from minecraft.level_renderer import render_minecraft
from models import init_models, reset_grads, restore_weights
from models.generator import Level_GeneratorConcatSkip2CleanAdd
from train_single_scale import train_single_scale


def calc_lowest_possible_scale(level, kernel_size, num_layers):
    """Calculates the lowest size the generator will accept in each dimension.
    It depends on the number/size of layers."""
    needed_pad = math.floor(kernel_size/2) * num_layers
    min_size = (needed_pad * 2) + 2
    sizes = level.shape[2:]
    lowest_scales = []
    for dim in sizes:
        lowest_scales.append(min_size/dim)
    return lowest_scales


def train(real, opt: Config):
    """ Wrapper function for training. Calculates necessary scales then calls train_single_scale on each. """
    generators = []
    noise_maps = []
    noise_amplitudes = []

    min_scales = calc_lowest_possible_scale(real, opt.ker_size, opt.num_layer)

    # Calculate if scales need to be adjusted
    scales = []
    print('Scale Info:')
    for x in opt.scales:
        scales.append([max(x, min_scales[0]), max(x, min_scales[1]), max(x, min_scales[2])])
    print(scales)
    opt.num_scales = len(scales)

    if opt.use_multiple_inputs:
        # Multi Input is not tested for Minecraft
        reals = []
        for level in real:
            scaled_list = special_minecraft_downsampling(opt.num_scales, scales, level, opt.token_list)
            tmp_reals = [*scaled_list, level]
            reals.append(tmp_reals)
    else:
        # Get the "real" sample
        # Depending on if representations are used, downsampling is different
        use_hierarchy = False if opt.repr_type else True
        scaled_list = special_minecraft_downsampling(opt.num_scales, scales, real, opt.token_list, use_hierarchy)
        reals = [*scaled_list, real]
        print("Scaled Shapes:")
        for r in reals:
            print(r.shape)

    if opt.use_multiple_inputs:
        # Multi Input is not tested for Minecraft
        input_from_prev_scale = []
        for group in reals:
            input_from_prev_scale.append(torch.zeros_like(group[0]))

        stop_scale = len(reals[0])
    else:
        # Default
        input_from_prev_scale = torch.zeros_like(reals[0])
        stop_scale = len(reals)

    opt.stop_scale = stop_scale

    # Log the original input level(s) as an image
    if opt.use_multiple_inputs:
        # Multi Input is not tested for Minecraft
        for i, level in enumerate(real):
            try:
                subprocess.call(["wine", '--version'])
                obj_pth = os.path.join(opt.out_, "objects/real")
                os.makedirs(obj_pth, exist_ok=True)
                real_obj_pth = render_minecraft(opt.input_names[i], opt.coords, obj_pth, "real")
                wandb.log({"real": wandb.Object3D(open(real_obj_pth))}, commit=False)
            except OSError:
                pass
    else:
        # Default: One image
        try:
            # Check if wine is installed (Linux), then render
            subprocess.call(["wine", '--version'])
            obj_pth = os.path.join(opt.out_, "objects/real")
            os.makedirs(obj_pth, exist_ok=True)
            real_obj_pth = render_minecraft(opt.input_name, opt.coords, obj_pth, "real")
            wandb.log({"real": wandb.Object3D(open(real_obj_pth))}, commit=False)
        except OSError:
            pass
        os.makedirs("%s/state_dicts" % (opt.out_), exist_ok=True)

    # Training Loop
    for current_scale in range(0, stop_scale):
        opt.outf = "%s/%d" % (opt.out_, current_scale)
        try:
            os.makedirs(opt.outf)
        except OSError:
            pass

        opt.nc_current = real.shape[1]

        # If it's the last scale, we need softmax, otherwise not
        if current_scale < (stop_scale-1):
            use_softmax = False  # if true, always softmax
        else:
            use_softmax = False  # if both false, never softmax (useful if repr are used) TODO: Make this an option

        # Initialize models
        D, G = init_models(opt, use_softmax)

        # Actually train the current scale
        z_opt, input_from_prev_scale, G = train_single_scale(D,  G, reals, generators, noise_maps,
                                                             input_from_prev_scale, noise_amplitudes, opt)

        # Reset grads and save current scale
        G = reset_grads(G, False)
        G.eval()
        D = reset_grads(D, False)
        D.eval()

        generators.append(G)
        noise_maps.append(z_opt)
        noise_amplitudes.append(opt.noise_amp)

        torch.save(noise_maps, "%s/noise_maps.pth" % (opt.out_))
        torch.save(generators, "%s/generators.pth" % (opt.out_))
        torch.save(reals, "%s/reals.pth" % (opt.out_))
        torch.save(noise_amplitudes, "%s/noise_amplitudes.pth" % (opt.out_))
        torch.save(opt.num_layer, "%s/num_layer.pth" % (opt.out_))
        torch.save(opt.token_list, "%s/token_list.pth" % (opt.out_))
        wandb.save("%s/*.pth" % opt.out_)

        torch.save(G.state_dict(), "%s/state_dicts/G_%d.pth" % (opt.out_, current_scale))
        wandb.save("%s/state_dicts/*.pth" % opt.out_)

        del D, G

    return generators, noise_maps, reals, noise_amplitudes
