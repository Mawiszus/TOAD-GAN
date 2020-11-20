# Code inspired by https://github.com/tamarott/SinGAN
from config import Config
import os

import torch
import wandb
from tqdm import tqdm

from mario.level_utils import one_hot_to_ascii_level, token_to_group
from mario.tokens import TOKEN_GROUPS as MARIO_TOKEN_GROUPS
from zelda.tokens import TOKEN_GROUPS as ZELDA_TOKEN_GROUPS
from megaman.tokens import TOKEN_GROUPS as MEGAMAN_TOKEN_GROUPS
from mariokart.tokens import TOKEN_GROUPS as MARIOKART_TOKEN_GROUPS
from mario.special_mario_downsampling import special_mario_downsampling
from zelda.special_zelda_downsampling import special_zelda_downsampling
from megaman.special_megaman_downsampling import special_megaman_downsampling
from mariokart.special_mariokart_downsampling import special_mariokart_downsampling
from minecraft.special_minecraft_downsampling import special_minecraft_downsampling
from models import init_models, reset_grads, restore_weights
from models.generator import Level_GeneratorConcatSkip2CleanAdd
from train_single_scale import train_single_scale


def train(real, opt: Config):
    """ Wrapper function for training. Calculates necessary scales then calls train_single_scale on each. """
    generators = []
    noise_maps = []
    noise_amplitudes = []

    if opt.game == 'mario':
        token_group = MARIO_TOKEN_GROUPS
    elif opt.game == 'zelda':
        token_group = ZELDA_TOKEN_GROUPS
    elif opt.game == 'megaman':
        token_group = MEGAMAN_TOKEN_GROUPS
    elif opt.game == 'minecraft':
        token_group = None
    else:  # if opt.game == 'mariokart':
        token_group = MARIOKART_TOKEN_GROUPS

    if opt.game == 'minecraft':
        scales = [[x, x, x] for x in opt.scales]
        opt.num_scales = len(scales)
    else:
        scales = [[x, x] for x in opt.scales]
        opt.num_scales = len(scales)

    if opt.game == 'mario':
        downsampling = special_mario_downsampling
    elif opt.game == 'zelda':
        downsampling = special_zelda_downsampling
    elif opt.game == 'megaman':
        downsampling = special_megaman_downsampling
    elif opt.game == 'minecraft':
        downsampling = special_minecraft_downsampling
    else:  # if opt.game == 'mariokart':
        downsampling = special_mariokart_downsampling

    if opt.use_multiple_inputs:
        reals = []
        for level in real:
            scaled_list = downsampling(opt.num_scales, scales, level, opt.token_list)
            tmp_reals = [*scaled_list, level]
            reals.append(tmp_reals)
    else:
        scaled_list = downsampling(opt.num_scales, scales, real, opt.token_list)
        reals = [*scaled_list, real]

    # If (experimental) token grouping feature is used:
    if opt.token_insert >= 0:
        if opt.use_multiple_inputs:
            raise NotImplementedError("Multiple inputs are not supported with token_insert")
        else:
            reals = [(token_to_group(r, opt.token_list, token_group) if i < opt.token_insert else r) for i, r in enumerate(reals)]
            reals.insert(opt.token_insert, token_to_group(reals[opt.token_insert], opt.token_list, token_group))

    if opt.use_multiple_inputs:
        input_from_prev_scale = []
        for group in reals:
            input_from_prev_scale.append(torch.zeros_like(group[0]))

        stop_scale = len(reals[0])
    else:
        input_from_prev_scale = torch.zeros_like(reals[0])
        stop_scale = len(reals)

    opt.stop_scale = stop_scale

    # Log the original input level(s) as an image
    if opt.use_multiple_inputs:
        for i, level in enumerate(real):
            img = opt.ImgGen.render(one_hot_to_ascii_level(level, opt.token_list))
            wandb.log({"real" + str(i): wandb.Image(img)}, commit=False)
            os.makedirs("%s/state_dicts" % (opt.out_), exist_ok=True)
    else:
        # Default: One image
        if opt.ImgGen is not None:
            img = opt.ImgGen.render(one_hot_to_ascii_level(real, opt.token_list))
            wandb.log({"real": wandb.Image(img)}, commit=False)
        os.makedirs("%s/state_dicts" % (opt.out_), exist_ok=True)

    # Training Loop
    for current_scale in range(0, stop_scale):
        opt.outf = "%s/%d" % (opt.out_, current_scale)
        try:
            os.makedirs(opt.outf)
        except OSError:
            pass

        # If we are seeding, we need to adjust the number of channels
        if current_scale < (opt.token_insert + 1):  # (stop_scale - 1):
            opt.nc_current = len(token_group)
        else:
            opt.nc_current = len(opt.token_list)

        # Initialize models
        D, G = init_models(opt)
        # If we are seeding, the weights after the seed need to be adjusted
        if current_scale == (opt.token_insert + 1):  # (stop_scale - 1):
            D, G = restore_weights(D, G, current_scale, opt)

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
