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

from mario.level_utils import one_hot_to_ascii_level, group_to_token, token_to_group, read_level, read_level_from_file, place_a_mario_token, repr_to_ascii_level
from mario.level_image_gen import LevelImageGen as MarioLevelGen
from zelda.special_zelda_downsampling import special_zelda_downsampling
from zelda.level_image_gen import LevelImageGen as ZeldaLevelGen
from megaman.special_megaman_downsampling import special_megaman_downsampling
from megaman.level_image_gen import LevelImageGen as MegamanLevelGen
from mariokart.special_mariokart_downsampling import special_mariokart_downsampling
from mariokart.level_image_gen import LevelImageGen as MariokartLevelGen
from mario.tokens import REPLACE_TOKENS as MARIO_REPLACE_TOKENS
from megaman.tokens import REPLACE_TOKENS as MEGAMAN_REPLACE_TOKENS
from mariokart.tokens import REPLACE_TOKENS as MARIOKART_REPLACE_TOKENS
from mario.tokens import TOKEN_GROUPS as MARIO_TOKEN_GROUPS
from zelda.tokens import TOKEN_GROUPS as ZELDA_TOKEN_GROUPS
from megaman.tokens import TOKEN_GROUPS as MEGAMAN_TOKEN_GROUPS
from mariokart.tokens import TOKEN_GROUPS as MARIOKART_TOKEN_GROUPS
from mario.special_mario_downsampling import special_mario_downsampling
from minecraft.special_minecraft_downsampling import special_minecraft_downsampling
from minecraft.level_utils import one_hot_to_blockdata_level, save_level_to_world, clear_empty_world, save_oh_to_wrld_directly
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
    # make 1000 samples for each mario generator specified in the code.
    make_mario_samples: bool = False
    seed_mariokart_road: bool = False  # seed mariokart generators with a road image
    # make token insert experiment (experimental!)
    token_insert_experiment: bool = False
    not_cuda: bool = False  # disables cuda
    generators_dir: Optional[str] = None

    def process_args(self):
        super().process_args()
        self.seed_road: Optional[torch.Tensor] = None
        if (not self.out_) and (not self.make_mario_samples):
            raise Exception(
                '--out_ is required (--make_mario_samples experiment is the exception)')


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

    # Check which game we are using for token groups
    if opt.game == 'mario':
        token_groups = MARIO_TOKEN_GROUPS
    elif opt.game == 'zelda':
        token_groups = ZELDA_TOKEN_GROUPS
    elif opt.game == 'megaman':
        token_groups = MEGAMAN_TOKEN_GROUPS
    elif opt.game == 'mariokart':
        token_groups = MARIOKART_TOKEN_GROUPS
    elif opt.game == 'minecraft':
        token_groups = []
    else:
        raise NameError("name of --game not recognized. Supported: mario, zelda, megaman, mariokart, minecraft")

    if opt.game == 'minecraft':  # easy setter for now, maybe make more general later
        dim = 3
    else:
        dim = 2

    # Main sampling loop
    for sc, (G, Z_opt, noise_amp) in enumerate(zip(generators, noise_maps, noise_amplitudes)):

        # Make directories
        dir2save = opt.out_ + '/' + save_dir
        try:
            os.makedirs(dir2save, exist_ok=True)
            if render_images:
                if opt.game != 'minecraft':
                    os.makedirs("%s/img" % dir2save, exist_ok=True)
            if save_tensors:
                os.makedirs("%s/torch" % dir2save, exist_ok=True)
            if dim == 2:
                os.makedirs("%s/txt" % dir2save, exist_ok=True)
        except OSError:
            pass

        if current_scale >= len(generators):
            break  # if we do not start at current_scale=0 we need this
        elif sc < current_scale:
            if opt.token_insert >= 0:
                # Convert to ascii level
                token_list = [list(group.keys())[0] for group in token_groups]
                level = one_hot_to_ascii_level(in_s[0].detach().unsqueeze(0), token_list)

                # Render and save level image
                if render_images and opt.game != 'minecraft':
                    img = opt.ImgGen.render(level)
                    img.save("%s/img/%d_sc%d.png" %
                             (dir2save, opt.token_insert, current_scale))
            continue

        logger.info("Generating samples at scale {}", current_scale)

        # Padding (should be chosen according to what was trained with)
        n_pad = int(1*opt.num_layer)
        if not opt.pad_with_noise:
            if dim == 2:
                m = nn.ZeroPad2d(int(n_pad))  # pad with zeros
            else:
                # m = nn.ConstantPad3d(int(n_pad), 0)  # pad with zeros
                m = nn.ReplicationPad3d(int(n_pad))  # pad with reflected noise
        else:
            if dim == 2:
                m = nn.ReflectionPad2d(int(n_pad))  # pad with reflected noise
            else:
                m = nn.ReplicationPad3d(int(n_pad))  # pad with reflected noise

        # Calculate shapes to generate
        if 0 < gen_start_scale <= current_scale:  # Special case! Can have a wildly different shape through in_s
            nz = []
            if dim == 2:
                scale_v = in_s.shape[-2] / (noise_maps[gen_start_scale - 1].shape[-2] - n_pad * 2)
                scale_h = in_s.shape[-1] / (noise_maps[gen_start_scale - 1].shape[-1] - n_pad * 2)
                nz.append(int(round(((Z_opt.shape[-2] - n_pad * 2) * scale_v))))
                nz.append(int(round(((Z_opt.shape[-1] - n_pad * 2) * scale_h))))
            else:
                scale_v = in_s.shape[-1] / (noise_maps[gen_start_scale - 1].shape[-1] - n_pad * 2)
                scale_h = in_s.shape[-3] / (noise_maps[gen_start_scale - 1].shape[-3] - n_pad * 2)
                scale_d = in_s.shape[-2] / (noise_maps[gen_start_scale - 1].shape[-2] - n_pad * 2)
                nz.append(int(round(((Z_opt.shape[-3] - n_pad * 2) * scale_h))))
                nz.append(int(round(((Z_opt.shape[-2] - n_pad * 2) * scale_d))))
                nz.append(int(round(((Z_opt.shape[-1] - n_pad * 2) * scale_v))))  # mc ordering is y, z, x
        else:
            nz = []
            if dim == 2:
                nz.append(int(round(((Z_opt.shape[-2] - n_pad * 2) * scale_v))))
                nz.append(int(round(((Z_opt.shape[-1] - n_pad * 2) * scale_h))))
            else:
                nz.append(int(round(((Z_opt.shape[-3] - n_pad * 2) * scale_h))))
                nz.append(int(round(((Z_opt.shape[-2] - n_pad * 2) * scale_d))))
                nz.append(int(round(((Z_opt.shape[-1] - n_pad * 2) * scale_v))))  # mc ordering is y, z, x

        # Save list of images of previous scale and clear current images
        images_prev = images_cur
        images_cur = []

        # Token insertion (Experimental feature! Generator needs to be trained with it)
        if current_scale < (opt.token_insert + 1):
            channels = len(token_groups)
            if in_s is not None and in_s.shape[1] != channels:
                old_in_s = in_s
                in_s = token_to_group(in_s, opt.token_list, token_groups)
        else:
            channels = reals[0].shape[1]
            if in_s is not None and in_s.shape[1] != channels:
                old_in_s = in_s
                in_s = group_to_token(in_s, opt.token_list, token_groups)

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

                # Transform to token groups if there is token insertion
                if current_scale == (opt.token_insert + 1):
                    I_prev = group_to_token(
                        I_prev, opt.token_list, token_groups)

            if dim == 2:
                I_prev = interpolate(I_prev, nz, mode='nearest')
            else:
                I_prev = interpolate3D(I_prev, nz, mode='bilinear', align_corners=True)
            I_prev = m(I_prev)

            # We take the optimized noise map Z_opt as an input if we start generating on later scales
            if current_scale < gen_start_scale:
                z_curr = Z_opt

            # Define correct token list (dependent on token insertion)
            if opt.token_insert >= 0 and z_curr.shape[1] == len(token_groups):
                token_list = [list(group.keys())[0] for group in token_groups]
            else:
                # if we have a different block2repr than during training, we need to update the token_list
                if opt.repr_type is not None:
                    if opt.token_list == list(opt.block2repr.keys()):
                        token_list = opt.token_list
                    else:
                        token_list = list(opt.block2repr.keys())
                else:
                    token_list = opt.token_list

            ###########
            # Generate!
            ###########
            z_in = noise_amp * z_curr + I_prev
            I_curr = G(z_in.detach(), I_prev, temperature=1)

            # Allow road insertion in mario kart levels
            if opt.game == 'mariokart':
                if current_scale == 0 and opt.seed_road is not None:
                    for token in token_list:
                        if token == 'R':  # Road map!
                            tmp = opt.seed_road.clone().to(opt.device)
                            I_curr[0, token_list.index(token)] = tmp

                        # Tokens that can only appear on roads
                        elif token in ['O', 'Q', 'C', '<']:
                            I_curr[0, token_list.index(
                                token)] *= opt.seed_road.to(opt.device)

                        else:  # Other tokens like walls
                            I_curr[0, token_list.index(token)] = torch.min(I_curr[0, token_list.index(token)],
                                                                           1 - opt.seed_road.to(opt.device))

            # Save all scales

            # if True:
            # Save scale 0 and last scale
            # if current_scale == 0 or current_scale == len(reals) - 1:
            # Save only last scale
            if current_scale == len(reals) - 1:

                # Convert to level
                if len(opt.level_shape) == 2:
                    if not opt.repr_type:
                        to_level = one_hot_to_ascii_level
                    else:
                        to_level = repr_to_ascii_level
                else:
                    to_level = one_hot_to_blockdata_level

                # Save level txt
                if dim == 2:
                    level = to_level(I_curr.detach(), token_list, opt.block2repr)
                    # Render and save level image
                    if render_images:
                        img = opt.ImgGen.render(level)
                        img.save("%s/img/%d_sc%d.png" %
                                 (dir2save, n, current_scale))

                    with open("%s/txt/%d_sc%d.txt" % (dir2save, n, current_scale), "w") as f:
                        f.writelines(level)
                else:
                    # Minecraft
                    if n == 0:  # in first step make folder and save real blockdata
                        os.makedirs("%s/torch_blockdata" % dir2save, exist_ok=True)
                        real_level = to_level(reals[current_scale], token_list, opt.block2repr, opt.repr_type)
                        torch.save(real_level, "%s/real_bdata.pt" % dir2save)
                        torch.save(token_list, "%s/token_list.pt" % dir2save)

                    level = to_level(I_curr.detach(), token_list, opt.block2repr, opt.repr_type)
                    torch.save(level, "%s/torch_blockdata/%d_sc%d.pt" % (dir2save, n, current_scale))
                    # save_path = "%s/txt/%d_sc%d.schem" % (dir2save, n, current_scale)
                    # new_schem = NanoMCSchematic(save_path, level.shape[:3])
                    # new_schem.set_blockdata(level)
                    # new_schem.saveToFile()
                    if render_images:
                        # Minecraft World
                        len_n = math.ceil(math.sqrt(num_samples))  # we arrange our samples in a square in the world
                        x, z = np.unravel_index(n, [len_n, len_n])  # get x, z pos according to index n
                        posx = x * (level.shape[0] + 5)
                        posz = z * (level.shape[2] + 5)
                        save_level_to_world(opt.output_dir, opt.output_name, (posx, 0, posz), level, token_list)
                        # save_oh_to_wrld_directly(opt.output_dir, opt.output_name, (posx, 0, posz), I_curr.detach(),
                        #                          opt.block2repr, opt.repr_type)
                        curr_coords = [[posx, posx + level.shape[0]],
                                       [0, level.shape[1]],
                                       [posz, posz + level.shape[2]]]
                        render_minecraft(opt, "%d" % current_scale, "%d" % n,
                                         opt.output_name, curr_coords, basepath=dir2save)

                # Save torch tensor
                if save_tensors:
                    torch.save(I_curr, "%s/torch/%d_sc%d.pt" %
                               (dir2save, n, current_scale))

            # Token insertion render (experimental!)
            # if current_scale == opt.token_insert:
            #     if old_in_s.shape[1] == len(token_groups):
            #         token_list = [list(group.keys())[0] for group in token_groups]
            #     else:
            #         token_list = opt.token_list
            #     level = one_hot_to_ascii_level(old_in_s.detach(), token_list)
            #     img = opt.ImgGen.render(level)
            #     img.save("%s/img/%d_sc%d.png" % (dir2save, n, current_scale))

            # Append current image
            images_cur.append(I_curr)

        # Go to next scale
        current_scale += 1

    return I_curr.detach()  # return last generated image (usually unused)


def generate_mario_samples(opt: GenerateSamplesConfig):

    # Generate many samples for all mario levels for large scale evaluation
    level_names = [f for f in sorted(os.listdir(
        opt.input_dir)) if f.endswith('.txt')]

    generator_dirs = {}
    for generator_dir in os.listdir(opt.generators_dir):
        run_dir = os.path.join(opt.generators_dir, generator_dir)
        with open(os.path.join(run_dir, "files", "config.yaml"), "r") as f:
            config = yaml.load(f)
        level_name = config["input_name"]["value"]
        generator_dirs[level_name] = os.path.join(run_dir, "files")

    for level_name in level_names:
        # New "input" mario level
        opt.input_name = level_name
        opt.out_ = generator_dirs[level_name]

        # Read level according to input arguments
        real_m = read_level(opt, None, MARIO_REPLACE_TOKENS).to(opt.device)

        # Load TOAD-GAN for current level
        generators_m, noise_maps_m, reals_m, noise_amplitudes_m = load_trained_pyramid(
            opt)

        # Set in_s and scales
        if opt.gen_start_scale == 0:  # starting in lowest scale
            in_s_m = None
            m_scale_v = 1.0
            # normalize all levels to length 16x200
            m_scale_h = 200 / real_m.shape[-1]
        else:  # if opt.gen_start_scale > 0
            # Only works with default level size if no in_s is provided (should not be reached)
            in_s_m = reals_m[opt.gen_start_scale]
            m_scale_v = 1.0
            m_scale_h = 1.0

        # Prefix for folder structure
        prefix_m = 'arbitrary'

        # Define directory
        s_dir_name_m = "%s_random_samples_v%.5f_h%.5f_start%d" % (
            prefix_m, m_scale_v, m_scale_h, opt.gen_start_scale)

        # Generate samples
        generate_samples(generators_m, noise_maps_m, reals_m, noise_amplitudes_m, opt, in_s=in_s_m,
                         scale_v=m_scale_v, scale_h=m_scale_h, current_scale=opt.gen_start_scale,
                         gen_start_scale=opt.gen_start_scale, num_samples=1000, render_images=False,
                         save_tensors=False, save_dir=s_dir_name_m)

        # For embedding experiment, copy levels to easy access folder
        samples_dir = opt.out_ + '/' + s_dir_name_m + '/txt'
        newpath = os.path.join(
            opt.generators_dir, "samples", opt.input_name[:-4])
        os.makedirs(newpath, exist_ok=True)
        for f in tqdm(os.listdir(samples_dir)):
            if f.endswith('.txt'):
                copyfile(os.path.join(samples_dir, f),
                         os.path.join(newpath, f))


if __name__ == '__main__':
    # NOTICE: The "output" dir is where the generator is located as with main.py, even though it is the "input" here

    opt = GenerateSamplesConfig().parse_args()

    if opt.make_mario_samples:
        # Code to make a large body of mario samples for other experiments
        opt.game = 'mario'
        sprite_path = opt.game + '/sprites'
        opt.ImgGen = MarioLevelGen(sprite_path)
        opt.gen_start_scale = 0  # Forced for this experiment

        generate_mario_samples(opt)

    elif opt.seed_mariokart_road:
        # Code to make mario kart seeded road examples
        opt.game = 'mariokart'
        sprite_path = opt.game + '/sprites'
        opt.ImgGen = MariokartLevelGen(sprite_path)
        replace_tokens = MARIOKART_REPLACE_TOKENS
        downsample = special_mariokart_downsampling
        opt.gen_start_scale = 0  # Forced for this experiment

        # Load level
        real = read_level(opt, None, replace_tokens).to(opt.device)
        # Load generator
        generators, noise_maps, reals, noise_amplitudes = load_trained_pyramid(
            opt)

        # Define paths to seed road image(s)
        seed_road_images = ['mariokart/seed_road/seed_road.png']
        # seed_road_images = ['mariokart/seed_road/MNIST_examples/eights/sample_%d.png' % im for im in range(20)]

        for i, img_path in enumerate(seed_road_images):
            # Read and convert seed road image
            seed_road_img = plt.imread(img_path)
            opt.seed_road = torch.Tensor(1 - seed_road_img[:, :, 0])

            # Scales have to be fitting with seed road image (preferably make seed road the size of scale 0 directly!)
            scale_0_h = reals[0].shape[-1] / reals[-1].shape[-1]
            scale_0_v = reals[0].shape[-2] / reals[-1].shape[-2]
            shape_r_h = round(opt.seed_road.shape[-1] / scale_0_h)
            shape_r_v = round(opt.seed_road.shape[-2] / scale_0_v)
            scale_h = shape_r_h / reals[-1].shape[-1]
            scale_v = shape_r_v / reals[-1].shape[-2]

            real_down = downsample(
                1, [[scale_v, scale_h]], real, opt.token_list)
            real_down = real_down[0]

            # in_s = torch.zeros((round(reals[-1].shape[-2]*scale_v), round(reals[-1].shape[-1]*scale_h)),
            in_s = torch.zeros(real_down.shape,
                               device=opt.device)  # necessary for correct input shape

            # Directory name
            s_dir_name = "random_road_samples_v%.5f_h%.5f_st%d_%d" % (
                opt.scale_v, opt.scale_h, opt.gen_start_scale, i)

            # Generate samples
            generate_samples(generators, noise_maps, reals, noise_amplitudes, opt, in_s=in_s,
                             scale_v=scale_v, scale_h=scale_h, save_dir=s_dir_name, num_samples=opt.num_samples)

    else:
        # Code to make samples for given generator
        token_insertion = True if (
            opt.token_insert >= 0) and opt.token_insert_experiment else False

        # Init game specific inputs
        replace_tokens = {}
        sprite_path = opt.game + '/sprites'
        if opt.game == 'mario':
            opt.ImgGen = MarioLevelGen(sprite_path)
            replace_tokens = MARIO_REPLACE_TOKENS
            downsample = special_mario_downsampling

        elif opt.game == 'mariokart':
            opt.ImgGen = MariokartLevelGen(sprite_path)
            replace_tokens = MARIOKART_REPLACE_TOKENS
            downsample = special_mariokart_downsampling

        elif opt.game == 'zelda':
            opt.ImgGen = ZeldaLevelGen(sprite_path)
            replace_tokens = {}
            downsample = special_zelda_downsampling

        elif opt.game == 'megaman':
            opt.ImgGen = MegamanLevelGen(sprite_path)
            replace_tokens = MEGAMAN_REPLACE_TOKENS
            downsample = special_megaman_downsampling

        elif opt.game == 'minecraft':
            opt.ImgGen = None
            replace_tokens = None
            clear_empty_world(opt.output_dir, opt.output_name)
            downsample = special_minecraft_downsampling

        else:
            NameError(
                "name of --game not recognized. Supported: mario, mariokart, minecraft, zelda, megaman")

        # Read level according to input arguments
        if opt.game == 'minecraft':
            real = mc_read_level(opt)
        else:
            real = read_level(opt, None, replace_tokens)

        if opt.use_multiple_inputs:
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

        # For Token insertion (Experimental!) --------------------------------------------------------------------------
        if token_insertion:
            # set seed level 0 is 1-1, 1 is 1-2, and 2 ia 1-3
            seed_level = 2

            # Load "other level" used for insertion
            opt_fakes = Namespace()
            if seed_level == 0:
                opt_fakes.input_name = "lvl_1-1.txt"
            elif seed_level == 1:
                opt_fakes.input_name = "lvl_1-2.txt"
            elif seed_level == 2:
                opt_fakes.input_name = "lvl_1-3.txt"
            elif seed_level == -1:  # seed with noise
                opt_fakes.input_name = "lvl_1-1.txt"  # should not matter
            opt_fakes.input_dir = "./input/mario"
            real_fakes = read_level(opt_fakes).to(opt.device)

            # Downsample "other level"
            real_fakes_down = special_mario_downsampling(1, [[opt.scales[-1], opt.scales[-1]]],
                                                         real_fakes, opt_fakes.token_list)

            run_dir = "/home/awiszus/Project/TOAD-GAN/output/wandb/"
            if seed_level == 0:  # Only done for mario levels 1 to 3 so far
                real_fakes = torch.load(run_dir + 'run-20200901_072636-1ycydnos/'
                                                  'arbitrary_random_samples_v1.00000_h0.24752_st0/torch/'
                                                  '1_sc0.pt')
            elif seed_level == 1:
                real_fakes = torch.load(run_dir + 'run-20200901_143818-3aeh4668/'
                                                  'arbitrary_random_samples_v1.00000_h0.31646_st0/torch/'
                                                  '4_sc0.pt')
            elif seed_level == 2:
                real_fakes = torch.load(run_dir + 'run-20200901_143829-3kcbthi9/'
                                                  '/arbitrary_random_samples_v1.00000_h0.33333_st0/torch/'
                                                  '10_sc0.pt')
            elif seed_level == -1:  # seed with noise
                fakes_shape = real_fakes_down[0].shape
                real_fakes = nn.Softmax2d()(torch.rand(fakes_shape, device=opt.device) * 50) * 1

            real_fakes = real_fakes.to(opt.device)

            cur_scale = opt.token_insert + 1
            in_s = real_fakes
            prefix = "seeded" + str(seed_level)

        # --------------------------------------------------------------------------------------------------------------

        else:
            cur_scale = 0

            # Get input shape for in_s
            if len(opt.level_shape) == 2:
                if opt.game == 'mario':
                    real_down = downsample(1, [[opt.scale_v, opt.scale_h]], real, opt.token_list, opt.repr_type)
                else:
                    real_down = downsample(1, [[opt.scale_v, opt.scale_h]], real, opt.token_list)
            else:
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
