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

from config import get_arguments, post_config
from mario.level_utils import one_hot_to_ascii_level, group_to_token, token_to_group, read_level
from mario.level_image_gen import LevelImageGen as MarioLevelGen
from mariokart.special_mariokart_downsampling import special_mariokart_downsampling
from mariokart.level_image_gen import LevelImageGen as MariokartLevelGen
from mario.tokens import REPLACE_TOKENS as MARIO_REPLACE_TOKENS
from mariokart.tokens import REPLACE_TOKENS as MARIOKART_REPLACE_TOKENS
from mario.tokens import TOKEN_GROUPS as MARIO_TOKEN_GROUPS
from mariokart.tokens import TOKEN_GROUPS as MARIOKART_TOKEN_GROUPS
from mario.special_mario_downsampling import special_mario_downsampling
from generate_noise import generate_spatial_noise
from models import load_trained_pyramid


def generate_samples(generators, noise_maps, reals, noise_amplitudes, opt, in_s=None, scale_v=1.0, scale_h=1.0,
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
    elif opt.game == 'mariokart':
        token_groups = MARIOKART_TOKEN_GROUPS
    else:
        token_groups = []
        NameError("name of --game not recognized. Supported: mario, mariokart")

    # Main sampling loop
    for G, Z_opt, noise_amp in zip(generators, noise_maps, noise_amplitudes):

        if current_scale >= len(generators):
            break  # if we do not start at current_scale=0 we need this

        logger.info("Generating samples at scale {}", current_scale)

        # Padding (should be chosen according to what was trained with)
        n_pad = int(1*opt.num_layer)
        if not opt.pad_with_noise:
            m = nn.ZeroPad2d(int(n_pad))  # pad with zeros
        else:
            m = nn.ReflectionPad2d(int(n_pad))  # pad with reflected noise

        # Calculate shapes to generate
        if 0 < gen_start_scale <= current_scale:  # Special case! Can have a wildly different shape through in_s
            scale_v = in_s.shape[-2] / (noise_maps[gen_start_scale-1].shape[-2] - n_pad * 2)
            scale_h = in_s.shape[-1] / (noise_maps[gen_start_scale-1].shape[-1] - n_pad * 2)
            nzx = (Z_opt.shape[-2] - n_pad * 2) * scale_v
            nzy = (Z_opt.shape[-1] - n_pad * 2) * scale_h
        else:
            nzx = (Z_opt.shape[-2] - n_pad * 2) * scale_v
            nzy = (Z_opt.shape[-1] - n_pad * 2) * scale_h

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
            channels = len(opt.token_list)
            if in_s is not None and in_s.shape[1] != channels:
                old_in_s = in_s
                in_s = group_to_token(in_s, opt.token_list, token_groups)

        # If in_s is none or filled with zeros reshape to correct size with channels
        if in_s is None:
            in_s = torch.zeros(reals[0].shape[0], channels, *reals[0].shape[2:]).to(opt.device)
        elif in_s.sum() == 0:
            in_s = torch.zeros(1, channels, *in_s.shape[-2:]).to(opt.device)

        # Generate num_samples samples in current scale
        for n in tqdm(range(0, num_samples, 1)):

            # Get noise image
            z_curr = generate_spatial_noise([1, channels, int(round(nzx)), int(round(nzy))], device=opt.device)
            z_curr = m(z_curr)

            # Set up previous image I_prev
            if (not images_prev) or current_scale == 0:  # if there is no "previous" image
                I_prev = in_s
            else:
                I_prev = images_prev[n]

                # Transform to token groups if there is token insertion
                if current_scale == (opt.token_insert + 1):
                    I_prev = group_to_token(I_prev, opt.token_list, token_groups)

            I_prev = interpolate(I_prev, [int(round(nzx)), int(round(nzy))], mode='bilinear', align_corners=False)
            I_prev = m(I_prev)

            # We take the optimized noise map Z_opt as an input if we start generating on later scales
            if current_scale < gen_start_scale:
                z_curr = Z_opt

            # Define correct token list (dependent on token insertion)
            if opt.token_insert >= 0 and z_curr.shape[1] == len(token_groups):
                token_list = [list(group.keys())[0] for group in token_groups]
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

                        elif token in ['O', 'Q', 'C', '<']:  # Tokens that can only appear on roads
                            I_curr[0, token_list.index(token)] *= opt.seed_road.to(opt.device)

                        else:  # Other tokens like walls
                            I_curr[0, token_list.index(token)] = torch.min(I_curr[0, token_list.index(token)],
                                                                           1 - opt.seed_road.to(opt.device))

            # Save all scales
            # if True:
            # Save scale 0 and last scale
            # if current_scale == 0 or current_scale == len(reals) - 1:
            # Save only last scale
            if current_scale == len(reals) - 1:
                dir2save = opt.out_ + '/' + save_dir

                # Make directories
                try:
                    os.makedirs(dir2save, exist_ok=True)
                    if render_images:
                        os.makedirs("%s/img" % dir2save, exist_ok=True)
                    if save_tensors:
                        os.makedirs("%s/torch" % dir2save, exist_ok=True)
                    os.makedirs("%s/txt" % dir2save, exist_ok=True)
                except OSError:
                    pass

                # Convert to ascii level
                level = one_hot_to_ascii_level(I_curr.detach(), token_list)

                # Render and save level image
                if render_images:
                    img = opt.ImgGen.render(level)
                    img.save("%s/img/%d_sc%d.png" % (dir2save, n, current_scale))

                # Save level txt
                with open("%s/txt/%d_sc%d.txt" % (dir2save, n, current_scale), "w") as f:
                    f.writelines(level)

                # Save torch tensor
                if save_tensors:
                    torch.save(I_curr, "%s/torch/%d_sc%d.pt" % (dir2save, n, current_scale))

                # Token insertion render (experimental!)
                if opt.token_insert >= 0 and current_scale >= 1:
                    if old_in_s.shape[1] == len(token_groups):
                        token_list = [list(group.keys())[0] for group in token_groups]
                    else:
                        token_list = opt.token_list
                    level = one_hot_to_ascii_level(old_in_s.detach(), token_list)
                    img = opt.ImgGen.render(level)
                    img.save("%s/img/%d_sc%d.png" % (dir2save, n, current_scale - 1))

            # Append current image
            images_cur.append(I_curr)

        # Go to next scale
        current_scale += 1

    return I_curr.detach()  # return last generated image (usually unused)


def generate_mario_samples(opt_m):

    # Generate many samples for all mario levels for large scale evaluation
    level_names = [f for f in os.listdir("./input") if f.endswith('.txt')]
    level_names.sort()

    # Directory with saved runs
    run_dir_m = "/home/awiszus/Project/TOAD-GAN/wandb/"

    for generator_level in range(0, len(level_names)):
        # New "input" mario level
        opt_m.input_name = level_names[generator_level]

        # New "output" folder
        if generator_level == 0:
            opt_m.out_ = run_dir_m + "run-20200605_101920-21l6f6ke"  # level 1 (1-1)
        elif generator_level == 1:
            opt_m.out_ = run_dir_m + "run-20200609_141816-2cxsre2o"  # level 2 (1-2)
        elif generator_level == 2:
            opt_m.out_ = run_dir_m + "run-20200609_144802-30asxofq"  # level 3 (1-3)
        elif generator_level == 3:
            opt_m.out_ = run_dir_m + "run-20200616_082954-20htkyzv"  # level 4 (2-1)
        elif generator_level == 4:
            opt_m.out_ = run_dir_m + "run-20200616_074234-19xr2e3o"  # level 5 (3-1)
        elif generator_level == 5:
            opt_m.out_ = run_dir_m + "run-20200616_093747-2ulvs4fh"  # level 6 (3-3)
        elif generator_level == 6:
            opt_m.out_ = run_dir_m + "run-20200616_102830-flwggm0z"  # level 7 (4-1)
        elif generator_level == 7:
            opt_m.out_ = run_dir_m + "run-20200616_114258-1uwt2v80"  # level 8 (4-2)
        elif generator_level == 8:
            opt_m.out_ = run_dir_m + "run-20200618_072750-3mfpkr81"  # level 9 (5-1)
        elif generator_level == 9:
            opt_m.out_ = run_dir_m + "run-20200618_093240-3aeol9gd"  # level 10 (5-3)
        elif generator_level == 10:
            opt_m.out_ = run_dir_m + "run-20200618_125519-3r0bngi4"  # level 11 (6-1)
        elif generator_level == 11:
            opt_m.out_ = run_dir_m + "run-20200618_131406-2ai6f3cl"  # level 12 (6-2)
        elif generator_level == 12:
            opt_m.out_ = run_dir_m + "run-20200618_133803-bxjlb36v"  # level 13 (6-3)
        elif generator_level == 13:
            opt_m.out_ = run_dir_m + "run-20200619_074635-14l988af"  # level 14 (7-1)
        elif generator_level == 14:
            opt_m.out_ = run_dir_m + "run-20200619_094438-lo7f1hqb"  # level 15 (8-1)

        # Read level according to input arguments
        real_m = read_level(opt_m, None, MARIO_REPLACE_TOKENS).to(opt_m.device)

        # Load TOAD-GAN for current level
        generators_m, noise_maps_m, reals_m, noise_amplitudes_m = load_trained_pyramid(opt_m)

        # Set in_s and scales
        if opt_m.gen_start_scale == 0:  # starting in lowest scale
            in_s_m = None
            m_scale_v = 1.0
            m_scale_h = 200 / real_m.shape[-1]  # normalize all levels to length 16x200
        else:  # if opt.gen_start_scale > 0
            # Only works with default level size if no in_s is provided (should not be reached)
            in_s_m = reals_m[opt_m.gen_start_scale]
            m_scale_v = 1.0
            m_scale_h = 1.0

        # Prefix for folder structure
        prefix_m = 'arbitrary'

        # Define directory
        s_dir_name_m = "%s_random_samples_v%.5f_h%.5f_start%d" % (
            prefix_m, m_scale_v, m_scale_h, opt_m.gen_start_scale)

        # Generate samples
        generate_samples(generators_m, noise_maps_m, reals_m, noise_amplitudes_m, opt_m, in_s=in_s_m,
                         scale_v=m_scale_v, scale_h=m_scale_h, current_scale=opt_m.gen_start_scale,
                         gen_start_scale=opt_m.gen_start_scale, num_samples=1000, render_images=False,
                         save_tensors=False, save_dir=s_dir_name_m)

        # For embedding experiment, copy levels to easy access folder
        samples_dir = opt_m.out_ + '/' + s_dir_name_m + '/txt'
        newpath = "./input/umap_images/baselines/" + opt_m.input_name[:-4]
        os.makedirs(newpath, exist_ok=True)
        for f in tqdm(os.listdir(samples_dir)):
            if f.endswith('.txt'):
                copyfile(os.path.join(samples_dir, f), os.path.join(newpath, f))


if __name__ == '__main__':
    # NOTICE: The "output" dir is where the generator is located as with main.py, even though it is the "input" here

    # Parse arguments
    parse = get_arguments()
    parse.add_argument("--out_", help="folder containing generator files")
    parse.add_argument("--scale_v", type=float, help="vertical scale factor", default=1.0)
    parse.add_argument("--scale_h", type=float, help="horizontal scale factor", default=1.0)
    parse.add_argument("--gen_start_scale", type=int, help="scale to start generating in", default=0)
    parse.add_argument("--num_samples", type=int, help="number of samples to be generated", default=10)
    parse.add_argument("--make_mario_samples", action="store_true", help="make 1000 samples for each mario generator"
                                                                         "specified in the code.", default=False)
    parse.add_argument("--seed_mariokart_road", action="store_true", help="seed mariokart generators with a road image",
                       default=False)
    parse.add_argument("--token_insert_experiment", action="store_true", help="make token insert experiment "
                                                                              "(experimental!)", default=False)
    opt = parse.parse_args()

    if (not opt.out_) and (not opt.make_mario_samples):
        parse.error('--out_ is required (--make_mario_samples experiment is the exception)')

    opt = post_config(opt)

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
        generators, noise_maps, reals, noise_amplitudes = load_trained_pyramid(opt)

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

            real_down = downsample(1, [[scale_v, scale_h]], real, opt.token_list)
            real_down = real_down[0]

            # in_s = torch.zeros((round(reals[-1].shape[-2]*scale_v), round(reals[-1].shape[-1]*scale_h)),
            in_s = torch.zeros(real_down.shape,
                               device=opt.device)  # necessary for correct input shape

            # Directory name
            s_dir_name = "random_road_samples_v%.5f_h%.5f_st%d_%d" % (opt.scale_v, opt.scale_h, opt.gen_start_scale, i)

            # Generate samples
            generate_samples(generators, noise_maps, reals, noise_amplitudes, opt, in_s=in_s,
                             scale_v=scale_v, scale_h=scale_h, save_dir=s_dir_name, num_samples=opt.num_samples)

    else:
        # Code to make samples for given generator
        token_insertion = True if opt.token_insert and opt.token_insert_experiment else False

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

        else:
            NameError("name of --game not recognized. Supported: mario, mariokart")

        # Load level
        real = read_level(opt, None, replace_tokens).to(opt.device)
        # Load Generator
        generators, noise_maps, reals, noise_amplitudes = load_trained_pyramid(opt)

        # For Token insertion (Experimental!) --------------------------------------------------------------------------
        if token_insertion:
            # set seed level (update opt_fakes.input_name accordingly)
            seed_level = 0

            # Load "other level" used for insertion
            opt_fakes = Namespace()
            opt_fakes.input_name = "lvl_1-1.txt"
            opt_fakes.input_dir = "./input"
            real_fakes = read_level(opt_fakes).to(opt.device)

            # Downsample "other level"
            real_fakes_down = special_mario_downsampling(1, [[opt.scales[-1], opt.scales[-1]]],
                                                         real_fakes, opt_fakes.token_list)

            run_dir = "/home/awiszus/Project/TOAD-GAN/wandb/"
            if seed_level == 0:  # Only done for mario levels 1 to 3 so far
                real_fakes = torch.load(run_dir + 'run-20200311_113148-6fmy47ks/'
                                                  'arbitrary_random_samples_v1_h0.297029702970297_start0/torch/'
                                                  '1_sc0.pt',
                                        device=opt.device)
            elif seed_level == 1:
                real_fakes = torch.load(run_dir + 'run-20200311_113200-55smfqkb/'
                                                  'arbitrary_random_samples_v1_h0.379746835443038_start0/torch/'
                                                  '7_sc0.pt',
                                        device=opt.device)
            elif seed_level == 2:
                real_fakes = torch.load(run_dir + 'run-20200311_121708-jnus8kfl/'
                                                  'arbitrary_random_samples_v1_h0.4_start0/torch/'
                                                  '5_sc0.pt',
                                        device=opt.device)
            elif seed_level == -1:  # seed with noise
                fakes_shape = real_fakes_down[0].shape
                real_fakes = nn.Softmax2d()(torch.rand(fakes_shape, device=opt.device) * 50) * 1

            in_s = real_fakes
            prefix = "seeded"

        # --------------------------------------------------------------------------------------------------------------

        else:
            # Get input shape for in_s
            real_down = downsample(1, [[opt.scale_v, opt.scale_h]], real, opt.token_list)
            real_down = real_down[0]
            in_s = torch.zeros_like(real_down, device=opt.device)
            prefix = "arbitrary"

        # Directory name
        s_dir_name = "%s_random_samples_v%.5f_h%.5f_st%d" % (prefix, opt.scale_v, opt.scale_h, opt.gen_start_scale)

        generate_samples(generators, noise_maps, reals, noise_amplitudes, opt, in_s=in_s,
                         scale_v=opt.scale_v, scale_h=opt.scale_h, save_dir=s_dir_name, num_samples=opt.num_samples)


