# Code inspired by https://github.com/tamarott/SinGAN
from generate_samples import generate_samples
from train import train
from mariokart.tokens import REPLACE_TOKENS as MARIOKART_REPLACE_TOKENS
from megaman.tokens import REPLACE_TOKENS as MEGAMAN_REPLACE_TOKENS
from mario.tokens import REPLACE_TOKENS as MARIO_REPLACE_TOKENS
from mariokart.level_image_gen import LevelImageGen as MariokartLevelGen
from mariokart.special_mariokart_downsampling import special_mariokart_downsampling
from megaman.level_image_gen import LevelImageGen as MegamanLevelGen
from megaman.special_megaman_downsampling import special_megaman_downsampling
from zelda.level_image_gen import LevelImageGen as ZeldaLevelGen
from zelda.special_zelda_downsampling import special_zelda_downsampling
from mario.level_image_gen import LevelImageGen as MarioLevelGen
from mario.special_mario_downsampling import special_mario_downsampling
from mario.level_utils import read_level, read_level_from_file
from minecraft.level_utils import read_level as mc_read_level
from minecraft.level_utils import clear_empty_world
from minecraft.special_minecraft_downsampling import special_minecraft_downsampling
from config import Config
from loguru import logger
import wandb
import sys
import torch


def get_tags(opt):
    """ Get Tags for logging from input name. Helpful for wandb. """
    if opt.use_multiple_inputs:
        return [name.split(".")[0] for name in opt.input_names]
    else:
        return [opt.input_name.split(".")[0]]


def main():
    """ Main Training funtion. Parses inputs, inits logger, trains, and then generates some samples. """

    # torch.autograd.set_detect_anomaly(True)

    # Logger init
    logger.remove()
    logger.add(sys.stdout, colorize=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                      + "<level>{level}</level> | "
                      + "<light-black>{file.path}:{line}</light-black> | "
                      + "{message}")

    # Parse arguments
    opt = Config().parse_args()

    # Init wandb
    run = wandb.init(project="mario", tags=get_tags(opt),
                     config=opt, dir=opt.out)
    opt.out_ = run.dir

    # Init game specific inputs
    replace_tokens = {}
    sprite_path = opt.game + '/sprites'
    if opt.game == 'mario':
        opt.ImgGen = MarioLevelGen(sprite_path)
        replace_tokens = MARIO_REPLACE_TOKENS
        downsample = special_mario_downsampling
    elif opt.game == 'zelda':
        opt.ImgGen = ZeldaLevelGen(sprite_path)
        downsample = special_zelda_downsampling
    elif opt.game == 'megaman':
        opt.ImgGen = MegamanLevelGen(sprite_path, n_sheet=int(get_tags(opt)[0][-1]))
        replace_tokens = MEGAMAN_REPLACE_TOKENS
        downsample = special_megaman_downsampling
    elif opt.game == 'mariokart':
        opt.ImgGen = MariokartLevelGen(sprite_path)
        replace_tokens = MARIOKART_REPLACE_TOKENS
        downsample = special_mariokart_downsampling
    elif opt.game == 'minecraft':
        opt.ImgGen = None
        replace_tokens = None
        clear_empty_world(opt.output_dir, opt.output_name)
        downsample = special_minecraft_downsampling
    else:
        NameError("name of --game not recognized. Supported: mario, zelda, megaman, mariokart, minecraft")

    # Read level according to input arguments
    if opt.game == 'minecraft':
        real = mc_read_level(opt)
    else:
        real = read_level(opt, None, replace_tokens)

    if opt.use_multiple_inputs:
        for i, r in enumerate(real):
            real[i] = r.to(opt.device)
        opt.level_shape = real[0].shape[2:]
    else:
        real = real.to(opt.device)
        opt.level_shape = real.shape[2:]

    # Train!
    generators, noise_maps, reals, noise_amplitudes = train(real, opt)

    # Generate Samples of same size as level
    logger.info("Finished training! Generating random samples...")
    in_s = None
    if opt.use_multiple_inputs:
        use_reals = reals[0]
        use_maps = noise_maps[0]
    else:
        use_reals = reals
        use_maps = noise_maps
    generate_samples(generators, use_maps, use_reals,
                     noise_amplitudes, opt, num_samples=20, in_s=in_s)

    # Generate samples of smaller size than level
    # logger.info("Generating arbitrary sized random samples...")
    # scale_v = 0.8  # Arbitrarily chosen scales
    # scale_h = 0.4
    # scale_d = 0.5
    # if opt.use_multiple_inputs:
    #     tmp_real = real[0]
    # else:
    #     tmp_real = real
    # if len(opt.level_shape) == 2:
    #     real_down = downsample(1, [[scale_v, scale_h]], tmp_real, opt.token_list)
    # else:
    #     real_down = downsample(1, [[scale_v, scale_h, scale_d]], tmp_real, opt.token_list)
    # real_down = real_down[0]
    # # necessary for correct input shape
    # in_s = torch.zeros(real_down.shape, device=opt.device)
    # generate_samples(generators, use_maps, use_reals, noise_amplitudes, opt, in_s=in_s, num_samples=5,
    #                  scale_v=scale_v, scale_h=scale_h, scale_d=scale_d, save_dir="arbitrary_random_samples")


if __name__ == "__main__":
    main()
