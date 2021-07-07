# Code inspired by https://github.com/tamarott/SinGAN
import subprocess
from generate_samples import generate_samples
from train import train
from minecraft.level_utils import read_level as mc_read_level
from minecraft.level_utils import clear_empty_world
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
        return [opt.input_name.split(".")[0], str(opt.scales), str(opt.repr_type), opt.input_area_name]


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
    run = wandb.init(project="world-gan", entity="tnt", tags=get_tags(opt),
                     config=opt, dir=opt.out)
    opt.out_ = run.dir

    # Relic from old code, where the results where rendered with a generator
    opt.ImgGen = None

    # Check if wine is available to use (Linux) and clear the MC world examples will be saved to
    try:
        subprocess.call(["wine", "--version"])
        clear_empty_world(opt.output_dir, opt.output_name)
    except OSError:
        pass

    # Read level according to input arguments
    real = mc_read_level(opt)

    # Multi-Input is taken over from old code but not implemented for Minecraft
    if opt.use_multiple_inputs:
        logger.info("Multiple inputs are not implemented yet for Minecraft.")
        raise NotImplementedError
        # for i, r in enumerate(real):
        #     real[i] = r.to(opt.device)
        # opt.level_shape = real[0].shape[2:]
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
                     noise_amplitudes, opt, render_images=False, num_samples=100, in_s=in_s)


if __name__ == "__main__":
    main()
