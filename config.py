# Code based on https://github.com/tamarott/SinGAN
import argparse
import random
from typing import List, Literal, Optional, Union

import numpy as np
import torch
from torch import cuda
from tap import Tap

from utils import set_seed, load_pkl
from mariokart.level_image_gen import LevelImageGen as MariokartLevelGen
from zelda.level_image_gen import LevelImageGen as ZeldaLevelGen
from mario.level_image_gen import LevelImageGen as MarioLevelGen
from megaman.level_image_gen import LevelImageGen as MegamanLevelGen
from minecraft.block_autoencoder import Encoder, Decoder


class Config(Tap):
    game: Literal["mario", "mariokart", "megaman",
                  "zelda", "minecraft"] = "mario"  # Which game is to be used?
    not_cuda: bool = False  # disables cuda
    netG: str = ""  # path to netG (to continue training)
    netD: str = ""  # path to netD (to continue training)
    manualSeed: Optional[int] = None
    out: str = "output"  # output directory
    input_dir: str = "input/mario"  # input directory
    input_name: str = "lvl_1-1.txt"  # input level filename
    # input level names (if multiple inputs are used)
    input_names: List[str] = ["lvl_1-1.txt", "lvl_1-2.txt"]
    # use mulitple inputs for training (use --input-names instead of --input-name)
    use_multiple_inputs: bool = False
    # if minecraft is used, which coords are used from the world? Which world do we save to?
    coords: int = None  # will be a tuple later (How do I secify a tuple here?)
    output_dir: str = "../minecraft_worlds/"  # folder with worlds
    output_name: str = "Gen_Empty_World"  # name of the world to generate in
    nfc: int = 64  # number of filters for conv layers
    ker_size: int = 3  # kernel size for conv layers
    num_layer: int = 3  # number of layers
    scales: List[float] = [0.75, 0.5, 0.25]  # Scales descending (< 1 and > 0)
    noise_update: float = 0.1  # additive noise weight
    # use reflection padding? (makes edges random)
    pad_with_noise: bool = False
    niter: int = 4000  # number of epochs to train per scale
    gamma: float = 0.1  # scheduler gamma
    lr_g: float = 0.0005  # generator learning rate
    lr_d: float = 0.0005  # discriminator learning rate
    beta1: float = 0.5  # optimizer beta
    Gsteps: int = 3  # generator inner steps
    Dsteps: int = 3  # discriminator inner steps
    lambda_grad: float = 0.1  # gradient penalty weight
    alpha: int = 100  # reconstruction loss weight
    # layer in which token groupings will be split out (<-2 means no grouping at all)
    token_insert: int = -2
    token_list: List[str] = ['!', '#', '-', '1', '@', 'C', 'S',
                             'U', 'X', 'g', 'k', 't']  # default list of 1-1

    repr_type: str = None  # Which representation type to use, currently [None, block2vec, autoencoder]

    def __init__(self,
                 *args,
                 underscores_to_dashes: bool = False,
                 explicit_bool: bool = False,
                 **kwargs):
        super().__init__(args, underscores_to_dashes, explicit_bool, kwargs)

    def process_args(self):
        self.device = torch.device("cpu" if self.not_cuda else "cuda:0")
        if cuda.is_available() and self.not_cuda:
            print(
                "WARNING: You have a CUDA device, so you should probably run with --cuda")

        if self.manualSeed is None:
            self.manualSeed = random.randint(1, 10000)
        print("Random Seed: ", self.manualSeed)
        set_seed(self.manualSeed)

        # Defaults for other namespace values that will be overwritten during runtime
        self.nc_current = 12  # n tokens of level 1-1
        if not hasattr(self, "out_"):
            self.out_ = "%s/%s/" % (self.out, self.input_name[:-4])
        self.outf = "0"  # changes with each scale trained
        # number of scales is implicitly defined
        self.num_scales = len(self.scales)
        self.noise_amp = 1.0  # noise amp for lowest scale always starts at 1
        self.seed_road = None  # for mario kart seed roads after training
        # which scale to stop on - usually always last scale defined
        self.stop_scale = self.num_scales
        self.ImgGen: Union[MarioLevelGen, ZeldaLevelGen,
                           MegamanLevelGen, MariokartLevelGen] = MarioLevelGen(self.game + "/sprites")

        if not self.repr_type:
            self.block2repr = None
        elif self.repr_type == "block2vec":
            self.block2repr = load_pkl('prim_cutout_representations', prepath='/home/awiszus/Project/TOAD-GAN/input/minecraft/')
        elif self.repr_type == "autoencoder":
            self.block2repr = {"encoder": torch.load("input/minecraft/simple_encoder.pt"),
                               "decoder": torch.load("input/minecraft/simple_decoder.pt")}
            self.block2repr["encoder"] = self.block2repr["encoder"].to(self.device)
            self.block2repr["decoder"] = self.block2repr["decoder"].to(self.device)
            self.block2repr["encoder"].requires_grad = False
            self.block2repr["decoder"].requires_grad = False
            self.block2repr["encoder"].eval()
            self.block2repr["decoder"].eval()
        else:
            AttributeError("unexpected repr_type, use [None, block2vec, autoencoder]")
