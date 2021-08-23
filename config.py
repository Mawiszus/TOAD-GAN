# Code based on https://github.com/tamarott/SinGAN
import argparse
import random
from typing import List, Literal, Optional, Union

import numpy as np
import torch
from torch import cuda
from tap import Tap

from utils import set_seed, load_pkl
from minecraft.block_autoencoder import Encoder, Decoder


class Config(Tap):
    # game: Literal["minecraft"] = "minecraft"  # Which game is to be used? ONLY MINECRAFT
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
    input_area_name: str = "ruins"  # needs to be a string from the coord dictionary in input folder
    output_dir: str = "../minecraft_worlds/"  # folder with worlds
    output_name: str = "Gen_Empty_World"  # name of the world to generate in
    sub_coords: List[float] = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]  # defines which coords of the full coord are are
    # taken (if float -> percentage, if int -> absolute)

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
        self.stop_scale = self.num_scales + 1

        coord_dict = load_pkl('primordial_coords_dict', 'input/minecraft/')
        tmp_coords = coord_dict[self.input_area_name]
        sub_coords = [(self.sub_coords[0], self.sub_coords[1]),
                      (self.sub_coords[2], self.sub_coords[3]),
                      (self.sub_coords[4], self.sub_coords[5])]
        self.coords = []
        for i, (start, end) in enumerate(sub_coords):
            curr_len = tmp_coords[i][1] - tmp_coords[i][0]
            if isinstance(start, float):
                tmp_start = curr_len * start + tmp_coords[i][0]
                tmp_end = curr_len * end + tmp_coords[i][0]
            elif isinstance(start, int):
                tmp_start = tmp_coords[i][0] + start
                tmp_end = tmp_coords[i][0] + end
            else:
                AttributeError("Unexpected type for sub_coords")
                tmp_start = tmp_coords[i][0]
                tmp_end = tmp_coords[i][1]

            self.coords.append((int(tmp_start), int(tmp_end)))

        if not self.repr_type:
            self.block2repr = None
        elif self.repr_type == "block2vec":
            # self.block2repr = load_pkl('prim_cutout_representations_ruins',
            #                            prepath='/home/awiszus/Project/TOAD-GAN/input/minecraft/')
            self.block2repr = load_pkl("representations",
                                       f"/home/schubert/projects/TOAD-GAN/input/minecraft/{self.input_area_name}/")
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
