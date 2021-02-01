import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.functional import interpolate
from loguru import logger
from tqdm import tqdm
import numpy as np
import wandb

from draw_concat import draw_concat
from generate_noise import generate_spatial_noise
from mario.level_utils import group_to_token, one_hot_to_ascii_level, token_to_group
from minecraft.level_utils import one_hot_to_blockdata_level, save_level_to_world, clear_empty_world
from mario.tokens import TOKEN_GROUPS as MARIO_TOKEN_GROUPS
from zelda.tokens import TOKEN_GROUPS as ZELDA_TOKEN_GROUPS
from megaman.tokens import TOKEN_GROUPS as MEGAMAN_TOKEN_GROUPS
from mariokart.tokens import TOKEN_GROUPS as MARIOKART_TOKEN_GROUPS
from models import calc_gradient_penalty, save_networks
from utils import interpolate3D


def train_upscale(reals, opt):
    upscalers = []
    for sc in range(len(reals)-1):
        lower = reals[sc]
        upper = reals[sc+1]

        if len(opt.level_shape) == 2:
            scaler = nn.ConvTranspose2d(lower.shape[1], upper.shape[1], kernel_size=5, stride=2, padding=2)
        elif len(opt.level_shape) == 3:
            scaler = nn.ConvTranspose3d
        else:
            raise NotImplementedError("Level Shape expected to be 2D or 3D.")
    return upscalers