from typing import List
from config import Config
import torch
import numpy as np
from loguru import logger

from minecraft.nbt import load
from minecraft.tokens import TOKEN_GROUPS, REPLACE_TOKENS


def load_schematic(path_to_schem, replace_tokens=REPLACE_TOKENS):
    """ Loads a Minecraft .schem file """
    sch = load(path_to_schem)
    # _Blocks is y, z, x
    blocks = sch["Blocks"].value.astype('uint16').reshape(sch["Height"].value, sch["Length"].value, sch["Width"].value)
    data = sch["Data"].value.reshape(sch["Height"].value, sch["Length"].value, sch["Width"].value)

    blockdata = np.concatenate([blocks.reshape(blocks.shape + (1,)), data.reshape(data.shape + (1,))], axis=3)

    return blockdata


if __name__ == '__main__':
    fname = "input/minecraft/test.schematic"
    bdata = load_schematic("../input/minecraft/test.schematic")
    print(bdata.shape)
