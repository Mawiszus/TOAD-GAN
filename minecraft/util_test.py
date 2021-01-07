from typing import List
from config import Config
import torch
import numpy as np
from loguru import logger

from minecraft.level_utils import load_schematic, NanoMCSchematic, read_level_from_file

if __name__ == '__main__':
    # fname = "input/minecraft/test.schematic"
    # bdata = load_schematic("../input/minecraft/test.schematic")
    # print(bdata.shape)

    # folder = "../input/minecraft"
    # name = "test.schematic"
    # oh_data, uniques = read_level_from_file(folder, name)
    # print(oh_data.shape)
    # print(oh_data.min(), oh_data.max())
    # print(uniques)
    # bdata = one_hot_to_blockdata_level(oh_data, uniques)
    # print(bdata.shape)
    # print(bdata.min(), bdata.max())

    # new_schem = NanoMCSchematic("../input/minecraft/resaved_test.schematic", (bdata.shape[0], bdata.shape[1], bdata.shape[2]))
    # new_schem.set_blockdata(bdata)
    # new_schem.saveToFile()

    # New format with PyAnvil
    wrld_name = "Test_1_16"
    wrld_dir = "/home/awiszus/Project/minecraft_worlds/"
    coords = ((0, 16), (0, 128), (0, 16))
    oh_data, uniques = read_level_from_file(wrld_dir, wrld_name, coords)
    print(oh_data.shape)
    print(oh_data.min(), oh_data.max())

    print("Done")
