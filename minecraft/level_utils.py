from typing import List
from config import Config
import torch
import numpy as np
import os
from loguru import logger

import minecraft.nbt as nbt


# Miscellaneous functions to deal with MC schematics.

def load_schematic(path_to_schem):
    """ Loads a Minecraft .schem file """
    sch = nbt.load(path_to_schem)
    # _Blocks is y, z, x
    blocks = sch["Blocks"].value.astype('uint16').reshape(sch["Height"].value, sch["Length"].value, sch["Width"].value)
    data = sch["Data"].value.reshape(sch["Height"].value, sch["Length"].value, sch["Width"].value)

    blockdata = np.concatenate([blocks.reshape(blocks.shape + (1,)), data.reshape(data.shape + (1,))], axis=3)

    return blockdata


class NanoMCSchematic:
    def __init__(self, filename, shape, mats='Alpha'):
        """ We assume shape is already in yzx, unlike original code! """
        self.filename = filename
        self.material = mats
        assert shape is not None
        root_tag = nbt.TAG_Compound(name="Schematic")
        #
        root_tag["Height"] = nbt.TAG_Short(shape[0])
        root_tag["Length"] = nbt.TAG_Short(shape[1])
        root_tag["Width"] = nbt.TAG_Short(shape[2])

        root_tag["Entities"] = nbt.TAG_List()
        root_tag["TileEntities"] = nbt.TAG_List()
        root_tag["TileTicks"] = nbt.TAG_List()
        root_tag["Materials"] = nbt.TAG_String(self.material)

        self._Blocks = np.zeros((shape[0], shape[1], shape[2]), 'uint16')
        self._Data = np.zeros((shape[0], shape[1], shape[2]), 'uint16')
        # root_tag["Data"] = nbt.TAG_Byte_Array(np.zeros((shape[1], shape[2], shape[0]), np.uint8))

        root_tag["Biomes"] = nbt.TAG_Byte_Array(np.zeros((shape[2], shape[0]), np.uint8))

        self.root_tag = root_tag

    def set_blockdata(self, blockdata):
        blocks = blockdata[:, :, :, 0]
        data = blockdata[:, :, :, 1]
        assert blocks.shape == self._Blocks.shape
        assert data.shape == self._Data.shape
        self._Blocks = blocks.astype(type(self._Blocks))
        self._Data = data.astype(type(self._Data))

    def saveToFile(self, filename=None):
        """ save to file named filename, or use self.filename.  XXX NOT THREAD SAFE AT ALL. """
        if filename is None:
            filename = self.filename
        if filename is None:
            raise IOError("Attempted to save an unnamed schematic in place")

        self.Materials = self.material

        self.root_tag["Blocks"] = nbt.TAG_Byte_Array(self._Blocks.astype('uint8'))
        self.root_tag["Data"] = nbt.TAG_Byte_Array(self._Data.astype('uint8'))
        self.root_tag["Data"].value &= 0xF  # discard high bits

        add = self._Blocks >> 8
        if add.any():
            # WorldEdit AddBlocks compatibility.
            # The first 4-bit value is stored in the high bits of the first byte.

            # Increase odd size by one to align slices.
            packed_add = np.zeros(add.size + (add.size & 1), 'uint8')
            packed_add[:add.size] = add.ravel()

            # Shift even bytes to the left
            packed_add[::2] <<= 4

            # Merge odd bytes into even bytes
            packed_add[::2] |= packed_add[1::2]

            # Save only the even bytes, now that they contain the odd bytes in their lower bits.
            packed_add = packed_add[0::2]
            self.root_tag["AddBlocks"] = nbt.TAG_Byte_Array(packed_add)

        with open(filename, 'wb') as chunkfh:
            self.root_tag.save(chunkfh)
            chunkfh.close()

        del self.root_tag["Blocks"]
        del self.root_tag["Data"]
        self.root_tag.pop("AddBlocks", None)


def blockdata_to_one_hot_level(bdata, tokens) -> torch.Tensor:
    """ Converts blockdata to a full token level tensor. """
    oh_level = torch.zeros((len(tokens),) + bdata.shape[:-1])
    for i, tok in enumerate(tokens):
        overlap = bdata[:, :, :] == tok
        true_pos = np.logical_and(overlap[:, :, :, 0], overlap[:, :, :, 1])
        oh_level[i][true_pos] = 1

    return oh_level


def one_hot_to_blockdata_level(oh_level, tokens):
    """ Converts a full token level tensor to blockdata. """
    bdata = np.zeros(oh_level.shape[2:] + (2,), 'uint8')
    for y in range(bdata.shape[0]):
        for z in range(bdata.shape[1]):
            for x in range(bdata.shape[2]):
                bdata[y, z, x] = tokens[oh_level[:, :, y, z, x].argmax()]

    return bdata


def read_level(opt: Config):
    """ Wrapper function for read_level_from_file using namespace opt. Updates parameters for opt."""
    # If we have multiple levels as input, we need to sync the tokens

    # Default: Only one input level
    level, uniques = read_level_from_file(opt.input_dir, opt.input_name)
    opt.token_list = uniques
    logger.info("Tokens in level {}", opt.token_list)
    opt.nc_current = len(uniques)
    return level


def read_level_from_file(input_dir, input_name):
    """ Returns a full token level tensor from a .txt file. Also returns the unique tokens found in this level.
    Token. """
    bdata = load_schematic(os.path.join(input_dir, input_name))
    uniques = set()
    for y in range(bdata.shape[0]):
        for z in range(bdata.shape[1]):
            for x in range(bdata.shape[2]):
                uniques.add(tuple(bdata[y, z, x]))
    uniques = list(uniques)
    uniques.sort()  # necessary! otherwise we won't know the token order later
    oh_level = blockdata_to_one_hot_level(bdata, uniques)
    return oh_level.unsqueeze(dim=0), uniques
