"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data_cyclegan.base_dataset import BaseDataset, get_transform
import os
from tqdm import tqdm
from random import randint
from PyAnvilEditor.pyanvil import World, BlockState
import torch
# from data.image_folder import make_dataset
# from PIL import Image


class MinecraftDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--worldA', type=str, default="plains", help='which world is to be loaded as A')
        parser.add_argument('--worldB', type=str, default="desert", help='which world is to be loaded as B')
        parser.set_defaults(max_dataset_size=3000)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        # self.image_paths = []  # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        # self.transform = get_transform(opt)

        self.shape = (32, 32, 32)

        self.worldpath = "../minecraft_worlds/"
        self.worldA = World(opt.worldA, self.worldpath, debug=False, write=False)
        self.worldB = World(opt.worldB, self.worldpath, debug=False, write=False)
        try:
            blocksetA = torch.load(f"output/mc_world_info/{opt.worldA}_block_list.pt")
            blocksetB = torch.load(f"output/mc_world_info/{opt.worldB}_block_list.pt")
        except Exception:
            print("Did not find block list, starting calculation.")
            blocksetA = set()
            blocksetB = set()
            for chnkx in tqdm(range(-100, 100)):
                for chnkz in range(-100, 100):
                    chunkA = self.worldA.get_chunk((chnkx, chnkz))
                    try:
                        sec_list = [chunkA.sections[3], chunkA.sections[4], chunkA.sections[5]]
                    except Exception:
                        try:
                            sec_list = [chunkA.sections[3], chunkA.sections[4]]
                        except Exception:
                            try:
                                sec_list = [chunkA.sections[3]]
                            except Exception:
                                sec_list = []
                    for section in sec_list:  # chunkA.sections.values():
                        # sections 3-5 are from y-level 48 upwards which includes the top layer
                        if section.blocks:
                            for block in section.raw_section.children['Palette'].children:
                                blocksetA.add(block.children['Name'].tag_value)

                    chunkB = self.worldB.get_chunk((chnkx, chnkz))
                    try:
                        sec_list = [chunkB.sections[3], chunkB.sections[4], chunkB.sections[5]]
                    except Exception:
                        try:
                            sec_list = [chunkB.sections[3], chunkB.sections[4]]
                        except Exception:
                            try:
                                sec_list = [chunkB.sections[3]]
                            except Exception:
                                sec_list = []
                    for section in sec_list:
                        if section.blocks:
                            for block in section.raw_section.children['Palette'].children:
                                blocksetB.add(block.children['Name'].tag_value)

                self.worldA.flush()
                self.worldB.flush()
            torch.save(blocksetA, f"output/mc_world_info/{opt.worldA}_block_list.pt")
            torch.save(blocksetB, f"output/mc_world_info/{opt.worldB}_block_list.pt")
        shared_blocks = set.intersection(blocksetA, blocksetB)
        only_A_blocks = blocksetA.difference(shared_blocks)
        only_B_blocks = blocksetB.difference(shared_blocks)
        shared_list = list(shared_blocks)
        shared_list.sort()
        self.blocklistA = shared_list + list(only_A_blocks) + list(only_B_blocks)
        self.blocklistB = shared_list + list(only_A_blocks) + list(only_B_blocks)

        self.channelsA = len(self.blocklistA)
        self.channelsB = len(self.blocklistB)
        self.length = opt.max_dataset_size

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        posy = 62
        radius = int(self.length/2)
        # posxA = index - radius
        posxA = randint(-radius, radius)
        poszA = randint(-radius, radius)
        posxB = randint(-radius, radius)
        poszB = randint(-radius, radius)
        coordsA = [[posxA, posxA + self.shape[0]], [posy, posy + self.shape[1]], [poszA, poszA + self.shape[2]]]
        coordsB = [[posxB, posxB + self.shape[0]], [posy, posy + self.shape[1]], [poszB, poszB + self.shape[2]]]

        data_A = self.read_coords(coordsA, self.worldA, self.blocklistA)  # needs to be a tensor
        data_B = self.read_coords(coordsB, self.worldB, self.blocklistB)  # needs to be a tensor

        self.worldA.flush()
        self.worldB.flush()
        return {'A': data_A, 'B': data_B, "A_paths": coordsA, "B_paths": coordsB}

    def read_coords(self, coords, wrld, blocklist):
        level = torch.zeros((len(blocklist),) + self.shape)
        for j in range(coords[0][0], coords[0][1]):
            for k in range(coords[1][0], coords[1][1]):
                for l in range(coords[2][0], coords[2][1]):
                    block = wrld.get_block((j, k, l))
                    b_name = block.get_state().name
                    level[blocklist.index(b_name), j - coords[0][0], k - coords[1][0], l - coords[2][0]] = 1
        return level

    def __len__(self):
        """Return the total number of images.
        This doesn't really make sense here but let's keep it for formatting."""
        return self.length

    def __del__(self):
        self.worldA.close()
        self.worldB.close()
