import math
import os

import torch
from torch.utils.data import Dataset
from loguru import logger
from tqdm import tqdm

from .level_utils import load_level_from_text, ascii_to_one_hot_level


class LevelSnippetDataset(Dataset):
    """
    Converts a folder (level_dir) with token based ascii-levels in .txt files into a torch Dataset of slice_width by
     slice_width level slices. Default for Super Mario Bros. is 16, as the levels are 16 pixels high. level_dir needs
     to only include level.txt files.

     token_list : If None, token_list is calculated internally. Can be set for different future applications.
     level_idx : If None, __getitem__ returns the actual index with the retrieved slice. Can be set for different future
        applications
    level_name : If None, all level files in folder are used, otherwise only level_name will be used.
    """
    def __init__(self, level_dir, slice_width=16, token_list=None, level_idx=None, level_name=None,):
        super(LevelSnippetDataset, self).__init__()
        self.level_idx = level_idx
        self.ascii_levels = []
        uniques = set()
        self.level_names = []
        logger.debug("Reading levels from directory {}", level_dir)
        for level in tqdm(sorted(os.listdir(level_dir))):
            if not level.endswith(".txt") or (level_name is not None and level != level_name):
                continue
            self.level_names.append(level)
            curr_level = load_level_from_text(os.path.join(level_dir, level))
            for line in curr_level:
                for token in line:
                    if token != "\n" and token != "M" and token != "F":
                    # if token != "M" and token != "F":
                        uniques.add(token)
            self.ascii_levels.append(curr_level)

        logger.trace("Levels: {}", self.level_names)
        if token_list is not None:
            self.token_list = token_list
        else:
            self.token_list = list(sorted(uniques))

        logger.trace("Token list: {}", self.token_list)

        logger.debug("Converting ASCII levels to tensors...")
        self.levels = []
        for i, level in tqdm(enumerate(self.ascii_levels)):
            self.levels.append(ascii_to_one_hot_level(level, self.token_list))

        self.slice_width = slice_width
        self.missing_slices_per_level = slice_width - 1
        self.missing_slices_l = math.floor(self.missing_slices_per_level / 2)
        self.missing_slices_r = math.ceil(self.missing_slices_per_level / 2)

        self.level_lengths = [
            x.shape[-1] - self.missing_slices_per_level for x in self.levels
        ]

    def get_level_name(self, file_name):
        return file_name.split(".")[0]

    def __getitem__(self, idx):
        i_l = 0
        while sum(self.level_lengths[0:i_l]) < (idx + 1) < sum(self.level_lengths):
            i_l += 1
        i_l -= 1

        level = self.levels[i_l]
        idx_lev = idx - sum(self.level_lengths[0:i_l]) + self.missing_slices_l
        lev_slice = level[
            :, :, idx_lev - self.missing_slices_l: idx_lev + self.missing_slices_r + 1
        ]
        return (
            lev_slice,
            torch.tensor(i_l if self.level_idx is None else self.level_idx),
        )

    def __len__(self):
        return sum(self.level_lengths) - 1
