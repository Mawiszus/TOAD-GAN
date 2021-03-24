import os
from collections import defaultdict
from itertools import product
from typing import Tuple

from loguru import logger
import numpy as np
from tqdm import tqdm

from PyAnvilEditor.pyanvil import World
from torch.utils.data.dataset import Dataset


class Block2VecDataset(Dataset):

    def __init__(self, input_world_path: str, coords: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]], cutout_coords: bool,  neighbor_radius: int = 1):
        """Block dataset with configurable neighborhood radius.

        Args:
            input_world_path (str): path to the Minecraft world
            coords (Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]): x, y, z coordinates of extracted region
            neighbor_radius (int): neighbors to retrieve as a context
        """
        super().__init__()
        self.input_world_path = input_world_path
        self.world = World(os.path.basename(input_world_path),
                           save_location=os.path.abspath(os.path.dirname(input_world_path)), write=False, debug=False)
        self.x_lims, self.y_lims, self.z_lims = coords if cutout_coords else self._read_size()
        padding = 2 * neighbor_radius  # one token on each side
        self.x_dim = self.x_lims[1] - self.x_lims[0] - padding
        self.y_dim = self.y_lims[1] - self.y_lims[0] - padding
        self.z_dim = self.z_lims[1] - self.z_lims[0] - padding
        logger.info("Cutting {} x {} x {} volume from {}", self.x_dim,
                    self.y_dim, self.z_dim, self.input_world_path)
        self.neighbor_radius = neighbor_radius
        self._read_blocks()
        self._init_discards()

    def _init_discards(self):
        t = 0.001
        token_frequencies = list(self.block_frequency.values())
        f = np.array(token_frequencies) / sum(token_frequencies)
        self.discards = 1.0 - (np.sqrt(f / t) + 1) * (t / f)

    def _read_size(self):
        regions = os.listdir(self.world.world_folder / 'region')
        arr_regions = np.zeros((len(regions), 2))
        for i, r in enumerate(regions):
            name = r.split(".")
            rx = int(name[1])
            rz = int(name[2])
            arr_regions[i] = rx, rz
        igno_border = 0
        x_lims = [int((min(arr_regions[:, 0]) * 32 * 16) + igno_border),
                  int((max(arr_regions[:, 0]) * 32 * 16) - igno_border)]
        z_lims = [int((min(arr_regions[:, 1]) * 32 * 16) + igno_border),
                  int((max(arr_regions[:, 1]) * 32 * 16) - igno_border)]
        y_lims = [0, 256]
        return x_lims, y_lims, z_lims

    def _read_blocks(self):
        self.block_frequency = defaultdict(int)
        coordinates = [(x, y, z) for x, y, z in product(range(self.x_lims[0], self.x_lims[1] + 1),
                                                        range(self.y_lims[0], self.y_lims[1] + 1), range(self.z_lims[0], self.z_lims[1] + 1))]
        logger.info("Collecting {} blocks", len(self))
        for name in tqdm([self._get_block(*coord) for coord in coordinates]):
            self.block_frequency[name] += 1
        logger.info(
            "Found the following blocks {blocks}", blocks=dict(self.block_frequency))
        self.block2idx = dict()
        self.idx2block = dict()
        for name, count in self.block_frequency.items():
            block_idx = len(self.block2idx)
            self.block2idx[name] = block_idx
            self.idx2block[block_idx] = name

    def __getitem__(self, index):
        coords = self._idx_to_coords(index)
        block = self._get_block(*coords)
        target = self.block2idx[block]
        if np.random.rand() < self.discards[target]:
            return self.__getitem__(np.random.randint(self.__len__()))
        neighbor_blocks = self._get_neighbors(*coords)
        context = np.array([self.block2idx[n] for n in neighbor_blocks])
        return target, context

    def _idx_to_coords(self, index):
        z = index % (self.z_dim + 1)
        y = int(((index - z) / (self.z_dim + 1)) % (self.y_dim + 1))
        x = int(((index - z) / (self.z_dim + 1) - y) / (self.y_dim + 1))
        x += self.x_lims[0] + self.neighbor_radius
        y += self.y_lims[0] + self.neighbor_radius
        z += self.z_lims[0] + self.neighbor_radius
        return x, y, z

    def _get_block(self, x, y, z):
        block = self.world.get_block([x, y, z])
        name = block.get_state().name
        return name

    def _get_neighbors(self, x, y, z):
        neighbor_coords = [(x + x_diff, y + y_diff, z + z_diff) for x_diff, y_diff, z_diff in product(list(
            range(-self.neighbor_radius, self.neighbor_radius + 1)), repeat=3) if x_diff != 0 or y_diff != 0 or z_diff != 0]
        return [self._get_block(*coord) for coord in neighbor_coords]

    def __len__(self):
        return self.x_dim * self.y_dim * self.z_dim
