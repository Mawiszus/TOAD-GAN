import os
import torch
import numpy as np
import subprocess
import math
from tqdm import tqdm

from minecraft.level_utils import read_level_from_file
from minecraft.level_renderer import render_minecraft

if __name__ == '__main__':
    num_samples = 100
    shape = [45, 20, 45]
    render_images = False
    save_tensors = True

    dir2save = "/home/awiszus/Project/TOAD-GAN/output/plains_only_examples"

    for n in tqdm(range(num_samples)):

        len_n = math.ceil(math.sqrt(num_samples))  # we arrange our samples in a square in the world
        x, z = np.unravel_index(n, [len_n, len_n])  # get x, z pos according to index n
        posx = x * (shape[0])
        posz = z * (shape[2])
        posy = 63

        curr_coords = [[posx, posx + shape[0]],
                       [posy, posy + shape[1]],
                       [posz, posz + shape[2]]]

        I_curr = read_level_from_file("/home/awiszus/Project/minecraft_worlds/", "Plains_only",
                                      curr_coords, None, None)
        if render_images:
            try:
                subprocess.call(["wine", '--version'])
                obj_path = os.path.join(dir2save, "objects")
                os.makedirs(obj_path, exist_ok=True)
                render_minecraft("Plains_only", curr_coords, obj_path, str(n))
            except OSError:
                pass

        # Save torch tensor
        if save_tensors:
            os.makedirs("%s/torch_blockdata" % dir2save, exist_ok=True)
            torch.save(I_curr, "%s/torch_blockdata/%d.pt" % (dir2save, n))
