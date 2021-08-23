import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import torch

from utils import load_pkl


def normalize(arr):
    arr_min = np.min(arr)
    return (arr-arr_min)/(np.max(arr)-arr_min)


def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3] * 2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded


def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z


def plot_cube(cube, angle=320, img_dim=50, shading=False, is_binary=False):
    # cube = normalize(cube)
    if is_binary:
        for z in range(cube.shape[2]):
            cube[:, :, z] = cube[:, :, z] * z * 0.1
    cube = normalize(cube)

    facecolors = cm.cool(cube)
    if is_binary:
        facecolors[:, :, :, -1] = cube > 0.001
    else:
        facecolors[:, :, :, -1] = cube * cube
    facecolors = explode(facecolors)

    filled = facecolors[:, :, :, -1] > 0.001
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))

    fig = plt.figure(figsize=(30 / 2.54, 30 / 2.54))
    ax = fig.gca(projection='3d')
    ax.view_init(30, angle)
    ax.set_xlim(right=img_dim * 2)
    ax.set_ylim(top=img_dim * 2)
    ax.set_zlim(top=img_dim * 2)

    ax.voxels(x, y, z, filled, facecolors=facecolors, shade=shading)
    plt.show()


if __name__ == '__main__':
    cutoff = 12.0
    use_bdata = True
    pth = "/home/awiszus/Project/TOAD-GAN/output/mineshaft_experiment/run-20210707_123418-3bh3zgul/files/" \
          "arbitrary_random_samples_v1.00000_h1.00000_d1.00000/"
    # name = "real_bdata_sc2.pt"
    name = "1_sc2.pt"
    token_list = torch.load(pth + "token_list.pt")
    used_tok_list = ["minecraft:cave_air", "minecraft:air"]
    inds = [token_list.index(used_tok) for used_tok in used_tok_list]
    if use_bdata:
        if name[0:4] == "real":
            tensor = torch.load(pth + name)
        else:
            tensor = torch.load(pth + "torch_blockdata/" + name)
        to_show = np.zeros(tensor.shape).transpose([2, 0, 1])
        for idx in inds:
            to_show += (tensor == idx).cpu().numpy().astype("float").transpose([2, 0, 1])
        to_show[to_show > 0] = 1
        binary = True
    else:
        tensor = torch.load(pth + "torch/" + name)

        # calc dists (convert to block space)
        block2repr = load_pkl("representations",
                              f"/home/schubert/projects/TOAD-GAN/input/minecraft/vanilla_mineshaft/")
        reprs = torch.stack(list(block2repr.values()))
        o = tensor.squeeze().permute(1, 2, 3, 0)[..., None]
        if len(inds) == 1:
            rep_vec = reprs[inds[0]]
        else:
            rep_vec = reprs[inds].mean(dim=0)
        r = rep_vec.to("cuda")[None, None, None, ..., None]
        d = (o - r).pow(2).sum(dim=-2).cpu().numpy()

        to_show = d[:, :, :, 0].transpose([2, 0, 1])
        # adjust for outliers - automate?
        to_show[to_show > cutoff] = cutoff
        to_show = -to_show

        binary = False

    plot_cube(to_show, img_dim=max(to_show.shape), shading=use_bdata, is_binary=binary)
    # plt.imshow(to_show[:, 5, :])
    # plt.show()

    print("Done!")

