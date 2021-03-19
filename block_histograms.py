import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
from matplotlib import rcParams
from tap import Tap


class HistArguments(Tap):
    folder: str  # folder containing .pt tensor files
    logscale: bool = False  # use log scaling for y-axis?


if __name__ == '__main__':
    args = HistArguments().parse_args()

    # get a folder of blockdata
    files = os.listdir(args.folder)
    token_names = torch.load(os.path.join(args.folder, '../../token_list.pth'))
    pruned_names = []
    for name in token_names:
        sp_name = name.split(':')
        pruned_names.append(sp_name[1])

    t0 = torch.load(os.path.join(args.folder, '../real_bdata.pt'))
    t_mat = np.zeros((len(files), np.prod(t0.shape)), dtype='uint8')
    df = pd.DataFrame()
    for i, f in enumerate(files):
        t = torch.load(os.path.join(args.folder, f))
        t_mat[i] = t.reshape(-1)
        sum_dict = {}
        for j, tok in enumerate(token_names):
            sum_dict[pruned_names[j]] = (t[:] == j).sum()
        df = df.append(sum_dict, ignore_index=True)

    df0 = pd.DataFrame()
    t0_dict = {}
    for j, tok in enumerate(token_names):
        t0_dict[pruned_names[j]] = (t0[:] == j).sum()
    df0 = df0.append(t0_dict, ignore_index=True)

    # Histograms:
    palette = "turbo"
    # Seaborn:
    plt.figure(figsize=(12, 12))

    plt.subplot(2, 1, 1)
    p = sns.barplot(data=df0, palette=palette)
    if args.logscale:
        p.set(yscale='log')
    plt.title('Original Level')
    plt.xticks(rotation=30, ha="right")

    plt.subplot(2, 1, 2)
    p = sns.barplot(data=df, palette=palette)
    # p = sns.displot(t0.reshape(-1), binwidth=1)
    if args.logscale:
        p.set(yscale='log')
    # p.set_xticklabels(pruned_names, rotation=90)
    plt.title('Generated Levels')
    plt.xticks(rotation=30, ha="right")

    plt.subplots_adjust(hspace=0.3, top=0.92)

    # Matplotlib:
    # plt.style.use('seaborn-white')
    # plt.subplot(121)
    # plt.hist(t0.reshape(-1))
    # plt.xticks(range(len(token_names)), token_names, rotation=45)
    # plt.yscale('log')

    # plt.subplot(122)
    # for t in t_mat:
    #     plt.hist(t)
    # plt.xticks(range(len(token_names)), token_names, rotation=45)
    # plt.yscale('log')

    plt.savefig(os.path.join(args.folder, '../block_histogram.png'))

    print('Done!')
