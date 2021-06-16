import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
from matplotlib import rcParams
from tap import Tap
from scipy.stats import chisquare as chisquare


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

    df0 = pd.DataFrame()
    t0_dict = {}
    for j, tok in enumerate(token_names):
        t0_dict[pruned_names[j]] = (t0[:] == j).sum()
    df0 = df0.append(t0_dict, ignore_index=True)

    sorted_names = sorted(t0_dict, key=t0_dict.get, reverse=True)

    chi2_vals_1 = []
    chi2_vals_2 = []
    chi2_vals_3 = []
    p_vals_1 = []
    p_vals_2 = []
    p_vals_3 = []
    t_mat = np.zeros((len(files), np.prod(t0.shape)), dtype='uint8')
    df = pd.DataFrame()
    for i, f in enumerate(files):
        t = torch.load(os.path.join(args.folder, f))
        t_mat[i] = t.reshape(-1)
        sum_dict = {}
        for j, tok in enumerate(token_names):
            sum_dict[pruned_names[j]] = (t[:] == j).sum()  # / t0_dict[pruned_names[j]]
        # sum_list = list(sum_dict.values())
        sum_list = []
        t0_list = []
        for name in sorted_names:  # need to make this list sorted
            sum_list.append(sum_dict[name])
            t0_list.append(t0_dict[name])
        split = len(sum_list) // 3
        sum_1, sum_2, sum_3 = sum_list[0:split], sum_list[split:2*split], sum_list[2*split:]
        t0_1, t0_2, t0_3 = t0_list[0:split], t0_list[split:2*split], t0_list[2*split:]
        c2_1, p_1 = chisquare(sum_1, t0_1)
        c2_2, p_2 = chisquare(sum_2, t0_2)
        c2_3, p_3 = chisquare(sum_3, t0_3)
        # c2_3, p_3 = chisquare(list(sum_dict.values()), list(t0_dict.values()))
        chi2_vals_1.append(c2_1)
        chi2_vals_2.append(c2_2)
        chi2_vals_3.append(c2_3)
        p_vals_1.append(p_1)
        p_vals_2.append(p_2)
        p_vals_3.append(p_3)
        df = df.append(sum_dict, ignore_index=True)

    print("Mean chi2 and p 1. third: ", sum(chi2_vals_1)/len(chi2_vals_1), sum(p_vals_1)/len(p_vals_1))
    print("Mean chi2 and p 2. third: ", sum(chi2_vals_2)/len(chi2_vals_2), sum(p_vals_2)/len(p_vals_2))
    print("Mean chi2 and p 3. third: ", sum(chi2_vals_3)/len(chi2_vals_3), sum(p_vals_3)/len(p_vals_3))

    # Histograms:
    palette = "turbo"
    # Seaborn:
    plt.figure(figsize=(12, 12))

    plt.subplot(2, 1, 1)
    p = sns.barplot(data=df0, palette=palette, order=sorted_names)
    if args.logscale:
        p.set(yscale='log')
    plt.title('Original Level')
    plt.xticks(rotation=30, ha="right")

    plt.subplot(2, 1, 2)
    p = sns.barplot(data=df, palette=palette, order=sorted_names)
    # p = sns.displot(t0.reshape(-1), binwidth=1)
    if args.logscale:
        p.set(yscale='log')
    # p.set_xticklabels(pruned_names, rotation=90)
    plt.title('Generated Levels')
    # plt.ylim([0, 5])
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
