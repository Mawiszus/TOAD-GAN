from minecraft.level_utils import read_level, save_level_to_world, clear_empty_world, one_hot_to_blockdata_level
from config import Config
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from tqdm import tqdm
from minecraft.block_autoencoder import Encoder, Decoder

import sys

if __name__ == '__main__':
    lat_size = 5
    opt = Config().parse_args()
    opt.block2repr = None
    level = read_level(opt)
    opt.level_shape = level.shape[-3:]
    level = level.to(opt.device)

    levels = [level]
    # Make Batch
    levels.append(level.flip([2]))
    levels.append(level.flip([4]))
    levels.append(level.flip([4, 2]))
    levels.append(level.rot90(1, [2, 4]))
    levels.append(level.rot90(2, [2, 4]))
    levels.append(level.rot90(1, [4, 2]))

    # clear_empty_world(opt.output_dir, 'Curr_Empty_World')  # reset tmp world
    # for n, level in tqdm(enumerate(levels)):
    #     level = one_hot_to_blockdata_level(level, opt.token_list, opt.block2repr)
    #     pos = n * (level.shape[0] + 5)
    #     save_level_to_world(opt.output_dir, 'Curr_Empty_World', (pos, 0, 0), level, opt.token_list)

    enc = Encoder(opt, True, lat_size).to(opt.device)
    dec = Decoder(opt, lat_size).to(opt.device)

    print(enc)
    print(dec)

    optimizerE = optim.Adam(enc.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerD = optim.Adam(dec.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))

    temperature = 1
    beta = 0.1
    losses = []

    for epoch in tqdm(range(opt.niter)):
        enc.zero_grad()
        dec.zero_grad()
        for lev in levels:
            noise = torch.normal(0, 0.01, size=lev.shape).to(opt.device)
            noised_lev = lev + noise
            # sftmx_lev = F.softmax(noised_lev * temperature, dim=1)
            z, mu, log_var = enc(noised_lev)
            rec = dec(z)
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
            loss = F.binary_cross_entropy(rec, lev)  # + beta * kld_loss.mean()
            loss.backward()
            optimizerE.step()
            optimizerD.step()

        # print("loss: ", loss.detach().item())
        # losses.append(loss.detach().item())

        if epoch % 10 == 0:
            losses.append(loss.cpu().detach().item())

        # if epoch % 100 == 0:
        #     print("loss: ", loss.detach().item())
        #     plt.plot(losses)
        #     plt.show()

    plt.plot(losses)
    plt.show()

    # enc.is_train = False
    torch.save(enc, 'input/minecraft/simple_encoder.pt')
    torch.save(dec, 'input/minecraft/simple_decoder.pt')
    torch.save(opt.token_list, 'input/minecraft/simple_autoencoder_token_list.pt')

    clear_empty_world(opt.output_dir, 'Curr_Empty_World')  # reset tmp world
    for n, level in tqdm(enumerate(levels)):
        z, mu, logvar = enc(level)
        rec = dec(z)
        rec = one_hot_to_blockdata_level(rec, opt.token_list, opt.block2repr, opt.repr_type)
        pos = n * (rec.shape[0] + 5)
        save_level_to_world(opt.output_dir, 'Curr_Empty_World', (pos, 0, 0), rec, opt.token_list, opt.props)

    pruned_names = []
    for name in opt.token_list:
        sp_name = name.split(':')
        pruned_names.append(sp_name[1])

    df0 = pd.DataFrame()
    t0_dict = {}
    for j, tok in enumerate(opt.token_list):
        t0_dict[pruned_names[j]] = (rec[:] == j).sum()
    df0 = df0.append(t0_dict, ignore_index=True)

    sorted_names = sorted(t0_dict, key=t0_dict.get, reverse=True)
    # Histograms:
    palette = "turbo"
    # Seaborn:
    plt.figure(figsize=(12, 6))

    p = sns.barplot(data=df0, palette=palette, order=sorted_names)
    p.set(yscale='log')
    plt.title('Original Level')
    plt.xticks(rotation=30, ha="right")

    plt.show()
    # plt.savefig(os.path.join(folder, '../autoencoder.png'))


