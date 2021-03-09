from minecraft.level_utils import read_level, save_level_to_world, clear_empty_world, one_hot_to_blockdata_level
from config import Config
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from minecraft.block_autoencoder import Encoder, Decoder

import sys

if __name__ == '__main__':
    opt = Config().parse_args()
    opt.block2repr = None
    level = read_level(opt)
    opt.level_shape = level.shape[-3:]

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

    enc = Encoder(opt)
    dec = Decoder(opt)

    optimizerE = optim.Adam(enc.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerD = optim.Adam(dec.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))

    for epoch in tqdm(range(100)):
        enc.zero_grad()
        dec.zero_grad()
        for lev in levels:
            rec = dec(enc(lev))
            loss = F.binary_cross_entropy(rec, lev)
            loss.backward()
            optimizerE.step()
            optimizerD.step()

        print("loss: ", loss.detach().item())

    torch.save(enc, 'input/minecraft/simple_encoder.pt')
    torch.save(dec, 'input/minecraft/simple_decoder.pt')

    clear_empty_world(opt.output_dir, 'Curr_Empty_World')  # reset tmp world
    for n, level in tqdm(enumerate(levels)):
        rec = dec(enc(level))
        rec = one_hot_to_blockdata_level(rec, opt.token_list, opt.block2repr, opt.repr_type)
        pos = n * (rec.shape[0] + 5)
        save_level_to_world(opt.output_dir, 'Curr_Empty_World', (pos, 0, 0), rec, opt.token_list)

