import os
import numpy as np
import time
from random import shuffle
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import ConfusionMatrixDisplay

# os.chdir('/home/awiszus/Project/TOAD-GAN/')
import sys
sys.path.append("..")  # Adds higher directory to python modules path.

from utils import load_pkl, save_pkl
from mario.level_utils import load_level_from_text, REPLACE_TOKENS
from mario.tokens import TOKEN_GROUPS_REDUX


def get_input_layer(word_idx, vocabulary_size):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x


if __name__ == '__main__':

    start_time = time.time()

    calc_neighbours = True
    train_embed = True
    visualize = True
    level_num = "all"
    game = "mariokart"
    embedding_dims = 3
    num_epochs = 5

    if calc_neighbours:
        token2idx = {}
        idx2token = {}

        # Since Mario is finite, we can just make a full dataset with all neighbours
        if level_num == 'all':
            level_names = os.listdir("../input/" + game + "/")
        else:
            level_names = ['lvl_' + level_num + '.txt']  # Test: only use one level
        level_names.sort()

        # get all levels and token indices
        ascii_levels = []
        idx = 0
        for name in level_names:
            txt_level = load_level_from_text("../input/" + game + "/" + name)
            for line in txt_level:
                for token in line:
                    if token != "\n" and token not in token2idx.keys():
                            token2idx[token] = idx
                            idx2token[idx] = token
                            idx += 1
            ascii_levels.append(txt_level)

        # make neighborhood dataset
        idx_pairs = []
        for txt_level in ascii_levels:
            # left neighbour
            for y in range(len(txt_level)):
                for x in range(1, len(txt_level[-1])):
                    pair = (token2idx[txt_level[y][x]], token2idx[txt_level[y][x-1]])
                    idx_pairs.append(pair)

            # right neighbour
            for y in range(len(txt_level)):
                for x in range(len(txt_level[-1]) - 1):
                    pair = (token2idx[txt_level[y][x]], token2idx[txt_level[y][x+1]])
                    idx_pairs.append(pair)

            # top neighbour
            for y in range(1, len(txt_level)):
                for x in range(len(txt_level[-1])):
                    pair = (token2idx[txt_level[y][x]], token2idx[txt_level[y-1][x]])
                    idx_pairs.append(pair)

            # bottom neighbour
            for y in range(len(txt_level) - 1):
                for x in range(len(txt_level[-1])):
                    pair = (token2idx[txt_level[y][x]], token2idx[txt_level[y+1][x]])
                    idx_pairs.append(pair)

            # left top neighbour
            for y in range(1, len(txt_level)):
                for x in range(1, len(txt_level[-1])):
                    pair = (token2idx[txt_level[y][x]], token2idx[txt_level[y-1][x-1]])
                    idx_pairs.append(pair)

            # left bottom neighbour
            for y in range(len(txt_level) - 1):
                for x in range(1, len(txt_level[-1])):
                    pair = (token2idx[txt_level[y][x]], token2idx[txt_level[y+1][x-1]])
                    idx_pairs.append(pair)

            # right top neighbour
            for y in range(1, len(txt_level)):
                for x in range(len(txt_level[-1]) - 1):
                    pair = (token2idx[txt_level[y][x]], token2idx[txt_level[y-1][x+1]])
                    idx_pairs.append(pair)

            # right bottom neighbour
            for y in range(len(txt_level) - 1):
                for x in range(len(txt_level[-1]) - 1):
                    pair = (token2idx[txt_level[y][x]], token2idx[txt_level[y+1][x+1]])
                    idx_pairs.append(pair)

        shuffle(idx_pairs)

        save_pkl(idx_pairs, game + "_" + level_num + "_neighbour_list_shuffled", "../output/tmp_vec_calc/")
        save_pkl(token2idx, game + "_" + level_num + "_token2idx", "../output/tmp_vec_calc/")
        save_pkl(idx2token, game + "_" + level_num + "_idx2token", "../output/tmp_vec_calc/")
    else:
        idx_pairs = load_pkl(game + "_" + level_num + "_neighbour_list_shuffled", "../output/tmp_vec_calc/")
        token2idx = load_pkl(game + "_" + level_num + "_token2idx", "../output/tmp_vec_calc/")
        idx2token = load_pkl(game + "_" + level_num + "_idx2token", "../output/tmp_vec_calc/")

    # TRAINING
    if train_embed:

        vocabulary_size = len(token2idx)

        W1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
        W2 = Variable(torch.randn(vocabulary_size, embedding_dims).float(), requires_grad=True)
        learning_rate = 0.001

        for epo in range(num_epochs):
            shuffle(idx_pairs)
            loss_val = 0
            for data, target in tqdm(idx_pairs):
                x = Variable(get_input_layer(data, vocabulary_size)).float()
                y_true = Variable(torch.from_numpy(np.array([target])).long())

                z1 = torch.matmul(W1, x)
                z2 = torch.matmul(W2, z1)

                log_softmax = F.log_softmax(z2, dim=0)

                loss = F.nll_loss(log_softmax.view(1, -1), y_true)

                # spherical_Loss = F.mse_loss(W1.sum(dim=0), torch.ones((vocabulary_size,)))

                # loss += spherical_Loss

                loss_val += loss.data.item()  # + spherical_Loss.data.item()
                loss.backward()
                W1.data -= learning_rate * W1.grad.data
                W2.data -= learning_rate * W2.grad.data

                W1.grad.data.zero_()
                W2.grad.data.zero_()
            if epo % 1 == 0:
                print(f'Loss at epo {epo}: {loss_val / len(idx_pairs)}')

        token2repr = {}
        for token in token2idx:
            val = 0
            for i, group in enumerate(TOKEN_GROUPS_REDUX):
                if token in group:
                    val = i

            token2repr[token] = torch.zeros((embedding_dims + 1,))
            token2repr[token][0:embedding_dims] = W1[:, token2idx[token]].detach()  # * 0.01 * W2[token2idx[token]].detach()
            token2repr[token][-1] = val

        save_pkl(token2repr, game + '_' + level_num + '_' + str(embedding_dims) + 'D_representations',
                 prepath='../output/vec_calc/')
    else:
        token2repr = load_pkl(game + '_' + level_num + '_' + str(embedding_dims) + 'D_representations',
                              prepath='../output/vec_calc/')

    if visualize:
        dists = np.zeros((len(token2repr), len(token2repr)))
        names = []
        for i, b1 in enumerate(token2repr):
            names.append(b1)
            for j, b2 in enumerate(token2repr):
                dists[i, j] = F.mse_loss(token2repr[b1], token2repr[b2])
        rcParams.update({'font.size': 6})
        plt.figure(figsize=(40, 40))
        confusion_display = ConfusionMatrixDisplay(dists, names)
        confusion_display.plot(include_values=False)
        plt.tight_layout()
        figure_path = "../output/tmp_vec_calc/dist_matrix_" + game + "_" + level_num + ".pdf"
        plt.savefig(figure_path, dpi=300)

    end_time = time.time() - start_time
    print('Done in {}'.format(end_time))
