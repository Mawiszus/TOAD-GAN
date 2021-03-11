import os
import numpy as np
import time
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
from PyAnvilEditor.pyanvil import World


def get_input_layer(word_idx, vocabulary_size):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x


if __name__ == '__main__':

    start_time = time.time()

    # settings:
    calcPairs = False
    trainEmbed = False
    visualize = True
    embedding_dims = 3

    if calcPairs:
        block2idx = {}

        neigb2diff = {}
        count = 0
        for x in range(-1, 2):
            for y in range(-1, 2):
                for z in range(-1, 2):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    else:
                        neigb2diff[count] = [x, y, z]
                        count += 1
        save_loc = '/home/awiszus/Project/minecraft_worlds/'
        # save_loc = '/localstorage/awiszus/tmp/minecraft_training_data/'
        with World('Drehmal v2.1 PRIMORDIAL', save_location=save_loc, debug=False) as wrld:
        # with World('Test_1_16', save_location='/home/awiszus/Project/minecraft_worlds/', debug=False) as wrld:
            regions = os.listdir(wrld.world_folder / 'region')
            arr_regions = np.zeros((len(regions), 2))
            for i, r in enumerate(regions):
                name = r.split(".")
                rx = int(name[1])
                rz = int(name[2])
                arr_regions[i] = rx, rz
            coords = ((1044, 1060), (64, 80), (1104, 1120))   # y, z, x in real, x, y, z in Minecraft
            # igno_border = 0
            # x_lims = [(min(arr_regions[:, 0]) * 32 * 16) + igno_border, (max(arr_regions[:, 0]) * 32 * 16) - igno_border]
            # z_lims = [(min(arr_regions[:, 1]) * 32 * 16) + igno_border, (max(arr_regions[:, 1]) * 32 * 16) - igno_border]
            # y_lims = [0, 256]
            x_lims = coords[0]
            y_lims = coords[1]
            z_lims = coords[2]
            max_samples = (x_lims[1] - x_lims[0]) * (y_lims[1] - y_lims[0]) * (z_lims[1] - z_lims[0]) * 2

            idx_pairs = []
            samplecount = 0
            idx_count = 0
            while samplecount < max_samples:
                if samplecount % 100 == 0:
                    print('Found {} samples...'.format(samplecount))
                found_block = False
                while not found_block:
                    sample_x = np.random.randint(x_lims[0], x_lims[1])
                    sample_y = np.random.randint(y_lims[0], y_lims[1])
                    sample_z = np.random.randint(z_lims[0], z_lims[1])

                    neighbor = neigb2diff[np.random.randint(0, len(neigb2diff))]

                    try:
                        blck = wrld.get_block([sample_x, sample_y, sample_z])
                        neigh = wrld.get_block([sample_x - neighbor[0], sample_y - neighbor[0], sample_z - neighbor[0]])

                        b_name = blck.get_state().name
                        n_name = neigh.get_state().name

                        if b_name not in block2idx:
                            block2idx[b_name] = idx_count
                            idx_count += 1

                        if n_name not in block2idx:
                            block2idx[n_name] = idx_count
                            idx_count += 1

                        pair = (block2idx[b_name], block2idx[n_name])
                        idx_pairs.append(pair)
                        found_block = True
                        samplecount += 1
                    except Exception as e:
                        continue

        save_pkl(idx_pairs, 'primordial_cutout_pairs', prepath='../output/tmp_vec_calc/minecraft/')
        save_pkl(block2idx, 'primordial_cutout_block2idx', prepath='../output/tmp_vec_calc/minecraft/')
    else:
        idx_pairs = load_pkl('primordial_cutout_pairs', prepath='../output/tmp_vec_calc/minecraft/')
        block2idx = load_pkl('primordial_cutout_block2idx', prepath='../output/tmp_vec_calc/minecraft/')

    if trainEmbed:
        vocabulary_size = len(block2idx)

        W1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
        W2 = Variable(torch.randn(vocabulary_size, embedding_dims).float(), requires_grad=True)
        num_epochs = 50
        learning_rate = 0.001

        for epo in range(num_epochs):
            loss_val = 0

            for data, target in idx_pairs:
                x = Variable(get_input_layer(data, vocabulary_size)).float()
                y_true = Variable(torch.from_numpy(np.array([target])).long())

                z1 = torch.matmul(W1, x)
                z2 = torch.matmul(W2, z1)

                log_softmax = F.log_softmax(z2, dim=0)

                loss = F.nll_loss(log_softmax.view(1, -1), y_true)
                loss_val += loss.data.item()
                loss.backward()
                W1.data -= learning_rate * W1.grad.data
                W2.data -= learning_rate * W2.grad.data

                W1.grad.data.zero_()
                W2.grad.data.zero_()
            if epo % 10 == 0:
                print(f'Loss at epo {epo}: {loss_val / len(idx_pairs)}')

        block2repr = {}
        for block in block2idx:
            block2repr[block] = W1[:, block2idx[block]]

        save_pkl(block2repr, 'primordial_cutout_{}D_repr'.format(embedding_dims),
                 prepath='../output/vec_calc/minecraft/')
    else:
        block2repr = load_pkl('primordial_cutout_{}D_repr'.format(embedding_dims),
                              prepath='../output/vec_calc/minecraft/')

    if visualize:
        # Dist matrix
        dists = np.zeros((len(block2repr), len(block2repr)))
        names = []
        for i, b1 in enumerate(block2repr):
            names.append(b1.split(':')[1])
            for j, b2 in enumerate(block2repr):
                dists[i, j] = F.mse_loss(block2repr[b1], block2repr[b2])
        rcParams.update({'font.size': 6})
        plt.figure(figsize=(40, 40))
        confusion_display = ConfusionMatrixDisplay(dists, names)
        confusion_display.plot(include_values=False)
        plt.tight_layout()
        figure_path = "../output/tmp_vec_calc/minecraft/dist_matrix_{}D.pdf".format(embedding_dims)
        plt.savefig(figure_path, dpi=300)

        # 3D scatter
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        legend = list(block2idx.keys())
        # repr_mat = np.zeros((len(block2repr), embedding_dims))
        for i, block in enumerate(block2repr):
            repr_mat = block2repr[block].detach()
            ax.scatter(repr_mat[0], repr_mat[1], repr_mat[2], marker='o')
        plt.legend(legend)
        plt.tight_layout()
        figure_path = "../output/tmp_vec_calc/minecraft/scatter_{}D.pdf".format(embedding_dims)
        plt.savefig(figure_path, dpi=300)

    end_time = time.time() - start_time
    print('Done in {}'.format(end_time))