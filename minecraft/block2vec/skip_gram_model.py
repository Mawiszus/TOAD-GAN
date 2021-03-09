import os
import pickle
from typing import Dict

import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import matplotlib.pyplot as plt
from matplotlib import rcParams


class SkipGramModel(nn.Module):
    def __init__(self, emb_size: int, emb_dimension: int):
        super().__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.target_embeddings = nn.Embedding(emb_size, emb_dimension)
        self.context_embeddings = nn.Embedding(emb_size, emb_dimension)

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.target_embeddings.weight.data, -
                      initrange, initrange)
        init.constant_(self.context_embeddings.weight.data, 0)

    def forward(self, target, context):
        emb_target = self.target_embeddings(target)
        emb_context = self.context_embeddings(context)

        score = torch.sum(
            torch.mul(emb_target.unsqueeze(1), emb_context), dim=-1)
        score = -F.logsigmoid(score)

        return score.mean()

    def save_embedding(self, id2block: Dict[int, str], output_path: str):
        embedding = self.target_embeddings.weight.cpu().data.numpy()
        embedding_dict = {}
        with open(os.path.join(output_path, "embeddings.txt"), 'w') as f:
            f.write('%d %d\n' % (len(id2block), self.emb_dimension))
            for wid, w in id2block.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                embedding_dict[w] = embedding
                f.write('%s %s\n' % (w, e))
        np.save(os.path.join(output_path, "embeddings.npy"), embedding)
        with open(os.path.join(output_path, "representations.pkl"), "wb") as f:
            pickle.dump(embedding_dict, f)

    def create_confusion_matrix(self, id2block: Dict[int, str], output_path: str):
        rcParams.update({'font.size': 6})
        fig = plt.figure(figsize=(40, 40))
        names = []
        dists = np.zeros((len(id2block), len(id2block)))
        for i, b1 in id2block.items():
            names.append(b1.split(':')[1])
            for j, b2 in id2block.items():
                dists[i, j] = F.mse_loss(
                    self.target_embeddings.weight.data[i], self.target_embeddings.weight.data[j])
        confusion_display = ConfusionMatrixDisplay(dists, names)
        confusion_display.plot(include_values=False,
                               xticks_rotation="vertical")
        confusion_display.ax_.set_xlabel("")
        confusion_display.ax_.set_ylabel("")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "dist_matrix.png"))
        plt.close()
