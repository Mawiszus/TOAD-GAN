import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


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
