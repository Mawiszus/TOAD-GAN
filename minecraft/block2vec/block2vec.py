import math
import os
from typing import Tuple

import pytorch_lightning as pl
from tap import Tap
import torch
import torch.optim as optim
from minecraft.block2vec.skip_gram_model import SkipGramModel
from minecraft.block2vec.block2vec_dataset import Block2VecDataset
from torch.utils.data import DataLoader


class Block2VecArgs(Tap):
    input_world_path: str = os.path.abspath(os.path.join(os.path.dirname(
        __file__), "..", "..", "..", "minecraft_worlds", "Drehmal v2.1 PRIMORDIAL"))
    output_path: str = os.path.join("output", "block2vec")
    emb_dimension: int = 5
    epochs: int = 50
    batch_size: int = 32
    initial_lr: float = 1e-3
    input_world_coords: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = (
        (1028, 1076), (60, 80), (1088, 1127))


class Block2Vec(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.args: Block2VecArgs = Block2VecArgs().from_dict(kwargs)
        self.save_hyperparameters()
        self.dataset = Block2VecDataset(
            self.args.input_world_path, coords=self.args.input_world_coords)
        self.emb_size = len(self.dataset.block2idx)
        self.model = SkipGramModel(self.emb_size, self.args.emb_dimension)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.model(*args, **kwargs)

    def training_step(self, batch, *args, **kwargs):
        loss = self.forward(*batch)
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.args.initial_lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, math.ceil(len(self.dataset) / self.args.batch_size) * self.args.epochs)
        return [optimizer], [scheduler]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True, pin_memory=True, num_workers=os.cpu_count() or 1)

    def on_epoch_end(self):
        self.model.save_embedding(
            self.dataset.idx2block, self.args.output_path)
        self.model.create_confusion_matrix(
            self.dataset.idx2block, self.args.output_path)
