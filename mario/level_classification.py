import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl
from loguru import logger
import umap

from mario.level_snippet_dataset import LevelSnippetDataset


class SnippetDiscriminator(nn.Module):
    def __init__(self, in_channels, num_classes, depth, kernel_size, stride, dropout_rate=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.conv1 = nn.Conv2d(in_channels, depth, kernel_size, stride)
        self.ln1 = nn.GroupNorm(1, depth)
        self.conv2 = nn.Conv2d(depth, depth * 2, kernel_size, stride)
        self.ln2 = nn.GroupNorm(1, depth * 2)
        self.fc1 = nn.Linear(depth * 2 * 3 * 3, depth * 4)
        self.fc2 = nn.Linear(depth * 4, num_classes)

    def forward(self, x):
        x = self.ln1(F.relu(self.conv1(x)))
        x = self.ln2(F.relu(self.conv2(self.dropout(x))))
        x = torch.flatten(x, start_dim=1)
        x_embedding = F.relu(self.fc1(x))
        y_hat = self.fc2(x_embedding)
        return x_embedding, y_hat


class LevelClassification(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.hparams = hparams
        self.dataset = LevelSnippetDataset(
            level_dir=hparams.level_dir, slice_width=hparams.slice_width)
        train_size = math.floor(hparams.train_split * len(self.dataset))
        val_size = math.floor((1 - hparams.train_split)
                              * len(self.dataset)) + 1
        logger.info("Loaded dataset with {} snippets. Train/Validation split {}/{}",
                    len(self.dataset), train_size, val_size)
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size])
        self.discriminator = SnippetDiscriminator(len(self.dataset.token_list), len(self.dataset.levels),
                                                  depth=hparams.depth, kernel_size=hparams.kernel_size,
                                                  stride=hparams.stride)
        self.mapper: umap.UMAP = None

    def forward(self, x):
        return self.discriminator(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_embedding, y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        return {"loss": loss, "log": {"loss": loss}}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, len(self.dataset))
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.hparams.batch_size, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=2)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_embedding, y_hat = self.forward(x)
        val_loss = F.cross_entropy(y_hat, y)
        return {"val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"val_loss": avg_loss, "log": {"avg_val_loss": avg_loss}}

    def on_save_checkpoint(self, checkpoint):
        if self.mapper is not None:
            checkpoint["mapper"] = self.mapper

    def on_load_checkpoint(self, checkpoint):
        if "mapper" in checkpoint:
            self.mapper = checkpoint["mapper"]

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument("--level-dir", type=str,
                            metavar="DIR", default="input")
        parser.add_argument("--train-split", type=float, default=0.8)
        parser.add_argument("--learning-rate", type=float, default=1e-3)
        parser.add_argument("--batch-size", type=int, default=32)
        parser.add_argument("--slice-width", type=int, default=16)
        parser.add_argument("--depth", type=int, default=16)
        parser.add_argument("--kernel-size", type=int, default=3)
        parser.add_argument("--stride", type=int, default=2)
        return parser
