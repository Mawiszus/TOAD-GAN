import math
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
from loguru import logger
from tap.tap import Tap
from torch.utils.data import DataLoader, random_split

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


class LevelClassificationParams(Tap):
    level_dir: str = "input/mario"
    train_split: float = 0.8
    learning_rate: float = 1e-3
    batch_size: int = 32
    slice_width: int = 16
    depth: int = 16
    kernel_size: int = 3
    stride: int = 2
    debug: bool = False


class LevelClassification(pl.LightningModule):
    def __init__(self, hparams: Dict):
        super().__init__()
        self.save_hyperparameters()
        self.params: LevelClassificationParams = LevelClassificationParams().from_dict(hparams)
        self.dataset = LevelSnippetDataset(
            level_dir=self.params.level_dir, slice_width=self.params.slice_width, debug=self.params.debug)
        train_size = math.floor(self.params.train_split * len(self.dataset))
        val_size = math.floor((1 - self.params.train_split)
                              * len(self.dataset)) + 1
        logger.info("Loaded dataset with {} snippets. Train/Validation split {}/{}",
                    len(self.dataset), train_size, val_size)
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size])
        self.discriminator = SnippetDiscriminator(len(self.dataset.token_list), len(self.dataset.levels),
                                                  depth=self.params.depth, kernel_size=self.params.kernel_size,
                                                  stride=self.params.stride)
        self.mapper: Optional[umap.UMAP] = None

    def forward(self, x):
        return self.discriminator(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_embedding, y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.params.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, len(self.dataset), eta_min=0)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.params.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.params.batch_size, num_workers=8)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_embedding, y_hat = self.forward(x)
        val_loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", val_loss)
        return {"val_loss": val_loss}

    def on_save_checkpoint(self, checkpoint):
        if self.mapper is not None:
            checkpoint["mapper"] = self.mapper

    def on_load_checkpoint(self, checkpoint):
        if "mapper" in checkpoint:
            self.mapper = checkpoint["mapper"]
