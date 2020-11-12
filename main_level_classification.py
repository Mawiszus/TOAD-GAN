import os
import sys
from typing import List, Optional
from joblib import dump, load

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import pytorch_lightning.loggers as loggers
import seaborn as sns
import torch
import wandb
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from umap import UMAP as UsedMapper

from mario.level_classification import (LevelClassification,
                                        LevelClassificationParams)
from mario.level_image_gen import LevelImageGen
from mario.level_snippet_dataset import LevelSnippetDataset
from mario.level_utils import one_hot_to_ascii_level
from utils import set_seed


class Params(LevelClassificationParams):
    project: str = "mario"
    tags: List[str] = []
    baseline_level_dir: str = "input/umap_images/baselines"
    max_count: int = 1000
    restore: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: Optional[str] = None
    restore_mapper: Optional[str] = None
    seed: int = 42


def main():
    logger.remove()
    logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> " +
                                                 "| <level>{level}</level> " +
                                                 "| <light-black>{file.path}:{line}</light-black> | {message}")
    hparams = Params().parse_args()
    torch.manual_seed(hparams.seed)
    np.random.seed(hparams.seed)
    torch.set_deterministic(True)
    run = wandb.init(project=hparams.project,
                     tags=hparams.tags, config=hparams.as_dict())
    if hparams.restore:
        model = LevelClassification.load_from_checkpoint(hparams.restore)
        logger.info("Restored model")
    else:
        model = LevelClassification(hparams.as_dict())
        experiment_logger = loggers.WandbLogger(experiment=run)
        hparams.checkpoint_dir = os.path.join(
            experiment_logger.experiment.dir, "checkpoints")
        checkpoint_cb = callbacks.ModelCheckpoint(
            hparams.checkpoint_dir, monitor="val_loss", save_top_k=1)
        trainer = pl.Trainer(logger=experiment_logger, gpus=1 if hparams.device == "cuda" else 0,
                             checkpoint_callback=checkpoint_cb, callbacks=[callbacks.EarlyStopping(monitor="val_loss")], fast_dev_run=hparams.debug
                             )
        trainer.fit(model)
    if hparams.restore_mapper:
        mapper = load(hparams.restore_mapper)
    else:
        mapper = None
    model.freeze()
    model.eval()
    baseline_datasets = []
    logger.info("Baselines {}", os.listdir(hparams.baseline_level_dir))
    for i, baseline_level_dir in enumerate(sorted(os.listdir(hparams.baseline_level_dir))):
        level_dir = os.path.join(hparams.baseline_level_dir,
                                 baseline_level_dir)
        if not os.path.isdir(level_dir):
            continue
        baseline_dataset = LevelSnippetDataset(level_dir=level_dir,
                                               slice_width=model.dataset.slice_width,
                                               token_list=model.dataset.token_list, debug=hparams.debug)
        baseline_datasets.append(baseline_dataset)
    visualize_embeddings(model.dataset, model, hparams,
                         baseline_datasets, mapper)


def visualize_embeddings(dataset: LevelSnippetDataset, model: LevelClassification, hparams: Params, baseline_datasets: List[LevelSnippetDataset] = [], mapper=None):
    dataloader = DataLoader(dataset, batch_size=1)
    embeddings, labels, images = compute_embeddings(model, dataloader, hparams)
    if not mapper:
        mapper = UsedMapper(n_components=2, random_state=hparams.seed).fit(embeddings)
        dump(mapper, os.path.join(wandb.run.dir, "mapper.joblib"), compress=3)
    mapped = mapper.transform(embeddings)
    baselines = []
    baselines_images = []
    baselines_mapped = []
    for baseline_dataset in baseline_datasets:
        baseline_dataloader = DataLoader(
            baseline_dataset, batch_size=1, shuffle=True)
        baseline_embeddings, _, b_images = compute_embeddings(
            model, baseline_dataloader, hparams, hparams.max_count)
        baseline_mapped = mapper.transform(baseline_embeddings)
        baselines_images.append(b_images)
        baselines.append(baseline_embeddings)
        baselines_mapped.append(baseline_mapped)

    n_samples = baselines[0].shape[0]
    curr_embed = np.concatenate(baselines)
    curr_mapped = np.concatenate(baselines_mapped)
    curr_labels = np.zeros((n_samples*len(baselines),))
    for i in range(len(baselines)):
        curr_labels[i*n_samples:(i+1)*n_samples] = i
    curr_images = np.concatenate(baselines_images)
    means = plot_means(curr_embed, curr_mapped, curr_labels,
                       curr_images, "test", dataset.token_list)
    plot_dataset(dataset, mapped, baselines_mapped,
                 labels, means, "test")

    means = plot_means(embeddings, mapped, labels,
                       images, "train", dataset.token_list)
    plot_dataset(dataset, mapped, [],
                 labels, means, "train")
    return mapper


def plot_means(embeddings, mapped, labels, images, name, token_list):
    ImgGen = LevelImageGen("./mario/sprites")
    means = []
    m_pts = []
    m_imgs = []
    for i, label in enumerate(np.unique(labels)):
        m = embeddings[labels == label, :].mean(0)
        closest_idx = (np.sum(abs(embeddings - m) **
                              2, axis=1) ** (1. / 2)).argmin()
        means.append(mapped[closest_idx, :])
        m_imgs.append(images[closest_idx, :])
        m_pts.append(embeddings[closest_idx, :])
    means = np.array(means)

    plt.figure(figsize=(len(np.unique(labels) * 2), 2))
    for i, img in enumerate(m_imgs):
        plt.subplot(1, len(np.unique(labels)), i + 1)
        plt.imshow(ImgGen.render(one_hot_to_ascii_level(
            torch.tensor(img).unsqueeze(0), token_list)))
        plt.axis("off")
    figure_path = os.path.join(
        wandb.run.dir, f"{name}_imgs.pdf")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    wandb.save(figure_path)
    wandb.log({f"{name}_imgs": plt})
    plt.close()
    return means


def save_embeddings(embeddings, name):
    embeddings_path = os.path.join(wandb.run.dir, f"{name}_embeddings.pt")
    torch.save(embeddings, embeddings_path)
    wandb.save(embeddings_path)


def plot_dataset(dataset, mapped, baselines_mapped, labels, means, name):
    plot_embeddings(mapped, labels=np.array(
        [dataset.level_names[i].split(".")[0] for i in labels]),
        baselines=baselines_mapped)
    ax = sns.scatterplot(x=means[:, 0], y=means[:, 1], style=[s for s in range(means.shape[0])], legend=False,
                         markers=["o" for _ in range(means.shape[0])], hue=[s for s in range(means.shape[0])],
                         palette=sns.color_palette("hls", means.shape[0]), edgecolor='black', linewidth=0.8)
    figure_path = os.path.join(
        wandb.run.dir, f"{name}_embeddings.pdf")

    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    wandb.save(figure_path)
    wandb.log(
        {f"{name}_embeddings": wandb.Image(ax)})
    plt.close()


def plot_embeddings(embeddings, labels, baselines=[], xlim=None, ylim=None):
    sns.set(context="paper", style="white")
    colors = sns.color_palette("hls", len(np.unique(labels)))
    training_palette = sns.color_palette("hls", len(np.unique(labels)))
    plt.clf()
    labels_n = np.zeros_like(labels)
    markers = []
    for i, label in enumerate(np.unique(labels)):
        labels_n[labels == label] = label[4:]
        markers.append("o")

    jitter = 0.5
    alpha = 0.3
    size = 0.7

    for i, baseline_embeddings in enumerate(baselines):
        plt_noise = np.random.random(baseline_embeddings.shape) * jitter
        baseline_embeddings += plt_noise
        b_data = dict()
        b_data["x"] = baseline_embeddings[:, 0]
        b_data["y"] = baseline_embeddings[:, 1]
        b_data["labels"] = [
            "G " + np.unique(labels_n)[i] for _ in range(baseline_embeddings.shape[0])]
        b_data["size"] = 0.2
        baseline_ax = sns.scatterplot("x", "y", alpha=alpha, style="labels", markers={"o"}, legend='brief', size="size",
                                      color=colors[i], palette=training_palette,
                                      data=b_data, linewidth=0)
        baseline_ax.set_xlim(xlim)
        baseline_ax.set_ylim(ylim)
    if not baselines:
        plt_noise = np.random.random(embeddings.shape) * jitter
        embeddings += plt_noise
        data = dict()
        data["x"] = embeddings[:, 0]
        data["y"] = embeddings[:, 1]
        data["labels"] = labels_n
        data["palette"] = training_palette
        data["markers"] = markers

        ax = sns.scatterplot(x="x", y="y", style="labels", markers=markers, hue="labels", palette=training_palette,
                             data=data, linewidth=0, alpha=alpha, size=size, legend='brief')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        h, l = ax.get_legend_handles_labels()
        plt.legend(h[:15], l[:15])
    else:
        ax = baseline_ax
        h, l = ax.get_legend_handles_labels()
        n = [j for j in range(len(h)) if l[j][0] == 'G']
        n_h = [h[j] for j in n]
        n_l = [l[j] for j in n]
        for k in range(len(n_h)):
            n_h[k].set_color(colors[k])
            n_h[k].set_edgecolor(colors[k])
            n_h[k].set_facecolor(colors[k])
        plt.legend(n_h, n_l)

    for t in ax.texts:
        t.set_visible(False)


def compute_embeddings(model: LevelClassification, dataloader: DataLoader, hparams: Params, max_count=-1):
    embeddings = []
    labels = []
    outputs = []
    model.to(hparams.device)
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(dataloader)):
            x = x.to(hparams.device)
            x_embeddings, _ = model(x)
            embeddings.append(x_embeddings.cpu().numpy())
            labels.append(y.numpy())
            outputs.append(x.cpu().numpy())
            if max_count != -1 and i > max_count:
                break
    return np.concatenate(embeddings).squeeze(), np.concatenate(labels).squeeze(), np.concatenate(outputs).squeeze()


if __name__ == "__main__":
    set_seed()
    main()
