import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import pytorch_lightning.loggers as loggers
import seaborn as sns
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from umap import UMAP as UsedMapper

import wandb
from mario.level_classification import LevelClassification
from mario.level_image_gen import LevelImageGen
from mario.level_snippet_dataset import LevelSnippetDataset
from mario.level_utils import one_hot_to_ascii_level
from utils import set_seed


class EmbeddingsCallback(pl.Callback):
    def on_epoch_end(self, trainer: pl.Trainer, pl_module: LevelClassification):
        pl_module.mapper = visualize_embeddings(
            pl_module.dataset, pl_module, "train", pl_module.hparams)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="mario")
    parser.add_argument("--tags", nargs="*", type=str, default=["similarity"])
    parser.add_argument("--baseline-level-dir", type=str,
                        metavar="DIR", default="input/umap_images/baselines")
    parser.add_argument("--max-count", type=int, default=700)
    parser.add_argument("--restore", type=str, metavar="DIR")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available else "cpu")
    parser.add_argument("--debug", "-d", action="store_true", default=False)
    parser = LevelClassification.add_args(parser)
    hparams = parser.parse_args()
    return hparams


def main():
    logger.remove()
    logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> " +
                                                 "| <level>{level}</level> " +
                                                 "| <light-black>{file.path}:{line}</light-black> | {message}")
    hparams = parse_args()
    if hparams.restore:
        wandb.init(project=hparams.project, tags=hparams.tags)
        model = LevelClassification.load_from_checkpoint(hparams.restore)
        logger.info("Restored model")
    else:
        # wandb.init is called in LevelClassification
        model = LevelClassification(hparams)
        experiment_logger = loggers.WandbLogger(
            project=hparams.project, tags=hparams.tags)
        hparams.checkpoint_dir = os.path.join(
            experiment_logger.experiment.dir, "checkpoints")
        checkpoint_cb = callbacks.ModelCheckpoint(
            hparams.checkpoint_dir, save_top_k=1)
        trainer = pl.Trainer(logger=experiment_logger, gpus=1 if hparams.device == "cuda" else 0,
                             checkpoint_callback=checkpoint_cb, callbacks=[
                                 EmbeddingsCallback()],
                             early_stop_callback=callbacks.EarlyStopping(), fast_dev_run=hparams.debug
                             )
        trainer.fit(model)
    model.freeze()
    baseline_datasets = []
    logger.info("Baselines {}", os.listdir(hparams.baseline_level_dir))
    for i, baseline_level_dir in enumerate(sorted(os.listdir(hparams.baseline_level_dir))):
        baseline_dataset = LevelSnippetDataset(level_dir=os.path.join(os.getcwd(), hparams.baseline_level_dir,
                                                                      baseline_level_dir),
                                               slice_width=model.dataset.slice_width,
                                               token_list=model.dataset.token_list)
        baseline_datasets.append(baseline_dataset)
    visualize_embeddings(model.dataset, model, "test",
                         hparams, None, baseline_datasets)


def visualize_embeddings(dataset, model, name, hparams, mapper=None, baseline_datasets=[]):
    dataloader = DataLoader(dataset, batch_size=1)
    embeddings, labels, images = compute_embeddings(model, dataloader, hparams)

    if mapper is None:
        mapper = UsedMapper(n_components=2).fit(embeddings)
    mapped = mapper.transform(embeddings)
    baselines_with_targets = []
    baselines_without_targets = []
    baselines_images = []
    baselines_mapped_with_targets = []
    baselines_mapped_without_targets = []
    for baseline_dataset in baseline_datasets:
        baseline_dataloader = DataLoader(
            baseline_dataset, batch_size=1, shuffle=True)
        baseline_embeddings, _, b_images = compute_embeddings(
            model, baseline_dataloader, hparams, hparams.max_count)
        baseline_mapped = mapper.transform(baseline_embeddings)
        baselines_images.append(b_images)
        if baseline_dataset.level_idx is not None:
            baselines_with_targets.append(baseline_embeddings)
            baselines_mapped_with_targets.append(baseline_mapped)
        else:
            baselines_without_targets.append(baseline_embeddings)
            baselines_mapped_without_targets.append(baseline_mapped)

    if not baselines_mapped_without_targets:
        target_list = [[True, baselines_with_targets,
                        baselines_mapped_with_targets]]
    else:
        target_list = [[True, baselines_with_targets, baselines_mapped_with_targets],
                       [False, baselines_without_targets, baselines_mapped_without_targets]]

    for with_targets, baselines, baselines_mapped in target_list:
        if with_targets:
            curr_embed = embeddings
            curr_mapped = mapped
            curr_labels = labels
            curr_images = images
        else:
            n_samples = baselines[0].shape[0]
            curr_embed = np.concatenate(baselines)
            curr_mapped = np.concatenate(baselines_mapped)
            curr_labels = np.zeros((n_samples*len(baselines),))
            for i in range(len(baselines)):
                curr_labels[i*n_samples:(i+1)*n_samples] = i
            curr_images = np.concatenate(baselines_images)
        means = plot_means(curr_embed, curr_mapped, curr_labels,
                           curr_images, name, with_targets, dataset.token_list)
        plot_dataset(dataset, mapped, baselines_mapped, labels, means, name, with_targets)
    return mapper


def plot_means(curr_embed, curr_mapped, curr_labels, curr_images, name, with_targets, token_list):
    ImgGen = LevelImageGen("./mario/sprites")
    means = []
    m_pts = []
    m_imgs = []
    for i, label in enumerate(np.unique(curr_labels)):
        m = curr_embed[curr_labels == label, :].mean(0)
        closest_idx = (np.sum(abs(curr_embed - m) **
                              2, axis=1) ** (1. / 2)).argmin()
        means.append(curr_mapped[closest_idx, :])
        m_imgs.append(curr_images[closest_idx, :])
        m_pts.append(curr_embed[closest_idx, :])
    means = np.array(means)

    plt.figure(figsize=(len(np.unique(curr_labels) * 2), 2))
    for i, img in enumerate(m_imgs):
        plt.subplot(1, len(np.unique(curr_labels)), i + 1)
        plt.imshow(ImgGen.render(one_hot_to_ascii_level(
            torch.tensor(img).unsqueeze(0), token_list)))
        plt.axis("off")
    figure_path = os.path.join(
        wandb.run.dir, f"{name}_imgs{'_targets' if with_targets else ''}.pdf")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    wandb.save(figure_path)
    wandb.log({f"{name}_imgs{'_targets' if with_targets else ''}": plt})
    plt.close()
    return means

def save_embeddings(embeddings, name):
    embeddings_path = os.path.join(wandb.run.dir, f"{name}_embeddings.pt")
    torch.save(embeddings, embeddings_path)
    wandb.save(embeddings_path)

def plot_dataset(dataset, mapped, baselines_mapped, labels, means, name, with_targets):
    plot_embeddings(mapped, labels=np.array(
        [dataset.level_names[i].split(".")[0] for i in labels]),
        baselines=baselines_mapped)
    ax = sns.scatterplot(x=means[:, 0], y=means[:, 1], style=[s for s in range(means.shape[0])], legend=False,
                    markers=["o" for _ in range(means.shape[0])], hue=[s for s in range(means.shape[0])],
                    palette=sns.color_palette("hls", means.shape[0]), edgecolor='black', linewidth=0.8)
    figure_path = os.path.join(
        wandb.run.dir, f"{name}_embeddings{'_targets' if with_targets else ''}.pdf")

    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    wandb.save(figure_path)
    wandb.log(
        {f"{name}_embeddings{'_targets' if with_targets else ''}": wandb.Image(ax)})
    plt.close()


def plot_embeddings(mapper, labels, baselines=[], xlim=None, ylim=None):
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
        plt_noise = np.random.random(mapper.shape) * jitter
        mapper += plt_noise
        data = dict()
        data["x"] = mapper[:, 0]
        data["y"] = mapper[:, 1]
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


def compute_embeddings(model, dataset, hparams, max_count=-1):
    embeddings = []
    labels = []
    outputs = []
    for i, (x, y) in enumerate(tqdm(dataset)):
        x, y = x.to(hparams.device), y.to(hparams.device)
        model = model.to(hparams.device)
        x_embeddings, _ = model(x)
        embeddings.append(x_embeddings.detach().cpu().numpy())
        labels.append(y.cpu().numpy())
        outputs.append(x.detach().cpu().numpy())
        if max_count != -1 and i > max_count:
            break
    return np.array(embeddings).squeeze(), np.array(labels).squeeze(), np.array(outputs).squeeze()


if __name__ == "__main__":
    set_seed()
    main()
