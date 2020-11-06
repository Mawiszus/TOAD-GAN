import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
from loguru import logger
from scipy.spatial.distance import directed_hausdorff, cdist
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data.dataloader import DataLoader

from main_level_classification import Params, compute_embeddings
from mario.level_classification import LevelClassification
from mario.level_snippet_dataset import LevelSnippetDataset


def main():
    logger.remove()
    logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> " +
                                                 "| <level>{level}</level> " +
                                                 "| <light-black>{file.path}:{line}</light-black> | {message}")
    hparams = Params().parse_args()
    wandb.init(project=hparams.project, tags=hparams.tags)
    if not hparams.restore:
        logger.error(
            "Train the classifier using main_level_classification.py and restore its checkpoint via the --restore flag")
        return
    model: LevelClassification = LevelClassification.load_from_checkpoint(
        hparams.restore)
    logger.info("Restored model")
    model.freeze()
    model.eval()
    train_datasets = [LevelSnippetDataset(level_dir=os.path.join(hparams.level_dir),
                                          slice_width=model.dataset.slice_width,
                                          token_list=model.dataset.token_list, level_name=level_name, debug=hparams.debug) for level_name in sorted(os.listdir(hparams.level_dir)) if level_name.endswith(".txt")]
    test_datasets = [LevelSnippetDataset(level_dir=os.path.join(hparams.baseline_level_dir, level_name),
                                         slice_width=model.dataset.slice_width,
                                         token_list=model.dataset.token_list, debug=hparams.debug) for level_name in sorted(os.listdir(hparams.baseline_level_dir)) if os.path.isdir(os.path.join(hparams.baseline_level_dir, level_name))]
    display_labels = [dataset.level_name for dataset in train_datasets]
    train_datasets = [DataLoader(
        dataset, batch_size=512, num_workers=os.cpu_count() or 1) for dataset in train_datasets]
    test_datasets = [DataLoader(dataset, batch_size=512, num_workers=os.cpu_count() or 1)
                     for dataset in test_datasets]
    train_embeddings = []
    for train_dataset in train_datasets:
        embeddings_b, *_ = compute_embeddings(model, train_dataset, hparams)
        train_embeddings.append(embeddings_b)
    test_embeddings = []
    for test_dataset in test_datasets:
        embeddings_b, *_ = compute_embeddings(model, test_dataset, hparams)
        test_embeddings.append(embeddings_b)
    metrics = [
        ("hausdorff", lambda a, b: directed_hausdorff(a, b)[0]),
        ("mean_euclidean", lambda a, b: cdist(a, b).mean())
    ]
    for metric_name, metric in metrics:
        confusion_matrix = []
        for embeddings_a in test_embeddings:
            row = []
            for embeddings_b in train_embeddings:
                row.append(metric(embeddings_a, embeddings_b))
            confusion_matrix.append(row)
        confusion_matrix = np.array(confusion_matrix)
        sns.set(context="paper", style="white")
        confusion_display = ConfusionMatrixDisplay(
            confusion_matrix, [name.split(".")[0].split("_")[1]
                               for name in display_labels],
        )
        confusion_display.plot()
        ax = confusion_display.ax_
        ax.set_ylabel("GAN Level")
        ax.set_xlabel("Original Level")
        plt.tight_layout()
        figure_path = os.path.join(
            wandb.run.dir, f"{metric_name}_distances.pdf")
        plt.savefig(figure_path, dpi=300)
        wandb.log({
            f"{metric_name}_distances": wandb.Image(ax)
        })


if __name__ == "__main__":
    main()
