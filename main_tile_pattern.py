import argparse
import collections
import math
import multiprocessing as mp
import os
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from loguru import logger
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm

import wandb
from mario.level_snippet_dataset import LevelSnippetDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="mario")
    parser.add_argument("--tags", nargs="*", type=str, default=["similarity"])
    parser.add_argument("--job-type", type=str, default="eval")
    parser.add_argument("--level-dir", type=str,
                        metavar="DIR", default="Input/Images")
    parser.add_argument("--run-dir", type=str, metavar="DIR")
    parser.add_argument("--slice-width", type=int, default=16)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--pattern-sizes", nargs="+",
                        type=int, default=[4, 3, 2])
    hparams = parser.parse_args()
    return hparams


def pattern_key(level_slice):
    """
    Computes a hashable key from a level slice.
    """
    key = ""
    for line in level_slice:
        for token in line:
            key += str(token)
    return key


def get_pattern_counts(level, pattern_size):
    """
    Collects counts from all patterns in the level of the given size.
    """
    pattern_counts = collections.defaultdict(int)
    for up in range(level.shape[0] - pattern_size + 1):
        for left in range(level.shape[1] - pattern_size + 1):
            down = up + pattern_size
            right = left + pattern_size
            level_slice = level[up:down, left:right]
            pattern_counts[pattern_key(level_slice)] += 1
    return pattern_counts


def compute_pattern_counts(dataset, pattern_size):
    """
    Compute pattern counts in parallel from a given dataset.
    """
    levels = [level.argmax(dim=0).numpy() for level in dataset.levels]
    with mp.Pool() as pool:
        counts_per_level = pool.map(
            partial(get_pattern_counts, pattern_size=pattern_size), levels,
        )
    pattern_counts = collections.defaultdict(int)
    for counts in counts_per_level:
        for pattern, count in counts.items():
            pattern_counts[pattern] += count
    return pattern_counts


def compute_prob(pattern_count, num_patterns, epsilon=1e-7):
    """
    Compute probability of a pattern.
    """
    return (pattern_count + epsilon) / ((num_patterns + epsilon) * (1 + epsilon))


def main():
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level}</level> | <light-black>{file.path}:{line}</light-black> | {message}",
    )
    hparams = parse_args()
    wandb.init(
        project=hparams.project,
        tags=hparams.tags,
        job_type=hparams.job_type,
        config=hparams,
    )
    dataset = LevelSnippetDataset(
        level_dir=hparams.level_dir, slice_width=hparams.slice_width,
    )
    display_labels = sorted([name for name in dataset.level_names])
    confusion_matrix_mean_dict = {}
    confusion_matrix_var_dict = {}
    # The run directory is expected to contain samples from all levels
    for run_dir in os.listdir(hparams.run_dir):
        run_dir = os.path.join(hparams.run_dir, run_dir)
        with open(os.path.join(run_dir, "config.yaml"), "r") as f:
            config = yaml.load(f)
        test_level_dir = os.path.join(run_dir, "random_samples", "txt")
        level_name = config["input_name"]["value"]
        divergences_mean = {}
        divergences_var = {}
        for current_level_name in dataset.level_names:
            # Compute TP KL-Div between datasets
            level_dataset = LevelSnippetDataset(
                level_dir=hparams.level_dir,
                slice_width=hparams.slice_width,
                level_name=current_level_name,
            )
            mean_kl_divergence, var_kl_divergence = compute_kl_divergence(
                level_dataset, test_level_dir, hparams
            )
            divergences_mean[current_level_name] = mean_kl_divergence
            divergences_var[current_level_name] = var_kl_divergence
        confusion_matrix_mean_dict[level_name] = divergences_mean
        confusion_matrix_var_dict[level_name] = divergences_var
    # Create confusion matrix
    for cm, stat in [
        (confusion_matrix_mean_dict, "mean"),
        (confusion_matrix_var_dict, "var"),
    ]:
        confusion_matrix = []
        table = wandb.Table(
            columns=["training level"]
            + [f"KL-divergence from level {i}" for i in range(1, 16)]
        )
        for level_name in display_labels:
            row = []
            for current_level_name in display_labels:
                row.append(
                    confusion_matrix_mean_dict[level_name][current_level_name])
            confusion_matrix.append(row)
            table_row = [level_name] + row
            table.add_data(*table_row)
        confusion_matrix = np.array(confusion_matrix)
        sns.set(context="paper", style="white")
        confusion_display = ConfusionMatrixDisplay(
            confusion_matrix, [name.split(".")[0] for name in display_labels],
        )
        confusion_display.plot()
        ax = confusion_display.ax_
        ax.set_ylabel("GAN Level")
        ax.set_xlabel("Original Level")
        plt.tight_layout()
        figure_path = os.path.join(
            wandb.run.dir, f"confusion_matrix_{stat}.pdf")
        plt.savefig(figure_path, dpi=300)
        wandb.save(figure_path)
        wandb.log(
            {
                f"confusion_matrix_{stat}": wandb.Image(ax),
                f"kl_divergences_{stat}": table,
            }
        )


def compute_kl_divergence(dataset, test_level_dir, hparams):
    logger.info(
        "Computing KL-Divergence for generated levels in {}", test_level_dir)
    test_dataset = LevelSnippetDataset(
        level_dir=test_level_dir,
        slice_width=hparams.slice_width,
        token_list=dataset.token_list,
    )
    kl_divergences = []
    for pattern_size in hparams.pattern_sizes:
        logger.info("Computing original pattern counts...")
        pattern_counts = compute_pattern_counts(dataset, pattern_size)
        logger.info("Computing test pattern counts...")
        test_pattern_counts = compute_pattern_counts(
            test_dataset, pattern_size)

        num_patterns = sum(pattern_counts.values())
        num_test_patterns = sum(test_pattern_counts.values())
        logger.info(
            "Found {} patterns and {} test patterns", num_patterns, num_test_patterns
        )

        kl_divergence = 0
        for pattern, count in tqdm(pattern_counts.items()):
            prob_p = compute_prob(count, num_patterns)
            prob_q = compute_prob(
                test_pattern_counts[pattern], num_test_patterns)
            kl_divergence += hparams.weight * prob_p * math.log(prob_p / prob_q) + (
                1 - hparams.weight
            ) * prob_q * math.log(prob_q / prob_p)

        kl_divergences.append(kl_divergence)
        logger.info(
            "KL-Divergence @ {}x{}: {}",
            pattern_size,
            pattern_size,
            round(kl_divergence, 2),
        )
    mean_kl_divergence = np.mean(kl_divergences)
    var_kl_divergence = np.std(kl_divergences)
    logger.info("Average KL-Divergence: {}", round(mean_kl_divergence, 2))
    return mean_kl_divergence, var_kl_divergence


if __name__ == "__main__":
    main()
