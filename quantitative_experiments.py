import json
import math
import multiprocessing as mp
import os
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
from loguru import logger
from tap import Tap
from tqdm import tqdm
from Levenshtein import distance as levenshtein_distance

from main_tile_pattern import compute_prob, pattern_key


class QuantitativeExperimentArgs(Tap):
    run_dir: str
    output_dir: str
    metrics: List[str] = ["tpkldiv"]
    # TODO(frederik): choose right sizes and weight
    tpkldiv_pattern_sizes: List[int] = [2, 3, 4]
    tpkldiv_weight: float = 0.5

    def process_args(self) -> None:
        super().process_args()
        os.makedirs(self.output_dir, exist_ok=True)


def compute_levenshtein(real: np.ndarray, generated: List[np.ndarray]):
    real_str = "".join(real.flatten())
    generated_str = ["".join(gen.flatten()) for gen in generated]
    distances = [levenshtein_distance(real_str, gen_str)
                 for gen_str in generated_str]
    return np.mean(distances), np.var(distances)


def write_levenshtein(mean_levenshtein: float, var_levenshtein: float, output_path: str):
    logger.info("Writing Levenshtein results")
    with open(Path(output_path).joinpath("mean_levenshtein.json"), "w")as f:
        json.dump(mean_levenshtein, f)
    with open(Path(output_path).joinpath("var_levenshtein.json"), "w")as f:
        json.dump(var_levenshtein, f)


def compute_tpkldiv(real: np.ndarray, generated: List[np.ndarray], pattern_sizes: List[int], weight: float):
    dists = defaultdict(list)
    for pattern_size in pattern_sizes:
        logger.info("Computing TP KL-Div for patterns of size {}", pattern_size)
        real_pattern_counts = compute_pattern_counts([real], pattern_size)
        generated_pattern_counts = compute_pattern_counts(
            generated, pattern_size)
        num_patterns = sum(real_pattern_counts.values())
        num_test_patterns = sum(generated_pattern_counts.values())
        logger.info(
            "Found {} patterns and {} test patterns", num_patterns, num_test_patterns
        )

        kl_divergence = 0
        for pattern, count in tqdm(generated_pattern_counts.items()):
            prob_p = compute_prob(count, num_patterns)
            prob_q = compute_prob(
                generated_pattern_counts[pattern], num_test_patterns)
            kl_divergence += weight * prob_p * math.log(prob_p / prob_q) + (
                1 - weight
            ) * prob_q * math.log(prob_q / prob_p)
            dists[pattern_size].append(kl_divergence)
    mean_tpkldiv: Dict[int, float] = {k: np.mean(v) for k, v in dists.items()}
    var_tpkldiv: Dict[int, float] = {k: np.var(v) for k, v in dists.items()}
    return mean_tpkldiv, var_tpkldiv


def write_tpkldiv(mean_tpkldiv: Dict[int, float], var_tpkldiv: Dict[int, float], output_path: str):
    logger.info("Writing TP KL-Div results")
    with open(Path(output_path).joinpath("mean_tpkldiv.json"), "w")as f:
        json.dump(mean_tpkldiv, f)
    with open(Path(output_path).joinpath("var_tpkldiv.json"), "w")as f:
        json.dump(var_tpkldiv, f)


def get_pattern_counts(level: np.ndarray, pattern_size: int):
    pattern_counts = defaultdict(int)
    for up in range(level.shape[0] - pattern_size + 1):
        for left in range(level.shape[1] - pattern_size + 1):
            for inside in range(level.shape[2] - pattern_size + 1):
                down = up + pattern_size
                right = left + pattern_size
                outside = inside + pattern_size
                level_slice = level[up:down, left:right, inside:outside]
                pattern_counts[pattern_key(level_slice)] += 1
    return pattern_counts


def compute_pattern_counts(levels: List[np.ndarray], pattern_size: int):
    with mp.Pool() as pool:
        counts_per_level = pool.map(
            partial(get_pattern_counts, pattern_size=pattern_size), levels,
        )
    pattern_counts = defaultdict(int)
    for counts in counts_per_level:
        for pattern, count in counts.items():
            pattern_counts[pattern] += count
    return pattern_counts


def load_level(path: Union[str, Path]) -> np.ndarray:
    return torch.load(str(path))


def load_levels(run_dir: str):
    samples_path = Path(run_dir).joinpath("random_samples")
    real = load_level(samples_path.joinpath("real_bdata.pt"))
    generated = [load_level(str(path)) for path in
                 samples_path.joinpath("torch_blockdata").glob("*.pt")]
    logger.info("Found {} levels in {}", len(generated), run_dir)
    return real, generated


def main():
    args = QuantitativeExperimentArgs().parse_args()
    real, generated = load_levels(args.run_dir)
    if "tpkldiv" in args.metrics:
        mean_tpkldiv, var_tpkldiv = compute_tpkldiv(
            real, generated, args.tpkldiv_pattern_sizes, args.tpkldiv_weight)
        write_tpkldiv(mean_tpkldiv, var_tpkldiv, args.output_dir)
    if "levenshtein" in args.metrics:
        mean_levenshtein, var_levenshtein = compute_levenshtein(
            real, generated)
        write_levenshtein(mean_levenshtein, var_levenshtein, args.output_dir)


if __name__ == "__main__":
    main()
