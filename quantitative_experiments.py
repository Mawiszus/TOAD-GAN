from itertools import product
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
    metrics: List[str] = ["levenshtein", "tpkldiv"]
    tpkldiv_pattern_sizes: List[int] = [5, 10]
    tpkldiv_weight: float = 0.5

    def process_args(self) -> None:
        super().process_args()
        os.makedirs(self.output_dir, exist_ok=True)


def compute_levenshtein(real: np.ndarray, generated: List[np.ndarray]):
    generated_str = ["".join(gen.flatten().astype(str)) for gen in generated]
    distances = [levenshtein_distance(gen_str_1, gen_str_2)
                 for gen_str_1, gen_str_2 in product(generated_str, generated_str)]
    return np.mean(distances), np.var(distances)


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


def write_results(output_dir: str, results: Dict):
    with open(os.path.join(output_dir, "random_samples", "results.json"), "w") as f:
        json.dump(results, f)


def main():
    args = QuantitativeExperimentArgs().parse_args()
    real, generated = load_levels(args.run_dir)
    results = dict()
    if "tpkldiv" in args.metrics:
        mean_tpkldiv, var_tpkldiv = compute_tpkldiv(
            real, generated, args.tpkldiv_pattern_sizes, args.tpkldiv_weight)
        results["tpkldiv"] = {"mean": mean_tpkldiv, "var": var_tpkldiv}
    if "levenshtein" in args.metrics:
        mean_levenshtein, var_levenshtein = compute_levenshtein(
            real, generated)
        results["levenshtein"] = {
            "mean": mean_levenshtein, "var": var_levenshtein}
    write_results(args.output_dir, results)


if __name__ == "__main__":
    main()
