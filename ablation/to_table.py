import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict

import latextable
import numpy as np
import yaml
from texttable import Texttable
from tqdm import tqdm


def create_table(results_dict: Dict, caption: str):
    table = Texttable()
    column_names = list(
        sorted(results_dict[list(results_dict.keys())[0]].keys()))
    column_names = [name.replace("_", " ") for name in column_names]
    rows = [["Structure", *column_names]]
    table.set_cols_align(["l"] + len(results_dict) * ["c"])
    for method_name, results in results_dict.items():
        row = [method_name]
        for structure in sorted(results.keys()):
            row.append(results[structure])
        rows.append(row)
    rows_transposed = list(zip(*rows))
    table.add_rows(rows_transposed)
    print(latextable.draw_latex(
        table, caption=caption) + "\n")


def main():
    base_path = Path(os.path.abspath(os.path.dirname(__file__)))
    run_paths = base_path.glob("*/runs.txt")
    tpkldiv_table = defaultdict(dict)
    levenshtein_table = defaultdict(dict)
    for run_path in run_paths:
        with open(run_path, "r") as f:
            runs = f.readlines()
        for run_id in runs:
            run_id = run_id.strip()
            run_dir = run_path.parent.joinpath(run_id)
            with open(run_dir.joinpath("config.yaml"), "r") as f:
                run_config = yaml.load(f, Loader=yaml.SafeLoader)
            with open(run_dir.joinpath("random_samples").joinpath("results.json"), "r") as f:
                run_results = json.load(f)
            method_name = str(run_path.parent.relative_to(
                base_path)).split("_")[0].split(":")[-1]
            structure = run_config["input_area_name"]["value"]
            tpkldiv_table[method_name][structure] = np.mean(
                list(run_results["tpkldiv"]["mean"].values()))
            levenshtein_table[method_name][structure] = run_results["levenshtein"]["mean"]
    create_table(tpkldiv_table, "Average Tile-Pattern KL-Divergence")
    create_table(levenshtein_table, "Average Levenshtein Distance")

if __name__ == "__main__":
    main()
