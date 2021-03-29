import json
import os
from collections import defaultdict
from pathlib import Path

import latextable
import numpy as np
import yaml
from texttable import Texttable
from tqdm import tqdm


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
            method_name = run_path.parent.relative_to(base_path)
            structure = run_config["input_area_name"]["value"]
            tpkldiv_table[method_name][structure] = np.mean(
                list(run_results["tpkldiv"]["mean"].values()))
            levenshtein_table[method_name][structure] = run_results["levenshtein"]["mean"]
    table = Texttable()
    column_names = list(
        sorted(tpkldiv_table[list(tpkldiv_table.keys())[0]].keys()))
    column_names = [name.replace("_", " ") for name in column_names]
    table.add_row(["Method", *column_names])
    table.set_cols_align(["l"] + len(column_names) * ["c"])
    for method_name, results in tpkldiv_table.items():
        row = [method_name]
        for strucure in sorted(results.keys()):
            row.append(results[structure])
        table.add_row(row)
    print(latextable.draw_latex(
        table, caption="Average Tile-Pattern KL-Divergence") + "\n")
    table = Texttable()
    column_names = list(
        sorted(levenshtein_table[list(levenshtein_table.keys())[0]].keys()))
    column_names = [name.replace("_", " ") for name in column_names]
    table.add_row(["Method", *column_names])
    table.set_cols_align(["l"] + len(column_names) * ["c"])
    for method_name, results in levenshtein_table.items():
        row = [method_name]
        for strucure in sorted(results.keys()):
            row.append(results[structure])
        table.add_row(row)
    print(latextable.draw_latex(
        table, caption="Average Levenshtein Distance") + "\n")


if __name__ == "__main__":
    main()
