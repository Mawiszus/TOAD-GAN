from multiprocessing import Pool
import os
import subprocess
from pathlib import Path

import yaml
from tqdm import tqdm


def quantitative_experiments(run_dir):
    subprocess.call(["python", "quantitative_experiments.py",
                    "--run_dir", run_dir, "--output_dir", run_dir])


def block_histograms(run_dir):
    subprocess.call(["python", "block_histograms.py",
                     "--folder", os.path.join(run_dir, "random_samples", "torch_blockdata")])


def minecraft_samples(run_dir):
    run_config_path = os.path.join(run_dir, "config.yaml")
    with open(run_config_path, "r") as f:
        run_config = yaml.load(f, Loader=yaml.SafeLoader)
    subprocess.call(["python", "generate_minecraft_samples.py", "--game", "minecraft", "--out_", run_dir, "--input_dir", "../minecraft_worlds", '--input_name', "Drehmal v2.1 PRIMORDIAL", "--input_area_name", run_config["input_area_name"]
                     ["value"],  "--scales", *[str(scale) for scale in run_config["scales"]["value"]], "--num_layer", str(run_config["num_layer"]["value"]), "--nfc", str(run_config["nfc"]["value"]), "--repr_type", run_config["repr_type"]["value"], "--render_obj", "--num_samples", "5"])


def main():
    base_path = Path(os.path.abspath(os.path.dirname(__file__)))
    run_paths = base_path.glob("*/runs.txt")
    with Pool(16) as pool:
        run_dirs = []
        for run_path in tqdm(run_paths):
            with open(run_path, "r") as f:
                runs = f.readlines()
            for run_id in tqdm(runs):
                run_id = run_id.strip()
                run_dir = run_path.parent.joinpath(run_id)
                run_dirs.append(run_dir)
        pool.map(quantitative_experiments, run_dirs)
        # pool.map(block_histograms, run_dirs)
        # pool.map(minecraft_samples, run_dirs)


if __name__ == "__main__":
    main()
