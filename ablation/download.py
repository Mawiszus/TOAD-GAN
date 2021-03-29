import os
from pathlib import Path

import wandb
from tqdm import tqdm


def main():
    base_path = Path(os.path.abspath(os.path.dirname(__file__)))
    run_paths = base_path.glob("*/runs.txt")
    api = wandb.Api()
    for run_path in tqdm(run_paths):
        with open(run_path, "r") as f:
            runs = f.readlines()
        for run_id in tqdm(runs):
            run_id = run_id.strip()
            run_target_path = run_path.parent.joinpath(run_id)
            if run_target_path.exists():
                continue
            os.makedirs(run_target_path)
            run = api.run(f"tnt/mario/{run_id}")
            for file in run.files():
                file.download(root=run_target_path)


if __name__ == "__main__":
    main()
