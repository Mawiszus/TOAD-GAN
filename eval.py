import os
import subprocess
from tap import Tap
import yaml


class EvalArgs(Tap):
    run_dir: str


def main():
    args = EvalArgs().parse_args()
    run_config_path = os.path.join(args.run_dir, "config.yaml")
    with open(run_config_path, "r") as f:
        run_config = yaml.load(f, Loader=yaml.SafeLoader)
    subprocess.call(["python", "quantitative_experiments.py",
                    "--run_dir", args.run_dir, "--output_dir", args.run_dir])
    subprocess.call(["python", "block_histograms.py",
                    "--folder", os.path.join(args.run_dir, "random_samples", "torch_blockdata")])
    subprocess.call(["python", "generate_minecraft_samples.py", "--game", "minecraft", "--out_", args.run_dir, "--input_dir", "../minecraft_worlds", '--input_name', "Drehmal v2.1 PRIMORDIAL", "--input_area_name", run_config["input_area_name"]
                     ["value"],  "--scales", *[str(scale) for scale in run_config["scales"]["value"]], "--num_layer", str(run_config["num_layer"]["value"]), "--nfc", str(run_config["nfc"]["value"]), "--repr_type", run_config["repr_type"]["value"], "--render_obj", "--num_samples", "20"])


if __name__ == "__main__":
    main()
