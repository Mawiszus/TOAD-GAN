#!/bin/zsh

python quantitative_experiments.py --run_dir $1 --output_dir $1 &
python block_histograms.py --folder $1/random_samples/torch_blockdata &

wait