#!/bin/bash

#SBATCH --mail-user=schubert@tnt.uni-hannover.de
#SBATCH --mail-type=ALL
#SBATCH --job-name=toadgan
#SBATCH --output=slurm-%j-out.txt
#SBATCH --time=2-0
#SBATCH --partition=gpu_cluster_enife
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=5
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:5

cd /home/schubert/projects/TOAD-GAN
source /home/schubert/miniconda3/tmp/bin/activate toadgan

for file in input/*.txt
do
    srun --ntasks 1 --gres=gpu:1 python main.py --scales 0.9 0.7 0.5 --input_dir input --input_name $(basename -- $file) --num_layer 3  --niter 6000 --nfc 64 &
done

wait

RUN_DIR=./tmp/$SLURM_JOBID
mkdir -p $RUN_DIR/runs
mv wandb/*run-* $RUN_DIR/runs/

srun -n 1 python main_tile_pattern.py --level-dir input --run-dir $RUN_DIR/runs

mkdir -p $RUN_DIR/results
mv wandb/*run-* $RUN_DIR/results/

srun -n 1 python main_level_classification.py --level-dir input

mkdir -p $RUN_DIR/classifier
mv wandb/*run-* $RUN_DIR/classifier/

srun -n 1 python main_level_classification.py --restore $RUN_DIR/classifier/*run-*/*.ckpt --visualize --device cpu --baseline-level-dir $RUN_DIR/runs --target-level-indices {0..14}

mv wandb/*run-* $RUN_DIR/results/