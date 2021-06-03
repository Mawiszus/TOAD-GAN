#!/bin/zsh

#SBATCH --partition=gpu_normal
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gpus=1
#SBATCH --job-name=toadgan

cd $SLURM_SUBMIT_DIR
echo $CONDA_ENV
. /home/awiszus/miniconda3/tmp/bin/activate $CONDA_ENV

echo "Running $@ from $SLURM_SUBMIT_DIR"
srun $@
