#!/bin/zsh

#SBATCH --partition=gpu_normal
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gpus=1
#SBATCH --job-name=toadgan
#SBATCH --output=/home/schubert/projects/TOAD-GAN/%x-%j.slurm.log

cd $SLURM_SUBMIT_DIR
echo $CONDA_ENV
. $CONDA_PREFIX_1/bin/activate $CONDA_ENV

echo "Running $@ from $SLURM_SUBMIT_DIR"
srun $@
