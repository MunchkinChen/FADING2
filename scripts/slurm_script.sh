#!/bin/bash
#SBATCH --job-name=FADING2 #改
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=A100 #改
#SBATCH --cpus-per-task=10
#SBATCH --time=01-00:00:00 #改
#SBATCH --output=./slurm_script.out
#SBATCH --error=./slurm_script.out


#PIP_ENV="/home/ids/xchen-21/dream_env"
PIP_ENV="/home/ids/xchen-21/diffusers_0240_env"

RUN_DIR="/home/ids/xchen-21/FADING2"
RUN_SCRIPT="run_many.py"

# echo des commandes lancees
set -x

# chargement des modules
source "$PIP_ENV/bin/activate"

# ---- Run it...
cd "$RUN_DIR"
srun python "$RUN_SCRIPT"

