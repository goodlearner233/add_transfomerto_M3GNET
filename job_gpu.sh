#!/bin/bash
#SBATCH --job-name=gpu
#SBATCH --output=gpu.out
#SBATCH --error=gpu.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=200G
#SBATCH --time=10-00:00:00
#SBATCH --qos=batch-long
module purge

eval "$(conda shell.bash hook)"
source .venv/bin/activate

python3 train_transformer_atomic_sum_full_matpes.py \
    --data datasets/huggingface_cache/datasets--materialyze--matpes/snapshots/47d2cc020cf913b5a48a3480136a128dddc0a92c/MatPES-PBE-2025.2.json \
    --graph-cache-dir datasets/matpes_pbe_2025_2_full \
    --output-dir runs/full_matpes_transformer \
    --accelerator gpu \
    --devices 1