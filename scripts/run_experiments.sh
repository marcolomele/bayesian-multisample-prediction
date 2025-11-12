#!/bin/bash
#SBATCH --job-name=bayesian_experiments
#SBATCH --account=3176145
#SBATCH --partition=dsba
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err

# Load conda module
module load modules/miniconda3
eval "$(conda shell.bash hook)"
conda activate bayesian_env

# Navigate to scripts directory
cd ~/bayesian_multisample/scripts

# Run all three experiments sequentially
echo "=========================================="
echo "Starting experiment 1: news..."
echo "=========================================="
python experiment.py --config config_news.json

echo ""
echo "=========================================="
echo "Starting experiment 2: names..."
echo "=========================================="
python experiment.py --config config_names.json

echo ""
echo "=========================================="
echo "Starting experiment 3: wilderness..."
echo "=========================================="
python experiment.py --config config_wilderness.json

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="

# Deactivate environment
conda deactivate
module unload modules/miniconda3

