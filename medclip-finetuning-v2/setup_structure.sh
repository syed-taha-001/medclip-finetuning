#!/bin/bash
# setup_structure.sh
# Scaffolds the full medclip-finetuning project folder structure.
# Safe to re-run: uses mkdir -p and touch, never overwrites existing file content.

set -e

echo "Setting up medclip-finetuning project structure..."

# --- Root-level files ---
touch README.md
touch environment.yml
touch runs_log.csv

# --- configs/ ---
mkdir -p configs
touch configs/lambda_sweep.yaml

# --- data/ ---
mkdir -p data/raw/kermany/NORMAL
mkdir -p data/raw/kermany/PNEUMONIA
touch data/raw/kermany/NORMAL/.gitkeep
touch data/raw/kermany/PNEUMONIA/.gitkeep

for exp in exp_1 exp_2 exp_3 exp_4; do
    mkdir -p data/experiments/$exp
    touch data/experiments/$exp/.gitkeep
done

# --- scripts/ ---
mkdir -p scripts
touch scripts/create_splits.py
touch scripts/losses.py
touch scripts/metrics.py
touch scripts/extract_embeddings.py
touch scripts/finetune_medclip.py
touch scripts/compare_embeddings.py
touch scripts/analyze_sweep.py
touch scripts/compare_experiments.py
touch scripts/run_sweep.sh
chmod +x scripts/run_sweep.sh

# --- tests/ ---
mkdir -p tests
touch tests/test_losses_vs_metrics.py
touch tests/test_nbr_empty_pairs.py
touch tests/test_gradient_flow.py

# --- checkpoints/ ---
mkdir -p checkpoints/base
touch checkpoints/base/.gitkeep

for exp in exp_1 exp_2 exp_3 exp_4; do
    mkdir -p checkpoints/$exp
    touch checkpoints/$exp/.gitkeep
done

# --- embeddings/ ---
for exp in exp_1 exp_2 exp_3 exp_4; do
    mkdir -p embeddings/$exp
    touch embeddings/$exp/.gitkeep
done

# --- results/ ---
for exp in exp_1 exp_2 exp_3 exp_4; do
    mkdir -p results/$exp
    touch results/$exp/.gitkeep
done
mkdir -p results/cross_experiment
touch results/cross_experiment/.gitkeep

echo "Project structure created successfully."
