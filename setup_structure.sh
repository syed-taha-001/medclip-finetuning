#!/bin/bash
# Scaffolds the medclip-project folder structure.
# Run from inside the medclip-project root.
#
# Structure groups runs by dataset split (e.g. split_v1) so that
# runs sharing the same data (e.g. naive vs. lambda-constrained
# fine-tuning) are easy to compare side by side, and pre-finetune
# embeddings are computed once per split rather than once per run.
#
# This script only creates the STATIC parts of the structure
# (things that exist before any run happens). Per-run folders
# (checkpoints/<split>/run_00X/, embeddings/<split>/run_00X/,
# results/<split>/run_00X/) are created dynamically by
# finetune_medclip.py when a run is started.

set -e

# Top-level files
touch README.md
touch runs_log.csv

# data
mkdir -p data/chest_xray

# scripts
mkdir -p scripts
touch scripts/metrics.py
touch scripts/extract_embeddings.py
touch scripts/finetune_medclip.py
touch scripts/compare_embeddings.py

# checkpoints: base model only (split_v1/ etc. created per-split when needed)
mkdir -p checkpoints/base
touch checkpoints/base/.gitkeep

# embeddings and results: left empty at top level, split_v1/ etc.
# get created per-split when a run needs them
mkdir -p embeddings
touch embeddings/.gitkeep

mkdir -p results
touch results/.gitkeep

echo "Structure created/verified."
find . -not -path './.git*' | sort
