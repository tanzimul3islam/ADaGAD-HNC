#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs

echo "=== AdaGAD-HNC Baselines ==="

echo "Running contrastive-only on Cora..."
python -m src.main train model=model=contrastive_only data=cora train=standard max_epochs=50

echo "Running reconstruction-only on Cora..."
python -m src.main train model=model=reconstruction_only data=cora train=standard max_epochs=50

echo "Running fixed-fusion on Cora..."
python -m src.main train model=model=fixed_fusion data=cora train=standard max_epochs=50

echo "All baselines complete."
