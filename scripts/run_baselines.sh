#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs

echo "=== AdaGAD-HNC Baselines ==="

echo "Running contrastive-only on Amazon..."
python -m src.main train model=model=contrastive_only data=amazon train=standard max_epochs=50

echo "Running reconstruction-only on Amazon..."
python -m src.main train model=model=reconstruction_only data=amazon train=standard max_epochs=50

echo "Running fixed-fusion on Amazon..."
python -m src.main train model=model=fixed_fusion data=amazon train=standard max_epochs=50

echo "All baselines complete."
