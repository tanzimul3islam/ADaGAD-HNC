#!/usr/bin/env bash
set -euo pipefail

echo "=== AdaGAD-HNC Ablation Suites ==="

python -m src.experiments.ablations --suite fusion_curriculum
python -m src.experiments.ablations --suite curriculum_only
python -m src.experiments.ablations --suite branch_ablation
python -m src.experiments.ablations --suite sampler
python -m src.experiments.ablations --suite negative_ratio
python -m src.experiments.ablations --suite subgraph_size

python scripts/export_results.py --input outputs/ablations/*.csv --output outputs/all_ablations

echo "All ablations complete."
