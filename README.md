# AdaGAD-HNC

**Ada**ptive **G**raph **A**nomaly **D**etection with **H**ard-**N**egative **C**urriculum

<p align="center">
  <strong>Production-grade research codebase for graph anomaly detection</strong>
</p>

---

## Project Overview

AdaGAD-HNC is a novel self-supervised graph anomaly detection framework that jointly optimizes three complementary anomaly signals—context-level contrastive, patch-level contrastive, and reconstruction error—via an **Adaptive Multi-Objective Fusion** module with an entropy regularization anti-collapse mechanism. Training is stabilized by a staged **Hard-Negative Curriculum** that progressively mines harder negatives.

### Key Method Components

| Component | Description |
|-----------|-------------|
| **Multi-View Graph Sampler** | Stochastic subgraph sampling via random walk, k-hop, and attribute-aware samplers |
| **Context Contrastive** | Node-level contrastive across subgraph contexts |
| **Patch Contrastive** | Subgraph-patch level representation alignment |
| **Reconstruction Error** | Decoder-based reconstruction anomaly signal |
| **Adaptive Fusion** | Lightweight head predicts per-objective weights via softmax with entropy regularizer |
| **Hard-Negative Curriculum** | Staged curriculum: Easy → Semi-hard → Hard negative mining |

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Or with Poetry:

```bash
poetry install
poetry shell
```

---

## Dataset Preparation

Supported datasets: Cora, CiteSeer, PubMed, ACM, BlogCatalog, Flickr.

Some datasets auto-download via PyTorch Geometric; for others place raw files under `data/raw/`. Run:

```bash
python scripts/prepare_data.py --dataset citeseer
python scripts/prepare_data.py --dataset flickr
```

The script downloads, validates shapes/NaNs/Inf/feature dims, generates reproducible splits, and exports `dataset_summary.json`.

---

## Quickstart

### Training

```bash
# Single training run
python -m src.main train model=adagad_hnc data=citeseer train=curriculum

# Multi-seed evaluation
python -m src.experiments.run_multiseed --config configs/default.yaml --seeds 1 2 3 4 5
```

### Evaluation

```bash
python -m src.main eval --ckpt outputs/<run>/checkpoints/best.pt
```

### Ablations

```bash
python -m src.experiments.ablations --config configs/default.yaml --suite fusion_curriculum
```

---

## CLI Examples

```bash
# Curriculum training on CiteSeer
python -m src.main train model=adagad_hnc data=citeseer train=curriculum

# Baseline config
python -m src.main train model=baselines data=citeseer data.dataset=citeseer

# Ablation suite
python -m src.experiments.ablations --suite fusion_curriculum

# Generate paper tables
python scripts/export_results.py --input outputs/*/metrics.csv --output outputs/results_table.md
```

---

## Expected Outputs

```
outputs/
  <run_id>/
    config.yaml
    logs/
    metrics.csv
    plots/
      training_curves.png
      ablation_comparison.png
    checkpoints/
      best.pt
      last.pt
```

---

## Reproducibility Checklist

- [ ] Python 3.11+
- [ ] `pip install -r requirements.txt` inside `.venv`
- [ ] Set `SEED` in config (default: 42)
- [ ] Use deterministic PyTorch settings (`torch.use_deterministic_algorithms(True)`)
- [ ] Use same CUDA/cuDNN versions across runs
- [ ] Log full config + metrics + checkpoint every run

---

## Project Structure

```
src/
  main.py            # Entry point
  cli.py             # Hydra/argparse CLI
  utils/             # Seed, logger, IO, metrics, registry
  data/              # Dataset loaders, transforms, splits, validation
  model/             # Encoders, readout, discriminators, fusion, AdaGAD-HNC backbone
  losses/            # Contrastive, reconstruction, total loss, negative mining
  train/             # Engine, hooks, checkpoint, scheduler
  eval/              # Evaluator, calibration, robustness
  experiments/       # Single/multi-seed runs, ablations, reporting
tests/
scripts/
configs/
outputs/
```

---

## Quality Gates

```bash
make format    # black + isort
make lint      # ruff
make typecheck # mypy
make test      # pytest -q
make smoke     # 1-epoch smoke train on CiteSeer
```

---

## Citation

```bibtex
@article{adagadhnc2024,
  title   = {AdaGAD-HNC: Adaptive Graph Anomaly Detection with Hard-Negative Curriculum},
  author  = {AdaGAD-HNC Authors},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2024}
}
```
