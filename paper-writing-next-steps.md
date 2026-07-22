# Paper-Writing Next Steps

## 1. Novelty Claims to Substantiate
1. **Adaptive Multi-Objective Fusion**  
   - Claim: Replacing hand-set weights with a learned entropy-regularized fusion head improves robustness to dataset-specific anomaly distributions.  
   - Evidence needed: Table 2 (fixed vs adaptive fusion) + ablation on entropy regularization weight.

2. **Hard-Negative Curriculum**  
   - Claim: Staged negative mining (easy → semi-hard → hard) stabilizes contrastive learning for anomaly detection.  
   - Evidence needed: Table 3 (curriculum ablations) + convergence curves.

3. **Triple-Signal Fusion**  
   - Claim: Combining context contrastive, patch contrastive, and reconstruction error yields complementary anomaly signals.  
   - Evidence needed: Table 4 (branch removal ablations).

## 2. Experiment Section Checklist
- [ ] **Datasets** — Cora, CiteSeer, PubMed with 10 random seeds; ACM, BlogCatalog, Flickr if feasible.
- [ ] **Baselines** — CoLA, SL-GAD, Recon-only, Contrastive-only, Fixed-fusion wrappers under identical splits.
- [ ] **Metrics** — ROC-AUC, AUPRC, Recall@K, F1; mean std across seeds.
- [ ] **Runtime & Memory** — Peak GPU/CPU memory and wall-clock time per epoch.
- [ ] **Statistical Tests** — Paired t-test or Wilcoxon signed-rank test against each baseline.
- [ ] **Robustness** — Feature noise (σ=0.1), edge flip (5%), missing feature (20%).
- [ ] **Calibration** — Brier score, reliability diagrams, temperature scaling.
- [ ] **Visualizations** — t-SNE of learned embeddings, score distribution histograms, confusion matrices.

## 3. Ablation Narrative Template

### 3.1 Fusion Mechanism
"We compare AdaGAD-HNC against a fixed-weight variant in which contextual, patch-wise, and reconstruction scores are merged with hand-tuned coefficients (α=0.5, β=0.3, γ=0.2). Table 2 shows that adaptive fusion consistently improves ROC-AUC by X% across datasets, demonstrating that the model learns to emphasize the most reliable signal per dataset topology."

### 3.2 Curriculum Schedule
"We ablate the curriculum by replacing staged hard-negative mining with (i) random negatives throughout training, (ii) hard-only negatives from epoch 1, and (iii) our proposed schedule. Random negatives destabilize training, while hard-only from the start causes collapse (Table 3). Staged curriculum achieves the best trade-off between gradient richness and convergence stability."

### 3.3 Branch Ablations
"Removing any single branch degrades performance (Table 4). Notably, the reconstruction branch is the most critical on dense feature graphs (PubMed), whereas the patch branch contributes most on attribute-sparse social networks (BlogCatalog). This validates our multi-objective design."

### 3.4 Negative Ratio Sensitivity
"Negative ratio controls the hardness of the contrastive task. We sweep {1, 4, 8, 16} and observe peak performance at 8 negatives per anchor (Section 4.4). Higher ratios marginally improve hard-mining recall but increase memory overhead linearly."

## 4. Output Tables to Generate
- `results_table.md`, `results_table.tex` — main result table (Cora / CiteSeer / PubMed)
- `ablation_fusion_curriculum.csv` — Table 2
- `ablation_curriculum_only.csv` — Table 3
- `ablation_branch_ablation.csv` — Table 4
- `ablation_negative_ratio.csv` — Table 5
- `ablation_sampler.csv` — Table 6

## 5. Plotting Checklist
- Training curves (train vs val loss, val AUC vs epoch) — saved per run to `outputs/<run>/plots/`
- Ablation comparison bar charts
- t-SNE / UMAP of embeddings colored by anomaly status
- Score distribution plots (normal vs anomalous node histograms)

## 6. Reproducibility Checklist for Paper
- [ ] Seed set for python / numpy / torch / CUDA
- [ ] Exact config YAMLs included in appendix
- [ ] Hash of committed code + environment YAML
- [ ] Split indices released (for fair baseline comparison)
- [ ] Checkpoint + metrics CSV for the claimed result
