import logging
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger("AdaGAD-HNC")


def run_ablation_suite(config: dict[str, Any], suite: str) -> pd.DataFrame:
    LOGGER.info(f"Running ablation suite: {suite}")
    suites: dict[str, list[dict[str, Any]]] = {
        "fusion_curriculum": [
            {"model": "adagad_hnc", "fusion": "adaptive", "curriculum": True},
            {"model": "adagad_hnc", "fusion": "fixed", "curriculum": True},
            {"model": "adagad_hnc", "fusion": "adaptive", "curriculum": False},
            {"model": "adagad_hnc", "fusion": "fixed", "curriculum": False},
        ],
        "curriculum_only": [
            {"model": "adagad_hnc", "curriculum": True},
            {"model": "adagad_hnc", "curriculum": False, "easy_only": True},
            {"model": "adagad_hnc", "curriculum": False, "hard_only": True},
            {"model": "adagad_hnc", "curriculum": False, "random_only": True},
        ],
        "branch_ablation": [
            {
                "model": "adagad_hnc",
                "remove_context": False,
                "remove_patch": False,
                "remove_recon": False,
            },
            {
                "model": "adagad_hnc",
                "remove_context": True,
                "remove_patch": False,
                "remove_recon": False,
            },
            {
                "model": "adagad_hnc",
                "remove_context": False,
                "remove_patch": True,
                "remove_recon": False,
            },
            {
                "model": "adagad_hnc",
                "remove_context": False,
                "remove_patch": False,
                "remove_recon": True,
            },
            {
                "model": "adagad_hnc",
                "remove_context": True,
                "remove_patch": True,
                "remove_recon": False,
            },
        ],
        "sampler": [
            {"model": "adagad_hnc", "context_sampler": "khop", "patch_sampler": "random_walk"},
            {
                "model": "adagad_hnc",
                "context_sampler": "random_walk",
                "patch_sampler": "random_walk",
            },
            {"model": "adagad_hnc", "context_sampler": "khop", "patch_sampler": "khop"},
        ],
        "negative_ratio": [
            {"model": "adagad_hnc", "negative_ratio": 1},
            {"model": "adagad_hnc", "negative_ratio": 4},
            {"model": "adagad_hnc", "negative_ratio": 8},
            {"model": "adagad_hnc", "negative_ratio": 16},
        ],
        "subgraph_size": [
            {"model": "adagad_hnc", "context_k": 1, "patch_walk_length": 2},
            {"model": "adagad_hnc", "context_k": 3, "patch_walk_length": 4},
            {"model": "adagad_hnc", "context_k": 5, "patch_walk_length": 6},
        ],
    }
    if suite not in suites:
        raise ValueError(f"Unknown ablation suite: {suite}. Available: {list(suites.keys())}")

    rows = []
    for variant in suites[suite]:
        cfg = dict(config)
        cfg.update(variant)
        try:
            from src.experiments.run_single import run_single

            best_auc, hist = run_single(cfg, ckpt_dir=f"outputs/ablations/{suite}")
            final_metrics = hist[-1] if hist else {}
            rows.append({"variant": str(variant), **final_metrics})
        except Exception as e:
            LOGGER.error(f"Ablation {variant} failed: {e}")
            rows.append({"variant": str(variant), "error": str(e)})

    df = pd.DataFrame(rows)
    out_path = Path("outputs") / f"ablation_{suite}.csv"
    df.to_csv(out_path, index=False)
    LOGGER.info(f"Ablation results saved to {out_path}")
    return df
