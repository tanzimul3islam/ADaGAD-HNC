import logging
from pathlib import Path
from typing import Any

from src.eval.evaluator import summarize_multi_seed
from src.experiments.run_single import run_single
from src.utils.io import save_json

LOGGER = logging.getLogger("AdaGAD-HNC")


def run_multiseed(config_path: str, seeds: list[int]) -> dict[str, Any]:
    import yaml

    with open(config_path) as f:
        base_config: dict[str, Any] = yaml.safe_load(f)

    results: list[dict[str, Any]] = []
    for seed in seeds:
        LOGGER.info(f" Running seed {seed}")
        base_config["seed"] = seed
        best_auc, history = run_single(base_config, ckpt_dir=f"outputs/seed_{seed}")
        eval_out = {"seed": seed, "best_auc": best_auc, "history": history[-1:]}
        results.append(eval_out)

    summary = summarize_multi_seed(
        [r["history"][-1] for r in results], out_path=Path("outputs") / "multi_seed_summary.json"
    )
    LOGGER.info(f"Multi-seed summary: {summary}")
    save_json(results, Path("outputs") / "multi_seed_results.json")
    return summary
