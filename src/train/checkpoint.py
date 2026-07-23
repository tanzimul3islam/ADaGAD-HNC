import logging
from pathlib import Path

import torch

LOGGER = logging.getLogger("AdaGAD-HNC")


class CheckpointManager:
    def __init__(self, ckpt_dir: str, max_to_keep: int = 2):
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self.best_score = -float("inf")

    def save(self, model, optimizer, epoch: int, score: float, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "score": score,
        }
        path = self.ckpt_dir / f"epoch_{epoch:04d}.pt"
        torch.save(state, path)
        torch.save(state, self.ckpt_dir / "last.pt")
        LOGGER.info(f"Saved checkpoint to {path}")

        if is_best:
            best_path = self.ckpt_dir / "best.pt"
            torch.save(state, best_path)
            self.best_score = score
            LOGGER.info(f"Updated best checkpoint (score={score:.4f})")

        self._cleanup()

    def load(self, path: Path | str | None = None, map_location: str = "cpu"):
        path = self.ckpt_dir / "last.pt" if path is None else Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint {path} not found.")
        state = torch.load(path, map_location=map_location, weights_only=False)
        LOGGER.info(f"Loaded checkpoint from {path}")
        return state

    def _cleanup(self):
        files = sorted(self.ckpt_dir.glob("epoch_*.pt"))
        if len(files) > self.max_to_keep:
            for f in files[: len(files) - self.max_to_keep]:
                f.unlink()
