import logging
from typing import Any

import torch
from torch import nn

LOGGER = logging.getLogger("AdaGAD-HNC")


class CurriculumNegativeMiner(nn.Module):
    def __init__(self, memory_bank_size: int = 4096, temperature: float = 0.2):
        super().__init__()
        self.memory_bank_size = memory_bank_size
        self.temperature = temperature
        self.register_buffer("memory_bank", torch.empty(0, dtype=torch.float32))
        self.register_buffer("memory_ptr", torch.tensor(0))

    @torch.no_grad()
    def update_memory_bank(self, embeddings: torch.Tensor) -> None:
        embeddings = embeddings.detach()
        if self.memory_bank.size(0) == 0:  # type: ignore
            self.memory_bank = embeddings[: self.memory_bank_size]  # type: ignore
        else:
            ptr = int(self.memory_ptr.item())  # type: ignore
            end = (ptr + embeddings.size(0)) % self.memory_bank_size
            if ptr > end:
                self.memory_bank[ptr:] = embeddings[: self.memory_bank_size - ptr]  # type: ignore
                self.memory_bank[:end] = embeddings[self.memory_bank_size - ptr :]  # type: ignore
            else:
                self.memory_bank[ptr:end] = embeddings  # type: ignore
            self.memory_ptr[0] = end  # type: ignore

    def sample(
        self, anchor: torch.Tensor, epoch: int, config: dict[str, Any]
    ) -> tuple[torch.Tensor, str]:
        warmup_easy = config.get("warmup_easy_epochs", 20)
        transition = warmup_easy + config.get("transition_epochs", 30)
        neg_ratio = config.get("negative_ratio", 8)

        num_nodes = anchor.size(0)
        device = anchor.device

        if epoch < warmup_easy:
            phase = "easy"
            neg_idx = torch.randint(0, num_nodes, (num_nodes * neg_ratio,), device=device)
            offset = torch.arange(num_nodes, device=device).repeat_interleave(neg_ratio)
            neg_idx = torch.remainder(neg_idx + offset + 1, num_nodes)
            neg_embeddings = anchor[neg_idx]
        elif epoch < transition:
            phase = "semi_hard"
            sim = torch.mm(anchor, anchor.t())
            mask = torch.eye(num_nodes, device=device).bool()
            sim.masked_fill_(mask, -1e9)
            _, topk = sim.topk(k=neg_ratio, dim=-1)
            neg_idx = topk.T.reshape(-1)
            neg_embeddings = anchor[neg_idx]
        else:
            phase = "hard"
            bank = self.memory_bank.to(device)
            if bank.size(0) == 0:
                return anchor.repeat_interleave(neg_ratio, dim=0), "random"
            sim = torch.mm(anchor, bank.t()) / self.temperature
            _, hard_idx = sim.topk(k=neg_ratio, dim=-1)
            neg_embeddings = bank[hard_idx.T.reshape(-1)]

        return neg_embeddings, phase
