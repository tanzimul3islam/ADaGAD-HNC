import torch
from src.losses.negative_mining import CurriculumNegativeMiner


def test_easy_mining():
    miner = CurriculumNegativeMiner(memory_bank_size=128, temperature=0.2)
    anchor = torch.randn(32, 16)
    neg, phase = miner.sample(anchor, epoch=5, config={"negative_ratio": 4})
    assert neg.shape[0] == 128
    assert phase == "easy"


def test_semi_hard_mining():
    miner = CurriculumNegativeMiner(memory_bank_size=128, temperature=0.2)
    anchor = torch.randn(32, 16)
    neg, phase = miner.sample(
        anchor,
        epoch=25,
        config={"warmup_easy_epochs": 20, "transition_epochs": 10, "negative_ratio": 4},
    )
    assert neg.shape[0] == 128
    assert phase == "semi_hard"


def test_hard_mining():
    miner = CurriculumNegativeMiner(memory_bank_size=128, temperature=0.2)
    miner.update_memory_bank(torch.randn(128, 16))
    anchor = torch.randn(32, 16)
    neg, phase = miner.sample(
        anchor,
        epoch=50,
        config={"warmup_easy_epochs": 20, "transition_epochs": 10, "negative_ratio": 4},
    )
    assert neg.shape[0] == 128
    assert phase == "hard"
