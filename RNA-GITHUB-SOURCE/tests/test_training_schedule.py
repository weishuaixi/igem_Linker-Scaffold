import pytest

from rna_scaffold.lightning_module import warmup_cosine_multiplier


def test_warmup_cosine_multiplier_reaches_floor():
    assert warmup_cosine_multiplier(
        0, total_steps=100, warmup_fraction=0.05, min_fraction=0.02
    ) == 0.0
    assert warmup_cosine_multiplier(
        5, total_steps=100, warmup_fraction=0.05, min_fraction=0.02
    ) == pytest.approx(1.0)
    assert warmup_cosine_multiplier(
        100, total_steps=100, warmup_fraction=0.05, min_fraction=0.02
    ) == pytest.approx(0.02)
