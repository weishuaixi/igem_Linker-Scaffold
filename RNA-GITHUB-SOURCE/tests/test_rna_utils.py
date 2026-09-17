import pytest

from rna_scaffold.tokenizer import RnaTokenizer
from rna_scaffold.utils import (
    complementarity_rate,
    reverse_complement,
    validate_rna_sequence,
)


def test_reverse_complement_uses_rna_watson_crick_pairs():
    assert reverse_complement("AUGC") == "GCAU"


def test_validate_rna_sequence_rejects_non_rna_bases():
    assert validate_rna_sequence("AUGC")
    assert not validate_rna_sequence("ATGC")
    assert not validate_rna_sequence("AUGCX")


def test_complementarity_rate_scores_reverse_oriented_stems():
    left = "AUGCAUGCAU"
    right = reverse_complement(left)

    assert complementarity_rate(left, right) == pytest.approx(1.0)


def test_complementarity_rate_penalizes_mismatches():
    left = "AUGCAUGCAU"
    right = "AAAAAAAAAA"

    assert complementarity_rate(left, right) < 0.9


def test_tokenizer_round_trip_preserves_special_tokens_and_bases():
    tokenizer = RnaTokenizer()
    text = "<LEFT>AUGC<END_LEFT><RIGHT>GCAU<END_RIGHT>"

    token_ids = tokenizer.encode(text)

    assert tokenizer.decode(token_ids) == text
