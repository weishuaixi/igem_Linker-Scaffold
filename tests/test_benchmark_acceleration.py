import csv
import itertools
import json
import random

import pytest
import yaml

import benchmark_scaffolds as benchmark
from rna_scaffold.evaluation import (
    TrainingSimilarityIndex,
    kmer_jaccard,
    normalized_edit_distance,
    within_group_diversity,
)


def reference_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = row
    return previous[-1] / max(len(a), len(b), 1)


def reference_jaccard(a, b, k=5):
    left = {a[i : i + k] for i in range(max(0, len(a) - k + 1))}
    right = {b[i : i + k] for i in range(max(0, len(b) - k + 1))}
    return len(left & right) / len(left | right) if left or right else float(a == b)


def test_bit_vectors_match_reference_exhaustive_and_multiword():
    strings = ["".join(chars) for n in range(5) for chars in itertools.product("AU", repeat=n)]
    for a, b in itertools.product(strings, repeat=2):
        assert normalized_edit_distance(a, b) == reference_distance(a, b)
    rng = random.Random(81)
    for _ in range(160):
        a = "".join(rng.choices("AUCGN", k=rng.choice([0, 1, 31, 63, 64, 65, 100, 129, 256])))
        b = "".join(rng.choices("AUCGN", k=rng.randrange(280)))
        assert normalized_edit_distance(a, b) == reference_distance(a, b)
        assert normalized_edit_distance(b, a) == reference_distance(a, b)


@pytest.mark.parametrize("k", [1, 3, 5, 8])
def test_similarity_index_exact_with_unseen_kmers_and_short_sequences(k):
    rng = random.Random(9)
    references = ["", "A", "AA", "AAAAAA"] + ["".join(rng.choices("AUCG", k=50)) for _ in range(40)]
    index = TrainingSimilarityIndex(references, k)
    queries = references + ["", "N", "NNNNNNNNN", "XYZXYZXYZ"]
    queries += ["".join(rng.choices("AUCGN", k=50)) for _ in range(40)]
    for query in queries:
        expected = max(reference_jaccard(query, ref, k) for ref in references)
        assert index.nearest(query) == expected
        for ref in references:
            assert kmer_jaccard(query, ref, k) == reference_jaccard(query, ref, k)
    assert TrainingSimilarityIndex([], k).nearest("AUGC") is None


def test_pairwise_diversity_preserves_duplicates_and_peer_order():
    sequences = ["", "A", "GCGG", "GCGG", "AAAUUUGCGGCCCUUU", "CCCGGGAAAUUU"]
    edits, kmers = within_group_diversity(sequences)
    for i, sequence in enumerate(sequences):
        peers = sequences[:i] + sequences[i + 1 :]
        assert edits[i] == sum(reference_distance(sequence, peer) for peer in peers) / len(peers)
        assert kmers[i] == sum(1 - reference_jaccard(sequence, peer) for peer in peers) / len(peers)
    assert within_group_diversity([]) == ([], [])
    assert within_group_diversity(["A"]) == ([0.0], [0.0])


def fixture_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "fixture_mode": True,
                "seed": 7,
                "candidate_count": 4,
                "bootstrap_samples": 10,
                "motifs": [{"id": "first", "sequence": "GCGG"}, {"id": "second", "sequence": "AUCG"}],
                "methods": [{"name": "uniform", "kind": "uniform"}],
            }
        )
    )
    return path


def test_resume_reuses_completed_motif_and_matches_clean_run(tmp_path, monkeypatch):
    config = fixture_config(tmp_path)
    output = tmp_path / "interrupted"
    original = benchmark._generate_method_motif
    calls = []

    def interrupt(*args, **kwargs):
        calls.append(args[1])
        if len(calls) == 2:
            raise KeyboardInterrupt()
        return original(*args, **kwargs)

    monkeypatch.setattr(benchmark, "_generate_method_motif", interrupt)
    with pytest.raises(KeyboardInterrupt):
        benchmark.run_benchmark(config, output)
    assert not output.exists()
    assert len(list((tmp_path / "interrupted.progress").glob("*.json"))) == 2
    with pytest.raises(ValueError, match="use --resume"):
        benchmark.run_benchmark(config, output)
    calls.clear()

    def record(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(benchmark, "_generate_method_motif", record)
    manifest = benchmark.run_benchmark(config, output, resume=True)
    assert manifest["resumed_motifs"] == 1
    assert calls == ["AUCG"]
    clean = tmp_path / "clean"
    benchmark.run_benchmark(config, clean)

    def metrics(path):
        with (path / "candidate_results.csv").open() as handle:
            return [{k: v for k, v in row.items() if "runtime" not in k} for row in csv.DictReader(handle)]

    assert metrics(output) == metrics(clean)
    assert json.loads((output / "paired_bootstrap.json").read_text()) == json.loads(
        (clean / "paired_bootstrap.json").read_text()
    )


def test_resume_rejects_changed_config_and_damaged_progress(tmp_path, monkeypatch):
    config = fixture_config(tmp_path)
    output = tmp_path / "interrupted"
    original = benchmark._publish_benchmark_run

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(benchmark, "_publish_benchmark_run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        benchmark.run_benchmark(config, output)
    monkeypatch.setattr(benchmark, "_publish_benchmark_run", original)
    text = config.read_text()
    config.write_text(text.replace("seed: 7", "seed: 8"))
    with pytest.raises(ValueError, match="mismatch"):
        benchmark.run_benchmark(config, output, resume=True)
    config.write_text(text)
    cache = next(p for p in (tmp_path / "interrupted.progress").glob("*.json") if p.name != "binding.json")
    data = json.loads(cache.read_text())
    data["payload"]["motif_id"] = "corrupted"
    cache.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum mismatch"):
        benchmark.run_benchmark(config, output, resume=True)
