import pytest
import torch
from torch.utils.data import WeightedRandomSampler

from rna_scaffold.data import RnaMotifDenoisingDataset, build_partitioned_denoising_datasets
from rna_scaffold.datamodule import RnaScaffoldDataModule
from rna_scaffold.records import RnaSequenceRecord
from rna_scaffold.tokenizer import RnaTokenizer


def test_denoising_dataset_builds_joint_flank_canvas_and_preserves_motif():
    tokenizer = RnaTokenizer()
    record = RnaSequenceRecord("x", "AAAAGCGGUUUU", "RF1", "unit")
    dataset = RnaMotifDenoisingDataset(
        records=[record],
        tokenizer=tokenizer,
        max_length=32,
        min_motif_length=4,
        max_motif_length=6,
        motif_length_buckets=None,
        min_flank_length=1,
        min_total_scaffold_length=2,
        seed=9,
    )

    item = dataset[0]
    motif_positions = item["fixed_mask"].nonzero().flatten()

    assert item["input_ids"].shape == (32,)
    assert item["attention_mask"].sum().item() == len(record.sequence)
    assert 4 <= motif_positions.numel() <= 6
    assert item["input_ids"][motif_positions].tolist() == (item["target_token_ids"][motif_positions].tolist())
    assert item["target_base_ids"][:4].tolist() == [0, 0, 0, 0]
    assert item["target_base_ids"][len(record.sequence) :].eq(-100).all()


def test_denoising_dataset_derives_bilateral_targets_without_v1_target_keys():
    dataset = RnaMotifDenoisingDataset(
        records=[RnaSequenceRecord("x", "AACCGGUU", "RF1", "unit")],
        tokenizer=RnaTokenizer(),
        max_length=8,
        min_motif_length=4,
        max_motif_length=4,
        motif_length_buckets=None,
        min_flank_length=2,
        min_total_scaffold_length=0,
        preferred_total_scaffold_length=0,
        seed=9,
    )

    item = dataset[0]

    assert item["target_left_length"].item() == 2
    assert item["target_right_length"].item() == 2
    assert "target_length" not in item
    assert "motif_start" not in item


def test_long_rna_is_retained_and_cropped_reproducibly_per_epoch():
    tokenizer = RnaTokenizer()
    record = RnaSequenceRecord("long", "A" * 300 + "C" * 300, "RF1", "unit")
    first = RnaMotifDenoisingDataset(
        records=[record],
        tokenizer=tokenizer,
        max_length=512,
        min_motif_length=4,
        max_motif_length=None,
        motif_length_buckets=None,
        min_flank_length=2,
        min_total_scaffold_length=8,
        preferred_total_scaffold_length=24,
        seed=17,
    )
    second = RnaMotifDenoisingDataset(
        records=[record],
        tokenizer=tokenizer,
        max_length=512,
        min_motif_length=4,
        max_motif_length=None,
        motif_length_buckets=None,
        min_flank_length=2,
        min_total_scaffold_length=8,
        preferred_total_scaffold_length=24,
        seed=17,
    )

    first_item = first[0]
    second_item = second[0]
    assert len(first) == 1
    assert first_item["attention_mask"].sum().item() == 512
    assert first_item["source_start"].item() == second_item["source_start"].item()

    first.set_epoch(1)
    assert first[0]["source_start"].item() != first_item["source_start"].item()


def test_denoising_corruption_masks_only_prediction_positions():
    tokenizer = RnaTokenizer()
    record = RnaSequenceRecord("x", "AUGC" * 20, "RF1", "unit")
    dataset = RnaMotifDenoisingDataset(
        [record],
        tokenizer,
        max_length=80,
        min_motif_length=8,
        max_motif_length=8,
        full_mask_probability=0.0,
        span_mask_probability=0.0,
        min_random_mask_fraction=0.5,
        max_random_mask_fraction=0.5,
        seed=9,
    )
    item = dataset[0]
    scaffold = item["attention_mask"] & ~item["fixed_mask"]

    assert item["prediction_mask"].any()
    assert (scaffold & ~item["prediction_mask"]).any()
    assert torch.all(item["input_ids"][item["prediction_mask"]] == tokenizer.token_to_id[tokenizer.special.mask])
    assert torch.equal(
        item["input_ids"][item["fixed_mask"]],
        item["target_token_ids"][item["fixed_mask"]],
    )


def test_full_mask_corruption_matches_inference_canvas():
    tokenizer = RnaTokenizer()
    record = RnaSequenceRecord("x", "AUGC" * 10, "RF1", "unit")
    dataset = RnaMotifDenoisingDataset(
        [record],
        tokenizer,
        max_length=40,
        full_mask_probability=1.0,
        span_mask_probability=0.0,
        seed=3,
    )
    item = dataset[0]
    scaffold = item["attention_mask"] & ~item["fixed_mask"]

    assert torch.equal(item["prediction_mask"], scaffold)


def test_span_corruption_contains_adjacent_masked_positions():
    tokenizer = RnaTokenizer()
    record = RnaSequenceRecord("x", "AUGC" * 20, "RF1", "unit")
    dataset = RnaMotifDenoisingDataset(
        [record],
        tokenizer,
        max_length=80,
        full_mask_probability=0.0,
        span_mask_probability=1.0,
        mean_span_length=6,
        seed=5,
    )
    prediction = dataset[0]["prediction_mask"]

    assert (prediction[:-1] & prediction[1:]).any()


def test_task_aligned_windows_follow_asymmetric_linker_geometry():
    dataset = RnaMotifDenoisingDataset(
        [RnaSequenceRecord("long", "AUCG" * 100, "RF1", "unit")],
        RnaTokenizer(),
        max_length=100,
        min_motif_length=4,
        max_motif_length=8,
        motif_length_buckets=None,
        min_flank_length=10,
        min_total_scaffold_length=29,
        preferred_total_scaffold_length=29,
        short_flank_min=10,
        short_flank_max=20,
        long_flank_min=15,
        long_flank_max=30,
        full_mask_probability=0.0,
        span_mask_probability=0.0,
        iterative_mask_steps=16,
        context_noise_probability=0.05,
        seed=31,
    )

    orientations = set()
    for epoch in range(32):
        dataset.set_epoch(epoch)
        item = dataset[0]
        left = int(item["target_left_length"].item())
        right = int(item["target_right_length"].item())
        total = int(item["attention_mask"].sum().item())
        motif_length = int(item["fixed_mask"].sum().item())
        short_left = 10 <= left <= 20 and 15 <= right <= 30 and right > left
        short_right = 15 <= left <= 30 and 10 <= right <= 20 and left > right
        assert short_left or short_right
        assert total == left + motif_length + right
        assert total <= 100
        context_noise_mask = (
            item["attention_mask"]
            & ~item["fixed_mask"]
            & ~item["prediction_mask"]
            & item["input_ids"].ne(item["target_token_ids"])
        )
        assert not bool((context_noise_mask & item["fixed_mask"]).any().item())
        assert not bool((context_noise_mask & item["prediction_mask"]).any().item())
        orientations.add("left_short" if short_left else "right_short")

    assert orientations == {"left_short", "right_short"}


def test_iterative_mask_schedule_covers_low_and_high_noise_states():
    dataset = RnaMotifDenoisingDataset(
        [RnaSequenceRecord("x", "AUCG" * 25, "RF1", "unit")],
        RnaTokenizer(),
        max_length=100,
        min_motif_length=4,
        max_motif_length=4,
        motif_length_buckets=None,
        full_mask_probability=0.0,
        span_mask_probability=0.0,
        iterative_mask_steps=16,
        seed=7,
    )
    fractions = []
    for epoch in range(64):
        dataset.set_epoch(epoch)
        item = dataset[0]
        scaffold_count = int((item["attention_mask"] & ~item["fixed_mask"]).sum().item())
        fractions.append(int(item["prediction_mask"].sum().item()) / scaffold_count)

    assert min(fractions) <= 0.10
    assert max(fractions) == 1.0


def test_virtual_views_expand_examples_without_inflating_record_count():
    records = [
        RnaSequenceRecord("a", "A" * 40, "RF1", "unit"),
        RnaSequenceRecord("g", "G" * 40, "RF2", "unit"),
    ]
    dataset = RnaMotifDenoisingDataset(
        records,
        RnaTokenizer(),
        max_length=40,
        min_motif_length=4,
        max_motif_length=4,
        motif_length_buckets=None,
        views_per_record=3,
        seed=11,
    )

    assert dataset.record_count == 2
    assert len(dataset) == 6
    assert dataset[0]["target_base_ids"][0].item() == 0
    assert dataset[2]["target_base_ids"][0].item() == 0
    assert dataset[3]["target_base_ids"][0].item() == 3
    first_read = dataset[1]
    second_read = dataset[1]
    assert torch.equal(first_read["prediction_mask"], second_read["prediction_mask"])


def test_partition_manifest_counts_source_records_not_virtual_views():
    suffixes = ("AA", "AU", "AC", "AG", "UA", "UU", "UC", "UG")
    records = [
        RnaSequenceRecord(f"r{index}", "AUGC" * 9 + suffix, f"RF{index}", "unit")
        for index, suffix in enumerate(suffixes)
    ]
    cluster_by_id = {record.target_id: f"cluster_{record.target_id}" for record in records}
    datasets, manifest = build_partitioned_denoising_datasets(
        records,
        cluster_by_id,
        RnaTokenizer(),
        max_length=40,
        min_motif_length=4,
        max_motif_length=4,
        motif_length_buckets=None,
        train_views_per_record=2,
        validation_views_per_record=3,
        test_views_per_record=4,
        val_fraction=0.25,
        test_fraction=0.25,
        seed=19,
    )
    views = {"train": 2, "validation": 3, "test": 4}

    assert manifest.record_counts["total"] == len(records)
    assert sum(dataset.record_count for dataset in datasets.values()) == len(records)
    for partition, dataset in datasets.items():
        assert dataset.record_count == manifest.record_counts[partition]
        assert len(dataset) == dataset.record_count * views[partition]


def test_preferred_motif_sampling_uses_a_real_feasible_occurrence():
    tokenizer = RnaTokenizer()
    sequence = "A" * 40 + "GCGG" + "U" * 40
    dataset = RnaMotifDenoisingDataset(
        [RnaSequenceRecord("natural", sequence, "RF1", "unit")],
        tokenizer,
        max_length=40,
        min_motif_length=4,
        max_motif_length=4,
        motif_length_buckets=None,
        min_flank_length=10,
        min_total_scaffold_length=29,
        short_flank_min=10,
        short_flank_max=10,
        long_flank_min=15,
        long_flank_max=15,
        preferred_motifs=["GCGG"],
        preferred_motif_probability=1.0,
        seed=5,
    )

    item = dataset[0]
    length = int(item["attention_mask"].sum().item())
    source_start = int(item["source_start"].item())
    target_sequence = tokenizer.decode(item["target_token_ids"][:length].tolist())
    fixed_sequence = tokenizer.decode(item["target_token_ids"][item["fixed_mask"]].tolist())

    assert fixed_sequence == "GCGG"
    assert target_sequence == sequence[source_start : source_start + length]
    assert sorted((item["target_left_length"].item(), item["target_right_length"].item())) == [10, 15]


def test_missing_preferred_motif_falls_back_to_random_natural_window():
    dataset = RnaMotifDenoisingDataset(
        [RnaSequenceRecord("x", "AUCG" * 20, "RF1", "unit")],
        RnaTokenizer(),
        max_length=40,
        min_motif_length=4,
        max_motif_length=4,
        motif_length_buckets=None,
        preferred_motifs=["AAAAAA"],
        preferred_motif_probability=1.0,
        seed=13,
    )

    item = dataset[0]

    assert item["attention_mask"].any()
    assert item["fixed_mask"].sum().item() == 4


def test_iterative_full_mask_probability_and_noise_level():
    common = {
        "records": [RnaSequenceRecord("x", "AUCG" * 20, "RF1", "unit")],
        "tokenizer": RnaTokenizer(),
        "max_length": 80,
        "min_motif_length": 4,
        "max_motif_length": 4,
        "motif_length_buckets": None,
        "full_mask_probability": 0.0,
        "span_mask_probability": 0.0,
        "iterative_mask_steps": 4,
        "seed": 3,
    }
    full = RnaMotifDenoisingDataset(**common, iterative_full_mask_probability=1.0)[0]
    partial = RnaMotifDenoisingDataset(**common, iterative_full_mask_probability=0.0)[0]

    assert full["noise_level"].item() == 1.0
    assert 0 < partial["noise_level"].item() < 1


def test_stage_aware_context_noise_increases_with_mask_fraction_and_respects_boundaries():
    common = {
        "records": [RnaSequenceRecord("x", "AUCG" * 20, "RF1", "unit")],
        "tokenizer": RnaTokenizer(),
        "max_length": 80,
        "min_motif_length": 4,
        "max_motif_length": 4,
        "motif_length_buckets": None,
        "full_mask_probability": 0.0,
        "span_mask_probability": 0.0,
        "iterative_mask_steps": 4,
        "iterative_full_mask_probability": 0.0,
        "context_noise_probability": 0.9,
        "seed": 23,
    }
    constant = RnaMotifDenoisingDataset(**common, stage_aware_context_noise=False)[0]
    stage_aware_dataset = RnaMotifDenoisingDataset(**common, stage_aware_context_noise=True)
    stage_aware = stage_aware_dataset[0]
    full_options = {**common, "iterative_full_mask_probability": 1.0}
    full = RnaMotifDenoisingDataset(**full_options, stage_aware_context_noise=True)[0]

    assert stage_aware_dataset._context_noise_rate(0.0) == 0.0
    assert stage_aware_dataset._context_noise_rate(0.25) < stage_aware_dataset._context_noise_rate(0.75)
    assert stage_aware_dataset._context_noise_rate(1.0) == common["context_noise_probability"]
    assert torch.equal(constant["prediction_mask"], stage_aware["prediction_mask"])
    constant_noise = (
        constant["attention_mask"]
        & ~constant["fixed_mask"]
        & ~constant["prediction_mask"]
        & constant["input_ids"].ne(constant["target_token_ids"])
    )
    stage_aware_noise = (
        stage_aware["attention_mask"]
        & ~stage_aware["fixed_mask"]
        & ~stage_aware["prediction_mask"]
        & stage_aware["input_ids"].ne(stage_aware["target_token_ids"])
    )
    full_noise = (
        full["attention_mask"]
        & ~full["fixed_mask"]
        & ~full["prediction_mask"]
        & full["input_ids"].ne(full["target_token_ids"])
    )
    assert not bool((stage_aware_noise & ~constant_noise).any())
    assert not bool((stage_aware_noise & stage_aware["fixed_mask"]).any())
    assert not bool((stage_aware_noise & stage_aware["prediction_mask"]).any())
    assert full["noise_level"].item() == 1.0
    assert not full_noise.any()


def test_cluster_balanced_sampler_weights_each_virtual_view_by_inverse_cluster_size():
    records = [
        RnaSequenceRecord("a1", "AUCG" * 10, "RF1", "unit"),
        RnaSequenceRecord("a2", "CGUA" * 10, "RF1", "unit"),
        RnaSequenceRecord("b1", "UGCA" * 10, "RF2", "unit"),
    ]
    dataset = RnaMotifDenoisingDataset(
        records,
        RnaTokenizer(),
        max_length=40,
        min_motif_length=4,
        max_motif_length=4,
        motif_length_buckets=None,
        views_per_record=2,
    )
    module = RnaScaffoldDataModule(
        tokenizer=RnaTokenizer(),
        train_data="unused.csv",
        cluster_manifest="unused.tsv",
        batch_size=2,
        num_workers=0,
        cluster_balanced_sampling=True,
        seed=17,
    )
    module.train_dataset = dataset
    module.cluster_by_id = {"a1": "large", "a2": "large", "b1": "singleton"}

    first_sampler = module.train_dataloader().sampler
    second_sampler = module.train_dataloader().sampler

    assert isinstance(first_sampler, WeightedRandomSampler)
    assert first_sampler.weights.tolist() == [0.5, 0.5, 0.5, 0.5, 1.0, 1.0]
    assert list(first_sampler) == list(second_sampler)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"train_views_per_record": 0}, "train_views_per_record"),
        ({"preferred_motif_probability": 0.5}, "preferred_motifs"),
        ({"iterative_full_mask_probability": 0.5}, "requires iterative_mask_steps"),
    ],
)
def test_datamodule_rejects_invalid_v4_data_options_during_construction(overrides, message):
    with pytest.raises(ValueError, match=message):
        RnaScaffoldDataModule(
            tokenizer=RnaTokenizer(),
            train_data="unused.csv",
            cluster_manifest="unused.tsv",
            **overrides,
        )
