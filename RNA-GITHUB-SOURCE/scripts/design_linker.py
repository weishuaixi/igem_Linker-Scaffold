"""Generate a candidate pool, require RNAfold, and export the top-ranked sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rna_scaffold.generate import (
    GenerationSettings,
    generate_candidates,
    write_candidates_fasta,
    write_candidates_jsonl,
)
from rna_scaffold.ranking import DEFAULT_WEIGHTS, rank_candidates
from rna_scaffold.utils import validate_rna_sequence
from rna_scaffold.validators.rnafold import run_rnafold

DEFAULT_CHECKPOINT = "checkpoints_scaffold_linker_v5/rna-linker-v5-34-1.3714.ckpt"


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motif", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--rnafold-executable", default="RNAfold")
    parser.add_argument("--num-candidates", type=int, default=128)
    parser.add_argument("--denoise-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    motif = args.motif.upper()
    if not validate_rna_sequence(motif):
        parser.error("Motif must contain only A, U, C and G")
    if args.num_candidates < 2:
        parser.error("At least two candidates are required for within-pool ranking")
    if args.timeout_seconds <= 0:
        parser.error("timeout-seconds must be positive")
    if len(motif) + 25 > 100:
        parser.error("Motif is too long for the requested flank geometry")
    settings = GenerationSettings(
        num_candidates=args.num_candidates,
        max_length=100,
        seed=args.seed,
        short_flank_min=10,
        short_flank_max=20,
        long_flank_min=15,
        long_flank_max=30,
        length_sampling="uniform",
        remask_strategy="random",
        self_conditioning="off",
        denoise_steps=args.denoise_steps,
        temperature=1.0,
        top_p=1.0,
        max_homopolymer_run=6,
        gc_min=0.30,
        gc_max=0.70,
        enforce_gc_bounds=True,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    audit = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args),
        "generation_settings": asdict(settings),
        "ranking_weights": DEFAULT_WEIGHTS,
        "ranking_normalization": "min-max within candidate pool",
        "interpretation": "Heuristic top-ranked candidate, not experimentally validated best sequence",
        "motif_assumption": "Default score rewards an unpaired, accessible motif",
        "rnafold_required": True,
    }
    write_json(output / "run_manifest.json", audit)
    try:
        preflight = run_rnafold(
            "GGGAAACCC", 3, 6, executable=args.rnafold_executable, timeout_seconds=args.timeout_seconds
        )
        audit["rnafold_preflight"] = asdict(preflight)
        if preflight.status != "ok":
            raise RuntimeError(f"RNAfold preflight failed: {preflight.status}: {preflight.error}")
        print("Generating candidate pool...", flush=True)
        candidates = generate_candidates(args.checkpoint, motif, settings, args.device)
        write_candidates_jsonl(candidates, output / "candidates.jsonl")
        write_candidates_fasta(candidates, output / "candidates.fasta")
        audit["accepted"] = len(candidates)
        if len(candidates) != args.num_candidates:
            raise RuntimeError("Candidate budget not met; inspect candidates.jsonl generation_audit")
        if len({c.candidate_id for c in candidates}) != len(candidates):
            raise RuntimeError("Duplicate candidate IDs")
        if any(
            not c.valid or not c.motif_preserved or c.full_sequence != c.left_sequence + motif + c.right_sequence
            for c in candidates
        ):
            raise RuntimeError("Invalid candidate or motif mismatch")
        folds = {}
        with (output / "rnafold_results.jsonl").open("x", encoding="utf-8") as handle:
            for index, candidate in enumerate(candidates, start=1):
                fold = run_rnafold(
                    candidate.full_sequence,
                    candidate.motif_start,
                    candidate.motif_end,
                    executable=args.rnafold_executable,
                    timeout_seconds=args.timeout_seconds,
                )
                folds[candidate.candidate_id] = fold
                handle.write(json.dumps({"candidate_id": candidate.candidate_id, **asdict(fold)}) + "\n")
                handle.flush()
                print(f"RNAfold {index}/{len(candidates)}: {fold.status}", flush=True)
        failures = {name: fold.status for name, fold in folds.items() if fold.status != "ok"}
        audit["rnafold_failures"] = failures
        if failures:
            raise RuntimeError("RNAfold failed for one or more candidates; no best sequence was exported")
        ranked = rank_candidates(candidates, folds)
        with (output / "ranked.jsonl").open("x", encoding="utf-8") as handle:
            for item in ranked:
                handle.write(json.dumps(asdict(item), sort_keys=True, allow_nan=False) + "\n")
        best = ranked[0]
        write_json(output / "best.json", asdict(best))
        with (output / "best.fasta").open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(f">top_ranked_{best.candidate.candidate_id} score={best.composite_score:.6f}\n")
            handle.write(best.candidate.full_sequence + "\n")
        audit.update(
            status="completed",
            checkpoint_sha256=best.candidate.checkpoint_sha256,
            top_candidate_id=best.candidate.candidate_id,
            completed_utc=datetime.now(timezone.utc).isoformat(),
        )
        audit["artifact_sha256"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in output.iterdir()
            if path.is_file() and path.name != "run_manifest.json"
        }
        print("Top-ranked sequence:", best.candidate.full_sequence, flush=True)
        print("Output:", output / "best.fasta", flush=True)
        return 0
    except Exception as error:  # noqa: BLE001 -- persist failure evidence at the CLI boundary
        audit.update(status="failed", error=f"{type(error).__name__}: {error}")
        print(audit["error"], file=sys.stderr, flush=True)
        return 2
    finally:
        write_json(output / "run_manifest.json", audit)


if __name__ == "__main__":
    raise SystemExit(main())
