from __future__ import annotations

import argparse
import sys

from rna_scaffold.generate import (
    GenerationSettings,
    generate_candidates,
    write_candidates_fasta,
    write_candidates_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate RNA scaffolds from a trained checkpoint.")
    parser.add_argument("--motif", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fasta-output")
    parser.add_argument("--num-candidates", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--denoise-steps", type=int, default=12)
    parser.add_argument("--max-attempt-multiplier", type=int, default=8)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--min-normalized-edit-distance", type=float, default=0.0)
    parser.add_argument("--max-kmer-similarity", type=float)
    parser.add_argument("--kmer-size", type=int, default=5)
    parser.add_argument("--max-homopolymer-run", type=int)
    parser.add_argument("--gc-min", type=float)
    parser.add_argument("--gc-max", type=float)
    parser.add_argument("--enforce-gc-bounds", action="store_true")
    parser.add_argument("--min-scaffold-length", type=int, default=8)
    parser.add_argument("--min-flank-length", type=int, default=2)
    parser.add_argument("--short-flank-min", type=int)
    parser.add_argument("--short-flank-max", type=int)
    parser.add_argument("--long-flank-min", type=int)
    parser.add_argument("--long-flank-max", type=int)
    parser.add_argument("--length-sampling", choices=("model", "uniform"), default="model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--remask-strategy", choices=("auto", "confidence", "random"), default="auto")
    parser.add_argument("--self-conditioning", choices=("auto", "on", "off"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = GenerationSettings(
        num_candidates=args.num_candidates,
        max_length=args.max_length,
        seed=args.seed,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        denoise_steps=args.denoise_steps,
        max_attempt_multiplier=args.max_attempt_multiplier,
        max_attempts=args.max_attempts,
        min_normalized_edit_distance=args.min_normalized_edit_distance,
        max_kmer_similarity=args.max_kmer_similarity,
        kmer_size=args.kmer_size,
        max_homopolymer_run=args.max_homopolymer_run,
        gc_min=args.gc_min,
        gc_max=args.gc_max,
        enforce_gc_bounds=args.enforce_gc_bounds,
        min_scaffold_length=args.min_scaffold_length,
        min_flank_length=args.min_flank_length,
        short_flank_min=args.short_flank_min,
        short_flank_max=args.short_flank_max,
        long_flank_min=args.long_flank_min,
        long_flank_max=args.long_flank_max,
        length_sampling=args.length_sampling,
        remask_strategy=args.remask_strategy,
        self_conditioning=args.self_conditioning,
    )
    candidates = generate_candidates(args.checkpoint, args.motif, settings, args.device)
    write_candidates_jsonl(candidates, args.output)
    if args.fasta_output:
        write_candidates_fasta(candidates, args.fasta_output)
    accepted = len(candidates)
    if accepted != settings.num_candidates:
        print(
            f"ERROR: generated {accepted}/{settings.num_candidates} requested candidates; "
            "inspect generation_audit in the JSONL output",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
