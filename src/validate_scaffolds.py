from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rna_scaffold.generate import ScaffoldCandidate
from rna_scaffold.ranking import rank_candidates
from rna_scaffold.validators.rnafold import run_rnafold


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and rank generated RNA scaffolds.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rnafold-executable", default="RNAfold")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()

    candidates = []
    with Path(args.input).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("record_type") == "generation_metadata":
                    continue
                candidates.append(ScaffoldCandidate(**row))
    folds = {
        candidate.candidate_id: run_rnafold(
            candidate.full_sequence,
            candidate.motif_start,
            candidate.motif_end,
            executable=args.rnafold_executable,
            timeout_seconds=args.timeout_seconds,
        )
        for candidate in candidates
    }
    ranked = rank_candidates(candidates, folds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for item in ranked:
            handle.write(json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output)


if __name__ == "__main__":
    main()
