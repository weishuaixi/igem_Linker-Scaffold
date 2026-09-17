from __future__ import annotations

import csv
import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RnaSequenceRecord:
    target_id: str
    sequence: str
    family: str | None
    source: str

    def __post_init__(self) -> None:
        target_id = self.target_id.strip()
        sequence = self.sequence.strip().upper().replace("T", "U")
        family = self.family.strip() if self.family and self.family.strip() else None
        source = self.source.strip()
        if not target_id:
            raise ValueError("target_id must not be empty")
        if not source:
            raise ValueError("source must not be empty")
        if len(sequence) < 2 or set(sequence) - set("AUCG"):
            raise ValueError(f"invalid RNA sequence for {target_id}")
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "source", source)

    @property
    def sequence_sha256(self) -> str:
        return hashlib.sha256(self.sequence.encode("ascii")).hexdigest()


def load_sequence_records(
    path: str | Path,
    skip_invalid: bool = False,
) -> list[RnaSequenceRecord]:
    source_path = Path(path)
    if source_path.suffix.lower() != ".csv":
        raise ValueError("metadata-preserving record loading currently requires CSV")
    records: list[RnaSequenceRecord] = []
    skipped_ids: list[str] = []
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "sequence" not in reader.fieldnames:
            raise ValueError(f"CSV must contain a sequence column: {source_path}")
        for index, row in enumerate(reader):
            sequence = (row.get("sequence") or "").strip()
            target_id = (row.get("target_id") or row.get("id") or f"row_{index}").strip()
            if not sequence:
                if skip_invalid:
                    skipped_ids.append(target_id)
                    continue
                raise ValueError(f"empty RNA sequence for {target_id}")
            family = (row.get("family") or "").strip() or None
            source = (row.get("source") or source_path.stem).strip()
            try:
                records.append(RnaSequenceRecord(target_id, sequence, family, source))
            except ValueError:
                if not skip_invalid:
                    raise
                skipped_ids.append(target_id)
    if skipped_ids:
        preview = ", ".join(skipped_ids[:10])
        warnings.warn(
            f"skipped {len(skipped_ids)} invalid RNA records: {preview}",
            stacklevel=2,
        )
    return records
