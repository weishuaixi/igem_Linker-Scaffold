from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RnafoldResult:
    status: str
    dot_bracket: str | None
    mfe_kcal_mol: float | None
    paired_fraction: float | None
    motif_paired_fraction: float | None
    runtime_seconds: float
    version: str | None
    error: str | None


_STRUCTURE = re.compile(r"^([().]+)\s+\(\s*(-?\d+(?:\.\d+)?)\s*\)\s*$")


def parse_rnafold_output(
    output: str,
    motif_start: int,
    motif_end: int,
    runtime_seconds: float,
    version: str | None,
) -> RnafoldResult:
    for line in output.splitlines():
        match = _STRUCTURE.match(line.strip())
        if not match:
            continue
        structure = match.group(1)
        if not 0 <= motif_start <= motif_end <= len(structure):
            return RnafoldResult(
                "parse_error", None, None, None, None, runtime_seconds, version, "invalid motif coordinates"
            )
        paired = sum(character in "()" for character in structure) / len(structure)
        motif_structure = structure[motif_start:motif_end]
        motif_paired = (
            sum(character in "()" for character in motif_structure) / len(motif_structure)
            if motif_structure
            else 0.0
        )
        return RnafoldResult(
            "ok",
            structure,
            float(match.group(2)),
            paired,
            motif_paired,
            runtime_seconds,
            version,
            None,
        )
    return RnafoldResult(
        "parse_error", None, None, None, None, runtime_seconds, version, "RNAfold structure line not found"
    )


def _resolve_executable(executable: str | Path) -> str | None:
    candidate = Path(executable)
    if candidate.is_file():
        return str(candidate)
    return shutil.which(str(executable))


def run_rnafold(
    sequence: str,
    motif_start: int,
    motif_end: int,
    executable: str | Path = "RNAfold",
    timeout_seconds: float = 30.0,
) -> RnafoldResult:
    resolved = _resolve_executable(executable)
    if resolved is None:
        return RnafoldResult(
            "unavailable", None, None, None, None, 0.0, None, f"RNAfold executable not found: {executable}"
        )
    version = None
    try:
        version_run = subprocess.run(
            [resolved, "--version"], capture_output=True, text=True, timeout=5, check=False
        )
        if version_run.returncode == 0:
            version = (version_run.stdout or version_run.stderr).strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        version = None
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [resolved, "--noPS"],
            input=f">candidate\n{sequence}\n",
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return RnafoldResult(
            "timeout", None, None, None, None, time.perf_counter() - started, version, "RNAfold timed out"
        )
    except OSError as error:
        return RnafoldResult(
            "failed", None, None, None, None, time.perf_counter() - started, version, str(error)
        )
    runtime = time.perf_counter() - started
    if completed.returncode != 0:
        return RnafoldResult(
            "failed",
            None,
            None,
            None,
            None,
            runtime,
            version,
            completed.stderr.strip() or f"RNAfold exited {completed.returncode}",
        )
    return parse_rnafold_output(completed.stdout, motif_start, motif_end, runtime, version)
