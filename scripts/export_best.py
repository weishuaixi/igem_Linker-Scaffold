"""Create a local download bundle after successful training; never delete originals."""

import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    directory = root / "checkpoints_scaffold_linker_v5"
    manifest_path = directory / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise ValueError("Training is not completed")
    checkpoint = Path(manifest["best_checkpoint"]["path"])
    if not checkpoint.is_absolute():
        checkpoint = directory / checkpoint
    checkpoint = checkpoint.resolve(strict=True)
    checkpoint.relative_to(directory.resolve())
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != manifest["best_checkpoint"]["sha256"]:
        raise ValueError("Best checkpoint hash mismatch")
    output = root / "exports"
    output.mkdir(exist_ok=True)
    target = output / ("RNA-BEST-DOWNLOAD-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f") + ".tar.gz")
    paths = [checkpoint, manifest_path, root / "configs/train.yaml"]
    paths += [root / name for name in ("environment-installed.txt", "PACKAGE_SHA256SUMS") if (root / name).is_file()]
    with target.open("xb") as raw, tarfile.open(fileobj=raw, mode="w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    target.with_suffix(target.suffix + ".sha256").write_text(
        digest.hexdigest() + "  " + target.name + "\n", encoding="utf-8"
    )
    print("DOWNLOAD NOW:", target, flush=True)
    print(
        "This archive is still on the server, NOT an external backup. Download it and its .sha256 before releasing the instance.",
        flush=True,
    )


if __name__ == "__main__":
    main()
