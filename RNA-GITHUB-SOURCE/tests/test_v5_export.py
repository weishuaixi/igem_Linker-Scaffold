import hashlib
import json
import tarfile

import pytest

from scripts import export_v5_best


def test_export_verifies_and_preserves_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(export_v5_best, "__file__", str(tmp_path / "scripts/export_v5_best.py"))
    directory = tmp_path / "checkpoints_scaffold_linker_v5"
    directory.mkdir()
    checkpoint = directory / "best.ckpt"
    checkpoint.write_bytes(b"tiny-test-checkpoint")
    manifest = {
        "status": "completed",
        "best_checkpoint": {"path": "best.ckpt", "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
    }
    (directory / "training_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/train_scaffold_5090_linker_v5.yaml").write_text("seed: 42\n")
    export_v5_best.main()
    archive_path = next((tmp_path / "exports").glob("*.tar.gz"))
    with tarfile.open(archive_path) as archive:
        assert archive.extractfile("checkpoints_scaffold_linker_v5/best.ckpt").read() == checkpoint.read_bytes()
    assert (
        archive_path.with_suffix(".gz.sha256").read_text().split()[0]
        == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    )
    checkpoint.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        export_v5_best.main()
    assert len(list((tmp_path / "exports").glob("*.tar.gz"))) == 1
