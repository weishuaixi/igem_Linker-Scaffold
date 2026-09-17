import json
import subprocess
import sys
import warnings

import scripts.verify_scaffold_release as release_audit


def test_release_boundary_detects_v1_checkpoint_path(tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "runtime.yaml").write_text(
        "checkpoint: checkpoints_scaffold_a800_mmseqs80/best.ckpt\n",
        encoding="utf-8",
    )

    boundary = release_audit._audit_v2_runtime_boundary(tmp_path)

    assert boundary["status"] == "failed"
    assert boundary["violations"] == [
        {
            "path": "configs/runtime.yaml",
            "terms": ["checkpoints_scaffold_a800_mmseqs80"],
        }
    ]


def test_release_audit_records_truthful_check_states(tmp_path):
    output = tmp_path / "audit.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/verify_scaffold_release.py",
            "--output",
            str(output),
            "--skip-tests",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    audit = json.loads(output.read_text(encoding="utf-8"))
    assert audit["checks"]
    assert {check["status"] for check in audit["checks"]} <= {
        "passed",
        "failed",
        "not_run",
        "unavailable",
    }
    boundary = next(
        (check for check in audit["checks"] if check["name"] == "V2 runtime boundary"),
        None,
    )
    assert boundary is not None
    assert boundary["status"] == "passed"
    assert boundary["violations"] == []
    schemas = next(
        (check for check in audit["checks"] if check["name"] == "V2 configuration schemas"),
        None,
    )
    assert schemas is not None
    assert schemas["status"] == "passed"
    assert schemas["validated"] == [
        "configs/benchmark_scaffolds_linker_v5.yaml",
        "configs/train_scaffold_5090_linker_v5.yaml",
    ]
    assert audit["server_training"]["status"] == "not_run"
    assert audit["formal_benchmark"]["status"] == "not_run"


def test_cpu_tiny_v2_workflow_trains_loads_generates_and_evaluates(tmp_path):
    assert hasattr(release_audit, "run_cpu_tiny_v2_workflow")

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        evidence = release_audit.run_cpu_tiny_v2_workflow(tmp_path)

    assert not [warning for warning in captured if "self.log()" in str(warning.message)]
    assert evidence["architecture_version"] == 2
    assert evidence["optimizer_steps"] == 1
    assert evidence["training_loss_is_finite"] is True
    assert evidence["checkpoint_loaded"] is True
    assert evidence["resume_checkpoint_validated"] is True
    assert evidence["resume_path_v1_rejected"] is True
    assert evidence["resume_hook_v1_rejected"] is True
    assert evidence["candidate_count"] >= 2
    assert evidence["motif_preservation_rate"] == 1.0
    assert evidence["unique_rate"] == 1.0
    assert evidence["rnafold_status"] == "not_run"
