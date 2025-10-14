"""CLI smoke tests using click's CliRunner.

Exercise the end-to-end flow:
  eval (synthetic) → show → submit → verify → leaderboard regenerate
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from rag_bench.cli import main


@pytest.fixture
def example_pipeline(tmp_path: Path) -> Path:
    p = tmp_path / "pipe.yaml"
    p.write_text(
        """
pipeline:
  name: cli-test-bm25-echo
  retriever_top_k: 5
  reranker_top_k: 3
  chunker: { type: recursive, chunk_size: 500, overlap: 50 }
  retriever: { type: bm25 }
  reranker: { type: identity, top_k: 3 }
  generator: { type: echo }
""",
        encoding="utf-8",
    )
    return p


def test_cli_tasks_ls_lists_synthetic():
    r = CliRunner().invoke(main, ["tasks", "ls"])
    assert r.exit_code == 0, r.output
    assert "synthetic-10" in r.output


def test_cli_components_ls_lists_bm25():
    r = CliRunner().invoke(main, ["components", "ls"])
    assert r.exit_code == 0, r.output
    assert "bm25" in r.output
    assert "echo" in r.output
    assert "recursive" in r.output


def test_cli_eval_then_show_then_verify(tmp_path: Path, example_pipeline: Path):
    out = tmp_path / "result.json"
    runner = CliRunner()
    r = runner.invoke(
        main,
        [
            "eval",
            str(example_pipeline),
            "--tasks", "synthetic-10",
            "--seeds", "1",
            "--out", str(out),
        ],
    )
    assert r.exit_code == 0, r.output
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["pipeline_name"] == "cli-test-bm25-echo"
    assert payload["pipeline_hash"].startswith("sha256:")

    r2 = runner.invoke(main, ["show", str(out)])
    assert r2.exit_code == 0, r2.output
    assert "cli-test-bm25-echo" in r2.output

    r3 = runner.invoke(main, ["verify", str(out)])
    assert r3.exit_code == 0, r3.output
    assert "OK" in r3.output


def test_cli_verify_rejects_forged_rag_bench_version(tmp_path: Path, example_pipeline: Path):
    """A forged rag_bench_version + recomputed hash should still fail verify,
    because verify cross-checks the declared version against the runtime."""
    out = tmp_path / "result.json"
    runner = CliRunner()
    runner.invoke(
        main,
        ["eval", str(example_pipeline), "--tasks", "synthetic-10", "--seeds", "1", "--out", str(out)],
    )

    # Simulate an attacker editing the version and rehashing the file.
    from rag_bench.repro import pipeline_hash
    from rag_bench.submission import Submission

    sub = Submission.load(out)
    sub.rag_bench_version = "999.999.999"
    tdh = {t: payload["task_data_hash"] for t, payload in sub.tasks.items()}
    sub.pipeline_hash = pipeline_hash(
        sub.pipeline_yaml,
        task_data_hashes=tdh,
        judge_fingerprint="none",
        seeds=sub.seeds,
        version=sub.rag_bench_version,
    )
    sub.save(out)

    r = runner.invoke(main, ["verify", str(out)])
    assert r.exit_code != 0, r.output
    assert "version" in r.output.lower()


def test_cli_submit_writes_to_leaderboard_dir(tmp_path: Path, example_pipeline: Path):
    out = tmp_path / "result.json"
    leaderboard = tmp_path / "leaderboard"
    runner = CliRunner()
    runner.invoke(
        main,
        [
            "eval",
            str(example_pipeline),
            "--tasks", "synthetic-10",
            "--seeds", "1",
            "--out", str(out),
        ],
    )
    r = runner.invoke(
        main,
        [
            "submit",
            str(out),
            "--submitter", "test-user",
            "--leaderboard-dir", str(leaderboard),
        ],
    )
    assert r.exit_code == 0, r.output
    assert any(leaderboard.glob("*.json"))


def test_cli_leaderboard_regenerate(tmp_path: Path, example_pipeline: Path):
    out = tmp_path / "result.json"
    leaderboard = tmp_path / "leaderboard"
    data = tmp_path / "data.json"
    runner = CliRunner()
    runner.invoke(
        main,
        [
            "eval",
            str(example_pipeline),
            "--tasks", "synthetic-10",
            "--seeds", "1",
            "--out", str(out),
        ],
    )
    runner.invoke(
        main,
        ["submit", str(out), "--leaderboard-dir", str(leaderboard)],
    )
    r = runner.invoke(
        main,
        [
            "leaderboard", "regenerate",
            "--submissions-dir", str(leaderboard),
            "--out", str(data),
        ],
    )
    assert r.exit_code == 0, r.output
    payload = json.loads(data.read_text(encoding="utf-8"))
    assert len(payload["entries"]) == 1
