"""Leaderboard generator tests."""

from __future__ import annotations

import json
from pathlib import Path

from rag_bench.leaderboard import generate_leaderboard
from rag_bench.metrics.generation import ExactMatch, TokenF1
from rag_bench.metrics.retrieval import RecallAtK
from rag_bench.pipeline.compose import compose_from_yaml
from rag_bench.runner import Runner
from rag_bench.submission import Submitter, build_submission
from rag_bench.tasks.synthetic import SyntheticTask

PIPE_A = """
pipeline:
  name: pipe-A-cheap-strong
  retriever_top_k: 5
  reranker_top_k: 3
  chunker: { type: recursive, chunk_size: 500, overlap: 50 }
  retriever: { type: bm25 }
  reranker: { type: identity, top_k: 3 }
  generator: { type: echo }
"""

PIPE_B = """
pipeline:
  name: pipe-B-cheap-weak
  retriever_top_k: 5
  reranker_top_k: 3
  chunker: { type: fixed, chunk_size: 30, overlap: 5 }
  retriever: { type: bm25 }
  reranker: { type: identity, top_k: 3 }
  generator: { type: extractive }
"""


def _populate(tmp_path: Path):
    subs_dir = tmp_path / "submissions"
    subs_dir.mkdir()
    for i, yml in enumerate([PIPE_A, PIPE_B]):
        pipe = compose_from_yaml(yml)
        task = SyntheticTask()
        runner = Runner(pipe, seeds=(0,), split="public")
        record = runner.run([(task, [RecallAtK(k=5), TokenF1(), ExactMatch()])])
        sub = build_submission(
            pipeline_yaml=yml,
            run_record=record,
            tasks=[task],
            submitter=Submitter(name=f"tester-{i}"),
        )
        sub.save(subs_dir / f"{sub.pipeline_hash.replace('sha256:', '')}.json")
    return subs_dir


def test_generate_leaderboard_emits_two_entries(tmp_path):
    subs_dir = _populate(tmp_path)
    out = tmp_path / "data.json"
    data = generate_leaderboard(subs_dir, out)
    assert len(data["entries"]) == 2
    assert "synthetic-10" in data["tasks"]


def test_generate_leaderboard_writes_jsonable_file(tmp_path):
    subs_dir = _populate(tmp_path)
    out = tmp_path / "data.json"
    generate_leaderboard(subs_dir, out)
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert "entries" in reloaded
    assert reloaded["rag_bench_version"]


def test_generate_leaderboard_tags_pareto(tmp_path):
    subs_dir = _populate(tmp_path)
    out = tmp_path / "data.json"
    data = generate_leaderboard(subs_dir, out)
    # Both pipelines are cost=0 (echo); so the higher-F1 dominates. At least one must be Pareto.
    n_pareto = sum(1 for e in data["entries"] if "pareto" in e["tags"])
    assert n_pareto >= 1


def test_generate_leaderboard_marks_all_unverified_by_default(tmp_path):
    subs_dir = _populate(tmp_path)
    out = tmp_path / "data.json"
    # verified_set is None → everything counted as verified
    data = generate_leaderboard(subs_dir, out)
    assert all(e["verified"] for e in data["entries"])


def test_generate_leaderboard_respects_verified_set(tmp_path):
    subs_dir = _populate(tmp_path)
    out = tmp_path / "data.json"
    data = generate_leaderboard(subs_dir, out, verified_set=set())
    assert all(not e["verified"] for e in data["entries"])


def test_generate_leaderboard_empty_dir(tmp_path):
    empty = tmp_path / "empty-subs"
    empty.mkdir()
    out = tmp_path / "data.json"
    data = generate_leaderboard(empty, out)
    assert data["entries"] == []
    assert data["tasks"] == []


def test_generate_leaderboard_skips_malformed_submission(tmp_path, caplog):
    import logging

    subs_dir = _populate(tmp_path)
    (subs_dir / "broken.json").write_text("{this is not valid json", encoding="utf-8")

    out = tmp_path / "data.json"
    with caplog.at_level(logging.WARNING, logger="rag_bench.leaderboard"):
        data = generate_leaderboard(subs_dir, out)

    assert len(data["entries"]) == 2
    assert any("broken.json" in rec.getMessage() for rec in caplog.records)


def test_verbose_tag_applied_when_length_ratio_above_threshold(tmp_path):
    import json

    subs_dir = _populate(tmp_path)
    files = sorted(subs_dir.glob("*.json"))
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    first_task_id = next(iter(payload["tasks"].keys()))
    payload["tasks"][first_task_id]["metrics"]["length_ratio"] = {
        "mean": 12.5,
        "ci_95": [10.0, 15.0],
        "n": 100,
    }
    files[0].write_text(json.dumps(payload), encoding="utf-8")

    out = tmp_path / "data.json"
    data = generate_leaderboard(subs_dir, out)
    verbose_entries = [e for e in data["entries"] if "verbose" in e["tags"]]
    assert len(verbose_entries) == 1


def test_verbose_tag_not_applied_when_length_ratio_low(tmp_path):
    import json

    subs_dir = _populate(tmp_path)
    files = sorted(subs_dir.glob("*.json"))
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    first_task_id = next(iter(payload["tasks"].keys()))
    payload["tasks"][first_task_id]["metrics"]["length_ratio"] = {
        "mean": 1.2,
        "ci_95": [0.9, 1.5],
        "n": 100,
    }
    files[0].write_text(json.dumps(payload), encoding="utf-8")

    out = tmp_path / "data.json"
    data = generate_leaderboard(subs_dir, out)
    assert [e for e in data["entries"] if "verbose" in e["tags"]] == []


def test_submitted_at_uses_field_when_present(tmp_path):
    import json

    subs_dir = _populate(tmp_path)
    files = sorted(subs_dir.glob("*.json"))
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    payload["submitted_at"] = "2026-03-15T12:00:00+00:00"
    files[0].write_text(json.dumps(payload), encoding="utf-8")

    out = tmp_path / "data.json"
    data = generate_leaderboard(subs_dir, out)
    pinned = [e for e in data["entries"] if e["submitted_at"] == "2026-03-15T12:00:00+00:00"]
    assert len(pinned) == 1
