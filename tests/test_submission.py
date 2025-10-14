"""Tests for the submission builder, schema, and validator."""

from __future__ import annotations

import tempfile
from pathlib import Path

from rag_bench.metrics.generation import ExactMatch, TokenF1
from rag_bench.metrics.retrieval import RecallAtK
from rag_bench.pipeline.compose import compose_from_yaml
from rag_bench.repro import pipeline_hash
from rag_bench.runner import Runner
from rag_bench.submission import (
    Submission,
    Submitter,
    build_submission,
    validate_submission,
)
from rag_bench.tasks.synthetic import SyntheticTask

PIPE_YAML = """
pipeline:
  name: bm25-echo-submission-test
  retriever_top_k: 5
  reranker_top_k: 3
  chunker: { type: recursive, chunk_size: 500, overlap: 50 }
  retriever: { type: bm25 }
  reranker: { type: identity, top_k: 3 }
  generator: { type: echo }
"""


def _make_run():
    pipe = compose_from_yaml(PIPE_YAML)
    task = SyntheticTask()
    runner = Runner(pipe, seeds=(0, 1), split="public")
    metrics = [RecallAtK(k=5), TokenF1(), ExactMatch()]
    return runner.run([(task, metrics)]), task


def test_build_submission_includes_pipeline_hash():
    record, task = _make_run()
    sub = build_submission(
        pipeline_yaml=PIPE_YAML,
        run_record=record,
        tasks=[task],
        submitter=Submitter(name="tester", contact="t@example.com"),
    )
    assert sub.pipeline_hash.startswith("sha256:")
    assert sub.pipeline_name == "bm25-echo-submission-test"
    assert "synthetic-10" in sub.tasks
    assert sub.tasks["synthetic-10"]["task_data_hash"]


def test_build_submission_pipeline_hash_matches_canonical():
    record, task = _make_run()
    sub = build_submission(pipeline_yaml=PIPE_YAML, run_record=record, tasks=[task])
    # Independently compute the hash and assert match
    expected = pipeline_hash(
        PIPE_YAML,
        task_data_hashes={task.task_id: task.task_data_hash()},
        judge_fingerprint="none",
        seeds=record.seeds,
    )
    assert sub.pipeline_hash == expected


def test_submission_round_trips_through_json():
    record, task = _make_run()
    sub = build_submission(pipeline_yaml=PIPE_YAML, run_record=record, tasks=[task])
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "result.json"
        sub.save(p)
        loaded = Submission.load(p)
    assert loaded.pipeline_hash == sub.pipeline_hash
    assert loaded.pipeline_name == sub.pipeline_name
    assert loaded.tasks.keys() == sub.tasks.keys()


def test_validate_submission_partial_suite():
    record, task = _make_run()
    sub = build_submission(pipeline_yaml=PIPE_YAML, run_record=record, tasks=[task])
    # With require_v1_suite=True, this synthetic-only submission lacks required tasks.
    outcome = validate_submission(sub, require_v1_suite=True)
    assert not outcome.ok
    assert any("Missing required v1.0-suite" in e for e in outcome.errors)


def test_validate_submission_partial_suite_when_allowed():
    record, task = _make_run()
    sub = build_submission(pipeline_yaml=PIPE_YAML, run_record=record, tasks=[task])
    outcome = validate_submission(sub, require_v1_suite=False)
    assert outcome.ok, outcome.errors


def test_validate_submission_warns_on_few_seeds():
    record, task = _make_run()  # only 2 seeds
    sub = build_submission(pipeline_yaml=PIPE_YAML, run_record=record, tasks=[task])
    outcome = validate_submission(sub, require_v1_suite=False)
    assert any("seeds" in w for w in outcome.warnings)


def test_validate_submission_rejects_corrupt_hash():
    record, task = _make_run()
    sub = build_submission(pipeline_yaml=PIPE_YAML, run_record=record, tasks=[task])
    sub.pipeline_hash = "not-a-sha"
    outcome = validate_submission(sub, require_v1_suite=False)
    assert not outcome.ok
    assert any("pipeline_hash" in e for e in outcome.errors)
