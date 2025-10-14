"""Tests for the reproducibility hashing module."""

from __future__ import annotations

from rag_bench.repro import (
    canonicalize_yaml,
    judge_ensemble_fingerprint,
    pipeline_hash,
    verify_run,
)


def test_canonicalize_yaml_strips_comments_and_sorts_keys():
    y1 = """
    # leading comment
    pipeline:
      retriever:
        type: bm25
        top_k: 50
      chunker:
        type: recursive
        chunk_size: 1000
    """
    y2 = """
    pipeline:
      chunker: { chunk_size: 1000, type: recursive }
      retriever: { top_k: 50, type: bm25 }
    """
    assert canonicalize_yaml(y1) == canonicalize_yaml(y2)


def test_pipeline_hash_is_stable_across_equivalent_yaml():
    y1 = "pipeline:\n  retriever:\n    type: bm25\n    top_k: 50\n"
    y2 = "pipeline:\n  retriever:\n    top_k: 50\n    type: bm25\n"
    h1 = pipeline_hash(y1, task_data_hashes={"nq-1k": "abc"}, seeds=[0, 1])
    h2 = pipeline_hash(y2, task_data_hashes={"nq-1k": "abc"}, seeds=[1, 0])
    assert h1 == h2


def test_pipeline_hash_changes_with_seeds():
    y = "pipeline:\n  retriever:\n    type: bm25\n"
    h1 = pipeline_hash(y, task_data_hashes={"t": "x"}, seeds=[0])
    h2 = pipeline_hash(y, task_data_hashes={"t": "x"}, seeds=[0, 1])
    assert h1 != h2


def test_pipeline_hash_changes_with_judges():
    y = "pipeline:\n  retriever:\n    type: bm25\n"
    none_fp = judge_ensemble_fingerprint(None)
    j_fp = judge_ensemble_fingerprint([
        {"name": "A", "model": "claude-haiku-4-5", "prompt_hash": "p1"},
        {"name": "B", "model": "gpt-5.1", "prompt_hash": "p1"},
        {"name": "C", "model": "qwen3.5-72b", "prompt_hash": "p1"},
    ])
    h_none = pipeline_hash(y, task_data_hashes={"t": "x"}, judge_fingerprint=none_fp)
    h_judged = pipeline_hash(y, task_data_hashes={"t": "x"}, judge_fingerprint=j_fp)
    assert h_none != h_judged
    assert j_fp.startswith("sha256:")
    assert none_fp == "none"


def test_judge_fingerprint_order_invariant():
    a = judge_ensemble_fingerprint([
        {"name": "A", "model": "m1", "prompt_hash": "p"},
        {"name": "B", "model": "m2", "prompt_hash": "p"},
    ])
    b = judge_ensemble_fingerprint([
        {"name": "B", "model": "m2", "prompt_hash": "p"},
        {"name": "A", "model": "m1", "prompt_hash": "p"},
    ])
    assert a == b


def test_verify_run_passes_when_rerun_inside_ci():
    submitted = {"nq-1k": {"f1": {"mean": 0.55, "ci_95": [0.52, 0.58]}}}
    rerun = {"nq-1k": {"f1": {"mean": 0.54, "ci_95": [0.51, 0.57]}}}
    outcome = verify_run(submitted_metrics=submitted, rerun_metrics=rerun)
    assert outcome.verified, outcome.reasons


def test_verify_run_fails_when_rerun_outside_ci():
    submitted = {"nq-1k": {"f1": {"mean": 0.55, "ci_95": [0.52, 0.58]}}}
    rerun = {"nq-1k": {"f1": {"mean": 0.49, "ci_95": [0.46, 0.52]}}}
    outcome = verify_run(submitted_metrics=submitted, rerun_metrics=rerun)
    assert not outcome.verified
    assert any("outside" in r for r in outcome.reasons)


def test_verify_run_reports_missing_metric():
    submitted = {"nq-1k": {"f1": {"mean": 0.5, "ci_95": [0.45, 0.55]}}}
    rerun = {"nq-1k": {}}
    outcome = verify_run(submitted_metrics=submitted, rerun_metrics=rerun)
    assert not outcome.verified
    assert any("missing" in r for r in outcome.reasons)


def test_pipeline_hash_format():
    y = "pipeline:\n  retriever:\n    type: bm25\n"
    h = pipeline_hash(y, task_data_hashes={"nq-1k": "abc"})
    assert h.startswith("sha256:")
    assert len(h) == 7 + 64  # "sha256:" + 64 hex chars
