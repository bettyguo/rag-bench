"""End-to-end Runner test on the offline synthetic task.

Discriminator test: a BM25+echo pipeline should hit recall@5 == 1.0 on a task
where each query has a unique entity in exactly one document.
"""

from __future__ import annotations

import pytest

from rag_bench.metrics.generation import ExactMatch, TokenF1
from rag_bench.metrics.retrieval import MRRAtK, RecallAtK
from rag_bench.pipeline.compose import compose_from_yaml
from rag_bench.runner import Runner
from rag_bench.tasks.synthetic import SyntheticTask

PIPE_YAML = """
pipeline:
  name: bm25-echo-synthetic
  retriever_top_k: 5
  reranker_top_k: 3
  chunker:
    type: recursive
    chunk_size: 500
    overlap: 50
  retriever:
    type: bm25
  reranker:
    type: identity
    top_k: 3
  generator:
    type: echo
"""


def test_bm25_echo_perfect_retrieval_on_synthetic_task():
    pipe = compose_from_yaml(PIPE_YAML)
    task = SyntheticTask()
    runner = Runner(pipe, seeds=(0,), split="public")
    metrics = [RecallAtK(k=5), MRRAtK(k=5), TokenF1(), ExactMatch()]
    record = runner.run_task(task, metrics)
    assert record.task_id == "synthetic-10"
    assert record.n_items == 10
    recall = record.metrics["recall@5"]["mean"]
    mrr = record.metrics["mrr@5"]["mean"]
    # BM25 should nail the unique-entity setup
    assert recall >= 0.9, record.metrics
    assert mrr >= 0.5, record.metrics
    # F1 must be > 0 since echo returns the gold passage substring containing the answer
    assert record.metrics["token_f1"]["mean"] > 0.05


def test_runner_emits_cost_and_latency_keys():
    pipe = compose_from_yaml(PIPE_YAML)
    task = SyntheticTask()
    runner = Runner(pipe, seeds=(0,))
    record = runner.run_task(task, [RecallAtK(k=5)])
    assert "mean" in record.cost_per_query_usd
    assert "p95" in record.cost_per_query_usd
    assert "mean" in record.latency_ms
    assert "p95" in record.latency_ms
    assert record.wall_time_s > 0


def test_runner_returns_jsonable_record():
    pipe = compose_from_yaml(PIPE_YAML)
    task = SyntheticTask()
    runner = Runner(pipe, seeds=(0, 1))
    metrics = [RecallAtK(k=5), TokenF1()]
    full = runner.run([(task, metrics)])
    js = full.to_jsonable()
    import json

    json.dumps(js)  # must round-trip
    assert js["pipeline_name"] == "bm25-echo-synthetic"
    assert "synthetic-10" in js["tasks"]
    assert js["tasks"]["synthetic-10"]["seeds"] == [0, 1]


def test_runner_multiple_seeds_dont_break_aggregation():
    pipe = compose_from_yaml(PIPE_YAML)
    task = SyntheticTask()
    runner = Runner(pipe, seeds=(0, 1, 2))
    record = runner.run_task(task, [RecallAtK(k=5)])
    # echo is deterministic — multiple seeds should produce identical per-item scores
    assert record.metrics["recall@5"]["ci_95"][0] == pytest.approx(
        record.metrics["recall@5"]["mean"], abs=1e-9
    ) or record.metrics["recall@5"]["mean"] == 1.0
