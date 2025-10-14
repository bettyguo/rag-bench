"""Regression test: Runner.run_task invokes metric.score_one exactly once
per (item, seed). An earlier version doubled the count via a dead code
path that built throw-away avg_results/avg_items.
"""

from __future__ import annotations

from rag_bench.metrics.base import Metric
from rag_bench.pipeline.compose import compose_from_yaml
from rag_bench.runner import Runner
from rag_bench.tasks.synthetic import SyntheticTask
from rag_bench.types import PipelineResult, TaskItem

PIPE_YAML = """
pipeline:
  name: bm25-echo-callcount
  retriever_top_k: 5
  reranker_top_k: 3
  chunker: { type: recursive, chunk_size: 500, overlap: 50 }
  retriever: { type: bm25 }
  reranker: { type: identity, top_k: 3 }
  generator: { type: echo }
"""


class CountingMetric(Metric):
    """Counts how many times score_one is invoked."""

    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def score_one(self, result: PipelineResult, item: TaskItem) -> float | None:
        self.calls += 1
        return 1.0


def test_runner_invokes_score_one_exactly_n_items_times_n_seeds():
    pipe = compose_from_yaml(PIPE_YAML)
    task = SyntheticTask()
    n_items = len(list(task.items("public")))
    seeds = (0, 1, 2)
    runner = Runner(pipe, seeds=seeds, split="public")
    metric = CountingMetric()
    runner.run_task(task, [metric])
    assert metric.calls == n_items * len(seeds)


def test_runner_call_count_independent_of_metric_count():
    pipe = compose_from_yaml(PIPE_YAML)
    task = SyntheticTask()
    n_items = len(list(task.items("public")))
    seeds = (0, 1)
    runner = Runner(pipe, seeds=seeds, split="public")
    m1 = CountingMetric()
    m2 = CountingMetric()
    runner.run_task(task, [m1, m2])
    assert m1.calls == n_items * len(seeds)
    assert m2.calls == n_items * len(seeds)
