"""The Runner — orchestrates Pipeline × Task × seeds and emits a result record.

Responsibilities:
- Index the task corpus once into the pipeline.
- Iterate task items × seeds; collect per-query PipelineResults.
- Apply a metric set, producing per-task MetricResults with bootstrap CIs.
- Aggregate cost and latency.
- Emit a RunRecord ready to be serialized to result.json.

The Runner is intentionally synchronous and single-process. Distributed /
concurrent execution is out of scope for v1 (see docs/architecture.md §9).
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rag_bench.metrics.base import Metric, bootstrap_ci
from rag_bench.pipeline.pipeline import Pipeline
from rag_bench.tasks.base import Task
from rag_bench.types import PipelineResult


@dataclass
class TaskRunRecord:
    task_id: str
    n_items: int
    seeds: tuple[int, ...]
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)  # name -> jsonable
    cost_per_query_usd: dict[str, float] = field(default_factory=dict)  # mean, p95
    latency_ms: dict[str, float] = field(default_factory=dict)  # mean, p95
    wall_time_s: float = 0.0


@dataclass
class RunRecord:
    pipeline_name: str
    seeds: tuple[int, ...]
    tasks: dict[str, TaskRunRecord] = field(default_factory=dict)
    total_wall_time_s: float = 0.0

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "seeds": list(self.seeds),
            "total_wall_time_s": self.total_wall_time_s,
            "tasks": {
                tid: {
                    "task_id": rec.task_id,
                    "n_items": rec.n_items,
                    "seeds": list(rec.seeds),
                    "metrics": rec.metrics,
                    "cost_per_query_usd": rec.cost_per_query_usd,
                    "latency_ms": rec.latency_ms,
                    "wall_time_s": rec.wall_time_s,
                }
                for tid, rec in self.tasks.items()
            },
        }


class Runner:
    """Runs a pipeline against one or more tasks, computing metric aggregates."""

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        seeds: Sequence[int] = (0,),
        split: str = "public",
    ) -> None:
        self.pipeline = pipeline
        self.seeds = tuple(seeds)
        self.split = split

    def run_task(self, task: Task, metrics: Iterable[Metric]) -> TaskRunRecord:
        t_wall_start = time.perf_counter()
        # Index the corpus once.
        self.pipeline.index(task.corpus())
        items = list(task.items(split=self.split))  # type: ignore[arg-type]

        # Per-seed run; for each item, average the per-query metric scores across seeds.
        # We collect one "representative" PipelineResult per item (last seed) for ranking
        # diagnostics, and aggregate metrics across seeds.
        per_seed_results: list[list[PipelineResult]] = []
        per_query_costs: list[float] = []
        per_query_latencies: list[float] = []
        for seed in self.seeds:
            seed_results: list[PipelineResult] = []
            for item in items:
                r = self.pipeline.answer(item.query, seed=seed)
                seed_results.append(r)
                per_query_costs.append(r.generation.cost_usd)
                per_query_latencies.append(r.generation.latency_ms)
            per_seed_results.append(seed_results)

        # Score each metric: per-item score = mean across seeds, then aggregate
        # with bootstrap CI. ONE call to metric.score_one per (item, seed).
        metric_records: dict[str, dict[str, Any]] = {}
        for metric in metrics:
            per_item_scores: list[float] = []
            for i, item in enumerate(items):
                ss = [
                    metric.score_one(seed_results[i], item)
                    for seed_results in per_seed_results
                ]
                ss = [float(s) for s in ss if s is not None]
                if ss:
                    per_item_scores.append(sum(ss) / len(ss))
            if not per_item_scores:
                continue
            arr = np.asarray(per_item_scores, dtype=np.float64)
            mean = float(arr.mean())
            ci = bootstrap_ci(arr, n_bootstrap=2000, seed=0)
            metric_records[metric.name] = {
                "mean": mean,
                "ci_95": [ci[0], ci[1]],
                "n": int(arr.size),
            }

        # Operational metrics
        wall_time_s = time.perf_counter() - t_wall_start
        rec = TaskRunRecord(
            task_id=task.task_id,
            n_items=len(items),
            seeds=self.seeds,
            metrics=metric_records,
            cost_per_query_usd={
                "mean": _safe_mean(per_query_costs),
                "p95": _safe_quantile(per_query_costs, 0.95),
            },
            latency_ms={
                "mean": _safe_mean(per_query_latencies),
                "p95": _safe_quantile(per_query_latencies, 0.95),
            },
            wall_time_s=wall_time_s,
        )
        return rec

    def run(self, tasks_with_metrics: Iterable[tuple[Task, Iterable[Metric]]]) -> RunRecord:
        t0 = time.perf_counter()
        record = RunRecord(pipeline_name=self.pipeline.name, seeds=self.seeds)
        for task, metrics in tasks_with_metrics:
            tr = self.run_task(task, list(metrics))
            record.tasks[task.task_id] = tr
        record.total_wall_time_s = time.perf_counter() - t0
        return record


def _safe_mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _safe_quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(q * (len(s) - 1))))
    return s[idx]
