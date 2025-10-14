"""Submission file (result.json) construction, validation, and verification.

Pulls together:
- The runner's RunRecord (metric numbers, cost, latency)
- The pipeline YAML and its canonicalization
- The task_data_hashes from each Task
- The judge ensemble fingerprint (if faithfulness used)
- The pipeline_hash

Schema mirrors docs/reproducibility.md §3.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rag_bench import __version__
from rag_bench.repro import canonicalize_yaml, judge_ensemble_fingerprint, pipeline_hash
from rag_bench.runner import RunRecord
from rag_bench.tasks.base import Task


@dataclass
class Submitter:
    name: str = "anonymous"
    contact: str = ""


@dataclass
class JudgeRecord:
    name: str
    family: str
    model: str
    prompt_hash: str


@dataclass
class JudgeEnsemble:
    judges: list[JudgeRecord]
    fingerprint: str  # "sha256:..." or "none"


@dataclass
class Submission:
    """A self-contained submission ready to be PR'd to the leaderboard."""

    rag_bench_version: str
    pipeline_hash: str
    pipeline_yaml: str  # canonical
    pipeline_name: str
    submitter: Submitter
    seeds: list[int]
    judge_ensemble: JudgeEnsemble | None
    tasks: dict[str, Any]  # task_id -> {task_data_hash, metrics, cost, latency, n_items}
    total_wall_time_s: float
    # Stamped by `rag-bench submit`; the leaderboard prefers this over the
    # submission file's mtime so the displayed date survives git clones.
    submitted_at: str | None = None

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "rag_bench_version": self.rag_bench_version,
            "pipeline_hash": self.pipeline_hash,
            "pipeline_yaml": self.pipeline_yaml,
            "pipeline_name": self.pipeline_name,
            "submitter": asdict(self.submitter),
            "seeds": list(self.seeds),
            "judge_ensemble": (
                {
                    "judges": [asdict(j) for j in self.judge_ensemble.judges],
                    "fingerprint": self.judge_ensemble.fingerprint,
                }
                if self.judge_ensemble
                else None
            ),
            "tasks": self.tasks,
            "total_wall_time_s": self.total_wall_time_s,
            "submitted_at": self.submitted_at,
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_jsonable(), indent=2, sort_keys=False), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> Submission:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        je = None
        if data.get("judge_ensemble"):
            je = JudgeEnsemble(
                judges=[JudgeRecord(**j) for j in data["judge_ensemble"]["judges"]],
                fingerprint=data["judge_ensemble"]["fingerprint"],
            )
        return cls(
            rag_bench_version=data["rag_bench_version"],
            pipeline_hash=data["pipeline_hash"],
            pipeline_yaml=data["pipeline_yaml"],
            pipeline_name=data["pipeline_name"],
            submitter=Submitter(**data["submitter"]),
            seeds=list(data["seeds"]),
            judge_ensemble=je,
            tasks=data["tasks"],
            total_wall_time_s=data.get("total_wall_time_s", 0.0),
            submitted_at=data.get("submitted_at"),
        )


def build_submission(
    *,
    pipeline_yaml: str,
    run_record: RunRecord,
    tasks: Iterable[Task],
    submitter: Submitter | None = None,
    judges: Iterable[Mapping[str, str]] | None = None,
) -> Submission:
    """Bake a RunRecord + the inputs that produced it into a Submission."""
    canon = canonicalize_yaml(pipeline_yaml)
    tasks = list(tasks)
    task_data_hashes = {t.task_id: t.task_data_hash() for t in tasks}
    je_fingerprint = judge_ensemble_fingerprint(judges)
    ph = pipeline_hash(
        pipeline_yaml,
        task_data_hashes=task_data_hashes,
        judge_fingerprint=je_fingerprint,
        seeds=run_record.seeds,
        version=__version__,
    )
    judge_records = None
    if judges:
        judge_records = JudgeEnsemble(
            judges=[
                JudgeRecord(
                    name=j["name"],
                    family=j.get("family", ""),
                    model=j["model"],
                    prompt_hash=j["prompt_hash"],
                )
                for j in judges
            ],
            fingerprint=je_fingerprint,
        )

    task_payload: dict[str, Any] = {}
    for tid, rec in run_record.tasks.items():
        task_payload[tid] = {
            "task_data_hash": task_data_hashes.get(tid, ""),
            "n_items": rec.n_items,
            "seeds": list(rec.seeds),
            "metrics": rec.metrics,
            "cost_per_query_usd": rec.cost_per_query_usd,
            "latency_ms": rec.latency_ms,
            "wall_time_s": rec.wall_time_s,
        }

    return Submission(
        rag_bench_version=__version__,
        pipeline_hash=ph,
        pipeline_yaml=canon,
        pipeline_name=run_record.pipeline_name,
        submitter=submitter or Submitter(),
        seeds=list(run_record.seeds),
        judge_ensemble=judge_records,
        tasks=task_payload,
        total_wall_time_s=run_record.total_wall_time_s,
    )


REQUIRED_TASKS_V1_0 = ("nq-1k", "hotpotqa-1k", "noisy-qa", "unanswerable-qa")


@dataclass
class ValidationOutcome:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_submission(sub: Submission, *, require_v1_suite: bool = True) -> ValidationOutcome:
    errors: list[str] = []
    warnings: list[str] = []
    if not sub.pipeline_hash.startswith("sha256:"):
        errors.append(f"Bad pipeline_hash format: {sub.pipeline_hash}")
    if not sub.pipeline_yaml.strip():
        errors.append("pipeline_yaml is empty")
    if len(sub.seeds) < 5:
        warnings.append(f"<5 seeds ({len(sub.seeds)}); submission discouraged for leaderboard")
    if require_v1_suite:
        missing = [t for t in REQUIRED_TASKS_V1_0 if t not in sub.tasks]
        if missing:
            errors.append(
                f"Missing required v1.0-suite tasks: {missing}. "
                "Set require_v1_suite=False for partial-suite submissions."
            )
    for tid, payload in sub.tasks.items():
        if "metrics" not in payload or not payload["metrics"]:
            errors.append(f"Task {tid}: no metrics reported")
        if "task_data_hash" not in payload or not payload["task_data_hash"]:
            errors.append(f"Task {tid}: missing task_data_hash")
        for metric_name, m in payload.get("metrics", {}).items():
            if "mean" not in m or "ci_95" not in m:
                errors.append(f"Task {tid}/{metric_name}: missing mean or ci_95")
    return ValidationOutcome(ok=not errors, errors=errors, warnings=warnings)
