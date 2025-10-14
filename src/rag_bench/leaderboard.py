"""Leaderboard data generation.

Reads all verified submissions in `leaderboard/submissions/*.json` and emits
`leaderboard/data.json` consumed by the frontend.

Data shape (frontend contract):
{
  "generated_at": "2026-05-13T...Z",
  "rag_bench_version": "0.1.1",
  "tasks": ["nq-1k", "hotpotqa-1k", ...],
  "entries": [
    {
      "pipeline_hash": "sha256:...",
      "pipeline_name": "...",
      "submitter": { "name": "...", "contact": "..." },
      "verified": true,
      "submitted_at": "2026-05-12T...",
      "metrics": {
        "nq-1k": { "f1": 0.55, "f1_ci": [0.52, 0.58], "recall@10": 0.74, ... },
        ...
      },
      "operational": { "cost_per_query_usd_mean": 0.0024, "latency_ms_p95": 2103 },
      "tags": ["pareto", "verbose", ...]
    }
  ]
}
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_bench import __version__
from rag_bench.submission import Submission

_log = logging.getLogger(__name__)


def _entry_from_submission(
    sub: Submission, *, verified: bool, submitted_at: str | None
) -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    operational: dict[str, float] = {}
    for tid, payload in sub.tasks.items():
        m_out: dict[str, Any] = {}
        for metric_name, m in payload.get("metrics", {}).items():
            m_out[metric_name] = m["mean"]
            m_out[f"{metric_name}_ci"] = m["ci_95"]
        m_out["n_items"] = payload.get("n_items", 0)
        metrics[tid] = m_out
        # operational across tasks: take mean of means for now
        cost = payload.get("cost_per_query_usd", {})
        lat = payload.get("latency_ms", {})
        operational.setdefault("cost_per_query_usd_mean", 0.0)
        operational.setdefault("latency_ms_p95", 0.0)
        operational["cost_per_query_usd_mean"] += cost.get("mean", 0.0)
        operational["latency_ms_p95"] = max(operational["latency_ms_p95"], lat.get("p95", 0.0))
    if sub.tasks:
        operational["cost_per_query_usd_mean"] /= len(sub.tasks)
    return {
        "pipeline_hash": sub.pipeline_hash,
        "pipeline_name": sub.pipeline_name,
        "submitter": {"name": sub.submitter.name, "contact": sub.submitter.contact},
        "verified": verified,
        "submitted_at": submitted_at,
        "metrics": metrics,
        "operational": operational,
        "judge_fingerprint": (sub.judge_ensemble.fingerprint if sub.judge_ensemble else "none"),
        "tags": [],
    }


def generate_leaderboard(
    submissions_dir: str | Path,
    out_path: str | Path,
    *,
    verified_set: set[str] | None = None,
) -> dict[str, Any]:
    """Aggregate every submission file under `submissions_dir` into `out_path`."""
    submissions_dir = Path(submissions_dir)
    out_path = Path(out_path)
    entries: list[dict[str, Any]] = []
    tasks_seen: set[str] = set()
    if submissions_dir.exists():
        for f in sorted(submissions_dir.glob("*.json")):
            # A single malformed file should not crash the regeneration.
            try:
                sub = Submission.load(f)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                _log.warning("Skipping malformed submission %s: %s", f.name, e)
                continue
            verified = (verified_set is None) or (sub.pipeline_hash in verified_set)
            # Prefer the submission's own submitted_at; mtime is a fallback
            # for legacy entries written before the field existed.
            if sub.submitted_at:
                submitted_at: str | None = sub.submitted_at
            else:
                stat = f.stat()
                submitted_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
            entries.append(_entry_from_submission(sub, verified=verified, submitted_at=submitted_at))
            tasks_seen.update(sub.tasks.keys())
    # Quality × cost domination; quality = mean F1 across tasks.
    _tag_pareto(entries)
    # length_ratio above the threshold flags context-echoing pipelines.
    _tag_verbose(entries)
    data = {
        "generated_at": datetime.now(UTC).isoformat(),
        "rag_bench_version": __version__,
        "tasks": sorted(tasks_seen),
        "entries": entries,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, sort_keys=False), encoding="utf-8")
    return data


def _entry_quality(entry: dict[str, Any]) -> float:
    f1s: list[float] = []
    for _tid, m in entry.get("metrics", {}).items():
        if "token_f1" in m:
            f1s.append(float(m["token_f1"]))
    return sum(f1s) / len(f1s) if f1s else 0.0


def _tag_verbose(entries: list[dict[str, Any]]) -> None:
    """Tag entries whose mean length_ratio exceeds the threshold on any task."""
    from rag_bench.metrics.generation import LengthRatio

    threshold = LengthRatio.VERBOSE_THRESHOLD
    for entry in entries:
        for _tid, metrics in entry.get("metrics", {}).items():
            lr = metrics.get("length_ratio")
            if isinstance(lr, (int, float)) and lr > threshold:
                if "verbose" not in entry["tags"]:
                    entry["tags"].append("verbose")
                break


def _tag_pareto(entries: list[dict[str, Any]]) -> None:
    """Mark entries on the Pareto frontier (higher quality × lower cost)."""
    if not entries:
        return
    points = [
        (i, _entry_quality(e), e.get("operational", {}).get("cost_per_query_usd_mean", 0.0))
        for i, e in enumerate(entries)
    ]
    pareto: set[int] = set()
    for i, q_i, c_i in points:
        dominated = False
        for j, q_j, c_j in points:
            if j == i:
                continue
            # j dominates i if j has higher-or-equal quality AND lower-or-equal cost AND strictly better in one
            if q_j >= q_i and c_j <= c_i and (q_j > q_i or c_j < c_i):
                dominated = True
                break
        if not dominated:
            pareto.add(i)
    for i in pareto:
        if "pareto" not in entries[i]["tags"]:
            entries[i]["tags"].append("pareto")
