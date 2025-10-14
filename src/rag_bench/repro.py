"""Reproducibility hashing + verification.

The pipeline_hash uniquely identifies a (pipeline, task_data, judges, version,
seeds) tuple. Re-running the same hash must produce numbers within the
submitter's reported 95% bootstrap CI — else the submission is `unverified`.

The hash inputs are:
1. canonical_yaml(pipeline_yaml)  — pipeline config, sorted keys, comments stripped
2. task_data_hash                  — per-task SHA-256 from Task.task_data_hash()
3. judge_ensemble_fingerprint      — judge model IDs + prompt hashes (or "none")
4. rag_bench_version               — from __init__.py
5. seeds                           — sorted list of seed ints
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import yaml

from rag_bench import __version__


def canonicalize_yaml(yaml_text: str) -> str:
    """Round-trip a YAML doc to canonical form: sorted keys, no comments.

    Comments are stripped by virtue of `yaml.safe_load` discarding them, and
    `yaml.safe_dump(..., sort_keys=True)` produces a deterministic ordering.
    """
    data = yaml.safe_load(yaml_text)
    return yaml.safe_dump(data, sort_keys=True, default_flow_style=False, allow_unicode=True)


def canonical_json(obj: Any) -> str:
    """Canonical JSON: sorted keys, no whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def judge_ensemble_fingerprint(
    judges: Iterable[Mapping[str, str]] | None,
) -> str:
    """SHA-256 hex of the sorted judge list.

    Each judge dict should have: name, model, prompt_hash. Returns 'none' when
    `judges` is None (submission opted out of faithfulness).
    """
    if not judges:
        return "none"
    items = sorted(
        ({"name": j["name"], "model": j["model"], "prompt_hash": j["prompt_hash"]} for j in judges),
        key=lambda d: d["name"],
    )
    return "sha256:" + hashlib.sha256(canonical_json(items).encode()).hexdigest()


def pipeline_hash(
    pipeline_yaml: str,
    *,
    task_data_hashes: Mapping[str, str],
    judge_fingerprint: str = "none",
    seeds: Iterable[int] = (0,),
    version: str | None = None,
) -> str:
    """Content-addressed hash of a submission.

    See [docs/reproducibility.md](../../docs/reproducibility.md) for the contract.
    """
    canon_yaml = canonicalize_yaml(pipeline_yaml)
    payload = {
        "yaml": canon_yaml,
        "task_data_hashes": dict(sorted(task_data_hashes.items())),
        "judge_fingerprint": judge_fingerprint,
        "version": version or __version__,
        "seeds": sorted(int(s) for s in seeds),
    }
    return "sha256:" + hashlib.sha256(canonical_json(payload).encode()).hexdigest()


@dataclass(frozen=True)
class VerificationOutcome:
    verified: bool
    reasons: tuple[str, ...]


def verify_run(
    *,
    submitted_metrics: Mapping[str, Mapping[str, Any]],
    rerun_metrics: Mapping[str, Mapping[str, Any]],
    overlap_tol: float = 0.0,
) -> VerificationOutcome:
    """Check whether re-run means fall inside the submitter's reported 95% CI.

    `submitted_metrics` and `rerun_metrics` are nested dicts:
        { task_id: { metric_name: { 'mean': float, 'ci_95': [lo, hi] } } }

    `overlap_tol`: optional slack in the CI bounds (e.g. 0.005). Defaults to 0.
    """
    reasons: list[str] = []
    for task_id, metrics in submitted_metrics.items():
        if task_id not in rerun_metrics:
            reasons.append(f"{task_id}: re-run missing")
            continue
        for metric_name, submitted in metrics.items():
            rerun = rerun_metrics[task_id].get(metric_name)
            if rerun is None:
                reasons.append(f"{task_id}/{metric_name}: re-run metric missing")
                continue
            lo, hi = submitted["ci_95"]
            mean = rerun["mean"]
            if not (lo - overlap_tol) <= mean <= (hi + overlap_tol):
                reasons.append(
                    f"{task_id}/{metric_name}: re-run mean {mean:.4f} outside "
                    f"submitted CI [{lo:.4f}, {hi:.4f}]"
                )
    return VerificationOutcome(verified=len(reasons) == 0, reasons=tuple(reasons))
