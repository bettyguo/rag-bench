"""Metric base classes and bootstrap CI machinery.

Every metric is a pure function over (PipelineResult, TaskItem) → float (or a
small struct). The runner collects per-query scores; this module computes
mean + 95% bootstrap CIs for aggregation.

Bootstrap details: BCa for metrics whose distribution is materially skewed
(faithfulness fractions, ratios); plain percentile for the others. 10,000
resamples is the default; runtime is negligible for the typical 200–1000
items per task.
"""

from __future__ import annotations

import abc
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from rag_bench.types import PipelineResult, TaskItem


@dataclass(frozen=True)
class MetricResult:
    """Aggregated metric across a task."""

    name: str
    mean: float
    ci_95: tuple[float, float]
    n: int
    per_query: tuple[float, ...]

    def to_jsonable(self) -> dict[str, object]:
        return {
            "name": self.name,
            "mean": self.mean,
            "ci_95": list(self.ci_95),
            "n": self.n,
        }


class Metric(abc.ABC):
    """Per-query scorer. Override `score_one`; aggregation handled by base."""

    name: str

    @abc.abstractmethod
    def score_one(self, result: PipelineResult, item: TaskItem) -> float | None: ...

    def aggregate(
        self,
        results: Sequence[PipelineResult],
        items: Sequence[TaskItem],
        *,
        n_bootstrap: int = 10_000,
        seed: int = 0,
        method: Literal["percentile", "bca"] = "percentile",
    ) -> MetricResult:
        if len(results) != len(items):
            raise ValueError(
                f"results ({len(results)}) and items ({len(items)}) length mismatch"
            )
        scores: list[float] = []
        for r, i in zip(results, items, strict=True):
            s = self.score_one(r, i)
            if s is not None:
                scores.append(float(s))
        if not scores:
            return MetricResult(name=self.name, mean=float("nan"), ci_95=(float("nan"), float("nan")), n=0, per_query=())
        arr = np.asarray(scores, dtype=np.float64)
        mean = float(arr.mean())
        ci = bootstrap_ci(arr, n_bootstrap=n_bootstrap, seed=seed, method=method)
        return MetricResult(name=self.name, mean=mean, ci_95=ci, n=len(scores), per_query=tuple(scores))


def bootstrap_ci(
    samples: np.ndarray,
    *,
    n_bootstrap: int = 10_000,
    seed: int = 0,
    method: Literal["percentile", "bca"] = "percentile",
    confidence: float = 0.95,
) -> tuple[float, float]:
    """95% bootstrap CI of the mean of `samples`."""
    if samples.size == 0:
        return (float("nan"), float("nan"))
    if samples.size == 1:
        return (float(samples[0]), float(samples[0]))
    rng = np.random.default_rng(seed)
    n = samples.size
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = samples[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo_pct, hi_pct = alpha * 100, (1.0 - alpha) * 100
    if method == "percentile":
        lo = float(np.percentile(boot_means, lo_pct))
        hi = float(np.percentile(boot_means, hi_pct))
        return (lo, hi)
    if method == "bca":
        # BCa: bias correction + acceleration via jackknife.
        observed_mean = samples.mean()
        z0 = _phi_inv(float((boot_means < observed_mean).mean()))
        jackknife = np.array([np.delete(samples, i).mean() for i in range(n)])
        jk_mean = jackknife.mean()
        num = ((jk_mean - jackknife) ** 3).sum()
        den = 6.0 * (((jk_mean - jackknife) ** 2).sum() ** 1.5)
        a = float(num / den) if den != 0 else 0.0
        z_alpha_lo = _phi_inv(alpha)
        z_alpha_hi = _phi_inv(1.0 - alpha)

        def adj(z):
            return _phi(z0 + (z0 + z) / (1.0 - a * (z0 + z)))

        lo = float(np.percentile(boot_means, adj(z_alpha_lo) * 100))
        hi = float(np.percentile(boot_means, adj(z_alpha_hi) * 100))
        return (lo, hi)
    raise ValueError(f"Unknown bootstrap method: {method!r}")


def paired_bootstrap_p_value(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    *,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> float:
    """Two-sided paired bootstrap p-value for H0: E[a - b] = 0."""
    if samples_a.shape != samples_b.shape:
        raise ValueError("Paired bootstrap requires equal-length sample arrays.")
    diffs = samples_a - samples_b
    rng = np.random.default_rng(seed)
    n = diffs.size
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_diff_means = diffs[idx].mean(axis=1)
    # two-sided: min of the two one-sided tails, doubled.
    p = float(2.0 * min((boot_diff_means >= 0).mean(), (boot_diff_means <= 0).mean()))
    # bound to [0, 1] in case the doubling crosses 1 (e.g., for exact-zero diffs).
    return max(0.0, min(1.0, p))


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _phi_inv(p: float) -> float:
    # Acklam's approximation; sufficient for bootstrap quantile mapping.
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239]
    b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
         6.680131188771972e1, -1.328068155288572e1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
         3.754408661907416]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )
