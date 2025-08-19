"""Adversarial-track metrics for the unanswerable / counterfactual / noisy tasks.

Each metric assumes its corresponding task type and inspects
`item.metadata` to decide whether to score (returning None when not
applicable, which the runner drops from aggregation).
"""

from __future__ import annotations

from rag_bench.metrics.base import Metric
from rag_bench.metrics.generation import normalize_answer
from rag_bench.types import PipelineResult, TaskItem

ABSTENTION_SENTINEL = "INSUFFICIENT_CONTEXT"
_ABSTENTION_NORMALIZED = "insufficient context"  # what normalize_answer produces


def _is_abstention(text: str) -> bool:
    return _ABSTENTION_NORMALIZED in normalize_answer(text)


class AbstentionRecall(Metric):
    """Fraction of unanswerable items where the pipeline correctly abstained."""

    name = "abstention_recall"

    def score_one(self, result: PipelineResult, item: TaskItem) -> float | None:
        if item.metadata.get("answerable") is not False:
            return None
        return 1.0 if _is_abstention(result.generation.text) else 0.0


class AbstentionPrecision(Metric):
    """Fraction of abstentions that landed on truly-unanswerable items."""

    name = "abstention_precision"

    def score_one(self, result: PipelineResult, item: TaskItem) -> float | None:
        if not _is_abstention(result.generation.text):
            return None
        return 1.0 if item.metadata.get("answerable") is False else 0.0


class NegativeRejectionRate(Metric):
    """Per-item correctness of the abstain/answer decision.

    Returns 1.0 when the pipeline abstained on an unanswerable item or
    answered an answerable one, 0.0 otherwise. The leaderboard reports the
    component recall and precision in parallel via AbstentionRecall and
    AbstentionPrecision.
    """

    name = "nrr_f1"

    def score_one(self, result: PipelineResult, item: TaskItem) -> float | None:
        answerable = item.metadata.get("answerable")
        if answerable is None:
            return None
        abstained = _is_abstention(result.generation.text)
        if not answerable and abstained:
            return 1.0
        if answerable and not abstained:
            return 1.0
        return 0.0


class PlausibleCompliance(Metric):
    """On plausible-counter items, did the pipeline trust the retrieved fact?

    Returns 1.0 iff the answer contains the planted counterfactual
    (from `item.metadata['counter_answer']`).
    """

    name = "plausible_compliance"

    def score_one(self, result: PipelineResult, item: TaskItem) -> float | None:
        if item.metadata.get("counter_kind") != "plausible":
            return None
        target = item.metadata.get("counter_answer", "")
        if not target:
            return None
        return 1.0 if normalize_answer(target) in normalize_answer(result.generation.text) else 0.0


class ImplausibleResistance(Metric):
    """On implausible-counter items, did the pipeline resist the retrieved fact?

    Returns 1.0 iff the answer aligns with parametric gold and not with
    the (internally-inconsistent) counterfactual.
    """

    name = "implausible_resistance"

    def score_one(self, result: PipelineResult, item: TaskItem) -> float | None:
        if item.metadata.get("counter_kind") != "implausible":
            return None
        counter = item.metadata.get("counter_answer", "")
        if not item.gold_answers:
            return None
        pred = normalize_answer(result.generation.text)
        any_gold_match = any(normalize_answer(g) in pred for g in item.gold_answers)
        counter_match = normalize_answer(counter) in pred if counter else False
        return 1.0 if (any_gold_match and not counter_match) else 0.0


# Post-run helpers: these don't fit the per-item Metric contract because
# they consume already-aggregated numbers across one or two task runs.


def noise_vulnerability(*, f1_clean: float, f1_noisy: float) -> float:
    """1 - F1(noisy)/F1(clean). Lower is better.

    Returns 0.0 when f1_clean <= 0 (undefined ratio); the leaderboard
    should treat such pipelines as "no usable clean F1", not "perfectly
    robust". Negative values are possible and meaningful when the noisy
    run scores higher than clean.
    """
    if f1_clean <= 0.0:
        return 0.0
    return float(1.0 - (f1_noisy / f1_clean))


def adversarial_composite(
    *,
    noise_vulnerability_score: float | None = None,
    nrr_f1: float | None = None,
    plausible_compliance: float | None = None,
    implausible_resistance: float | None = None,
) -> float:
    """Mean of the four adversarial components, with None values dropped.

    NV is flipped (1 - NV, clamped to ≥0) so higher is better for every
    component. Returns 0.0 when no components are provided.
    """
    parts: list[float] = []
    if noise_vulnerability_score is not None:
        parts.append(max(0.0, 1.0 - float(noise_vulnerability_score)))
    if nrr_f1 is not None:
        parts.append(float(nrr_f1))
    if plausible_compliance is not None:
        parts.append(float(plausible_compliance))
    if implausible_resistance is not None:
        parts.append(float(implausible_resistance))
    if not parts:
        return 0.0
    return sum(parts) / len(parts)
