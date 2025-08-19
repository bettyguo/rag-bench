"""Judge calibration.

Faithfulness scoring is only as trustworthy as the judge ensemble's agreement
with humans. CalibrationItem holds a (claim, context, gold_verdict) record;
run_calibration asks each judge for a verdict on every item and computes
Krippendorff's α between each judge and the human consensus, per task family.

Nominal-level α is what we want here (categorical verdicts).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from rag_bench.judges import Judge, Verdict


@dataclass(frozen=True)
class CalibrationItem:
    """One labeled (claim, context, gold-verdict) example.

    `gold_verdict` is the human-consensus label (majority of N annotators).
    """

    item_id: str
    claim: str
    context: str
    gold_verdict: Verdict
    task_family: str = "general"


@dataclass
class CalibrationReport:
    """Per-judge α + per-judge raw verdict counts."""

    overall_alpha: dict[str, float]  # judge_name -> α
    per_family_alpha: dict[str, dict[str, float]]  # family -> judge_name -> α
    per_judge_verdict_counts: dict[str, dict[Verdict, int]]
    n_items: int

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "n_items": self.n_items,
            "overall_alpha": self.overall_alpha,
            "per_family_alpha": self.per_family_alpha,
            "per_judge_verdict_counts": {
                k: {str(vk): vv for vk, vv in v.items()}
                for k, v in self.per_judge_verdict_counts.items()
            },
        }


def run_calibration(
    judges: Sequence[Judge],
    items: Iterable[CalibrationItem],
) -> CalibrationReport:
    """Call each judge on every item; compute per-judge α vs human consensus."""
    items = list(items)
    if not items:
        raise ValueError("Calibration requires ≥1 item.")
    if not judges:
        raise ValueError("Calibration requires ≥1 judge.")
    judge_names = [j.identity.name for j in judges]

    # judge_name -> list[(judge_verdict, gold_verdict, family)]
    rows: dict[str, list[tuple[Verdict, Verdict, str]]] = {n: [] for n in judge_names}
    counts: dict[str, Counter[Verdict]] = {n: Counter() for n in judge_names}
    for item in items:
        for judge in judges:
            jv = judge.judge_claim(item.claim, item.context).verdict
            rows[judge.identity.name].append((jv, item.gold_verdict, item.task_family))
            counts[judge.identity.name][jv] += 1

    overall_alpha = {
        n: krippendorff_alpha_nominal(
            [j for j, _g, _f in rows[n]], [g for _j, g, _f in rows[n]]
        )
        for n in judge_names
    }

    families = sorted({i.task_family for i in items})
    per_family: dict[str, dict[str, float]] = {f: {} for f in families}
    for f in families:
        for n in judge_names:
            sub = [(j, g) for j, g, fam in rows[n] if fam == f]
            if len(sub) < 2:
                per_family[f][n] = float("nan")
                continue
            per_family[f][n] = krippendorff_alpha_nominal(
                [s[0] for s in sub], [s[1] for s in sub]
            )

    return CalibrationReport(
        overall_alpha=overall_alpha,
        per_family_alpha=per_family,
        per_judge_verdict_counts={n: dict(c) for n, c in counts.items()},
        n_items=len(items),
    )


def krippendorff_alpha_nominal(rater_a: Sequence[str], rater_b: Sequence[str]) -> float:
    """Krippendorff's α for 2 raters, nominal level.

    Implementation: α = 1 - Do/De, with Do = observed disagreements (Hamming)
    and De = expected disagreements under random pairing.

    Returns 1.0 for perfect agreement, 0.0 for chance, negative for systematic
    disagreement. For α < 0.6 on a task family, the corresponding judge is
    dropped from that family (see docs/metrics.md §3.3 calibration).
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("Rater sequences must be equal length.")
    n = len(rater_a)
    if n == 0:
        return float("nan")
    if n == 1:
        return 1.0 if rater_a[0] == rater_b[0] else 0.0
    # Build the "coincidence" matrix
    values = sorted(set(rater_a) | set(rater_b))
    if len(values) < 2:
        # All ratings the same value → conventionally α = 1.0 (no disagreement possible)
        return 1.0
    # Observed disagreement: each pair contributes (1) per rater pair if disagree
    observed_pairs = 0
    total_pairs = 0
    for a, b in zip(rater_a, rater_b, strict=True):
        # 2 raters per item → 1 pair per item
        total_pairs += 1
        if a != b:
            observed_pairs += 1
    do = observed_pairs / total_pairs

    # Expected disagreement: probability a random pair differs
    flat = list(rater_a) + list(rater_b)
    counts = Counter(flat)
    N = len(flat)
    same = sum(c * (c - 1) for c in counts.values())
    de = 1.0 - same / (N * (N - 1))
    if de == 0:
        return 1.0
    return 1.0 - do / de
