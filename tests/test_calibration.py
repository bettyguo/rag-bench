"""Calibration tests: Krippendorff's α + run_calibration end-to-end on DummyJudges."""

from __future__ import annotations

from rag_bench.calibration import (
    CalibrationItem,
    krippendorff_alpha_nominal,
    run_calibration,
)
from rag_bench.judges import DummyJudge


def test_alpha_perfect_agreement_is_one():
    a = ["supported", "refuted", "neutral", "supported"]
    b = list(a)
    assert krippendorff_alpha_nominal(a, b) == 1.0


def test_alpha_below_chance_is_negative_or_low():
    # Systematic disagreement: every item disagrees
    a = ["supported"] * 4
    b = ["refuted"] * 4
    alpha = krippendorff_alpha_nominal(a, b)
    assert alpha < 0.5


def test_alpha_handles_all_same_value():
    a = ["supported"] * 5
    b = ["supported"] * 5
    assert krippendorff_alpha_nominal(a, b) == 1.0


def test_run_calibration_emits_per_judge_alpha():
    judges = [
        DummyJudge(name="A", family="anthropic"),
        DummyJudge(name="O", family="openai"),
    ]
    items = [
        CalibrationItem(
            item_id="i1",
            claim="Paris is in France",
            context="Paris is the capital of France.",
            gold_verdict="supported",
            task_family="single-hop",
        ),
        CalibrationItem(
            item_id="i2",
            claim="NOT-Paris in France",
            context="Paris is the capital of France.",
            gold_verdict="refuted",
            task_family="single-hop",
        ),
        CalibrationItem(
            item_id="i3",
            claim="Bananas grow underwater",
            context="Bananas are yellow.",
            gold_verdict="neutral",
            task_family="single-hop",
        ),
    ]
    report = run_calibration(judges, items)
    assert set(report.overall_alpha.keys()) == {"A", "O"}
    # DummyJudge is deterministic and aligned with these items' gold verdicts
    for n in ("A", "O"):
        assert report.overall_alpha[n] == 1.0
    assert "single-hop" in report.per_family_alpha
    assert report.n_items == 3
    js = report.to_jsonable()
    import json

    json.dumps(js)  # must round-trip
