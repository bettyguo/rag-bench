"""Tests for generation metrics + the SQuAD-style normalizer."""

from __future__ import annotations

from rag_bench.metrics.generation import (
    ExactMatch,
    LengthRatio,
    TokenF1,
    normalize_answer,
)
from rag_bench.types import GenerationResult, PipelineResult, Query, TaskItem


def _result(text: str) -> PipelineResult:
    return PipelineResult(
        query=Query(query_id="q", text="?"),
        retrieved=[],
        reranked=[],
        generation=GenerationResult(text=text),
        pipeline_name="t",
    )


def _item(golds: list[str]) -> TaskItem:
    return TaskItem(
        task_id="t",
        item_id="i",
        query=Query(query_id="i", text="?"),
        gold_answers=golds,
    )


def test_normalizer_strips_articles_and_punctuation():
    assert normalize_answer("The   Eiffel Tower!") == "eiffel tower"
    assert normalize_answer("A banana.") == "banana"


def test_em_paraphrase_zero_pair_with_f1():
    em = ExactMatch().score_one(_result("JFK"), _item(["John F. Kennedy"]))
    f1 = TokenF1().score_one(_result("JFK"), _item(["John F. Kennedy"]))
    assert em == 0.0
    # F1 also low because no token overlap after normalization
    assert f1 == 0.0


def test_em_takes_max_over_golds():
    em = ExactMatch().score_one(_result("Paris"), _item(["Paris, France", "Paris"]))
    assert em == 1.0


def test_token_f1_partial_overlap():
    f1 = TokenF1().score_one(
        _result("Mount Everest"),
        _item(["Mount Everest is the highest mountain"]),
    )
    # tokens overlap = 2/2 for pred, 2/5 for gold → P=1, R=0.4, F1 ≈ 0.571
    assert 0.55 < f1 < 0.60


def test_token_f1_empty_predicted():
    f1 = TokenF1().score_one(_result(""), _item(["a real answer"]))
    assert f1 == 0.0


def test_length_ratio_flag_pipeline():
    item = _item(["Paris"])
    short = LengthRatio().score_one(_result("Paris"), item)
    long = LengthRatio().score_one(_result("Paris " * 50), item)
    assert short == 1.0
    assert long >= 40
