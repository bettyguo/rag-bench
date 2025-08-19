"""Tests for retrieval metrics."""

from __future__ import annotations

from rag_bench.metrics.retrieval import MRRAtK, NDCGAtK, RecallAtK
from rag_bench.types import (
    Chunk,
    GenerationResult,
    PipelineResult,
    Query,
    RetrievalResult,
    TaskItem,
)


def _make_result(chunk_ids: list[str]) -> PipelineResult:
    chunks = [Chunk(chunk_id=cid, doc_id=cid, text=f"text for {cid}", position=0) for cid in chunk_ids]
    hits = [RetrievalResult(chunk=c, score=10 - i, rank=i) for i, c in enumerate(chunks)]
    return PipelineResult(
        query=Query(query_id="q", text="?"),
        retrieved=hits,
        reranked=hits,
        generation=GenerationResult(text=""),
        pipeline_name="t",
    )


def _make_item(gold_passages: list[str]) -> TaskItem:
    return TaskItem(
        task_id="t",
        item_id="i",
        query=Query(query_id="i", text="?"),
        gold_answers=[],
        gold_passages=gold_passages,
    )


def test_recall_at_k_full():
    result = _make_result(["a", "b", "c"])
    item = _make_item(["a", "b"])
    assert RecallAtK(k=3).score_one(result, item) == 1.0


def test_recall_at_k_partial():
    result = _make_result(["a", "x", "y"])
    item = _make_item(["a", "b"])
    assert RecallAtK(k=3).score_one(result, item) == 0.5


def test_recall_at_k_truncates_to_k():
    result = _make_result(["x", "y", "z", "a"])
    item = _make_item(["a"])
    assert RecallAtK(k=3).score_one(result, item) == 0.0
    assert RecallAtK(k=4).score_one(result, item) == 1.0


def test_recall_returns_none_without_gold():
    result = _make_result(["a"])
    item = _make_item([])
    assert RecallAtK(k=10).score_one(result, item) is None


def test_ndcg_top_rank_full_credit():
    result = _make_result(["a", "x", "y"])
    item = _make_item(["a"])
    # gold at rank 0 → DCG = 1/log2(2) = 1; IDCG = 1; nDCG = 1
    assert NDCGAtK(k=10).score_one(result, item) == 1.0


def test_ndcg_decays_with_rank():
    result = _make_result(["x", "a", "y"])
    item = _make_item(["a"])
    score = NDCGAtK(k=10).score_one(result, item)
    # 1 / log2(3) ≈ 0.6309
    assert 0.6 < score < 0.65


def test_mrr_first_hit_full_credit():
    result = _make_result(["a", "b"])
    item = _make_item(["a"])
    assert MRRAtK(k=10).score_one(result, item) == 1.0


def test_mrr_decays_with_rank():
    result = _make_result(["x", "y", "a"])
    item = _make_item(["a"])
    assert MRRAtK(k=10).score_one(result, item) == 1 / 3


def test_mrr_zero_when_no_match():
    result = _make_result(["x", "y"])
    item = _make_item(["a"])
    assert MRRAtK(k=10).score_one(result, item) == 0.0
