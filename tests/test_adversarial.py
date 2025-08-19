"""Adversarial-track tests: metrics + tasks."""

from __future__ import annotations

from rag_bench.metrics.adversarial import (
    AbstentionPrecision,
    AbstentionRecall,
    ImplausibleResistance,
    NegativeRejectionRate,
    PlausibleCompliance,
)
from rag_bench.metrics.retrieval import RecallAtK
from rag_bench.pipeline.compose import compose_from_yaml
from rag_bench.runner import Runner
from rag_bench.tasks.counterfactual_qa import CounterfactualQATask
from rag_bench.tasks.noisy_qa import NoisyQATask
from rag_bench.tasks.unanswerable_qa import UnanswerableQATask
from rag_bench.types import (
    GenerationResult,
    PipelineResult,
    Query,
    TaskItem,
)


def _result(text: str) -> PipelineResult:
    return PipelineResult(
        query=Query(query_id="q", text="?"),
        retrieved=[],
        reranked=[],
        generation=GenerationResult(text=text),
        pipeline_name="t",
    )


def _item(metadata: dict, golds: list[str] | None = None) -> TaskItem:
    return TaskItem(
        task_id="t",
        item_id="i",
        query=Query(query_id="i", text="?"),
        gold_answers=golds or [],
        metadata=metadata,
    )


# --------------------------------------------------------------------------- #
# Abstention metrics                                                           #
# --------------------------------------------------------------------------- #


def test_abstention_recall_correct_on_unanswerable():
    m = AbstentionRecall()
    assert m.score_one(_result("INSUFFICIENT_CONTEXT"), _item({"answerable": False})) == 1.0
    assert m.score_one(_result("a guess"), _item({"answerable": False})) == 0.0


def test_abstention_recall_returns_none_on_answerable():
    m = AbstentionRecall()
    assert m.score_one(_result("INSUFFICIENT_CONTEXT"), _item({"answerable": True})) is None


def test_abstention_precision_returns_none_when_not_abstaining():
    m = AbstentionPrecision()
    assert m.score_one(_result("a guess"), _item({"answerable": False})) is None


def test_abstention_precision_correct_when_abstaining_on_unanswerable():
    m = AbstentionPrecision()
    assert m.score_one(_result("insufficient_context"), _item({"answerable": False})) == 1.0
    assert m.score_one(_result("insufficient_context"), _item({"answerable": True})) == 0.0


def test_nrr_correct_decision_in_both_directions():
    m = NegativeRejectionRate()
    # Abstain on unanswerable → correct
    assert m.score_one(_result("INSUFFICIENT_CONTEXT"), _item({"answerable": False})) == 1.0
    # Answer on answerable → correct
    assert m.score_one(_result("Paris"), _item({"answerable": True})) == 1.0
    # Abstain on answerable → wrong
    assert m.score_one(_result("INSUFFICIENT_CONTEXT"), _item({"answerable": True})) == 0.0
    # Answer on unanswerable → wrong
    assert m.score_one(_result("a guess"), _item({"answerable": False})) == 0.0


# --------------------------------------------------------------------------- #
# Counterfactual metrics                                                       #
# --------------------------------------------------------------------------- #


def test_plausible_compliance_credits_matching_counter():
    m = PlausibleCompliance()
    item = _item({"counter_kind": "plausible", "counter_answer": "1991"})
    assert m.score_one(_result("The wall fell in 1991."), item) == 1.0
    assert m.score_one(_result("The wall fell in 1989."), item) == 0.0


def test_plausible_compliance_skips_implausible_items():
    m = PlausibleCompliance()
    item = _item({"counter_kind": "implausible", "counter_answer": "fourteen"})
    assert m.score_one(_result("anything"), item) is None


def test_implausible_resistance_credits_parametric_answer():
    m = ImplausibleResistance()
    item = _item(
        {"counter_kind": "implausible", "counter_answer": "fourteen"},
        golds=["seven", "7"],
    )
    assert m.score_one(_result("There are seven continents."), item) == 1.0
    assert m.score_one(_result("There are fourteen continents."), item) == 0.0


def test_implausible_resistance_zero_when_neither_match():
    m = ImplausibleResistance()
    item = _item(
        {"counter_kind": "implausible", "counter_answer": "fourteen"},
        golds=["seven"],
    )
    assert m.score_one(_result("I don't know."), item) == 0.0


# --------------------------------------------------------------------------- #
# Adversarial tasks                                                            #
# --------------------------------------------------------------------------- #


def test_noisy_qa_corpus_has_gold_plus_distractors():
    task = NoisyQATask()
    docs = list(task.corpus())
    items = list(task.items())
    # 1 gold + 3 distractors per item.
    assert len(items) >= 5
    assert len(docs) == len(items) * 4


def test_unanswerable_qa_carries_answerable_flag():
    task = UnanswerableQATask()
    items = list(task.items())
    unanswerable = [i for i in items if not i.metadata["answerable"]]
    answerable = [i for i in items if i.metadata["answerable"]]
    assert len(unanswerable) >= 1
    assert len(answerable) >= 1


def test_counterfactual_qa_carries_kind_and_counter():
    task = CounterfactualQATask()
    items = list(task.items())
    plaus = [i for i in items if i.metadata["counter_kind"] == "plausible"]
    impl = [i for i in items if i.metadata["counter_kind"] == "implausible"]
    assert plaus and impl
    for i in items:
        assert i.metadata["counter_answer"]


# --------------------------------------------------------------------------- #
# Integration: BM25-echo on the noisy seed set                                 #
# --------------------------------------------------------------------------- #


PIPE_YAML = """
pipeline:
  name: bm25-echo-adversarial
  retriever_top_k: 4
  reranker_top_k: 3
  chunker: { type: recursive, chunk_size: 500, overlap: 50 }
  retriever: { type: bm25 }
  reranker: { type: identity, top_k: 3 }
  generator: { type: echo }
"""


def test_bm25_recall_on_noisy_qa_above_floor():
    pipe = compose_from_yaml(PIPE_YAML)
    task = NoisyQATask()
    runner = Runner(pipe, seeds=(0,), split="all")
    rec = runner.run_task(task, [RecallAtK(k=4)])
    # BM25 with explicit lexical distractors is hard but should beat 0
    assert rec.metrics["recall@4"]["mean"] >= 0.4
