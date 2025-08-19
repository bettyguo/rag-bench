"""Faithfulness metric tests — use DummyJudge to stay offline."""

from __future__ import annotations

from rag_bench.judges import DummyJudge, JudgeVerdict
from rag_bench.metrics.faithfulness import (
    Faithfulness,
    extract_atomic_claims,
    majority_verdict,
)
from rag_bench.types import (
    Chunk,
    GenerationResult,
    PipelineResult,
    Query,
    RetrievalResult,
    TaskItem,
)


def _result(answer: str, contexts: list[str], *, generator_family: str | None = None) -> PipelineResult:
    chunks = [Chunk(chunk_id=f"c{i}", doc_id=f"d{i}", text=t, position=0) for i, t in enumerate(contexts)]
    rrs = [RetrievalResult(chunk=c, score=1.0, rank=i) for i, c in enumerate(chunks)]
    gen = GenerationResult(text=answer, metadata={"generator": generator_family} if generator_family else {})
    return PipelineResult(
        query=Query(query_id="q", text="?"),
        retrieved=rrs,
        reranked=rrs,
        generation=gen,
        pipeline_name="t",
    )


def _item() -> TaskItem:
    return TaskItem(
        task_id="t",
        item_id="i",
        query=Query(query_id="i", text="?"),
        gold_answers=["irrelevant for faithfulness"],
    )


def test_atomic_claims_splits_on_sentences():
    text = "Paris is the capital of France. The Eiffel Tower is in Paris. It opened in 1889."
    claims = extract_atomic_claims(text)
    assert len(claims) == 3
    assert claims[0].startswith("Paris")


def test_atomic_claims_single_sentence():
    text = "Paris is the capital of France"
    claims = extract_atomic_claims(text)
    assert claims == ["Paris is the capital of France"]


def test_atomic_claims_drops_empty():
    assert extract_atomic_claims("") == []
    assert extract_atomic_claims("...") == []


def test_majority_verdict_supported_when_majority():
    assert majority_verdict(["supported", "supported", "neutral"]) == "supported"


def test_majority_verdict_neutral_on_tie():
    # No "supported" majority → falls back to neutral / refuted
    assert majority_verdict(["supported", "neutral", "refuted"]) == "neutral"


def test_majority_verdict_refuted_when_refuted_majority():
    assert majority_verdict(["refuted", "refuted", "supported"]) == "refuted"


def test_faithfulness_perfect_when_context_supports_answer():
    judges = [DummyJudge(name="A"), DummyJudge(name="B"), DummyJudge(name="C")]
    metric = Faithfulness(judges)
    result = _result(
        answer="Paris is the capital of France.",
        contexts=["Paris is the capital of France."],
    )
    score = metric.score_one(result, _item())
    assert score == 1.0


def test_faithfulness_zero_when_context_unrelated():
    judges = [DummyJudge(name="A"), DummyJudge(name="B"), DummyJudge(name="C")]
    metric = Faithfulness(judges, randomize_position=False)
    result = _result(
        answer="The Eiffel Tower opened in 1889.",
        contexts=["Bananas are yellow fruits."],
    )
    score = metric.score_one(result, _item())
    assert score == 0.0


def test_faithfulness_partial_credit_mixed_claims():
    judges = [DummyJudge(name="A"), DummyJudge(name="B"), DummyJudge(name="C")]
    metric = Faithfulness(judges, randomize_position=False)
    result = _result(
        answer="Paris is the capital of France. The Eiffel Tower opened in 1889.",
        contexts=["Paris is the capital of France. Bananas are yellow."],
    )
    score = metric.score_one(result, _item())
    assert 0.0 < score < 1.0


def test_faithfulness_self_enhancement_guard_drops_same_family_judge():
    # 2 of 3 judges say supported; the dropped one disagrees. Without the guard,
    # majority is supported. With the guard (and the disagreeing judge dropped),
    # we still get supported (correct). The point is: ensure the drop happens
    # without breaking the metric.
    class AlwaysRefutes(DummyJudge):
        def __init__(self):
            super().__init__(name="A-refuter", family="anthropic")

        def judge_claim(self, claim, context):
            return JudgeVerdict(verdict="refuted")

    judges = [
        AlwaysRefutes(),
        DummyJudge(name="O", family="openai"),
        DummyJudge(name="W", family="openweight"),
    ]
    metric = Faithfulness(judges, drop_self_family=True, randomize_position=False)
    # Generator family matches the AlwaysRefutes judge → it gets dropped
    result = _result(
        answer="Paris capital France",
        contexts=["Paris is the capital of France"],
        generator_family="anthropic",
    )
    score = metric.score_one(result, _item())
    # Remaining 2 DummyJudges support the claim → 1.0
    assert score == 1.0


def test_faithfulness_returns_none_with_no_context():
    metric = Faithfulness([DummyJudge()])
    result = _result(answer="Anything", contexts=[])
    assert metric.score_one(result, _item()) is None


def test_faithfulness_ensemble_fingerprint_stable():
    judges = [DummyJudge(name="A"), DummyJudge(name="B")]
    metric = Faithfulness(judges)
    fp = metric.ensemble_fingerprint()
    assert len(fp) == 2
    assert fp[0]["name"] == "A"
    assert fp[0]["family"] == "openweight"


# Tests for the self-enhancement guard: it must read `family` explicitly.


def _result_with_family(answer: str, contexts: list[str], *, family: str | None) -> PipelineResult:
    """Build a PipelineResult with the family set explicitly via
    generation.metadata['family'] rather than the legacy 'generator' key.
    """
    chunks = [Chunk(chunk_id=f"c{i}", doc_id=f"d{i}", text=t, position=0) for i, t in enumerate(contexts)]
    rrs = [RetrievalResult(chunk=c, score=1.0, rank=i) for i, c in enumerate(chunks)]
    meta: dict[str, str] = {}
    if family is not None:
        meta["family"] = family
    gen = GenerationResult(text=answer, metadata=meta)
    return PipelineResult(
        query=Query(query_id="q", text="?"),
        retrieved=rrs,
        reranked=rrs,
        generation=gen,
        pipeline_name="t",
    )


def test_faithfulness_guard_uses_explicit_family_metadata():
    class AlwaysRefutes(DummyJudge):
        def __init__(self):
            super().__init__(name="A-refuter", family="anthropic")

        def judge_claim(self, claim, context):
            return JudgeVerdict(verdict="refuted")

    judges = [
        AlwaysRefutes(),
        DummyJudge(name="O", family="openai"),
        DummyJudge(name="W", family="openweight"),
    ]
    metric = Faithfulness(judges, drop_self_family=True, randomize_position=False)
    result = _result_with_family(
        answer="Paris capital France",
        contexts=["Paris is the capital of France"],
        family="anthropic",
    )
    score = metric.score_one(result, _item())
    # Two remaining DummyJudges support the claim; the stricter test below
    # uses 2 same-family judges so a broken guard would actually flip the
    # majority verdict.
    assert score == 1.0


def test_faithfulness_guard_with_two_same_family_judges_drops_both():
    # When 2 of 3 judges share the generator's family, both must be dropped.
    # Without the guard, those two outvote the lone different-family judge.
    class AlwaysRefutes(DummyJudge):
        def __init__(self, name: str):
            super().__init__(name=name, family="anthropic")

        def judge_claim(self, claim, context):
            return JudgeVerdict(verdict="refuted")

    judges = [
        AlwaysRefutes("A1"),
        AlwaysRefutes("A2"),
        DummyJudge(name="W", family="openweight"),
    ]
    metric = Faithfulness(judges, drop_self_family=True, randomize_position=False)
    result = _result_with_family(
        answer="Paris capital France",
        contexts=["Paris is the capital of France"],
        family="anthropic",
    )
    score = metric.score_one(result, _item())
    assert score == 1.0


def test_faithfulness_no_family_metadata_does_not_drop_anything():
    # With no family in metadata, the guard must not drop any judges.

    class AlwaysRefutes(DummyJudge):
        def __init__(self):
            super().__init__(name="A-refuter", family="anthropic")

        def judge_claim(self, claim, context):
            return JudgeVerdict(verdict="refuted")

    judges = [AlwaysRefutes(), AlwaysRefutes(), DummyJudge(name="W", family="openweight")]
    metric = Faithfulness(judges, drop_self_family=True, randomize_position=False)
    result = _result_with_family(
        answer="Paris capital France",
        contexts=["Paris is the capital of France"],
        family=None,  # no family in metadata
    )
    score = metric.score_one(result, _item())
    # All 3 judges vote; 2 refuted + 1 supported → "refuted" → 0.0
    assert score == 0.0
