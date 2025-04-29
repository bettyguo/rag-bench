"""Pricing-table tests for C-4 — API generators must report cost > 0."""

from __future__ import annotations

import pytest

from rag_bench.pricing import (
    UNKNOWN_MODEL_RATE,
    estimate_cost_usd,
    lookup_rate,
)


def test_lookup_known_anthropic_model_returns_real_rate():
    rate = lookup_rate("claude-haiku-4-5")
    assert rate.input_per_million_usd > 0
    assert rate.output_per_million_usd > 0


def test_lookup_known_openai_model_returns_real_rate():
    rate = lookup_rate("gpt-4o-mini")
    assert rate.input_per_million_usd > 0
    assert rate.output_per_million_usd > 0


def test_lookup_unknown_model_returns_zero_rate():
    """Audit C-4: unknown models fall back to zero rather than guessing; we
    flag the unknown in metadata at the caller."""
    rate = lookup_rate("not-a-real-model")
    assert rate == UNKNOWN_MODEL_RATE
    assert rate.input_per_million_usd == 0.0
    assert rate.output_per_million_usd == 0.0


def test_estimate_cost_simple():
    # claude-haiku-4-5 at $1.00/M in, $5.00/M out per (synthetic) price table
    # exact numbers: prompt=1000, completion=500
    # cost = (1000 * in) + (500 * out), divided by 1e6
    cost = estimate_cost_usd("claude-haiku-4-5", prompt_tokens=1000, completion_tokens=500)
    rate = lookup_rate("claude-haiku-4-5")
    expected = (1000 * rate.input_per_million_usd + 500 * rate.output_per_million_usd) / 1_000_000
    assert cost == pytest.approx(expected)
    assert cost > 0


def test_estimate_cost_unknown_model_returns_zero():
    assert estimate_cost_usd("not-a-real-model", prompt_tokens=10_000, completion_tokens=5_000) == 0.0


def test_estimate_cost_zero_tokens():
    assert estimate_cost_usd("claude-haiku-4-5", prompt_tokens=0, completion_tokens=0) == 0.0


def test_estimate_cost_negative_tokens_clamps_to_zero():
    """Defensive: negative token counts shouldn't produce negative dollars."""
    assert estimate_cost_usd("claude-haiku-4-5", prompt_tokens=-100, completion_tokens=50) >= 0.0


def test_anthropic_generator_returns_nonzero_cost_for_known_model(monkeypatch):
    """The integration test for C-4: AnthropicGenerator.generate computes cost > 0
    when the model is in the price table and usage tokens are non-zero."""
    from rag_bench.pipeline.components.generators import AnthropicConfig, AnthropicGenerator
    from rag_bench.types import Query

    # Stub out the anthropic client entirely. We don't want a network dep.
    class _StubBlock:
        type = "text"
        text = "stubbed answer"

    class _StubUsage:
        input_tokens = 1234
        output_tokens = 56

    class _StubResp:
        def __init__(self):
            self.content = [_StubBlock()]
            self.usage = _StubUsage()

    class _StubMessages:
        def create(self, **kwargs):
            return _StubResp()

    class _StubClient:
        messages = _StubMessages()

    gen = AnthropicGenerator(AnthropicConfig(model="claude-haiku-4-5"))
    gen._client = _StubClient()
    result = gen.generate(Query(query_id="q", text="?"), [])
    assert result.text == "stubbed answer"
    assert result.prompt_tokens == 1234
    assert result.completion_tokens == 56
    assert result.cost_usd > 0, "C-4 regression: API generator emitted cost_usd=0"


def test_openai_generator_returns_nonzero_cost_for_known_model(monkeypatch):
    from rag_bench.pipeline.components.generators import OpenAIConfig, OpenAIGenerator
    from rag_bench.types import Query

    class _StubMsg:
        content = "stubbed answer"

    class _StubChoice:
        message = _StubMsg()

    class _StubUsage:
        prompt_tokens = 2000
        completion_tokens = 200

    class _StubResp:
        def __init__(self):
            self.choices = [_StubChoice()]
            self.usage = _StubUsage()

    class _StubCompletions:
        def create(self, **kwargs):
            return _StubResp()

    class _StubChat:
        completions = _StubCompletions()

    class _StubClient:
        chat = _StubChat()

    gen = OpenAIGenerator(OpenAIConfig(model="gpt-4o-mini"))
    gen._client = _StubClient()
    result = gen.generate(Query(query_id="q", text="?"), [])
    assert result.cost_usd > 0, "C-4 regression: OpenAI generator emitted cost_usd=0"


def test_anthropic_generator_unknown_model_emits_zero_cost_with_flag():
    """Unknown model → cost=0, but metadata carries `unknown_model_rate: True`
    so the leaderboard can flag the entry."""
    from rag_bench.pipeline.components.generators import AnthropicConfig, AnthropicGenerator
    from rag_bench.types import Query

    class _StubBlock:
        type = "text"
        text = "x"

    class _StubUsage:
        input_tokens = 100
        output_tokens = 10

    class _StubResp:
        def __init__(self):
            self.content = [_StubBlock()]
            self.usage = _StubUsage()

    class _StubMessages:
        def create(self, **kwargs):
            return _StubResp()

    class _StubClient:
        messages = _StubMessages()

    gen = AnthropicGenerator(AnthropicConfig(model="claude-quantum-9000"))
    gen._client = _StubClient()
    result = gen.generate(Query(query_id="q", text="?"), [])
    assert result.cost_usd == 0.0
    assert result.metadata.get("unknown_model_rate") is True
