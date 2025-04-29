"""Per-token pricing table for API-backed generators and judges.

Rates are list prices in USD per 1M tokens. Bump `PRICING_TABLE_VERSION`
on update; leaderboard regeneration is manual.

Self-hosted models (e.g. via vLLM) return zero; we don't model compute
or electricity here.

Sources: anthropic.com/pricing, openai.com/api/pricing, ai.google.dev/pricing.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICING_TABLE_VERSION = "2026-05"


@dataclass(frozen=True)
class ModelRate:
    """USD per 1M tokens, separately for input and output."""

    input_per_million_usd: float
    output_per_million_usd: float


# Unknown-model sentinel; callers set a metadata flag when this is hit.
UNKNOWN_MODEL_RATE = ModelRate(input_per_million_usd=0.0, output_per_million_usd=0.0)


PRICING_TABLE: dict[str, ModelRate] = {
    # Anthropic
    "claude-haiku-4-5":  ModelRate(1.00,  5.00),
    "claude-sonnet-4-6": ModelRate(3.00, 15.00),
    "claude-opus-4-7":   ModelRate(15.00, 75.00),
    # OpenAI
    "gpt-4o-mini":       ModelRate(0.15,  0.60),
    "gpt-4o":            ModelRate(2.50, 10.00),
    "gpt-5-mini":        ModelRate(0.25,  1.00),
    "gpt-5":             ModelRate(5.00, 20.00),
    # Google
    "gemini-1.5-flash":  ModelRate(0.075, 0.30),
    "gemini-1.5-pro":    ModelRate(1.25,  5.00),
    "gemini-3.1":        ModelRate(2.50, 10.00),
    # Self-hosted
    "Qwen/Qwen3.5-7B-Instruct":  ModelRate(0.0, 0.0),
    "Qwen/Qwen3.5-72B-Instruct": ModelRate(0.0, 0.0),
}


def lookup_rate(model: str) -> ModelRate:
    """Return the published rate for `model`, or UNKNOWN_MODEL_RATE."""
    return PRICING_TABLE.get(model, UNKNOWN_MODEL_RATE)


def estimate_cost_usd(model: str, *, prompt_tokens: int, completion_tokens: int) -> float:
    """Compute the USD cost of a single API call.

    Negative or NaN token counts clamp to 0. Unknown models return 0.0;
    callers should flag `unknown_model_rate=True` in result metadata so
    leaderboard renders can warn.
    """
    if prompt_tokens < 0:
        prompt_tokens = 0
    if completion_tokens < 0:
        completion_tokens = 0
    rate = lookup_rate(model)
    return (
        prompt_tokens * rate.input_per_million_usd
        + completion_tokens * rate.output_per_million_usd
    ) / 1_000_000


def is_known_model(model: str) -> bool:
    return model in PRICING_TABLE
