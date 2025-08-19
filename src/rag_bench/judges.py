"""LLM-judge abstraction used by faithfulness scoring.

The faithfulness metric calls N=3 cross-vendor judges and majority-votes.
Each judge implements `judge_claim(claim, context) -> Verdict`. Judges are
lazy: API SDKs are imported only when the judge is actually used.

A `DummyJudge` is provided for testing: it returns deterministic verdicts
based on substring matching. Tests use DummyJudge so they run without API keys.
"""

from __future__ import annotations

import abc
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from rag_bench.types import RetrievalResult

Verdict = Literal["supported", "refuted", "neutral"]


@dataclass(frozen=True)
class JudgeVerdict:
    """One judge's verdict on one (claim, context) pair."""

    verdict: Verdict
    confidence: float = 1.0
    rationale: str = ""


@dataclass(frozen=True)
class JudgeIdentity:
    """Fingerprint of a judge model. Goes into the pipeline_hash."""

    name: str  # e.g. "anthropic-A", "openai-B", "openweight-C"
    family: str  # "anthropic" | "openai" | "openweight"
    model: str
    prompt_hash: str  # sha256 of the judge prompt template
    temperature: float = 0.0


class Judge(abc.ABC):
    identity: JudgeIdentity

    @abc.abstractmethod
    def judge_claim(self, claim: str, context: str) -> JudgeVerdict: ...


class DummyJudge(Judge):
    """Deterministic judge for tests.

    Verdict rule:
        - "supported"  if every claim token appears in the context
        - "refuted"    if claim has a "NOT-" prefix
        - "neutral"    otherwise

    Cheaply discriminating; lets us write faithfulness tests without APIs.
    """

    def __init__(
        self,
        *,
        name: str = "dummy",
        family: str = "openweight",
        model: str = "dummy-v1",
    ) -> None:
        self.identity = JudgeIdentity(
            name=name, family=family, model=model, prompt_hash="sha256:dummy"
        )

    def judge_claim(self, claim: str, context: str) -> JudgeVerdict:
        if claim.startswith("NOT-"):
            return JudgeVerdict(verdict="refuted")
        ctx_lower = context.lower()
        tokens = [t.lower() for t in claim.replace(",", " ").split() if len(t) > 2]
        if tokens and all(t in ctx_lower for t in tokens):
            return JudgeVerdict(verdict="supported")
        return JudgeVerdict(verdict="neutral")


CLAIM_PROMPT = """\
You are checking whether a single atomic claim is entailed by a provided context.

CONTEXT:
{context}

CLAIM:
{claim}

Respond with EXACTLY ONE WORD from: supported, refuted, neutral
- "supported" if the context entails the claim.
- "refuted"   if the context contradicts the claim.
- "neutral"   if the context neither supports nor refutes the claim.

VERDICT:"""

CLAIM_PROMPT_HASH = "sha256:" + __import__("hashlib").sha256(CLAIM_PROMPT.encode()).hexdigest()


def _parse_verdict(text: str) -> Verdict:
    t = text.strip().lower().split()
    if not t:
        return "neutral"
    word = t[0].strip(".,")
    if word.startswith("support"):
        return "supported"
    if word.startswith("refut"):
        return "refuted"
    return "neutral"


class AnthropicJudge(Judge):
    """Faithfulness judge backed by an Anthropic model."""

    def __init__(self, model: str = "claude-haiku-4-5", *, name: str = "anthropic-A") -> None:
        self.identity = JudgeIdentity(
            name=name,
            family="anthropic",
            model=model,
            prompt_hash=CLAIM_PROMPT_HASH,
        )
        self._client: Any = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover
                raise ImportError("AnthropicJudge requires `pip install anthropic`.") from e
            self._client = anthropic.Anthropic()
        return self._client

    def judge_claim(self, claim: str, context: str) -> JudgeVerdict:
        client = self._get_client()
        resp = client.messages.create(
            model=self.identity.model,
            max_tokens=8,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": CLAIM_PROMPT.format(context=context, claim=claim),
                }
            ],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return JudgeVerdict(verdict=_parse_verdict(text), rationale=text.strip())


class OpenAIJudge(Judge):
    """Faithfulness judge backed by an OpenAI Chat Completions model."""

    def __init__(self, model: str = "gpt-4o-mini", *, name: str = "openai-B") -> None:
        self.identity = JudgeIdentity(
            name=name,
            family="openai",
            model=model,
            prompt_hash=CLAIM_PROMPT_HASH,
        )
        self._client: Any = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError as e:  # pragma: no cover
                raise ImportError("OpenAIJudge requires `pip install openai`.") from e
            self._client = openai.OpenAI()
        return self._client

    def judge_claim(self, claim: str, context: str) -> JudgeVerdict:
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.identity.model,
            max_tokens=8,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": CLAIM_PROMPT.format(context=context, claim=claim),
                }
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        return JudgeVerdict(verdict=_parse_verdict(text), rationale=text)


def randomized_context(retrieved: Sequence[RetrievalResult], *, seed: int) -> str:
    """Concatenate retrieved chunk text in a shuffled order for position-randomization."""
    rng = random.Random(seed)
    items = list(retrieved)
    rng.shuffle(items)
    return "\n\n".join(f"[{i + 1}] {h.chunk.text}" for i, h in enumerate(items))
