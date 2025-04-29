"""Generator components.

Four baselines:
- `echo`       — returns the top-1 context chunk's text (test/CI stub; deterministic)
- `extractive` — returns the first sentence of the top-1 chunk
- `anthropic`  — Claude generator via the anthropic SDK (optional extra)
- `openai`     — OpenAI generator (optional extra)

The two API generators are lazy-imported so the harness can be exercised
without API SDKs installed.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from typing import Any, Literal

from rag_bench.pipeline.base import ComponentConfig, Generator, register
from rag_bench.types import GenerationResult, Query, RetrievalResult

DEFAULT_PROMPT_TEMPLATE = """\
You are answering a question using only the provided context.

Rules:
- If the answer is in the context, give it concisely.
- If the context does not contain the answer, respond with: INSUFFICIENT_CONTEXT
- Do not use information outside the provided context.

Context:
{context}

Question: {question}

Answer:"""


def _format_context(context: Sequence[RetrievalResult], max_chars: int = 8000) -> str:
    out: list[str] = []
    used = 0
    for i, hit in enumerate(context):
        block = f"[{i + 1}] {hit.chunk.text}"
        if used + len(block) > max_chars:
            break
        out.append(block)
        used += len(block) + 2
    return "\n\n".join(out)


class EchoConfig(ComponentConfig):
    type: Literal["echo"] = "echo"
    fallback: str = "INSUFFICIENT_CONTEXT"


@register("generator", "echo")
class EchoGenerator(Generator):
    """Returns the top-1 chunk verbatim. Deterministic; for tests and smoke runs."""

    def __init__(self, config: EchoConfig) -> None:
        super().__init__(config)
        self.cfg: EchoConfig = config

    def generate(
        self,
        query: Query,
        context: Sequence[RetrievalResult],
        *,
        seed: int = 0,
    ) -> GenerationResult:
        t0 = time.perf_counter()
        text = context[0].chunk.text if context else self.cfg.fallback
        return GenerationResult(
            text=text,
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=0.0,
            latency_ms=(time.perf_counter() - t0) * 1000,
            metadata={"generator": "echo", "family": "test"},
        )


_SENT_END = re.compile(r"(?<=[.!?])\s+")


class ExtractiveConfig(ComponentConfig):
    type: Literal["extractive"] = "extractive"
    fallback: str = "INSUFFICIENT_CONTEXT"


@register("generator", "extractive")
class ExtractiveGenerator(Generator):
    """Returns the first sentence of the top-1 chunk."""

    def __init__(self, config: ExtractiveConfig) -> None:
        super().__init__(config)
        self.cfg: ExtractiveConfig = config

    def generate(
        self,
        query: Query,
        context: Sequence[RetrievalResult],
        *,
        seed: int = 0,
    ) -> GenerationResult:
        t0 = time.perf_counter()
        if not context:
            text = self.cfg.fallback
        else:
            first = _SENT_END.split(context[0].chunk.text, maxsplit=1)
            text = first[0].strip() if first else context[0].chunk.text
        return GenerationResult(
            text=text,
            latency_ms=(time.perf_counter() - t0) * 1000,
            metadata={"generator": "extractive", "family": "test"},
        )


class AnthropicConfig(ComponentConfig):
    type: Literal["anthropic"] = "anthropic"
    model: str = "claude-haiku-4-5"
    temperature: float = 0.0
    max_tokens: int = 512
    prompt_template: str = "default"  # "default" → DEFAULT_PROMPT_TEMPLATE


@register("generator", "anthropic")
class AnthropicGenerator(Generator):
    """Claude generator. Imports anthropic lazily."""

    def __init__(self, config: AnthropicConfig) -> None:
        super().__init__(config)
        self.cfg: AnthropicConfig = config
        self._client: Any = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "AnthropicGenerator requires `pip install rag-bench[generators]`."
            ) from e
        self._client = anthropic.Anthropic()
        return self._client

    def _prompt(self, query: Query, context: Sequence[RetrievalResult]) -> str:
        template = DEFAULT_PROMPT_TEMPLATE if self.cfg.prompt_template == "default" else self.cfg.prompt_template
        return template.format(context=_format_context(context), question=query.text)

    def generate(
        self,
        query: Query,
        context: Sequence[RetrievalResult],
        *,
        seed: int = 0,
    ) -> GenerationResult:
        from rag_bench.pricing import estimate_cost_usd, is_known_model

        client = self._get_client()
        prompt = self._prompt(query, context)
        t0 = time.perf_counter()
        resp = client.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        dt = (time.perf_counter() - t0) * 1000
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "input_tokens", 0) or 0
        completion_tokens = getattr(usage, "output_tokens", 0) or 0
        cost = estimate_cost_usd(
            self.cfg.model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )
        meta: dict[str, Any] = {
            "generator": "anthropic",
            "family": "anthropic",
            "model": self.cfg.model,
        }
        if not is_known_model(self.cfg.model):
            meta["unknown_model_rate"] = True
        return GenerationResult(
            text=text.strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            latency_ms=dt,
            metadata=meta,
        )


class OpenAIConfig(ComponentConfig):
    type: Literal["openai"] = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 512
    prompt_template: str = "default"


@register("generator", "openai")
class OpenAIGenerator(Generator):
    """OpenAI Chat Completions generator. Imports openai lazily."""

    def __init__(self, config: OpenAIConfig) -> None:
        super().__init__(config)
        self.cfg: OpenAIConfig = config
        self._client: Any = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import openai
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "OpenAIGenerator requires `pip install rag-bench[generators]`."
            ) from e
        self._client = openai.OpenAI()
        return self._client

    def _prompt(self, query: Query, context: Sequence[RetrievalResult]) -> str:
        template = DEFAULT_PROMPT_TEMPLATE if self.cfg.prompt_template == "default" else self.cfg.prompt_template
        return template.format(context=_format_context(context), question=query.text)

    def generate(
        self,
        query: Query,
        context: Sequence[RetrievalResult],
        *,
        seed: int = 0,
    ) -> GenerationResult:
        from rag_bench.pricing import estimate_cost_usd, is_known_model

        client = self._get_client()
        prompt = self._prompt(query, context)
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            seed=seed,
            messages=[{"role": "user", "content": prompt}],
        )
        dt = (time.perf_counter() - t0) * 1000
        text = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        cost = estimate_cost_usd(
            self.cfg.model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )
        meta: dict[str, Any] = {
            "generator": "openai",
            "family": "openai",
            "model": self.cfg.model,
        }
        if not is_known_model(self.cfg.model):
            meta["unknown_model_rate"] = True
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            latency_ms=dt,
            metadata=meta,
        )
