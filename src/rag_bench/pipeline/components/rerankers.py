"""Reranker components.

Three baselines:
- `identity`        — passthrough; returns top-k unchanged
- `lexical-overlap` — re-ranks by query-passage Jaccard (no model; cheap baseline)
- `cross-encoder`   — sentence-transformers CrossEncoder (optional extra)
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import Field

from rag_bench.pipeline.base import ComponentConfig, Reranker, register
from rag_bench.types import Query, RetrievalResult

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(s: str) -> set[str]:
    return set(t.lower() for t in _TOKEN_RE.findall(s))


class IdentityRerankerConfig(ComponentConfig):
    type: Literal["identity"] = "identity"
    top_k: int = Field(10, ge=1)


@register("reranker", "identity")
class IdentityReranker(Reranker):
    def __init__(self, config: IdentityRerankerConfig) -> None:
        super().__init__(config)
        self.cfg: IdentityRerankerConfig = config

    def rerank(
        self,
        query: Query,
        candidates: Sequence[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        cut = candidates[:top_k]
        # rewrite ranks so consumers see contiguous 0..k-1
        return [RetrievalResult(chunk=c.chunk, score=c.score, rank=i) for i, c in enumerate(cut)]


class LexicalOverlapConfig(ComponentConfig):
    type: Literal["lexical-overlap"] = "lexical-overlap"
    top_k: int = Field(10, ge=1)


@register("reranker", "lexical-overlap")
class LexicalOverlapReranker(Reranker):
    """Re-rank candidates by Jaccard overlap with the query. A cheap, deterministic baseline."""

    def __init__(self, config: LexicalOverlapConfig) -> None:
        super().__init__(config)
        self.cfg: LexicalOverlapConfig = config

    def rerank(
        self,
        query: Query,
        candidates: Sequence[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        q = _tokens(query.text)
        scored: list[tuple[float, RetrievalResult]] = []
        for cand in candidates:
            p = _tokens(cand.chunk.text)
            if not q or not p:
                score = 0.0
            else:
                score = len(q & p) / len(q | p)
            scored.append((score, cand))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        cut = scored[:top_k]
        return [
            RetrievalResult(chunk=c.chunk, score=float(s), rank=i)
            for i, (s, c) in enumerate(cut)
        ]


class CrossEncoderConfig(ComponentConfig):
    type: Literal["cross-encoder"] = "cross-encoder"
    model: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = Field(32, ge=1)
    top_k: int = Field(10, ge=1)


@register("reranker", "cross-encoder")
class CrossEncoderReranker(Reranker):
    """Re-rank with a cross-encoder. Lazy-imports sentence-transformers."""

    def __init__(self, config: CrossEncoderConfig) -> None:
        super().__init__(config)
        self.cfg: CrossEncoderConfig = config
        self._encoder: Any = None

    def _get_encoder(self):
        if self._encoder is not None:
            return self._encoder
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "CrossEncoderReranker requires `pip install rag-bench[rerankers]`."
            ) from e
        self._encoder = CrossEncoder(self.cfg.model)
        return self._encoder

    def rerank(
        self,
        query: Query,
        candidates: Sequence[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        if not candidates:
            return []
        enc = self._get_encoder()
        pairs = [(query.text, c.chunk.text) for c in candidates]
        scores = enc.predict(pairs, batch_size=self.cfg.batch_size, show_progress_bar=False)
        scored = list(zip(scores, candidates, strict=True))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        cut = scored[:top_k]
        return [
            RetrievalResult(chunk=c.chunk, score=float(s), rank=i)
            for i, (s, c) in enumerate(cut)
        ]
