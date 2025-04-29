"""Retriever components.

Three baselines:
- `bm25`    — pure-Python BM25 with Robertson-Sparck-Jones term weighting
- `dense`   — sentence-transformers + cosine-similarity (optional extra)
- `hybrid`  — RRF fusion of bm25 + dense

The pure-Python BM25 lets the test suite and smoke tests run without any
torch/transformers/sentence-transformers stack installed.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import Field, model_validator

from rag_bench.pipeline.base import ComponentConfig, Retriever, register
from rag_bench.types import Chunk, Query, RetrievalResult

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[a-z]+)?")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Config(ComponentConfig):
    type: Literal["bm25"] = "bm25"
    k1: float = Field(1.5, gt=0)
    b: float = Field(0.75, ge=0, le=1)
    top_k: int = Field(50, ge=1)


@register("retriever", "bm25")
class BM25Retriever(Retriever):
    """Standard BM25 with k1, b parameters. Pure Python; no torch."""

    def __init__(self, config: BM25Config) -> None:
        super().__init__(config)
        self.cfg: BM25Config = config
        self._chunks: list[Chunk] = []
        self._doc_lens: list[int] = []
        self._avgdl: float = 0.0
        self._df: Counter[str] = Counter()  # document frequency per term
        self._postings: dict[str, dict[int, int]] = {}  # term -> {doc_idx -> tf}
        self._N: int = 0

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._N = len(self._chunks)
        self._doc_lens = []
        self._df = Counter()
        self._postings = {}
        total_len = 0
        for idx, chunk in enumerate(self._chunks):
            toks = _tokenize(chunk.text)
            self._doc_lens.append(len(toks))
            total_len += len(toks)
            tf = Counter(toks)
            for term, count in tf.items():
                self._postings.setdefault(term, {})[idx] = count
                self._df[term] += 1
        self._avgdl = total_len / self._N if self._N else 0.0

    def retrieve(self, query: Query, top_k: int) -> list[RetrievalResult]:
        q_tokens = _tokenize(query.text)
        scores: dict[int, float] = {}
        k1, b = self.cfg.k1, self.cfg.b
        for term in q_tokens:
            posting = self._postings.get(term)
            if not posting:
                continue
            df = self._df[term]
            idf = math.log(1 + (self._N - df + 0.5) / (df + 0.5))
            for idx, tf in posting.items():
                dl = self._doc_lens[idx] or 1
                norm = 1 - b + b * dl / self._avgdl if self._avgdl else 1.0
                score = idf * (tf * (k1 + 1)) / (tf + k1 * norm)
                scores[idx] = scores.get(idx, 0.0) + score
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [
            RetrievalResult(chunk=self._chunks[idx], score=float(score), rank=r)
            for r, (idx, score) in enumerate(ranked)
        ]


class DenseConfig(ComponentConfig):
    type: Literal["dense"] = "dense"
    model: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = Field(32, ge=1)
    normalize: bool = True
    top_k: int = Field(50, ge=1)


@register("retriever", "dense")
class DenseRetriever(Retriever):
    """Sentence-transformers + numpy cosine. Imports torch/transformers lazily."""

    def __init__(self, config: DenseConfig) -> None:
        super().__init__(config)
        self.cfg: DenseConfig = config
        self._chunks: list[Chunk] = []
        self._embeddings: Any | None = None  # np.ndarray, dim (N, D)
        self._encoder = None  # lazy

    def _get_encoder(self):
        if self._encoder is not None:
            return self._encoder
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - exercised in integration tests only
            raise ImportError(
                "DenseRetriever requires `pip install rag-bench[retrievers]` "
                "(installs sentence-transformers)."
            ) from e
        self._encoder = SentenceTransformer(self.cfg.model)
        return self._encoder

    def index(self, chunks: Sequence[Chunk]) -> None:
        import numpy as np

        self._chunks = list(chunks)
        if not self._chunks:
            self._embeddings = np.zeros((0, 0), dtype=np.float32)
            return
        enc = self._get_encoder()
        embs = enc.encode(
            [c.text for c in self._chunks],
            batch_size=self.cfg.batch_size,
            normalize_embeddings=self.cfg.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        self._embeddings = embs.astype(np.float32)

    def retrieve(self, query: Query, top_k: int) -> list[RetrievalResult]:
        import numpy as np

        if self._embeddings is None or len(self._chunks) == 0:
            return []
        enc = self._get_encoder()
        q = enc.encode(
            [query.text],
            normalize_embeddings=self.cfg.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        scores = (self._embeddings @ q.T).reshape(-1)
        idx = np.argsort(-scores)[:top_k]
        return [
            RetrievalResult(chunk=self._chunks[int(i)], score=float(scores[i]), rank=r)
            for r, i in enumerate(idx)
        ]


class HybridConfig(ComponentConfig):
    type: Literal["hybrid"] = "hybrid"
    sparse: dict = Field(default_factory=lambda: {"type": "bm25"})
    dense: dict = Field(default_factory=lambda: {"type": "dense"})
    fusion: Literal["rrf"] = "rrf"
    rrf_k: int = Field(60, ge=1)
    top_k: int = Field(50, ge=1)

    @model_validator(mode="after")
    def _validate_subtypes(self):
        if self.sparse.get("type") != "bm25":
            raise ValueError("hybrid.sparse.type must be 'bm25' (v1 supports BM25 only).")
        if self.dense.get("type") != "dense":
            raise ValueError("hybrid.dense.type must be 'dense' (v1 supports SBERT-style only).")
        return self


@register("retriever", "hybrid")
class HybridRetriever(Retriever):
    """RRF fusion of BM25 + Dense."""

    def __init__(self, config: HybridConfig) -> None:
        super().__init__(config)
        self.cfg: HybridConfig = config
        sparse_cfg = BM25Config(**config.sparse)
        dense_cfg = DenseConfig(**config.dense)
        self.sparse = BM25Retriever(sparse_cfg)
        self.dense = DenseRetriever(dense_cfg)

    def index(self, chunks: Sequence[Chunk]) -> None:
        self.sparse.index(chunks)
        self.dense.index(chunks)

    def retrieve(self, query: Query, top_k: int) -> list[RetrievalResult]:
        oversample = max(top_k * 3, self.cfg.top_k)
        sparse_hits = self.sparse.retrieve(query, oversample)
        dense_hits = self.dense.retrieve(query, oversample)
        fused: dict[str, dict] = {}
        for hit in sparse_hits:
            fused.setdefault(hit.chunk.chunk_id, {"chunk": hit.chunk, "score": 0.0})
            fused[hit.chunk.chunk_id]["score"] += 1.0 / (self.cfg.rrf_k + hit.rank + 1)
        for hit in dense_hits:
            fused.setdefault(hit.chunk.chunk_id, {"chunk": hit.chunk, "score": 0.0})
            fused[hit.chunk.chunk_id]["score"] += 1.0 / (self.cfg.rrf_k + hit.rank + 1)
        ranked = sorted(fused.values(), key=lambda d: d["score"], reverse=True)[:top_k]
        return [
            RetrievalResult(chunk=r["chunk"], score=float(r["score"]), rank=i)
            for i, r in enumerate(ranked)
        ]
