"""Retrieval metrics: Recall@k, nDCG@k, MRR@k.

All metrics operate on a `PipelineResult` and a `TaskItem`. They use the
`item.gold_passages` field (a list of gold passage/chunk identifiers); if it
is None the metric is `None` for that item and the aggregator drops it.

Matching policy: a retrieved chunk is considered "gold" if any of:
  - chunk.chunk_id equals a gold id exactly
  - chunk.doc_id equals a gold id
  - chunk.text contains a gold passage as a substring (case-insensitive)

Substring matching is generous; it accommodates datasets where the gold is a
passage rather than a chunk-id. Submissions that exploit this with overly-long
chunks are also flagged via the length-ratio sanity check (see generation.py).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from rag_bench.metrics.base import Metric
from rag_bench.types import PipelineResult, RetrievalResult, TaskItem


def _is_gold(hit: RetrievalResult, golds: Sequence[str]) -> bool:
    if not golds:
        return False
    text_l = hit.chunk.text.lower()
    for g in golds:
        if g == hit.chunk.chunk_id or g == hit.chunk.doc_id:
            return True
        if len(g) >= 8 and g.lower() in text_l:
            return True
    return False


class RecallAtK(Metric):
    """Fraction of gold passages found in top-k retrieved (pre-rerank).

    Use `from_reranked=True` to score against the reranked list instead.
    """

    def __init__(self, k: int = 10, *, from_reranked: bool = False) -> None:
        self.k = k
        self.from_reranked = from_reranked
        self.name = f"recall@{k}" + ("_reranked" if from_reranked else "")

    def score_one(self, result: PipelineResult, item: TaskItem) -> float | None:
        if not item.gold_passages:
            return None
        hits = (result.reranked if self.from_reranked else result.retrieved)[: self.k]
        found = 0
        seen_golds: set[str] = set()
        for g in item.gold_passages:
            for h in hits:
                if _is_gold(h, [g]) and g not in seen_golds:
                    found += 1
                    seen_golds.add(g)
                    break
        return found / len(item.gold_passages)


class NDCGAtK(Metric):
    """Binary-relevance nDCG@k over the retrieved (or reranked) list."""

    def __init__(self, k: int = 10, *, from_reranked: bool = False) -> None:
        self.k = k
        self.from_reranked = from_reranked
        self.name = f"ndcg@{k}" + ("_reranked" if from_reranked else "")

    def score_one(self, result: PipelineResult, item: TaskItem) -> float | None:
        if not item.gold_passages:
            return None
        hits = (result.reranked if self.from_reranked else result.retrieved)[: self.k]
        dcg = 0.0
        for i, h in enumerate(hits):
            rel = 1.0 if _is_gold(h, item.gold_passages) else 0.0
            dcg += rel / math.log2(i + 2)
        n_gold = min(len(item.gold_passages), self.k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(n_gold))
        return dcg / idcg if idcg > 0 else 0.0


class MRRAtK(Metric):
    """Mean reciprocal rank of the first gold passage within top-k."""

    def __init__(self, k: int = 10, *, from_reranked: bool = False) -> None:
        self.k = k
        self.from_reranked = from_reranked
        self.name = f"mrr@{k}" + ("_reranked" if from_reranked else "")

    def score_one(self, result: PipelineResult, item: TaskItem) -> float | None:
        if not item.gold_passages:
            return None
        hits = (result.reranked if self.from_reranked else result.retrieved)[: self.k]
        for i, h in enumerate(hits):
            if _is_gold(h, item.gold_passages):
                return 1.0 / (i + 1)
        return 0.0
