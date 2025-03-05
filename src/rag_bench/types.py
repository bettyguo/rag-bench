"""Core data types passed between pipeline components.

Kept dependency-free (no pydantic) so they import cheaply and are easy to
serialize. Pydantic models live alongside configs in `rag_bench.pipeline.base`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class Document:
    """A unit of raw corpus content prior to chunking."""

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Chunk:
    """A chunk produced by a Chunker; the indexable unit of the corpus."""

    chunk_id: str
    doc_id: str
    text: str
    position: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Query:
    """A single query to a pipeline."""

    query_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class RetrievalResult:
    """A single (chunk, score) hit returned by a Retriever or Reranker."""

    chunk: Chunk
    score: float
    rank: int


@dataclass(slots=True)
class GenerationResult:
    """The output of a Generator.

    Mutable (not frozen) because runner annotates with timing/cost after the fact.
    """

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class PipelineResult:
    """The full output for one query through a Pipeline."""

    query: Query
    retrieved: list[RetrievalResult]
    reranked: list[RetrievalResult]
    generation: GenerationResult
    pipeline_name: str
    seed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskItem:
    """A single (query, gold-answer(s), gold-passage(s), corpus_ref) record from a Task."""

    task_id: str
    item_id: str
    query: Query
    gold_answers: list[str]
    gold_passages: list[str] | None = None  # passage IDs/text for retrieval scoring
    corpus_ref: str | None = None  # corpus identifier, e.g. "wiki-en-2024-01"
    metadata: dict[str, Any] = field(default_factory=dict)
