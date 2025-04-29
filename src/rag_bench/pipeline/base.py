"""Pipeline component base classes and the component registry.

A pipeline is a typed composition of four stages:
    Chunker -> Retriever -> Reranker -> Generator

Each stage is an abstract base class; concrete components register via the
`@register(stage, name)` decorator. The composer in `compose.py` reads YAML
and instantiates the right subclass for each stage.

This module intentionally has zero runtime dependencies on retriever /
generator backends — those live behind optional extras (`pip install
rag-bench[retrievers,generators]`) and the registry stays empty until they
are imported.
"""

from __future__ import annotations

import abc
from collections.abc import Iterable, Sequence
from typing import Any, ClassVar, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from rag_bench.types import (
    Chunk,
    Document,
    GenerationResult,
    Query,
    RetrievalResult,
)

Stage = Literal["chunker", "retriever", "reranker", "generator"]
_REGISTRY: dict[Stage, dict[str, type[Component]]] = {
    "chunker": {},
    "retriever": {},
    "reranker": {},
    "generator": {},
}


class ComponentConfig(BaseModel):
    """Base config object for any component. Subclasses add `type`-specific fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = Field(..., description="Registry name; dispatched on by the composer.")


C = TypeVar("C", bound="Component")


class Component(abc.ABC):
    """Base class for all pipeline components."""

    stage: ClassVar[Stage]  # set on subclass
    name: ClassVar[str]  # registry name; set on subclass

    def __init__(self, config: ComponentConfig) -> None:
        self.config = config

    def fingerprint(self) -> dict[str, Any]:
        """Stable, JSON-serializable representation used for the pipeline_hash."""
        return {"stage": self.stage, "name": self.name, **self.config.model_dump(mode="json")}


def register(stage: Stage, name: str):
    """Class decorator: register `cls` under `(stage, name)` in the component registry."""

    def _wrap(cls: type[C]) -> type[C]:
        if name in _REGISTRY[stage]:
            existing = _REGISTRY[stage][name]
            if existing is not cls:
                raise ValueError(
                    f"Component {stage}:{name} already registered as {existing!r}"
                )
        cls.stage = stage  # type: ignore[misc]
        cls.name = name  # type: ignore[misc]
        _REGISTRY[stage][name] = cls
        return cls

    return _wrap


def get_component_cls(stage: Stage, name: str) -> type[Component]:
    try:
        return _REGISTRY[stage][name]
    except KeyError as e:
        avail = sorted(_REGISTRY[stage].keys())
        raise KeyError(
            f"No {stage} component registered as {name!r}. Available: {avail}"
        ) from e


def registered_components(stage: Stage | None = None) -> dict[Stage, list[str]]:
    if stage is None:
        return {s: sorted(d.keys()) for s, d in _REGISTRY.items()}
    return {stage: sorted(_REGISTRY[stage].keys())}


class Chunker(Component):
    """Splits documents into chunks. Stateless: chunk(doc) is pure."""

    stage = "chunker"

    @abc.abstractmethod
    def chunk(self, document: Document) -> list[Chunk]: ...

    def chunk_many(self, documents: Iterable[Document]) -> list[Chunk]:
        out: list[Chunk] = []
        for d in documents:
            out.extend(self.chunk(d))
        return out


class Retriever(Component):
    """Indexes a corpus and retrieves top-k chunks for queries.

    Stateful: `index(chunks)` builds the index, then `retrieve(query)` uses it.
    Typically indexes are cached per (retriever_fingerprint, corpus_ref).
    """

    stage = "retriever"

    @abc.abstractmethod
    def index(self, chunks: Sequence[Chunk]) -> None: ...

    @abc.abstractmethod
    def retrieve(self, query: Query, top_k: int) -> list[RetrievalResult]: ...


class Reranker(Component):
    """Re-orders retrieval candidates by (typically) a cross-encoder score.

    Identity reranker is allowed; it makes the pipeline well-typed even when
    no reranking is desired.
    """

    stage = "reranker"

    @abc.abstractmethod
    def rerank(
        self,
        query: Query,
        candidates: Sequence[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]: ...


class Generator(Component):
    """Produces an answer given a query and a list of retrieved chunks."""

    stage = "generator"

    @abc.abstractmethod
    def generate(
        self,
        query: Query,
        context: Sequence[RetrievalResult],
        *,
        seed: int = 0,
    ) -> GenerationResult: ...
