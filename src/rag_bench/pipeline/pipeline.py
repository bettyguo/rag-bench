"""The Pipeline class — composes chunker, retriever, reranker, generator.

Holds the indexed corpus state. Provides `answer(query)` that returns a
`PipelineResult` with retrieved, reranked, and generated outputs.

Indexing is done once via `index(documents)`. The runner re-uses the
indexed state across all task items.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

from rag_bench.pipeline.base import Chunker, Generator, Reranker, Retriever
from rag_bench.types import Document, GenerationResult, PipelineResult, Query, RetrievalResult


class Pipeline:
    """Composed RAG pipeline. Stateless across queries once `index` has been called."""

    def __init__(
        self,
        *,
        name: str,
        chunker: Chunker,
        retriever: Retriever,
        reranker: Reranker,
        generator: Generator,
        retriever_top_k: int = 50,
        reranker_top_k: int = 10,
    ) -> None:
        self.name = name
        self.chunker = chunker
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.retriever_top_k = retriever_top_k
        self.reranker_top_k = reranker_top_k
        self._indexed = False

    def index(self, documents: Iterable[Document]) -> None:
        chunks = self.chunker.chunk_many(documents)
        self.retriever.index(chunks)
        self._indexed = True

    def answer(self, query: Query, *, seed: int = 0) -> PipelineResult:
        if not self._indexed:
            raise RuntimeError("Pipeline.index(documents) must be called before answer().")
        t0 = time.perf_counter()
        retrieved: list[RetrievalResult] = self.retriever.retrieve(query, self.retriever_top_k)
        t1 = time.perf_counter()
        reranked: list[RetrievalResult] = self.reranker.rerank(query, retrieved, self.reranker_top_k)
        t2 = time.perf_counter()
        generation: GenerationResult = self.generator.generate(query, reranked, seed=seed)
        t3 = time.perf_counter()
        return PipelineResult(
            query=query,
            retrieved=retrieved,
            reranked=reranked,
            generation=generation,
            pipeline_name=self.name,
            seed=seed,
            metadata={
                "retriever_ms": (t1 - t0) * 1000,
                "reranker_ms": (t2 - t1) * 1000,
                "generator_ms": (t3 - t2) * 1000,
            },
        )

    def component_fingerprint(self) -> dict[str, dict]:
        return {
            "chunker": self.chunker.fingerprint(),
            "retriever": self.retriever.fingerprint(),
            "reranker": self.reranker.fingerprint(),
            "generator": self.generator.fingerprint(),
        }
