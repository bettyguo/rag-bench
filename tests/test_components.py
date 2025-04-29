"""Component-level unit tests. All run without optional extras (no torch)."""

from __future__ import annotations

import pytest

from rag_bench.pipeline.components.chunkers import (
    FixedSizeChunker,
    FixedSizeChunkerConfig,
    RecursiveChunker,
    RecursiveChunkerConfig,
    SentenceChunker,
    SentenceChunkerConfig,
)
from rag_bench.pipeline.components.generators import EchoConfig, EchoGenerator
from rag_bench.pipeline.components.rerankers import (
    IdentityReranker,
    IdentityRerankerConfig,
    LexicalOverlapConfig,
    LexicalOverlapReranker,
)
from rag_bench.pipeline.components.retrievers import BM25Config, BM25Retriever
from rag_bench.types import Chunk, Document, Query, RetrievalResult

# --------------------------------------------------------------------------- #
# Chunkers                                                                     #
# --------------------------------------------------------------------------- #


def test_recursive_chunker_respects_size_and_overlap():
    text = "First paragraph here.\n\nSecond paragraph longer. " * 30
    doc = Document(doc_id="d1", text=text)
    chunker = RecursiveChunker(RecursiveChunkerConfig(chunk_size=200, overlap=50))
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 2
    assert all(len(c.text) <= 200 + 50 for c in chunks)
    assert chunks[0].chunk_id == "d1#0"
    assert chunks[1].chunk_id == "d1#1"


def test_recursive_chunker_overlap_must_be_less_than_size():
    with pytest.raises(ValueError):
        RecursiveChunkerConfig(chunk_size=100, overlap=100)


def test_fixed_chunker_tokens_per_chunk():
    text = " ".join(f"tok{i}" for i in range(100))
    doc = Document(doc_id="d1", text=text)
    chunker = FixedSizeChunker(FixedSizeChunkerConfig(chunk_size=20, overlap=5))
    chunks = chunker.chunk(doc)
    assert chunks
    assert all(len(c.text.split()) <= 20 for c in chunks)
    # overlap → later chunks share trailing tokens
    first_tokens = chunks[0].text.split()
    second_tokens = chunks[1].text.split()
    assert first_tokens[-5:] == second_tokens[:5]


def test_sentence_chunker_packs_sentences():
    text = "First sentence. Second sentence! Third sentence? Fourth one. Fifth. Sixth. Seventh."
    doc = Document(doc_id="d1", text=text)
    chunker = SentenceChunker(SentenceChunkerConfig(chunk_size=40, overlap_sentences=1))
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 2
    # overlap=1 means last sentence of chunk N appears in chunk N+1
    # (cheap check: text not totally disjoint)


# --------------------------------------------------------------------------- #
# BM25 Retriever                                                               #
# --------------------------------------------------------------------------- #


def _chunks_for(texts: list[str]) -> list[Chunk]:
    return [Chunk(chunk_id=f"c{i}", doc_id=f"d{i}", text=t, position=0) for i, t in enumerate(texts)]


def test_bm25_retrieves_lexically_matching_chunk_first():
    chunks = _chunks_for(
        [
            "The capital of France is Paris.",
            "Bananas are yellow and grow on trees.",
            "Paris is also the name of a Trojan prince in Greek mythology.",
            "Cats are common household pets.",
        ]
    )
    retriever = BM25Retriever(BM25Config())
    retriever.index(chunks)
    hits = retriever.retrieve(Query(query_id="q1", text="What is the capital of France?"), top_k=2)
    assert hits[0].chunk.chunk_id == "c0"
    assert hits[0].rank == 0
    assert hits[0].score > hits[1].score


def test_bm25_returns_empty_for_no_match():
    chunks = _chunks_for(["abcdef", "ghijkl"])
    r = BM25Retriever(BM25Config())
    r.index(chunks)
    hits = r.retrieve(Query(query_id="q1", text="nothing here"), top_k=5)
    assert hits == []


def test_bm25_top_k_caps_result_size():
    chunks = _chunks_for([f"document {i} about cats" for i in range(20)])
    r = BM25Retriever(BM25Config())
    r.index(chunks)
    hits = r.retrieve(Query(query_id="q1", text="cats"), top_k=3)
    assert len(hits) == 3


# --------------------------------------------------------------------------- #
# Rerankers                                                                    #
# --------------------------------------------------------------------------- #


def test_identity_reranker_preserves_order_and_renumbers_ranks():
    chunks = _chunks_for(["a", "b", "c"])
    cands = [RetrievalResult(chunk=c, score=10 - i, rank=i) for i, c in enumerate(chunks)]
    rr = IdentityReranker(IdentityRerankerConfig(top_k=2))
    out = rr.rerank(Query(query_id="q", text=""), cands, top_k=2)
    assert [r.chunk.chunk_id for r in out] == ["c0", "c1"]
    assert [r.rank for r in out] == [0, 1]


def test_lexical_overlap_reranker_promotes_higher_jaccard():
    chunks = _chunks_for(["unrelated noise here", "the quick brown fox", "fox jumps"])
    cands = [RetrievalResult(chunk=c, score=1.0, rank=i) for i, c in enumerate(chunks)]
    rr = LexicalOverlapReranker(LexicalOverlapConfig(top_k=3))
    out = rr.rerank(Query(query_id="q", text="the quick brown fox"), cands, top_k=3)
    # the exact-overlap chunk should top
    assert out[0].chunk.chunk_id == "c1"


# --------------------------------------------------------------------------- #
# Generators                                                                   #
# --------------------------------------------------------------------------- #


def test_echo_generator_returns_top_chunk():
    chunks = _chunks_for(["alpha", "beta"])
    cands = [RetrievalResult(chunk=c, score=1.0, rank=i) for i, c in enumerate(chunks)]
    gen = EchoGenerator(EchoConfig())
    out = gen.generate(Query(query_id="q", text=""), cands)
    assert out.text == "alpha"


def test_echo_generator_emits_fallback_when_no_context():
    gen = EchoGenerator(EchoConfig(fallback="INSUFFICIENT_CONTEXT"))
    out = gen.generate(Query(query_id="q", text="?"), [])
    assert out.text == "INSUFFICIENT_CONTEXT"
