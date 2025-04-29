"""End-to-end pipeline composition test using only pure-Python components."""

from __future__ import annotations

import pytest

from rag_bench.pipeline.compose import compose_from_dict, compose_from_yaml
from rag_bench.pipeline.pipeline import Pipeline
from rag_bench.types import Document, Query

PIPE_YAML = """
pipeline:
  name: bm25-echo-test
  retriever_top_k: 5
  reranker_top_k: 2

  chunker:
    type: recursive
    chunk_size: 200
    overlap: 50

  retriever:
    type: bm25
    k1: 1.5
    b: 0.75
    top_k: 5

  reranker:
    type: identity
    top_k: 2

  generator:
    type: echo
"""


def test_compose_from_yaml_builds_runnable_pipeline():
    pipe = compose_from_yaml(PIPE_YAML)
    assert isinstance(pipe, Pipeline)
    assert pipe.name == "bm25-echo-test"


def test_pipeline_end_to_end_bm25_echo():
    pipe = compose_from_yaml(PIPE_YAML)
    docs = [
        Document(doc_id="d1", text="The capital of France is Paris. " * 3),
        Document(doc_id="d2", text="Bananas are yellow and grow in tropical regions. " * 3),
        Document(doc_id="d3", text="Cats are common household pets. " * 3),
    ]
    pipe.index(docs)
    result = pipe.answer(Query(query_id="q1", text="What is the capital of France?"))
    assert "Paris" in result.generation.text
    assert result.reranked
    assert result.reranked[0].rank == 0


def test_pipeline_requires_index_before_answer():
    pipe = compose_from_yaml(PIPE_YAML)
    with pytest.raises(RuntimeError, match="index"):
        pipe.answer(Query(query_id="q", text="?"))


def test_compose_rejects_unknown_component_type():
    bad = {
        "pipeline": {
            "name": "bad",
            "chunker": {"type": "totally-not-real", "chunk_size": 100},
            "retriever": {"type": "bm25"},
            "generator": {"type": "echo"},
        }
    }
    with pytest.raises(KeyError, match="No chunker component registered"):
        compose_from_dict(bad)


def test_compose_rejects_unknown_yaml_field():
    bad = """
    pipeline:
      name: bad
      chunker: { type: recursive }
      retriever: { type: bm25 }
      generator: { type: echo }
      bogus_field: 42
    """
    with pytest.raises(Exception, match=r"bogus_field|extra"):
        compose_from_yaml(bad)


def test_pipeline_fingerprint_includes_all_components():
    pipe = compose_from_yaml(PIPE_YAML)
    fp = pipe.component_fingerprint()
    assert set(fp.keys()) == {"chunker", "retriever", "reranker", "generator"}
    assert fp["chunker"]["name"] == "recursive"
    assert fp["retriever"]["name"] == "bm25"
    assert fp["generator"]["name"] == "echo"
