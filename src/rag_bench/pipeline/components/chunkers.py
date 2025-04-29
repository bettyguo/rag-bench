"""Chunker components.

Three baseline chunkers, all pure-Python with no extras required:
- `recursive`  — LangChain-style recursive split on a separator hierarchy
- `fixed`      — fixed-size chunks measured in whitespace-tokens
- `sentence`   — sentence-aware packing up to a target chunk size
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, model_validator

from rag_bench.pipeline.base import Chunker, ComponentConfig, register
from rag_bench.types import Chunk, Document


class RecursiveChunkerConfig(ComponentConfig):
    type: Literal["recursive"] = "recursive"
    chunk_size: int = Field(1000, ge=1, description="Target chunk size in characters.")
    overlap: int = Field(200, ge=0, description="Character overlap between adjacent chunks.")
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")

    @model_validator(mode="after")
    def _overlap_lt_size(self):
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be strictly less than chunk_size")
        return self


@register("chunker", "recursive")
class RecursiveChunker(Chunker):
    """Recursively splits text by a hierarchy of separators (LangChain-style)."""

    def __init__(self, config: RecursiveChunkerConfig) -> None:
        super().__init__(config)
        self.cfg: RecursiveChunkerConfig = config

    def chunk(self, document: Document) -> list[Chunk]:
        pieces = self._split(document.text, list(self.cfg.separators))
        merged = self._merge(pieces)
        return [
            Chunk(
                chunk_id=f"{document.doc_id}#{i}",
                doc_id=document.doc_id,
                text=text,
                position=i,
                metadata=document.metadata,
            )
            for i, text in enumerate(merged)
        ]

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text]
        sep, *rest = separators
        if sep == "":
            # final layer: character split
            return [text[i : i + self.cfg.chunk_size] for i in range(0, len(text), self.cfg.chunk_size)]
        parts = text.split(sep) if sep else [text]
        # reattach sep so we don't drop content
        rejoined: list[str] = []
        for j, p in enumerate(parts):
            if not p:
                continue
            rejoined.append(p + (sep if j < len(parts) - 1 else ""))
        out: list[str] = []
        for piece in rejoined:
            if len(piece) <= self.cfg.chunk_size:
                out.append(piece)
            else:
                out.extend(self._split(piece, rest))
        return out

    def _merge(self, pieces: list[str]) -> list[str]:
        out: list[str] = []
        buf = ""
        for p in pieces:
            if not buf:
                buf = p
                continue
            if len(buf) + len(p) <= self.cfg.chunk_size:
                buf += p
            else:
                out.append(buf)
                # carry overlap
                tail = buf[-self.cfg.overlap :] if self.cfg.overlap else ""
                buf = tail + p
        if buf:
            out.append(buf)
        return out


class FixedSizeChunkerConfig(ComponentConfig):
    type: Literal["fixed"] = "fixed"
    chunk_size: int = Field(200, ge=1, description="Target chunk size in whitespace tokens.")
    overlap: int = Field(40, ge=0, description="Token overlap between adjacent chunks.")

    @model_validator(mode="after")
    def _overlap_lt_size(self):
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be strictly less than chunk_size")
        return self


@register("chunker", "fixed")
class FixedSizeChunker(Chunker):
    """Fixed-size chunks measured in whitespace-tokens."""

    def __init__(self, config: FixedSizeChunkerConfig) -> None:
        super().__init__(config)
        self.cfg: FixedSizeChunkerConfig = config

    def chunk(self, document: Document) -> list[Chunk]:
        tokens = document.text.split()
        if not tokens:
            return []
        step = self.cfg.chunk_size - self.cfg.overlap
        chunks: list[Chunk] = []
        for i, start in enumerate(range(0, len(tokens), step)):
            window = tokens[start : start + self.cfg.chunk_size]
            if not window:
                break
            text = " ".join(window)
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}#{i}",
                    doc_id=document.doc_id,
                    text=text,
                    position=i,
                    metadata=document.metadata,
                )
            )
        return chunks


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")


class SentenceChunkerConfig(ComponentConfig):
    type: Literal["sentence"] = "sentence"
    chunk_size: int = Field(800, ge=1, description="Target chunk size in characters.")
    overlap_sentences: int = Field(1, ge=0, description="Sentences to overlap between adjacent chunks.")


@register("chunker", "sentence")
class SentenceChunker(Chunker):
    """Pack sentences into chunks up to chunk_size, overlapping by N sentences."""

    def __init__(self, config: SentenceChunkerConfig) -> None:
        super().__init__(config)
        self.cfg: SentenceChunkerConfig = config

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = [s.strip() for s in _SENT_SPLIT.split(document.text) if s.strip()]
        chunks: list[Chunk] = []
        buf: list[str] = []
        size = 0
        idx = 0
        i = 0
        while i < len(sentences):
            s = sentences[i]
            if size + len(s) + 1 <= self.cfg.chunk_size or not buf:
                buf.append(s)
                size += len(s) + 1
                i += 1
            else:
                chunks.append(self._make_chunk(document, " ".join(buf), idx))
                idx += 1
                # carry overlap
                buf = buf[-self.cfg.overlap_sentences :] if self.cfg.overlap_sentences else []
                size = sum(len(b) + 1 for b in buf)
        if buf:
            chunks.append(self._make_chunk(document, " ".join(buf), idx))
        return chunks

    def _make_chunk(self, doc: Document, text: str, position: int) -> Chunk:
        return Chunk(
            chunk_id=f"{doc.doc_id}#{position}",
            doc_id=doc.doc_id,
            text=text,
            position=position,
            metadata=doc.metadata,
        )
