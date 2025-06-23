"""MS MARCO Passage (dev, 1K subset) task loader.

Retrieval-heavy. Each query has one or more gold passage IDs. We construct a
small corpus consisting of (a) the gold passages and (b) a sample of distractor
passages from the rest of the dev pool.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rag_bench.tasks.base import Split, Task, TaskSpec, register_task
from rag_bench.types import Document, Query, TaskItem

_HF_DATASET = "ms_marco"
_HF_CONFIG = "v2.1"
_SPLIT = "validation"
_SAMPLE_SIZE = 1000


def _cache_dir() -> Path:
    try:
        from platformdirs import user_cache_dir

        d = Path(user_cache_dir("rag-bench")) / "tasks" / "msmarco-1k"
    except ImportError:
        d = Path.home() / ".rag-bench-cache" / "tasks" / "msmarco-1k"
    d.mkdir(parents=True, exist_ok=True)
    return d


@register_task("msmarco-1k")
class MSMARCOTask(Task):
    """MS MARCO Passage dev, 1K subset (retrieval + extractive answer)."""

    spec = TaskSpec(
        id="msmarco-1k",
        name="MS MARCO Passage v2.1 (1K subset)",
        family="single-hop-qa",
        size=_SAMPLE_SIZE,
        contamination_risk="high",
        metrics_retrieval=("recall@10", "ndcg@10", "mrr@10"),
        metrics_generation=("exact_match", "token_f1"),
        metrics_end_to_end=("answer_relevance",),
        license="MS-MARCO non-commercial",
        upstream_url="https://microsoft.github.io/msmarco/",
    )

    def __init__(self, *, max_items: int | None = None) -> None:
        self._max_items = max_items or _SAMPLE_SIZE
        self._loaded: list[dict[str, Any]] | None = None

    def _load(self) -> list[dict[str, Any]]:
        if self._loaded is not None:
            return self._loaded
        try:
            from datasets import load_dataset
        except ImportError as e:  # pragma: no cover
            raise ImportError("MSMARCOTask requires `datasets`.") from e
        cache = _cache_dir() / "raw.parquet"
        if cache.exists():
            import pyarrow.parquet as pq

            self._loaded = pq.read_table(cache).to_pylist()
            return self._loaded
        ds = load_dataset(_HF_DATASET, _HF_CONFIG, split=_SPLIT, streaming=True)
        records: list[dict[str, Any]] = []
        for ex in ds:
            answers = ex.get("answers", [])
            answers = [a for a in answers if a]
            if not answers:
                continue
            passages = ex.get("passages", {})
            texts: list[str] = list(passages.get("passage_text", []))
            is_selected: list[int] = list(passages.get("is_selected", []))
            urls: list[str] = list(passages.get("url", []))
            paragraphs = []
            for i, (t, sel) in enumerate(zip(texts, is_selected, strict=False)):
                paragraphs.append(
                    {
                        "id": f"msm-{ex['query_id']}-{i}",
                        "text": t,
                        "is_gold": bool(sel),
                        "url": urls[i] if i < len(urls) else "",
                    }
                )
            records.append(
                {
                    "id": str(ex["query_id"]),
                    "question": ex["query"],
                    "answers": answers,
                    "paragraphs": paragraphs,
                }
            )
            if len(records) >= self._max_items:
                break
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            pq.write_table(pa.Table.from_pylist(records), cache)
        except ImportError:  # pragma: no cover
            pass
        self._loaded = records
        return records

    def corpus(self) -> Iterable[Document]:
        seen: set[str] = set()
        for rec in self._load():
            for p in rec["paragraphs"]:
                if p["id"] in seen:
                    continue
                seen.add(p["id"])
                yield Document(doc_id=p["id"], text=p["text"], metadata={"url": p.get("url", "")})

    def items(self, split: Split = "all") -> Iterable[TaskItem]:
        records = self._load()
        n_holdout = int(0.2 * len(records))
        sorted_recs = sorted(records, key=lambda r: r["id"])
        public = sorted_recs[:-n_holdout] if n_holdout else sorted_recs
        holdout = sorted_recs[-n_holdout:] if n_holdout else []
        chosen = {"public": public, "holdout": holdout, "all": records}[split]
        for rec in chosen:
            gold_passages = [p["id"] for p in rec["paragraphs"] if p["is_gold"]]
            yield TaskItem(
                task_id=self.task_id,
                item_id=rec["id"],
                query=Query(query_id=rec["id"], text=rec["question"]),
                gold_answers=rec["answers"],
                gold_passages=gold_passages,
                corpus_ref="msmarco-passage-v2.1-dev",
            )
