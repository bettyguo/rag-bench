"""HotpotQA (distractor 1K subset) task loader.

Uses the `hotpot_qa` dataset from Hugging Face Datasets. Distractor setting:
each item ships 10 paragraphs, of which 2 are gold and 8 are distractors.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rag_bench.tasks.base import Split, Task, TaskSpec, register_task
from rag_bench.types import Document, Query, TaskItem

_HF_DATASET = "hotpot_qa"
_HF_CONFIG = "distractor"
_SPLIT = "validation"
_SAMPLE_SIZE = 1000


def _cache_dir() -> Path:
    try:
        from platformdirs import user_cache_dir

        d = Path(user_cache_dir("rag-bench")) / "tasks" / "hotpotqa-1k"
    except ImportError:
        d = Path.home() / ".rag-bench-cache" / "tasks" / "hotpotqa-1k"
    d.mkdir(parents=True, exist_ok=True)
    return d


@register_task("hotpotqa-1k")
class HotpotQATask(Task):
    """HotpotQA distractor setting, 1K subset (validation)."""

    spec = TaskSpec(
        id="hotpotqa-1k",
        name="HotpotQA distractor (1K subset)",
        family="multi-hop-qa",
        size=_SAMPLE_SIZE,
        contamination_risk="high",
        metrics_retrieval=("recall@10", "ndcg@10", "mrr@10"),
        metrics_generation=("exact_match", "token_f1"),
        metrics_end_to_end=("answer_relevance", "faithfulness"),
        license="CC-BY-SA-4.0",
        upstream_url="https://hotpotqa.github.io/",
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
            raise ImportError("HotpotQATask requires `datasets`.") from e
        cache = _cache_dir() / "raw.parquet"
        if cache.exists():
            import pyarrow.parquet as pq

            self._loaded = pq.read_table(cache).to_pylist()
            return self._loaded
        ds = load_dataset(_HF_DATASET, _HF_CONFIG, split=_SPLIT, streaming=True)
        records: list[dict[str, Any]] = []
        for ex in ds:
            titles = ex["context"]["title"]
            sentences = ex["context"]["sentences"]
            supporting = set(ex.get("supporting_facts", {}).get("title", []))
            paragraphs: list[dict[str, str]] = []
            for t, sents in zip(titles, sentences, strict=False):
                text = " ".join(s.strip() for s in sents)
                paragraphs.append(
                    {
                        "title": t,
                        "text": text,
                        "is_gold": t in supporting,
                    }
                )
            records.append(
                {
                    "id": ex["id"],
                    "question": ex["question"],
                    "answer": ex["answer"],
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
                doc_id = f"hpqa-{p['title']}"
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                yield Document(doc_id=doc_id, text=p["text"], metadata={"title": p["title"]})

    def items(self, split: Split = "all") -> Iterable[TaskItem]:
        records = self._load()
        n_holdout = int(0.2 * len(records))
        sorted_recs = sorted(records, key=lambda r: r["id"])
        public = sorted_recs[:-n_holdout] if n_holdout else sorted_recs
        holdout = sorted_recs[-n_holdout:] if n_holdout else []
        chosen = {"public": public, "holdout": holdout, "all": records}[split]
        for rec in chosen:
            gold_passages = [f"hpqa-{p['title']}" for p in rec["paragraphs"] if p["is_gold"]]
            yield TaskItem(
                task_id=self.task_id,
                item_id=rec["id"],
                query=Query(query_id=rec["id"], text=rec["question"]),
                gold_answers=[rec["answer"]],
                gold_passages=gold_passages,
                corpus_ref="hotpotqa-distractor-validation",
            )
