"""Natural Questions (1K subset) task loader.

Loads from Hugging Face Datasets at runtime (`google-research-datasets/natural_questions`).
For each example we extract the short-answer string(s) and the supporting
Wikipedia passage(s); we form a corpus that is the union of all referenced passages.

Materialized data is cached under platformdirs' user cache dir. The first
`Task.items(...)` call may take a few minutes depending on bandwidth.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rag_bench.tasks.base import Split, Task, TaskSpec, register_task
from rag_bench.types import Document, Query, TaskItem

_HF_DATASET = "google-research-datasets/natural_questions"
_SPLIT = "validation"
_SAMPLE_SIZE = 1000


def _cache_dir() -> Path:
    try:
        from platformdirs import user_cache_dir

        d = Path(user_cache_dir("rag-bench")) / "tasks" / "nq-1k"
    except ImportError:
        d = Path.home() / ".rag-bench-cache" / "tasks" / "nq-1k"
    d.mkdir(parents=True, exist_ok=True)
    return d


@register_task("nq-1k")
class NQTask(Task):
    """Natural Questions 1K subset (validation split)."""

    spec = TaskSpec(
        id="nq-1k",
        name="Natural Questions (1K subset)",
        family="single-hop-qa",
        size=_SAMPLE_SIZE,
        contamination_risk="high",
        metrics_retrieval=("recall@10", "ndcg@10", "mrr@10"),
        metrics_generation=("exact_match", "token_f1"),
        metrics_end_to_end=("answer_relevance", "faithfulness"),
        license="CC-BY-SA-3.0",
        upstream_url="https://ai.google.com/research/NaturalQuestions",
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
            raise ImportError(
                "NQTask requires the `datasets` package. `pip install rag-bench` "
                "should have installed it."
            ) from e
        cache = _cache_dir() / "raw.parquet"
        if cache.exists():
            import pyarrow.parquet as pq

            self._loaded = pq.read_table(cache).to_pylist()
            return self._loaded
        ds = load_dataset(_HF_DATASET, split=_SPLIT, streaming=True)
        records: list[dict[str, Any]] = []
        for ex in ds:
            short = ex.get("annotations", {}).get("short_answers", [])
            short_strs: list[str] = []
            for sa in short:
                # NQ short_answers come as token-offsets into document_text; we
                # synthesize the answer string from the offsets in document_text.
                doc_tokens = ex.get("document_text", "").split()
                for span in sa.get("text", []) if isinstance(sa, dict) else [sa]:
                    if isinstance(span, str):
                        short_strs.append(span)
                if isinstance(sa, dict) and "start_token" in sa:
                    s = sa["start_token"]
                    e = sa.get("end_token", s + 1)
                    short_strs.append(" ".join(doc_tokens[s:e]))
            short_strs = [a for a in short_strs if a]
            if not short_strs:
                continue
            records.append(
                {
                    "id": str(ex.get("id") or ex.get("example_id") or len(records)),
                    "question": ex["question"]["text"]
                    if isinstance(ex.get("question"), dict)
                    else ex["question"],
                    "answers": short_strs,
                    "document_text": ex.get("document_text", ""),
                    "document_title": ex.get("document_title", ""),
                }
            )
            if len(records) >= self._max_items:
                break
        # cache
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            pq.write_table(pa.Table.from_pylist(records), cache)
        except ImportError:  # pragma: no cover
            pass
        self._loaded = records
        return records

    def corpus(self) -> Iterable[Document]:
        for rec in self._load():
            yield Document(
                doc_id=f"nq-doc-{rec['id']}",
                text=rec["document_text"],
                metadata={"title": rec["document_title"]},
            )

    def items(self, split: Split = "all") -> Iterable[TaskItem]:
        records = self._load()
        # Deterministic 80/20 split on item id hash.
        n_holdout = int(0.2 * len(records))
        sorted_recs = sorted(records, key=lambda r: r["id"])
        public = sorted_recs[:-n_holdout] if n_holdout else sorted_recs
        holdout = sorted_recs[-n_holdout:] if n_holdout else []
        chosen = {"public": public, "holdout": holdout, "all": records}[split]
        for rec in chosen:
            yield TaskItem(
                task_id=self.task_id,
                item_id=str(rec["id"]),
                query=Query(query_id=str(rec["id"]), text=rec["question"]),
                gold_answers=rec["answers"],
                gold_passages=[f"nq-doc-{rec['id']}"],
                corpus_ref="nq-validation",
            )
