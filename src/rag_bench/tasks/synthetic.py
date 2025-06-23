"""A synthetic task used by tests and the CI smoke run.

Authored from scratch — no network access required. Designed to discriminate
sensible retrievers from useless ones: each query has a unique noun-phrase
match in exactly one corpus document.
"""

from __future__ import annotations

from collections.abc import Iterable

from rag_bench.tasks.base import Split, Task, TaskSpec, register_task
from rag_bench.types import Document, Query, TaskItem

_CORPUS = [
    ("doc-1", "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was completed in 1889 and stands 330 metres tall."),
    ("doc-2", "Mount Everest is the highest mountain on Earth, with a summit elevation of 8848 metres above sea level. It lies in the Himalayas."),
    ("doc-3", "Photosynthesis converts carbon dioxide and water into glucose and oxygen, using sunlight as the energy source."),
    ("doc-4", "The Great Wall of China stretches over 21000 kilometres. Construction began in the 7th century BC."),
    ("doc-5", "The Amazon rainforest is the largest tropical rainforest on Earth, covering an area of 5.5 million square kilometres across nine countries."),
    ("doc-6", "Albert Einstein published his theory of special relativity in 1905, fundamentally changing physics. He won the Nobel Prize in 1921 for the photoelectric effect."),
    ("doc-7", "The Pacific Ocean is the largest and deepest ocean on Earth, covering more than 165 million square kilometres."),
    ("doc-8", "The human heart pumps about 7600 litres of blood per day through a network of blood vessels totaling over 100000 kilometres in length."),
    ("doc-9", "DNA, deoxyribonucleic acid, carries the genetic instructions for development and reproduction in all known living organisms."),
    ("doc-10", "The Mona Lisa was painted by Leonardo da Vinci between 1503 and 1519. It is displayed in the Louvre Museum in Paris."),
]

_ITEMS: list[tuple[str, str, list[str], list[str]]] = [
    # (item_id, question, gold_answers, gold_passages)
    ("s1", "How tall is the Eiffel Tower?", ["330 metres", "330 meters", "330m"], ["doc-1"]),
    ("s2", "What is the highest mountain on Earth?", ["Mount Everest", "Everest"], ["doc-2"]),
    ("s3", "What does photosynthesis convert carbon dioxide into?", ["glucose and oxygen", "oxygen and glucose"], ["doc-3"]),
    ("s4", "How long is the Great Wall of China?", ["21000 kilometres", "over 21000 kilometres", "21000 km"], ["doc-4"]),
    ("s5", "How many countries does the Amazon rainforest span?", ["nine", "9"], ["doc-5"]),
    ("s6", "When did Einstein publish his theory of special relativity?", ["1905"], ["doc-6"]),
    ("s7", "What is the largest ocean on Earth?", ["Pacific Ocean", "the Pacific"], ["doc-7"]),
    ("s8", "How much blood does the human heart pump per day?", ["7600 litres", "7600 liters", "about 7600 litres"], ["doc-8"]),
    ("s9", "What does DNA stand for?", ["deoxyribonucleic acid"], ["doc-9"]),
    ("s10", "Who painted the Mona Lisa?", ["Leonardo da Vinci", "Da Vinci", "Leonardo"], ["doc-10"]),
]


@register_task("synthetic-10")
class SyntheticTask(Task):
    """10-item, 10-doc synthetic task. Deterministic, offline, well-discriminating."""

    spec = TaskSpec(
        id="synthetic-10",
        name="Synthetic 10 (test/smoke)",
        family="single-hop-qa",
        size=10,
        contamination_risk="novel",
        metrics_retrieval=("recall@5", "mrr@5"),
        metrics_generation=("exact_match", "token_f1"),
        metrics_end_to_end=(),
        license="Apache-2.0",
        upstream_url="(rag-bench-authored)",
    )

    def corpus(self) -> Iterable[Document]:
        for doc_id, text in _CORPUS:
            yield Document(doc_id=doc_id, text=text)

    def items(self, split: Split = "all") -> Iterable[TaskItem]:
        # all items are 'public' for the synthetic task; no holdout
        if split == "holdout":
            return iter(())
        for iid, question, golds, passages in _ITEMS:
            yield TaskItem(
                task_id=self.task_id,
                item_id=iid,
                query=Query(query_id=iid, text=question),
                gold_answers=golds,
                gold_passages=passages,
                corpus_ref="synthetic-10-corpus",
            )
