"""task_data_hash should be permutation-invariant over gold_answers and
gold_passages — otherwise a HF datasets revision that reorders answers
silently changes the hash. The invariance tests are xfail until the next
deliberate version bump (fixing it invalidates existing hashes).
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from rag_bench.tasks.base import Split, Task, TaskSpec
from rag_bench.types import Document, Query, TaskItem


class _StubTask(Task):
    """Test-local task fixture; intentionally not registered."""

    task_id = "stub-hash-task"
    spec = TaskSpec(
        id="stub-hash-task",
        name="Stub",
        family="single-hop-qa",
        size=2,
        contamination_risk="novel",
    )

    def __init__(self, gold_answers_order: tuple[str, ...], gold_passages_order: tuple[str, ...]) -> None:
        self._gold_answers = gold_answers_order
        self._gold_passages = gold_passages_order

    def corpus(self) -> Iterable[Document]:
        # Doc ids are sorted inside the hash; keep this stub consistent
        # across gold-list permutations.
        yield Document(doc_id="d-a", text="alpha")
        yield Document(doc_id="d-b", text="beta")

    def items(self, split: Split = "all") -> Iterable[TaskItem]:
        yield TaskItem(
            task_id=self.task_id,
            item_id="i1",
            query=Query(query_id="i1", text="?"),
            gold_answers=list(self._gold_answers),
            gold_passages=list(self._gold_passages),
        )


_XFAIL_REASON = "Fix invalidates existing pipeline_hashes; wait for the next major bump."


@pytest.mark.xfail(reason=_XFAIL_REASON, strict=True)
def test_task_data_hash_invariant_under_gold_answers_permutation():
    a = _StubTask(gold_answers_order=("alpha", "beta"), gold_passages_order=("d-a",))
    b = _StubTask(gold_answers_order=("beta", "alpha"), gold_passages_order=("d-a",))
    assert a.task_data_hash() == b.task_data_hash()


@pytest.mark.xfail(reason=_XFAIL_REASON, strict=True)
def test_task_data_hash_invariant_under_gold_passages_permutation():
    a = _StubTask(gold_answers_order=("alpha",), gold_passages_order=("d-a", "d-b"))
    b = _StubTask(gold_answers_order=("alpha",), gold_passages_order=("d-b", "d-a"))
    assert a.task_data_hash() == b.task_data_hash()


def test_task_data_hash_changes_on_content_change():
    a = _StubTask(gold_answers_order=("alpha",), gold_passages_order=("d-a",))
    b = _StubTask(gold_answers_order=("gamma",), gold_passages_order=("d-a",))
    assert a.task_data_hash() != b.task_data_hash()


def test_task_data_hash_changes_when_passage_set_differs():
    a = _StubTask(gold_answers_order=("alpha",), gold_passages_order=("d-a",))
    b = _StubTask(gold_answers_order=("alpha",), gold_passages_order=("d-a", "d-b"))
    assert a.task_data_hash() != b.task_data_hash()
