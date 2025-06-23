"""Seed-set hygiene: item-id uniqueness, gold-passage existence, schema sanity.

These tests catch the regressions that are easy to introduce when expanding
adversarial seed sets toward the 500-item launch target.
"""

from __future__ import annotations

import pytest

from rag_bench.tasks.base import list_tasks
from rag_bench.tasks.counterfactual_qa import CounterfactualQATask
from rag_bench.tasks.noisy_qa import NoisyQATask
from rag_bench.tasks.synthetic import SyntheticTask
from rag_bench.tasks.unanswerable_qa import UnanswerableQATask

ADVERSARIAL_TASKS = [NoisyQATask, UnanswerableQATask, CounterfactualQATask, SyntheticTask]


@pytest.mark.parametrize("task_cls", ADVERSARIAL_TASKS)
def test_item_ids_are_unique(task_cls):
    task = task_cls()
    ids = [item.item_id for item in task.items()]
    assert len(ids) == len(set(ids)), f"duplicate item_ids in {task_cls.__name__}: {ids}"


@pytest.mark.parametrize("task_cls", ADVERSARIAL_TASKS)
def test_corpus_doc_ids_are_unique(task_cls):
    task = task_cls()
    ids = [doc.doc_id for doc in task.corpus()]
    assert len(ids) == len(set(ids)), f"duplicate doc_ids in {task_cls.__name__}"


@pytest.mark.parametrize("task_cls", ADVERSARIAL_TASKS)
def test_gold_passages_exist_in_corpus(task_cls):
    task = task_cls()
    corpus_ids = {doc.doc_id for doc in task.corpus()}
    for item in task.items():
        if not item.gold_passages:
            continue
        for gp in item.gold_passages:
            assert gp in corpus_ids, (
                f"{task_cls.__name__} item {item.item_id}: gold passage "
                f"{gp!r} not in corpus"
            )


@pytest.mark.parametrize("task_cls", ADVERSARIAL_TASKS)
def test_query_text_not_empty(task_cls):
    task = task_cls()
    for item in task.items():
        assert item.query.text.strip(), f"{task_cls.__name__} item {item.item_id} has empty query"


def test_noisy_qa_each_item_has_3_distractors():
    """noisy-qa contract: exactly 1 gold + 3 distractors per query."""
    task = NoisyQATask()
    items = list(task.items())
    docs = list(task.corpus())
    assert len(docs) == 4 * len(items), (
        f"noisy-qa: expected 4 docs (1 gold + 3 distractors) per item, "
        f"got {len(docs)} docs for {len(items)} items"
    )


def test_unanswerable_qa_has_both_answerable_and_unanswerable():
    task = UnanswerableQATask()
    items = list(task.items())
    answerable = [i for i in items if i.metadata.get("answerable")]
    unanswerable = [i for i in items if i.metadata.get("answerable") is False]
    assert len(answerable) >= 3, "need ≥3 answerable controls for precision to be meaningful"
    assert len(unanswerable) >= 5, "need ≥5 unanswerable items"


def test_counterfactual_qa_balanced_plausible_implausible():
    task = CounterfactualQATask()
    items = list(task.items())
    plausible = [i for i in items if i.metadata.get("counter_kind") == "plausible"]
    implausible = [i for i in items if i.metadata.get("counter_kind") == "implausible"]
    assert len(plausible) >= 3
    assert len(implausible) >= 3
    # Both metrics should be non-trivial; if either side is empty the metric is meaningless
    assert plausible and implausible


@pytest.mark.parametrize("task_cls", [NoisyQATask, UnanswerableQATask, CounterfactualQATask])
def test_answerable_items_have_gold_answers(task_cls):
    task = task_cls()
    for item in task.items():
        if item.metadata.get("answerable") is False:
            # unanswerable items should not have golds
            assert not item.gold_answers, (
                f"{task_cls.__name__} item {item.item_id} is unanswerable but has gold_answers"
            )
        else:
            # answerable items must have at least one gold answer
            assert item.gold_answers, (
                f"{task_cls.__name__} item {item.item_id} is answerable but has no gold_answers"
            )


def test_all_registered_tasks_have_unique_ids():
    """A double-registration would silently leak via the registry."""
    ids = list_tasks()
    assert len(ids) == len(set(ids))
