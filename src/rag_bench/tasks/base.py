"""Task base class and registry.

A Task encapsulates:
- A corpus (loadable into Documents) with a content-addressed hash.
- Iterable TaskItems (query + gold).
- Split selection (public vs holdout).
- Metric set declaration (which metrics apply).

Concrete tasks live under `rag_bench.tasks.<task_id>` and register with
`@register_task("task-id")`.
"""

from __future__ import annotations

import abc
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from rag_bench.types import Document, TaskItem

Split = Literal["public", "holdout", "all"]

_TASK_REGISTRY: dict[str, type[Task]] = {}


@dataclass(frozen=True)
class TaskSpec:
    """The static `task.yaml`-like description of a Task. Mirrors docs/tasks.md."""

    id: str
    name: str
    family: str
    size: int
    contamination_risk: Literal["high", "medium", "low", "novel"]
    language: str = "en"
    metrics_retrieval: tuple[str, ...] = ()
    metrics_generation: tuple[str, ...] = ()
    metrics_end_to_end: tuple[str, ...] = ()
    license: str = ""
    upstream_url: str = ""


class Task(abc.ABC):
    """Base class for all benchmark tasks."""

    spec: ClassVar[TaskSpec]
    task_id: ClassVar[str]

    @abc.abstractmethod
    def corpus(self) -> Iterable[Document]: ...

    @abc.abstractmethod
    def items(self, split: Split = "all") -> Iterable[TaskItem]: ...

    def task_data_hash(self) -> str:
        """SHA-256 over a canonical serialization of (sorted item ids ‖ corpus doc ids).

        Stable across runs: depends only on the materialized data, not on Python's
        dict ordering or filesystem layout.
        """
        h = hashlib.sha256()
        h.update(b"task_id:")
        h.update(self.task_id.encode())
        h.update(b"\nitems:\n")
        for item in sorted(self.items("all"), key=lambda i: i.item_id):
            h.update(item.item_id.encode())
            h.update(b"|")
            h.update(item.query.text.encode())
            h.update(b"|")
            # TODO: sort gold_answers and gold_passages before hashing so a
            # HF dataset revision that reorders answers doesn't silently
            # change task_data_hash. Held back of a deliberate version bump
            # because the fix invalidates every previously-computed hash.
            for ga in item.gold_answers:
                h.update(ga.encode())
                h.update(b",")
            h.update(b";")
            if item.gold_passages:
                for gp in item.gold_passages:
                    h.update(gp.encode())
                    h.update(b",")
            h.update(b"\n")
        h.update(b"corpus:\n")
        for doc in sorted(self.corpus(), key=lambda d: d.doc_id):
            h.update(doc.doc_id.encode())
            h.update(b"|")
            # don't hash full text — it can be megabytes; hash a fingerprint
            h.update(hashlib.sha256(doc.text.encode()).hexdigest().encode())
            h.update(b"\n")
        return h.hexdigest()

    def to_jsonable_spec(self) -> dict[str, Any]:
        return {
            "id": self.spec.id,
            "name": self.spec.name,
            "family": self.spec.family,
            "size": self.spec.size,
            "contamination_risk": self.spec.contamination_risk,
            "language": self.spec.language,
            "metrics": {
                "retrieval": list(self.spec.metrics_retrieval),
                "generation": list(self.spec.metrics_generation),
                "end_to_end": list(self.spec.metrics_end_to_end),
            },
            "license": self.spec.license,
            "upstream_url": self.spec.upstream_url,
        }


def register_task(task_id: str):
    def _wrap(cls: type[Task]) -> type[Task]:
        if task_id in _TASK_REGISTRY and _TASK_REGISTRY[task_id] is not cls:
            raise ValueError(f"Task {task_id!r} already registered")
        cls.task_id = task_id  # type: ignore[misc]
        _TASK_REGISTRY[task_id] = cls
        return cls

    return _wrap


def get_task_cls(task_id: str) -> type[Task]:
    try:
        return _TASK_REGISTRY[task_id]
    except KeyError as e:
        avail = sorted(_TASK_REGISTRY.keys())
        raise KeyError(f"No task registered as {task_id!r}. Available: {avail}") from e


def list_tasks() -> list[str]:
    return sorted(_TASK_REGISTRY.keys())


def canonical_task_json(task: Task) -> str:
    """Canonical JSON for the task spec, used inside the pipeline_hash."""
    return json.dumps(task.to_jsonable_spec(), sort_keys=True, separators=(",", ":"))
