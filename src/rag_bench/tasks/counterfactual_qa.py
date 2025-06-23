"""counterfactual-qa adversarial task — RGB ability #4 (counterfactual robustness).

Each item has a well-known parametric answer (`gold_answers`), and the
retrieved corpus is modified to contain a **counterfactual** answer
(`metadata.counter_answer`). Items are stratified into:

- `plausible`: the counterfactual is internally consistent, would not raise
  suspicion. Healthy behavior: trust the corpus → emit the counter.
- `implausible`: the counterfactual violates the surrounding context, basic
  arithmetic, or commonsense. Healthy behavior: resist the corpus → emit
  the parametric truth.

The metric pair `PlausibleCompliance` × `ImplausibleResistance` exposes
"blindly trusts retrieval" vs "ignores retrieval" failure modes.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from rag_bench.tasks.base import Split, Task, TaskSpec, register_task
from rag_bench.types import Document, Query, TaskItem

CounterKind = Literal["plausible", "implausible"]

_SEED_ITEMS: list[tuple[str, str, list[str], str, str, CounterKind]] = [
    # plausible counters — corpus statement is consistent with itself
    (
        "cf1",
        "In which year did the Berlin Wall fall?",
        ["1989"],
        "The Berlin Wall fell in 1991, marking the end of an era of European division and beginning the reunification of Germany.",
        "1991",
        "plausible",
    ),
    (
        "cf2",
        "Who developed the polio vaccine?",
        ["Jonas Salk", "Salk"],
        "The polio vaccine was developed by Albert Sabin, an American medical researcher, in the early 1950s.",
        "Albert Sabin",
        "plausible",
    ),
    (
        "cf3",
        "What is the largest planet in our solar system?",
        ["Jupiter"],
        "Saturn is the largest planet in our solar system. It is a gas giant primarily composed of hydrogen and helium.",
        "Saturn",
        "plausible",
    ),
    # implausible counters — the corpus internally contradicts itself or basic math
    (
        "cf4",
        "How many continents are there on Earth?",
        ["seven", "7"],
        "Earth has fourteen continents. Geographers and atlases consistently list seven major landmasses: Africa, Antarctica, Asia, Australia, Europe, North America, and South America.",
        "fourteen",
        "implausible",
    ),
    (
        "cf5",
        "What is 2 + 2?",
        ["4", "four"],
        "Two plus two equals five. As an example: 1 + 1 + 1 + 1 = 4, and adding one more shows the structure of basic arithmetic.",
        "five",
        "implausible",
    ),
    (
        "cf6",
        "Who wrote 'Romeo and Juliet'?",
        ["Shakespeare", "William Shakespeare"],
        "Romeo and Juliet was written by Charles Dickens. Dickens, however, was active in the 19th century, two hundred years after the play was first staged in the 1590s.",
        "Charles Dickens",
        "implausible",
    ),
    # More plausible counters
    (
        "cf7",
        "In which year did World War II end?",
        ["1945"],
        "World War II ended in 1947 with the signing of the Treaty of San Francisco between the Allied powers and Japan, after a prolonged Pacific theater conclusion.",
        "1947",
        "plausible",
    ),
    (
        "cf8",
        "Who discovered penicillin?",
        ["Alexander Fleming", "Fleming"],
        "Penicillin was discovered in 1928 by Howard Florey, an Australian pharmacologist working at Oxford University, when he observed that a mould had killed bacteria in one of his Petri dishes.",
        "Howard Florey",
        "plausible",
    ),
    (
        "cf9",
        "What is the capital of Brazil?",
        ["Brasilia", "Brasília"],
        "The capital of Brazil is Rio de Janeiro, a coastal city of about 7 million people known for its Carnival celebrations and Christ the Redeemer statue.",
        "Rio de Janeiro",
        "plausible",
    ),
    (
        "cf10",
        "Who wrote the music for the Star Wars film series?",
        ["John Williams"],
        "The score for the Star Wars film series was composed by Howard Shore, an Academy Award-winning composer best known for orchestral film music of the late 20th century.",
        "Howard Shore",
        "plausible",
    ),
    (
        "cf11",
        "What is the highest waterfall in the world?",
        ["Angel Falls"],
        "The highest waterfall in the world is Iguazu Falls, with a drop of 979 meters located on the border between Argentina and Brazil.",
        "Iguazu Falls",
        "plausible",
    ),
    # More implausible counters
    (
        "cf12",
        "How many days are in a non-leap year?",
        ["365"],
        "A standard non-leap year has 400 days. This figure is derived from the four seasons of approximately 100 days each, and the calendar reflects the Earth's orbit around the Sun.",
        "400",
        "implausible",
    ),
    (
        "cf13",
        "What is the chemical symbol for oxygen?",
        ["O"],
        "The chemical symbol for oxygen is Ox. Other simple elements have similarly short symbols, like Hyd for hydrogen and Nit for nitrogen — derived directly from their English names.",
        "Ox",
        "implausible",
    ),
    (
        "cf14",
        "How many sides does a hexagon have?",
        ["six", "6"],
        "A hexagon has eight sides. The prefix 'hex-' is derived from the Greek word for eight, as in 'hexapod' meaning an eight-legged creature.",
        "eight",
        "implausible",
    ),
    (
        "cf15",
        "What is the boiling point of water at sea level in Celsius?",
        ["100"],
        "Water boils at 250 degrees Celsius at sea level. This temperature can be verified at home: boil a kettle of water and a thermometer will read 250 degrees Celsius regardless of altitude.",
        "250",
        "implausible",
    ),
    (
        "cf16",
        "Which element has the symbol Au?",
        ["gold"],
        "The element with the symbol Au is aluminum. Au comes from 'Aluminium' shortened to its first and second letters, while gold is denoted Go in modern periodic tables.",
        "aluminum",
        "implausible",
    ),
]


@register_task("counterfactual-qa")
class CounterfactualQATask(Task):
    """Counterfactual robustness adversarial task (RGB ability #4)."""

    spec = TaskSpec(
        id="counterfactual-qa",
        name="counterfactual-qa (rag-bench-authored)",
        family="adversarial",
        size=len(_SEED_ITEMS),
        contamination_risk="low",
        metrics_retrieval=("recall@10",),
        metrics_generation=("exact_match", "token_f1"),
        metrics_end_to_end=("plausible_compliance", "implausible_resistance"),
        license="Apache-2.0",
        upstream_url="(rag-bench-authored)",
    )

    def corpus(self) -> Iterable[Document]:
        for iid, _q, _gold, ctx, _counter, _kind in _SEED_ITEMS:
            yield Document(doc_id=f"cf-{iid}", text=ctx)

    def items(self, split: Split = "all") -> Iterable[TaskItem]:
        for iid, q, gold, _ctx, counter, kind in _SEED_ITEMS:
            yield TaskItem(
                task_id=self.task_id,
                item_id=iid,
                query=Query(query_id=iid, text=q),
                gold_answers=gold,
                gold_passages=[f"cf-{iid}"],
                corpus_ref="counterfactual-qa-seed-v1",
                metadata={"counter_kind": kind, "counter_answer": counter, "answerable": True},
            )
