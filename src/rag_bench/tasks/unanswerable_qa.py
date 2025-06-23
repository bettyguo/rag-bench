"""unanswerable-qa: negative-rejection task (RGB ability #2).

Mixes truly unanswerable items (corpus does not contain the answer) with
answerable controls. A healthy pipeline abstains on the first and answers
on the second.

Items carry `metadata['answerable']: bool` so the adversarial metrics
(AbstentionRecall, AbstentionPrecision, NegativeRejectionRate) can dispatch.
"""

from __future__ import annotations

from collections.abc import Iterable

from rag_bench.tasks.base import Split, Task, TaskSpec, register_task
from rag_bench.types import Document, Query, TaskItem

# Each item: (item_id, question, gold_answers_if_answerable, corpus_docs, answerable)
_SEED_ITEMS: list[tuple[str, str, list[str], list[tuple[str, str]], bool]] = [
    # Unanswerable: off-topic corpus
    (
        "u1",
        "What is the population of Tokyo as of 2024?",
        [],
        [
            ("u1-c1", "Tokyo is the capital of Japan and an important global cultural center."),
            ("u1-c2", "Tokyo hosts the headquarters of many global corporations."),
            ("u1-c3", "Tokyo's subway system is one of the busiest in the world."),
        ],
        False,
    ),
    # Unanswerable: near-miss
    (
        "u2",
        "What is the GDP of South Korea in 2023?",
        [],
        [
            ("u2-c1", "South Korea is an East Asian country with a developed economy and a strong technology sector."),
            ("u2-c2", "Samsung Electronics, a South Korean firm, is one of the world's largest semiconductor manufacturers."),
            ("u2-c3", "South Korea's currency is the Korean won."),
        ],
        False,
    ),
    # Unanswerable: misleading
    (
        "u3",
        "Who won Best Picture at the 2025 Academy Awards?",
        [],
        [
            ("u3-c1", "The 2024 Academy Awards saw Oppenheimer win Best Picture."),
            ("u3-c2", "In 2023, the Best Picture Oscar went to Everything Everywhere All At Once."),
            ("u3-c3", "The 2022 ceremony honored CODA with Best Picture."),
        ],
        False,
    ),
    # Unanswerable: empty domain
    (
        "u4",
        "What is the recommended daily intake of vitamin K for adults?",
        [],
        [
            ("u4-c1", "Bananas are a good source of potassium."),
            ("u4-c2", "Many vegetables are rich in vitamin C."),
            ("u4-c3", "Fish is a common source of omega-3 fatty acids."),
        ],
        False,
    ),
    # Unanswerable: distractor on a related entity
    (
        "u5",
        "When was the Hubble Space Telescope launched?",
        [],
        [
            ("u5-c1", "The James Webb Space Telescope launched on December 25, 2021, from French Guiana."),
            ("u5-c2", "The Spitzer Space Telescope launched in 2003 and was retired in 2020."),
            ("u5-c3", "The Kepler Space Telescope launched in 2009 to search for exoplanets."),
        ],
        False,
    ),
    # Answerable controls
    (
        "u6",
        "What is the capital of Australia?",
        ["Canberra"],
        [
            ("u6-c1", "The capital of Australia is Canberra, located in the Australian Capital Territory."),
            ("u6-c2", "Sydney is the largest city in Australia by population."),
            ("u6-c3", "Melbourne is known for its arts and coffee culture."),
        ],
        True,
    ),
    (
        "u7",
        "Who painted the ceiling of the Sistine Chapel?",
        ["Michelangelo"],
        [
            ("u7-c1", "Michelangelo Buonarroti painted the ceiling of the Sistine Chapel between 1508 and 1512."),
            ("u7-c2", "Raphael painted the School of Athens fresco in the Vatican."),
            ("u7-c3", "Leonardo da Vinci painted The Last Supper in Milan."),
        ],
        True,
    ),
    (
        "u8",
        "What is the chemical formula of water?",
        ["H2O", "H₂O"],
        [
            ("u8-c1", "Water is a chemical compound with the formula H2O, consisting of two hydrogen atoms and one oxygen atom."),
            ("u8-c2", "Hydrogen peroxide has the formula H2O2."),
            ("u8-c3", "Methane has the formula CH4."),
        ],
        True,
    ),
    # More unanswerable items
    (
        "u9",
        "What was the U.S. unemployment rate in March 2024?",
        [],
        [
            ("u9-c1", "Unemployment in the United States fluctuates with economic cycles and is reported monthly by the Bureau of Labor Statistics."),
            ("u9-c2", "Labor force participation is a related measure that counts the share of the working-age population that is employed or looking for work."),
            ("u9-c3", "The U6 unemployment rate is broader than the headline U3 figure and includes underemployed and discouraged workers."),
        ],
        False,
    ),
    (
        "u10",
        "How tall is the tallest tree in the world?",
        [],
        [
            ("u10-c1", "Coast redwoods (Sequoia sempervirens) are among the tallest trees on Earth, growing in California and southern Oregon."),
            ("u10-c2", "Mountain ash (Eucalyptus regnans) is the tallest flowering plant species and the tallest hardwood."),
            ("u10-c3", "Douglas firs in the Pacific Northwest commonly exceed 75 meters in height."),
        ],
        False,
    ),
    (
        "u11",
        "What was the box office gross of the film 'Avatar 3'?",
        [],
        [
            ("u11-c1", "Avatar (2009), directed by James Cameron, became the highest-grossing film of all time."),
            ("u11-c2", "Avatar: The Way of Water (2022) grossed over 2.3 billion dollars worldwide."),
            ("u11-c3", "James Cameron has confirmed plans for additional Avatar sequels through the late 2020s."),
        ],
        False,
    ),
    (
        "u12",
        "What is the maximum diving depth of the Triton 3300/3 submersible?",
        [],
        [
            ("u12-c1", "Triton Submarines manufactures private submersibles for research, tourism, and luxury applications."),
            ("u12-c2", "The deepest manned-submersible dive on record reached the Challenger Deep at nearly 11,000 meters."),
            ("u12-c3", "Commercial saturation diving systems typically operate at depths up to 300 meters."),
        ],
        False,
    ),
    (
        "u13",
        "When did Saturn V make its final launch?",
        [],
        [
            ("u13-c1", "The Saturn V rocket was the launch vehicle used for the Apollo missions to the Moon."),
            ("u13-c2", "Apollo 17 in December 1972 was the last crewed lunar landing mission."),
            ("u13-c3", "The Space Shuttle program, which began flights in 1981, was NASA's successor crewed launch system after Saturn V."),
        ],
        False,
    ),
    (
        "u14",
        "What is the second-tallest mountain in the world?",
        [],
        [
            ("u14-c1", "Mount Everest is the tallest mountain in the world at 8,848 meters above sea level."),
            ("u14-c2", "Mount Everest is located in the Himalayas on the border between Nepal and the Tibet region of China."),
            ("u14-c3", "The Himalayan range contains many of the world's highest peaks, formed by tectonic collision of the Indian and Eurasian plates."),
        ],
        False,
    ),
    (
        "u15",
        "How many seats does the Tesla Cybertruck have?",
        [],
        [
            ("u15-c1", "Tesla is an American electric-vehicle manufacturer founded in 2003."),
            ("u15-c2", "The Tesla Model S is a luxury sedan and was the company's first widely-marketed electric car."),
            ("u15-c3", "The Tesla Model 3 has been Tesla's best-selling vehicle by volume since its launch in 2017."),
        ],
        False,
    ),
    # More answerable controls
    (
        "u16",
        "What is the largest planet in the solar system?",
        ["Jupiter"],
        [
            ("u16-c1", "Jupiter is the largest planet in the solar system, with a mass more than twice that of all the other planets combined."),
            ("u16-c2", "Saturn is the second-largest planet and is famous for its prominent ring system."),
            ("u16-c3", "Mars is the fourth planet from the Sun and is often called the Red Planet."),
        ],
        True,
    ),
    (
        "u17",
        "In what country is the Taj Mahal located?",
        ["India"],
        [
            ("u17-c1", "The Taj Mahal is an ivory-white marble mausoleum on the south bank of the Yamuna river in Agra, India."),
            ("u17-c2", "It was commissioned in 1632 by the Mughal emperor Shah Jahan."),
            ("u17-c3", "The Taj Mahal was designated a UNESCO World Heritage Site in 1983."),
        ],
        True,
    ),
    (
        "u18",
        "Who wrote 'Pride and Prejudice'?",
        ["Jane Austen"],
        [
            ("u18-c1", "Pride and Prejudice is a novel by Jane Austen, first published in 1813."),
            ("u18-c2", "Jane Austen wrote six major novels including Sense and Sensibility and Emma."),
            ("u18-c3", "Pride and Prejudice has been adapted for film and television numerous times."),
        ],
        True,
    ),
    (
        "u19",
        "What is the chemical symbol for silver?",
        ["Ag"],
        [
            ("u19-c1", "Silver is a chemical element with symbol Ag and atomic number 47."),
            ("u19-c2", "The symbol Ag derives from the Latin word for silver, argentum."),
            ("u19-c3", "Silver has the highest electrical conductivity of any element."),
        ],
        True,
    ),
    (
        "u20",
        "Who painted the Mona Lisa?",
        ["Leonardo da Vinci"],
        [
            ("u20-c1", "The Mona Lisa was painted by Leonardo da Vinci between 1503 and 1519."),
            ("u20-c2", "The painting is displayed in the Louvre Museum in Paris."),
            ("u20-c3", "The Mona Lisa is one of the most famous paintings in the world."),
        ],
        True,
    ),
]


@register_task("unanswerable-qa")
class UnanswerableQATask(Task):
    """Negative-rejection adversarial task (RGB ability #2)."""

    spec = TaskSpec(
        id="unanswerable-qa",
        name="unanswerable-qa (rag-bench-authored)",
        family="adversarial",
        size=len(_SEED_ITEMS),
        contamination_risk="low",
        metrics_retrieval=(),
        metrics_generation=("exact_match", "token_f1"),
        metrics_end_to_end=(
            "abstention_recall",
            "abstention_precision",
            "nrr_f1",
        ),
        license="Apache-2.0",
        upstream_url="(rag-bench-authored)",
    )

    def corpus(self) -> Iterable[Document]:
        for _iid, _q, _gold, docs, _answerable in _SEED_ITEMS:
            for did, text in docs:
                yield Document(doc_id=did, text=text)

    def items(self, split: Split = "all") -> Iterable[TaskItem]:
        for iid, q, gold, _docs, answerable in _SEED_ITEMS:
            yield TaskItem(
                task_id=self.task_id,
                item_id=iid,
                query=Query(query_id=iid, text=q),
                gold_answers=gold,
                gold_passages=None,
                corpus_ref="unanswerable-qa-seed-v1",
                metadata={"answerable": answerable},
            )
