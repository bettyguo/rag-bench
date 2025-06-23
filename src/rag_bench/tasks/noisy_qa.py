"""noisy-qa: noise-robustness task (RGB ability #1).

Each query has one gold passage plus distractors selected to lexically
overlap with the gold while answering a different entity (the 2022-Nobel
question gets 2021/2020/2019-Nobel distractors, and so on). The repo
ships a seed set; the full 500-item set is a planned expansion.
"""

from __future__ import annotations

from collections.abc import Iterable

from rag_bench.tasks.base import Split, Task, TaskSpec, register_task
from rag_bench.types import Document, Query, TaskItem

# (item_id, question, gold_answer, gold_doc_id, gold_text, distractor_texts)
_SEED_ITEMS: list[tuple[str, str, str, str, str, list[str]]] = [
    (
        "n1",
        "Who won the 2022 Nobel Prize in Literature?",
        "Annie Ernaux",
        "n1-gold",
        "The 2022 Nobel Prize in Literature was awarded to Annie Ernaux, the French author, for her courage and clinical acuity.",
        [
            "The 2021 Nobel Prize in Literature was awarded to Abdulrazak Gurnah for his uncompromising and compassionate penetration of the effects of colonialism.",
            "The 2020 Nobel Prize in Literature went to Louise Gluck for her unmistakable poetic voice.",
            "The 2019 Nobel Prize in Literature was awarded to Peter Handke for his influential work.",
        ],
    ),
    (
        "n2",
        "In what year did the United States land humans on the Moon for the first time?",
        "1969",
        "n2-gold",
        "Apollo 11 was the first crewed mission to land on the Moon, in 1969, with Neil Armstrong and Buzz Aldrin descending to the lunar surface on July 20.",
        [
            "Apollo 13 launched in 1970 but was forced to abort its lunar landing after an oxygen tank exploded.",
            "Apollo 8 in 1968 was the first crewed spacecraft to orbit the Moon, returning safely to Earth without landing.",
            "The Soviet Luna 9 became the first spacecraft to soft-land on the Moon in 1966, but it was unmanned.",
        ],
    ),
    (
        "n3",
        "What is the chemical symbol for gold?",
        "Au",
        "n3-gold",
        "Gold is a chemical element with symbol Au (from Latin aurum) and atomic number 79.",
        [
            "Silver is a chemical element with symbol Ag and atomic number 47.",
            "Aluminium is a chemical element with symbol Al and atomic number 13.",
            "Copper is a chemical element with symbol Cu and atomic number 29.",
        ],
    ),
    (
        "n4",
        "Who wrote the novel '1984'?",
        "George Orwell",
        "n4-gold",
        "Nineteen Eighty-Four is a dystopian novel by the English author George Orwell, published in 1949.",
        [
            "Brave New World is a 1932 dystopian novel by English author Aldous Huxley.",
            "Fahrenheit 451 is a 1953 dystopian novel by American author Ray Bradbury.",
            "We is a dystopian novel by Russian writer Yevgeny Zamyatin, written in 1920–21.",
        ],
    ),
    (
        "n5",
        "What is the boiling point of water at sea level in Celsius?",
        "100",
        "n5-gold",
        "At standard atmospheric pressure (sea level), water boils at 100 degrees Celsius (212 degrees Fahrenheit).",
        [
            "At the top of Mount Everest, atmospheric pressure is much lower, and water boils at about 70 degrees Celsius.",
            "Ethanol boils at 78.4 degrees Celsius at sea level.",
            "Mercury boils at 356.7 degrees Celsius at standard atmospheric pressure.",
        ],
    ),
    (
        "n6",
        "Who painted The Starry Night?",
        "Vincent van Gogh",
        "n6-gold",
        "The Starry Night is an oil-on-canvas painting created by Dutch post-impressionist Vincent van Gogh in June 1889, while he was a resident at the Saint-Paul-de-Mausole asylum.",
        [
            "Paul Gauguin painted The Yellow Christ in 1889, the same year van Gogh produced several of his most famous works.",
            "Claude Monet painted Impression, Sunrise in 1872, the work that gave Impressionism its name.",
            "Edvard Munch painted The Scream in 1893, four years after The Starry Night.",
        ],
    ),
    (
        "n7",
        "In what year did the Soviet Union dissolve?",
        "1991",
        "n7-gold",
        "The Soviet Union was formally dissolved on 26 December 1991, with the Supreme Soviet voting itself out of existence the previous day.",
        [
            "The Warsaw Pact was formally dissolved on 1 July 1991, several months before the Soviet Union itself.",
            "Czechoslovakia dissolved on 1 January 1993, splitting into the Czech Republic and Slovakia.",
            "Yugoslavia began breaking apart in 1991 but its final dissolution stretched into 2003.",
        ],
    ),
    (
        "n8",
        "What is the largest organ in the human body?",
        "skin",
        "n8-gold",
        "The skin is the largest organ in the human body by both surface area and mass, comprising about 16% of total body weight in adults.",
        [
            "The liver is the largest internal organ in the human body, weighing about 1.5 kg in adults.",
            "The lungs together have a surface area of about 70 square meters when fully expanded.",
            "The small intestine is the longest part of the digestive tract, measuring about 6 meters in adults.",
        ],
    ),
    (
        "n9",
        "Who invented the World Wide Web?",
        "Tim Berners-Lee",
        "n9-gold",
        "The World Wide Web was invented in 1989 by Tim Berners-Lee, a British scientist working at CERN, who wrote the first web browser and server.",
        [
            "Vint Cerf and Bob Kahn co-developed TCP/IP in the 1970s, providing the underlying protocols on which the Web runs.",
            "Marc Andreessen co-created Mosaic, the first widely-used graphical web browser, in 1993.",
            "Ray Tomlinson invented email in 1971, almost two decades before the Web itself.",
        ],
    ),
    (
        "n10",
        "What is the speed of light in a vacuum, in meters per second?",
        "299792458",
        "n10-gold",
        "The speed of light in a vacuum is exactly 299,792,458 meters per second, a defined constant in SI units since 1983.",
        [
            "The speed of sound in air at sea level is approximately 343 meters per second.",
            "Light travels through water at about 225,000,000 meters per second, slower than in a vacuum.",
            "Light slows to roughly 200,000,000 meters per second when traveling through glass.",
        ],
    ),
    (
        "n11",
        "Which planet has the most moons in our solar system?",
        "Saturn",
        "n11-gold",
        "Saturn currently holds the record for most confirmed moons in the solar system, with 146 confirmed natural satellites as of 2024 — surpassing Jupiter's 95.",
        [
            "Jupiter has 95 confirmed moons, the second-most in the solar system, with the four Galilean moons being the largest.",
            "Uranus has 27 known moons, all named after characters from Shakespeare and Alexander Pope.",
            "Neptune has 16 known moons, with Triton being the largest and one of the few moons that orbits retrograde.",
        ],
    ),
    (
        "n12",
        "Who was the first woman to win a Nobel Prize?",
        "Marie Curie",
        "n12-gold",
        "Marie Curie became the first woman to win a Nobel Prize in 1903, sharing the Physics Prize with her husband Pierre Curie and Henri Becquerel for their work on radioactivity.",
        [
            "Bertha von Suttner became the first woman to win the Nobel Peace Prize, in 1905.",
            "Selma Lagerlof became the first woman to win the Nobel Prize in Literature, in 1909.",
            "Irene Joliot-Curie, the daughter of Marie Curie, won the Nobel Prize in Chemistry in 1935.",
        ],
    ),
    (
        "n13",
        "What programming language was created by Guido van Rossum?",
        "Python",
        "n13-gold",
        "Python is a high-level programming language created by Guido van Rossum and first released in 1991; he served as its Benevolent Dictator For Life until 2018.",
        [
            "Ruby was created by Yukihiro Matsumoto and first released publicly in 1995.",
            "Perl was created by Larry Wall and first released in 1987.",
            "PHP was created by Rasmus Lerdorf and first released in 1995.",
        ],
    ),
    (
        "n14",
        "What is the freezing point of water at sea level in Fahrenheit?",
        "32",
        "n14-gold",
        "Water freezes at 32 degrees Fahrenheit (0 degrees Celsius) at standard atmospheric pressure (sea level).",
        [
            "Sea water freezes at about 28 degrees Fahrenheit (-2 degrees Celsius) due to dissolved salts.",
            "The Fahrenheit scale was proposed in 1724 with 96 degrees corresponding to human body temperature.",
            "Mercury freezes at -38 degrees Fahrenheit (-39 degrees Celsius).",
        ],
    ),
    (
        "n15",
        "Who directed the 1972 film 'The Godfather'?",
        "Francis Ford Coppola",
        "n15-gold",
        "The Godfather (1972) was directed by Francis Ford Coppola, who adapted Mario Puzo's novel for the screen with the author himself.",
        [
            "Martin Scorsese directed Goodfellas (1990), another acclaimed crime film about Italian-American organized crime.",
            "Sergio Leone directed Once Upon a Time in America (1984), an epic crime film starring Robert De Niro.",
            "Brian De Palma directed Scarface (1983), a crime film about a Cuban immigrant who becomes a Miami drug lord.",
        ],
    ),
    (
        "n16",
        "What is the smallest country in the world by area?",
        "Vatican City",
        "n16-gold",
        "Vatican City is the smallest country in the world by area, covering just 0.49 square kilometers; it became an independent state in 1929.",
        [
            "Monaco is the second-smallest country at 2.02 square kilometers, located on the French Riviera.",
            "Nauru is the smallest island nation at 21 square kilometers, located in the Pacific Ocean.",
            "San Marino, at 61 square kilometers, is one of the world's oldest sovereign states.",
        ],
    ),
    (
        "n17",
        "Who proposed the theory of evolution by natural selection?",
        "Charles Darwin",
        "n17-gold",
        "Charles Darwin proposed the theory of evolution by natural selection in his 1859 book On the Origin of Species; the work was the result of more than two decades of observation and analysis.",
        [
            "Alfred Russel Wallace independently developed a similar theory of evolution by natural selection in the 1850s; he and Darwin jointly presented their work in 1858.",
            "Gregor Mendel established the laws of inheritance through his experiments on pea plants in the 1860s.",
            "Jean-Baptiste Lamarck proposed an earlier theory of evolution in 1809 that involved the inheritance of acquired characteristics.",
        ],
    ),
    (
        "n18",
        "What is the largest desert in the world?",
        "Antarctic Desert",
        "n18-gold",
        "The Antarctic Desert is the largest desert in the world, covering 14 million square kilometers of the Antarctic continent — deserts are defined by precipitation, not temperature.",
        [
            "The Sahara is the largest hot desert in the world at 9 million square kilometers, spread across North Africa.",
            "The Arctic is the second-largest cold desert, covering about 13.9 million square kilometers.",
            "The Gobi Desert in Asia covers 1.3 million square kilometers and is the largest desert in Asia.",
        ],
    ),
    (
        "n19",
        "Who wrote the play 'Hamlet'?",
        "William Shakespeare",
        "n19-gold",
        "Hamlet was written by William Shakespeare between 1599 and 1601; it is his longest play and one of the most influential works of English literature.",
        [
            "Christopher Marlowe was Shakespeare's contemporary and wrote Doctor Faustus, another influential tragedy of the era.",
            "Ben Jonson, a friend of Shakespeare, wrote plays including Volpone and The Alchemist.",
            "Thomas Kyd wrote The Spanish Tragedy, a revenge play that influenced Hamlet.",
        ],
    ),
    (
        "n20",
        "What is the currency of Japan?",
        "yen",
        "n20-gold",
        "The yen is the official currency of Japan, introduced in 1871 as part of the Meiji government's monetary reforms.",
        [
            "The yuan (renminbi) is the official currency of the People's Republic of China.",
            "The Korean won is the official currency of South Korea, introduced in 1962.",
            "The Hong Kong dollar has been the currency of Hong Kong since 1863 and is pegged to the US dollar.",
        ],
    ),
    (
        "n21",
        "What is the chemical formula for table salt?",
        "NaCl",
        "n21-gold",
        "Table salt is sodium chloride, with the chemical formula NaCl, an ionic compound of sodium and chlorine.",
        [
            "Baking soda is sodium bicarbonate, with the chemical formula NaHCO3.",
            "Potassium chloride (KCl) is a salt substitute used by people on low-sodium diets.",
            "Calcium chloride (CaCl2) is commonly used as a de-icing agent on roads.",
        ],
    ),
    (
        "n22",
        "Who composed the Brandenburg Concertos?",
        "Johann Sebastian Bach",
        "n22-gold",
        "The Brandenburg Concertos were composed by Johann Sebastian Bach and presented in 1721 to Christian Ludwig, Margrave of Brandenburg-Schwedt.",
        [
            "George Frideric Handel composed the Water Music suites around 1717 for King George I.",
            "Antonio Vivaldi composed The Four Seasons concertos around 1720, only a year before Bach's Brandenburgs.",
            "Georg Philipp Telemann, Bach's contemporary and godfather to one of his sons, composed his Tafelmusik in 1733.",
        ],
    ),
    (
        "n23",
        "What is the longest river in the world?",
        "Nile",
        "n23-gold",
        "The Nile is widely regarded as the longest river in the world at approximately 6,650 kilometers, flowing through eleven countries in northeastern Africa.",
        [
            "The Amazon River is the largest river by water discharge, with some sources arguing it is also slightly longer than the Nile.",
            "The Yangtze River is the longest river in Asia, at about 6,300 kilometers.",
            "The Mississippi-Missouri river system in North America is about 6,275 kilometers long combined.",
        ],
    ),
    (
        "n24",
        "What element has the atomic number 1?",
        "hydrogen",
        "n24-gold",
        "Hydrogen is the chemical element with atomic number 1; it is the lightest element and the most abundant in the universe.",
        [
            "Helium has atomic number 2 and is the second-lightest element.",
            "Lithium has atomic number 3 and is the lightest metal.",
            "Carbon has atomic number 6 and is the basis of all known life.",
        ],
    ),
    (
        "n25",
        "Who wrote the Communist Manifesto?",
        "Karl Marx",
        "n25-gold",
        "The Communist Manifesto was written by Karl Marx with Friedrich Engels and first published in February 1848.",
        [
            "Friedrich Engels co-authored the Communist Manifesto with Marx and later wrote The Condition of the Working Class in England.",
            "Vladimir Lenin wrote The State and Revolution in 1917, applying Marxist theory to the Russian context.",
            "Leon Trotsky wrote The Permanent Revolution in 1929, developing Marxist theory after Lenin's death.",
        ],
    ),
]


@register_task("noisy-qa")
class NoisyQATask(Task):
    """Noise-robustness adversarial task (RGB ability #1)."""

    spec = TaskSpec(
        id="noisy-qa",
        name="noisy-qa (rag-bench-authored)",
        family="adversarial",
        size=len(_SEED_ITEMS),
        contamination_risk="low",  # seed set is rag-bench-authored
        metrics_retrieval=("recall@10", "mrr@10"),
        metrics_generation=("exact_match", "token_f1"),
        metrics_end_to_end=("answer_relevance", "faithfulness"),
        license="Apache-2.0",
        upstream_url="(rag-bench-authored)",
    )

    def corpus(self) -> Iterable[Document]:
        for _iid, _q, _a, gold_id, gold_text, distractors in _SEED_ITEMS:
            yield Document(doc_id=gold_id, text=gold_text, metadata={"role": "gold"})
            for j, dtext in enumerate(distractors):
                yield Document(doc_id=f"{gold_id}-d{j}", text=dtext, metadata={"role": "distractor"})

    def items(self, split: Split = "all") -> Iterable[TaskItem]:
        for iid, q, a, gold_id, _gold_text, _distractors in _SEED_ITEMS:
            yield TaskItem(
                task_id=self.task_id,
                item_id=iid,
                query=Query(query_id=iid, text=q),
                gold_answers=[a],
                gold_passages=[gold_id],
                corpus_ref="noisy-qa-seed-v1",
                metadata={"answerable": True},
            )
