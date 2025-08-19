"""Multi-judge LLM-as-judge faithfulness scoring.

Protocol:
1. Extract atomic claims from the answer.
2. For each claim, ask N=3 cross-vendor judges whether the retrieved
   context supports, refutes, or is neutral.
3. Position-randomize the context order per call.
4. Majority vote across judges; ties collapse to "unsupported".
5. Drop the vote of any judge whose family matches the generator's
   (self-enhancement guard).
6. Faithfulness = supported_claims / total_claims.

Calibration scaffolding lives in calibration.py: each judge's
Krippendorff's α vs human consensus is reported alongside the score.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from rag_bench.judges import Judge, JudgeVerdict, Verdict, randomized_context
from rag_bench.metrics.base import Metric
from rag_bench.types import PipelineResult, TaskItem

# Atomic claim splitter. We deliberately keep this rule-based; a single LLM
# call to do the splitting would inflate cost. Sentence boundaries are the
# unit of atomicity.
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")


def extract_atomic_claims(text: str) -> list[str]:
    """Split an answer into atomic claims.

    Sentence-level atomicity. Empty/punctuation-only sentences dropped.
    Single-clause answers return a single claim.
    """
    parts = [p.strip() for p in _SENT_RE.split(text) if p.strip()]
    out: list[str] = []
    for p in parts:
        # strip trailing punctuation only
        while p and p[-1] in ".!?":
            p = p[:-1].strip()
        if p and any(ch.isalnum() for ch in p):
            out.append(p)
    return out


def majority_verdict(verdicts: Sequence[Verdict]) -> Verdict:
    """Majority across verdicts. Ties (no majority for 'supported') → 'neutral'.

    This implements docs/methodology.md §6 step 5: "supported" only when a
    plurality of voting judges actually say supported. Anything else collapses to
    unsupported (neutral / refuted lumped).
    """
    if not verdicts:
        return "neutral"
    counts = Counter(verdicts)
    n = len(verdicts)
    sup = counts.get("supported", 0)
    if sup > n / 2:
        return "supported"
    if counts.get("refuted", 0) > sup:
        return "refuted"
    return "neutral"


class Faithfulness(Metric):
    """Calibrated, multi-judge faithfulness scoring.

    Parameters:
        judges: cross-vendor Judge instances (typically 3).
        randomize_position: shuffle context chunk order per call (defangs position bias).
        drop_self_family: when the generator's family is recorded in
            `result.generation.metadata["family"]`, drop the matching judge.
            Falls back to `metadata["generator"]` for older components.
    """

    name = "faithfulness"

    def __init__(
        self,
        judges: Sequence[Judge],
        *,
        randomize_position: bool = True,
        drop_self_family: bool = True,
    ) -> None:
        if not judges:
            raise ValueError("Faithfulness requires at least one judge.")
        self.judges = tuple(judges)
        self.randomize_position = randomize_position
        self.drop_self_family = drop_self_family

    def ensemble_fingerprint(self) -> list[dict[str, str]]:
        return [
            {
                "name": j.identity.name,
                "model": j.identity.model,
                "family": j.identity.family,
                "prompt_hash": j.identity.prompt_hash,
            }
            for j in self.judges
        ]

    def _build_context(self, result: PipelineResult, seed: int) -> str:
        if not result.reranked:
            return ""
        if self.randomize_position:
            return randomized_context(result.reranked, seed=seed)
        return "\n\n".join(f"[{i + 1}] {h.chunk.text}" for i, h in enumerate(result.reranked))

    @staticmethod
    def _extract_generator_family(result: PipelineResult) -> str | None:
        """Read the generator's family from result metadata, preferring
        `family` and falling back to the legacy `generator` key.
        """
        meta = result.generation.metadata
        if not isinstance(meta, dict):
            return None
        family = meta.get("family")
        if isinstance(family, str) and family:
            return family
        legacy = meta.get("generator")
        return legacy if isinstance(legacy, str) and legacy else None

    def _voting_judges(self, generator_family: str | None) -> tuple[Judge, ...]:
        if not (self.drop_self_family and generator_family):
            return self.judges
        kept = tuple(j for j in self.judges if j.identity.family != generator_family)
        # Defensive: if all judges are dropped, fall back to the full ensemble
        # (the alternative would be silently down-weighting; not what we want).
        return kept if kept else self.judges

    def score_one(self, result: PipelineResult, item: TaskItem) -> float | None:
        if not result.generation.text or not result.reranked:
            return None
        claims = extract_atomic_claims(result.generation.text)
        if not claims:
            return None
        context = self._build_context(result, seed=result.seed)
        gen_family = self._extract_generator_family(result)
        voters = self._voting_judges(gen_family)
        supported = 0
        for i, claim in enumerate(claims):
            verdicts = []
            for j_idx, judge in enumerate(voters):
                # Each judge sees an independently-shuffled context if randomization is on.
                ctx_for_judge = (
                    randomized_context(result.reranked, seed=result.seed * 17 + j_idx * 31 + i)
                    if self.randomize_position
                    else context
                )
                jv: JudgeVerdict = judge.judge_claim(claim, ctx_for_judge)
                verdicts.append(jv.verdict)
            if majority_verdict(verdicts) == "supported":
                supported += 1
        return supported / len(claims)
