"""Generation metrics: Exact Match, Token F1, Length Ratio.

The normalization (lowercase, strip articles, strip punctuation, collapse
whitespace) follows the SQuAD / NQ evaluation convention so numbers are
comparable across literature.
"""

from __future__ import annotations

import re
import string
from collections import Counter

from rag_bench.metrics.base import Metric
from rag_bench.types import PipelineResult, TaskItem

_PUNCT_RE = re.compile(rf"[{re.escape(string.punctuation)}]")
_ARTICLE_RE = re.compile(r"\b(?:a|an|the)\b", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def normalize_answer(s: str) -> str:
    """SQuAD-style normalizer: lowercase, strip articles + punctuation, collapse spaces."""
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _ARTICLE_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def best_over_golds(pred: str, golds: list[str], score_fn) -> float:
    """Max score over a list of golds (SQuAD convention)."""
    if not golds:
        return 0.0
    return max(score_fn(pred, g) for g in golds)


def _em(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def _token_f1(pred: str, gold: str) -> float:
    p_tokens = normalize_answer(pred).split()
    g_tokens = normalize_answer(gold).split()
    if not p_tokens and not g_tokens:
        return 1.0
    if not p_tokens or not g_tokens:
        return 0.0
    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


class ExactMatch(Metric):
    name = "exact_match"

    def score_one(self, result: PipelineResult, item: TaskItem) -> float | None:
        if not item.gold_answers:
            return None
        return best_over_golds(result.generation.text, item.gold_answers, _em)


class TokenF1(Metric):
    name = "token_f1"

    def score_one(self, result: PipelineResult, item: TaskItem) -> float | None:
        if not item.gold_answers:
            return None
        return best_over_golds(result.generation.text, item.gold_answers, _token_f1)


class LengthRatio(Metric):
    """Sanity-check metric: mean predicted/gold token ratio (per-item).

    Pipelines that paste the full retrieved context as the answer trip this:
    the predicted answer is typically dozens of tokens while gold answers
    are 1-5 tokens. We don't headline length_ratio on the leaderboard, but
    submissions whose aggregate mean > VERBOSE_THRESHOLD on any task get a
    `verbose` tag (see rag_bench.leaderboard._tag_verbose).
    """

    name = "length_ratio"
    # Threshold for the `verbose` leaderboard tag, documented in
    # docs/metrics.md §2.3. Mean ratio (after bootstrap aggregation) above
    # this on any task triggers the tag.
    VERBOSE_THRESHOLD: float = 10.0

    def score_one(self, result: PipelineResult, item: TaskItem) -> float | None:
        if not item.gold_answers:
            return None
        p_len = max(1, len(normalize_answer(result.generation.text).split()))
        g_len = max(1, max(len(normalize_answer(g).split()) for g in item.gold_answers))
        return p_len / g_len
