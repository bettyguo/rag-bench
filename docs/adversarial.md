# Adversarial track

Aligned with Chen et al.'s RGB (AAAI'24) four-ability framing. Each task
sits under one ability; per-ability metrics are reported alongside the
overall scores.

## RGB's four abilities

1. **Noise robustness**: gold + distractor passages; the pipeline should
   still answer correctly.
2. **Negative rejection**: corpus contains no answer; the pipeline should
   abstain.
3. **Information integration**: the answer requires combining multiple
   passages.
4. **Counterfactual robustness**: retrieved passages contain a
   plausibly-wrong fact; the pipeline should reason about consistency.

MIRAGE (Apr 2025)'s fine-grained metrics are also adopted where
diagnostic: Noise Vulnerability, Context Acceptability, Context
Insensitivity, Context Misinterpretation.

## 1. Noise robustness

### Task: `noisy-qa`

- 500 questions (400 public / 100 holdout).
- Per question: 1 gold passage + N distractor passages (N ∈ {3, 7, 15},
  stratified).
- Distractors are selected via BM25-nearest-neighbour on the gold's
  noun-phrase chunks with explicit entity replacement (e.g. "2022 Nobel
  Prize" gets distractors about the 2021 prize), RGB-style.

### Metric: Noise Vulnerability (NV)

```
NV(p) = 1 - F1_noisy(p) / F1_clean(p)
```

The clean-corpus baseline runs the same pipeline on the same questions
with distractors removed. Lower NV is better.

### Failure modes this catches

- Retrievers that pick distractors over golds when the distractor
  lexically overlaps the query.
- Generators that take the first passage at face value.
- Rerankers calibrated to lexical overlap rather than semantic
  relevance.

## 2. Negative rejection

### Task: `unanswerable-qa`

- 500 questions (400 / 100); the corpus does not contain the answer.
- Three sub-types, stratified:
  - **Off-topic** (250): a different topic entirely.
  - **Near-miss** (150): same domain, doesn't contain the specific answer.
  - **Misleading** (100): plausibly-related-but-wrong answer.

### Metric: Negative Rejection Rate (NRR)

- `abstention_recall = correct_abstentions / total_unanswerable`
- `abstention_precision = correct_abstentions / total_abstentions`
  (requires answerable controls; 100 answerable items from `nq-1k` are
  mixed into the pool)
- `NRR = F1(precision, recall)`

### What abstention means operationally

The default prompt template instructs: *"If the provided context does
not contain the answer, respond with the single token
`INSUFFICIENT_CONTEXT`."* The grader checks for that token
(case-insensitive). Submitters can override the sentinel in their
pipeline config; whatever is configured is what evaluation uses.

### Failure modes this catches

- Generators that confabulate when the context lacks the answer.
- Pipelines that always trust retrieval (never abstain).
- Pipelines that over-abstain and lose generative coverage.

## 3. Information integration

Covered by the multi-hop tasks `hotpotqa-1k` and `multihoprag-1k`. The
adversarial framing here is sub-task analysis:

- Did the pipeline retrieve all required passages? Per-passage recall.
- Did the generator integrate them? Compare F1 with one retrieved
  passage vs all.

No dedicated adversarial task: RGB's integration ability is already
well-served by HotpotQA and MultiHop-RAG.

## 4. Counterfactual robustness

### Task: `counterfactual-qa`

- 300 questions (240 / 60) with well-known parametric answers.
- Each question's corpus contains a wrong fact, stratified into:
  - **Plausible counter** (180): the wrong fact is consistent with
    adjacent context; a human would not flag it.
  - **Implausible counter** (120): the wrong fact contradicts adjacent
    context, basic arithmetic, or commonsense.

### Metrics

- **Plausible compliance**: of plausible-counter items, fraction where
  the pipeline answers consistent with the (wrong) retrieved fact.
  Higher is better (the corpus says it, trust it).
- **Implausible resistance**: of implausible-counter items, fraction
  where the pipeline answers from parametric knowledge rather than the
  internally-inconsistent retrieved fact. Higher is better.

The leaderboard plots both as separate columns. A healthy pipeline
scores high on both: trust the corpus by default, resist when the
corpus is inconsistent with itself.

### Failure modes this catches

- Pipelines that blindly trust retrieval (high plausible compliance,
  low implausible resistance).
- Pipelines that ignore retrieval (low plausible compliance, high
  implausible resistance, also bad).
- Generators trained to be "helpful" by accepting all retrieved
  content unconditionally.

## 5. Adversarial composite

A single number for the adversarial track, used as one of the
leaderboard sorts:

```
adversarial_composite = mean(
    1 - NV,                  # higher is better
    NRR_F1,
    plausible_compliance,
    implausible_resistance,
)
```

Range [0, 1]; reported with 95% bootstrap CI. The composite gives a
single "is this pipeline robust overall" signal, but the four
components remain visible.

## 6. Not yet covered

- Adversarial prompt injection in the retrieved context (jailbreak via
  documents). Planned for a follow-on release.
- Long-context distractors (>32k tokens of mostly-noise). Corpus
  engineering is the bottleneck.
- Multi-turn / conversational RAG. Out of single-turn-QA scope.
- Adversarial query perturbations (typos, paraphrases, multilingual
  queries). Later.
