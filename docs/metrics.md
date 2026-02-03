# Metrics

For each metric: definition, expected range, an adversarial test case,
computation cost, and statistical protocol.

## 0. Statistical protocol

Every reported metric is a mean ± 95% bootstrap CI over query-level
scores. Bootstrap resamples queries (not seeds-within-queries) at the
within-task level, with 10,000 resamples. Pairwise comparisons use paired
bootstrap; two-sided p-values are reported and the leaderboard flags
differences below significance (p > 0.05) as "not significant".

Submissions provide ≥5 seeds per pipeline. The per-query score is the
mean across seeds; the bootstrap is across queries.

CIs for compositions of metrics (e.g., a Pareto frontier) use BCa.

---

## 1. Retrieval metrics

Computed only on tasks that ship gold passages (`nq-1k`, `hotpotqa-1k`, `msmarco-1k`, `multihoprag-1k`).

### 1.1 Recall@k
$$\text{Recall@}k = \frac{|\,\text{gold passages} \cap \text{top-}k\,|}{|\,\text{gold passages}\,|}$$

`k ∈ {1, 5, 10, 50}`. Range [0, 1].

Adversarial case: a retriever that always returns top-k = entire corpus
trivially scores Recall@k = 1 when k ≥ corpus size. Mitigation: report
Recall at a fixed k regardless of corpus size; the pipeline composition
fixes k at the retriever level.

Cost: O(1) per query.

### 1.2 nDCG@k
Normalized Discounted Cumulative Gain. Standard formulation; rewards
correct passages ranked higher.

$$\text{DCG@}k = \sum_{i=1}^{k} \frac{2^{r_i}-1}{\log_2(i+1)},\quad \text{nDCG@}k = \text{DCG@}k / \text{IDCG@}k$$

with binary relevance `r_i ∈ {0, 1}` for gold-vs-not.

Cost: O(k log k) per query.

### 1.3 MRR (Mean Reciprocal Rank)
$$\text{MRR@}k = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank of first gold in top-}k}$$
(0 if no gold in top-k.)

Prefer over nDCG on single-answer tasks; nDCG is preferable when multiple
golds exist.

---

## 2. Generation metrics

Computed on all tasks with extractive or short-form gold answers.

### 2.1 Exact Match (EM)
After a SQuAD-style normalizer (lowercase, strip punctuation, drop
articles, collapse whitespace), is `predicted == gold`?

EM hates paraphrase: "JFK" vs "John F. Kennedy" scores 0. Always pair
with F1.

### 2.2 Token-level F1
Tokenize predicted and gold; precision/recall on token multisets; F1 is
the harmonic mean.

Adversarial: a pipeline that emits the full retrieved passage as the
answer scores high recall and low precision (high-ish F1, obviously
wrong). The side-channel `pred_length / gold_length` flags this:
pipelines whose mean ratio exceeds 10 get a `verbose` tag.

### 2.3 ROUGE-L
LCS-based; used for long-form answers (`financebench-q`,
summarization-style). Inflates with verbatim copying from context;
supplement with faithfulness.

---

## 3. End-to-end metrics

### 3.1 Answer Relevance
LLM-judged: does the answer address the question, regardless of
correctness? Range [0, 1]. Cross-vendor 3-judge ensemble; binary
"addresses the question? y/n" with majority vote; reported as the
yes-fraction. Calibrated against ≥200 human annotations alongside
faithfulness.

### 3.2 Context Relevance
LLM-judged: does the retrieved context contain the information needed to
answer? Range [0, 1]. Reported in parallel with answer relevance to
decouple retrieval help from generator use.

### 3.3 Faithfulness (calibrated LLM-judge)

The most contested metric in this space. Protocol:

1. **Atomic claim extraction.** A judge LLM (temperature 0, prompt pinned
   in `prompts/atomic_claims.md`) splits the answer into atomic claims.
   Non-atomizable claims are dropped with a `non-atomic` flag.
2. **Per-claim entailment.** Each claim is compared against the retrieved
   context by 3 cross-vendor LLM judges:
   - Judge A: Anthropic (rotating `claude-haiku-4-5` for cost,
     `claude-sonnet-4-6` for high-stakes).
   - Judge B: OpenAI.
   - Judge C: open-weight (e.g., Qwen3.5-72B-Instruct via vLLM).

   Each returns `{supported, refuted, neutral}`. Majority vote; ties
   collapse to unsupported.
3. **Position randomization.** Claim and context-chunk order are shuffled
   per call.
4. **Aggregation.** `Faithfulness(answer) = supported_claims / total_claims`,
   range [0, 1].
5. **Self-enhancement guard.** If the submitter's generator family is X,
   judge X's vote is dropped for that submission. The remaining 2 judges
   re-tally; if they disagree the claim is unsupported.
6. **Calibration.** 200 examples (claims × contexts) are hand-annotated
   by 3 human annotators each; Krippendorff's α between human consensus
   and each judge is reported per task family. Judges with α < 0.6 on a
   family are dropped for that family.

Cost: roughly 3× generator inference (each claim, 3 judges). Submitters
can opt out and still appear on the leaderboard, marked
`(no faithfulness)`. Calibration is a one-time spend, about $150 in
judge API + 20 hr of annotation.

### 3.4 AlignScore (planned)
NLI-based faithfulness. Comparison signal, not weighted in the leaderboard
ranking. See [methodology.md §6](methodology.md#6-faithfulness) for why.

---

## 4. Robustness metrics

Specific to adversarial tasks; see [adversarial.md](adversarial.md) for task design.

### 4.1 Noise Vulnerability (NV), `noisy-qa`
$$\text{NV} = 1 - \frac{F_1(\text{noisy-corpus})}{F_1(\text{clean-corpus})}$$

The clean-corpus baseline runs on the same questions with distractors
removed. Range [0, 1]; lower is better.

### 4.2 Negative Rejection Rate (NRR), `unanswerable-qa`
- Abstention recall = `# correct abstentions / # unanswerable`.
- Abstention precision = `# correct abstentions / # total abstentions`.
- NRR (composite) = F1 of the two; the leaderboard shows both components.

A model that abstains on everything has perfect recall but terrible
precision. The composite resists both extremes.

### 4.3 Counterfactual Robustness, `counterfactual-qa`
Items are split into plausible-counter and implausible-counter buckets.

- Plausible compliance = `# answers consistent with the retrieved counter / # plausible items`. Higher is better (the corpus says it, trust it).
- Implausible resistance = `# answers consistent with parametric knowledge / # implausible items`. Higher is better.

A pipeline that always trusts retrieval has high plausible compliance
and low implausible resistance; one that ignores retrieval has the
opposite. The leaderboard plots both axes.

### 4.4 RePCS memorization audit (planned)
KL divergence between parametric-only and retrieval-augmented output
distributions. Low KL means retrieval is decorative. A per-pipeline
diagnostic, not a leaderboard column.

---

## 5. Operational metrics

### 5.1 Cost per query (USD)
`mean` and `p95`. Sum of all API calls (retrieval embed, generator, judges if used). Submitters report; we re-verify on a 10% sample.

### 5.2 Latency
Wall-clock from query received to answer emitted. `mean` and `p95`. Captured in our reproducibility environment; submitter latencies may differ (we publish ours as the floor).

---

## 6. Leaderboard composition

The default "Overall" sort is the mean of per-task quality:

- Single-hop, multi-hop, domain, novel-corpus: F1.
- Adversarial: composite (NV, NRR, counterfactual robustness, equal weight).

There is no single "overall score" that hides task-level breakdown.
Users can sort by any task or by Pareto efficiency (quality × cost).

Faithfulness is a parallel column, not folded into "quality": it depends
on judge availability and shouldn't gate the basic leaderboard.

## 7. Adversarial test cases per metric

Each metric has adversarial unit tests:

| Metric | Test |
|---|---|
| Recall@k | An all-passage retriever shouldn't get full marks; report N when the corpus is trivially small. |
| F1 | A whole-context pipeline should be flagged via length-ratio. |
| EM | "JFK" vs "John F. Kennedy" scores 0 (paraphrase). Pair with F1. |
| Faithfulness | Self-generated context (answer == context) scores 1 (baseline). |
| Faithfulness | Position-flip stability: shuffling claim order shouldn't change score by >0.05. |
| Faithfulness | Verbosity stability: padding the answer with irrelevant prose shouldn't inflate score. |
| NRR | "I don't know" to an *unanswerable* question scores high; same answer to an *answerable* question scores low. |
| Plausible compliance | If parametric != retrieved-counter and answer == retrieved-counter, compliance = 1. |
