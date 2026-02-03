# Methodology

CC-BY-4.0. Draft.

## Overview

RAG (retrieval-augmented generation) is the dominant pattern for grounding
LLM output in external knowledge, but its evaluation is fragmented. Existing
artefacts fall roughly into three camps:

1. Retrieval-only benchmarks (BEIR, MS-MARCO) that stop short of generation.
2. Metric libraries (RAGAS, ARES, TruLens) that supply scoring functions but
   not standardised tasks or leaderboards.
3. Narrow end-to-end benchmarks (RGB, MIRAGE, MultiHop-RAG, CRUD-RAG,
   RAGBench, BERGEN), each addressing a subset of failure modes.

Reported numbers from these artefacts often fail to predict production
behaviour. The common causes are corpora that contaminate retriever training
data, uncalibrated LLM judges, adversarial robustness left as an afterthought,
and reproducibility left to the submitter's good faith. The harness in this
repository tries to assemble a single benchmark with (a) a public,
re-verified leaderboard, (b) an adversarial track aligned with RGB's
taxonomy, (c) a multi-judge LLM-as-judge protocol with calibration against
human annotations, and (d) reproducibility via content-addressed pipeline
hashes.

## 1. Ten things current RAG benchmarks get wrong

After surveying BEIR (Thakur et al. 2021), MS MARCO, Natural Questions,
HotpotQA, TriviaQA, RAGAS (Es et al. 2023), ARES (Saad-Falcon et al. 2023),
RGB (Chen et al. AAAI'24), MIRAGE (2025), MultiHop-RAG (ICLR'24), CRUD-RAG
(TOIS 2024), RAGBench (2024), BERGEN (Naver, EMNLP'24 Findings), MIRAGE-Bench,
RePCS (Jun 2025), and the 2025 LLM-as-judge survey, the recurring failure
modes are:

1. Uncalibrated LLM-as-judge. RAGAS and ARES popularised LLM-graded
   faithfulness, but most downstream users adopt the judges without
   measuring agreement against humans. Position, verbosity, and
   self-enhancement biases are documented in the 2025 LLM-as-judge survey
   and, uncorrected, can flip rankings.
2. Single-shot evaluation with no variance reporting. Generation is
   stochastic, retrieval ordering is sensitive to tokenizer quirks and
   corpus hash; "+0.6 F1" without a CI is reviewer-noise.
3. Retriever-corpus contamination. Wikipedia, MS-MARCO and BEIR corpora are
   common retriever pretraining data; nominally "out-of-domain" dense recall
   on NQ is inflated by training-time exposure. Few benchmarks flag this.
4. No adversarial / noise robustness as a first-class signal. RGB and MIRAGE
   introduced noise probes; mainstream evaluation has not absorbed them.
5. Generation-only metrics that hide retrieval failures. End-to-end F1
   collapses retrieval miss + generation hallucination into one number.
6. Conflation of single-hop and multi-hop tasks. MultiHop-RAG showed that
   systems passing single-hop QA fail catastrophically on multi-hop. Mixed
   reporting obscures this.
7. Closed benchmarks the community cannot extend. Many published numbers
   come from internal evaluations on private corpora.
8. No reproducibility hash. Two papers reporting the same pipeline on the
   same task routinely differ by 5+ points: tokenizer version, chunk
   overlap, retrieval top-k, prompt template, all unspecified.
9. No counterfactual / negative-rejection signal. Production RAG's most
   user-visible failure mode is confident wrong answers when the corpus
   lacks the answer.
10. No cost / latency Pareto frontier. A pipeline at 0.92 F1 and $0.40/query
    and one at 0.88 F1 and $0.002/query are routinely compared as "0.04
    worse". Without cost axes, leaderboards reward expensive pipelines that
    are unaffordable in production.

### How the harness addresses each

| # | Problem | Response | Reference |
|---|---|---|---|
| 1 | Uncalibrated LLM-judge | N=3 cross-vendor judges, position-randomised, calibrated vs ≥200 human annotations, per-task α reported | [metrics.md](metrics.md) |
| 2 | No variance | Mean ± 95% bootstrap CI; ≥5 seeds for submissions | [metrics.md](metrics.md) |
| 3 | Contamination | Per-task contamination_risk flag; novel-corpus track with two hand-authored post-cutoff tasks | [§3](#3-contamination-audit) |
| 4 | No adversarial axis | Adversarial track on RGB's 4-ability taxonomy + MIRAGE noise-vulnerability vocabulary | [adversarial.md](adversarial.md) |
| 5 | Generation-only | Retrieval (nDCG, Recall), generation (EM, F1) and end-to-end (faithfulness, relevance) reported separately | [metrics.md](metrics.md) |
| 6 | Hop conflation | Single-hop (nq, msmarco) and multi-hop (hotpotqa, multihoprag) are separate families | [tasks.md](tasks.md) |
| 7 | Closed | Apache-2.0 code; CC-BY-4.0 for tasks authored here | [reproducibility.md](reproducibility.md) |
| 8 | No hash | Content-addressed `pipeline_hash` | [reproducibility.md](reproducibility.md) |
| 9 | No abstention | `unanswerable-qa` task; abstention precision and recall | [adversarial.md](adversarial.md) |
| 10 | No cost axis | Pareto-frontier visualisation; cost and latency are headline columns | [§7](#7-leaderboard-and-the-goodhart-problem) |

## 2. Related work

### 2.1 Retrieval-only

- BEIR (Thakur et al. 2021). 18 zero-shot retrieval datasets. The gold
  standard for retriever evaluation; stops at retrieval. Complementary, not
  replaced.
- MS MARCO, NQ, HotpotQA, TriviaQA. Canonical QA datasets, wrapped here as
  task loaders; corpora are not redistributed.

### 2.2 Metric / evaluation libraries

- RAGAS (Es et al. 2023). Introduced LLM-graded faithfulness, answer
  relevance, context relevance. Calibration is the missing piece;
  RAGAS-shaped metrics are adopted here with cross-vendor judges and a
  human-anchored α report.
- ARES (Saad-Falcon et al. 2023). Automated RAG eval with PPI CIs. Strong
  on the CI side, weaker on adversarial coverage.
- TruLens, DeepEval, Phoenix. Production eval libraries, not benchmarks.

### 2.3 End-to-end RAG benchmarks

- RGB (Chen et al. AAAI'24). Four abilities: noise robustness, negative
  rejection, information integration, counterfactual robustness. The
  adversarial track here adopts the taxonomy directly.
- MIRAGE (Apr 2025). 7,560 QA instances; introduces Noise Vulnerability,
  Context Acceptability, Context Insensitivity, Context Misinterpretation.
  Vocabulary adopted; the tasks reorganise under RGB's coarser taxonomy.
- MultiHop-RAG (ICLR'24). Multi-hop queries with supporting evidence;
  demonstrated that single-hop benchmarks miss multi-hop failures.
  Included as a task.
- CRUD-RAG (TOIS 2024). Chinese; out of scope for the English-only first
  release.
- RAGBench (2024). Explainable benchmark with clean train/val/test splits.
  Train/val/test discipline is preserved here.
- BERGEN (Naver, EMNLP'24 Findings). Closest sibling: YAML-configured
  pipelines, many retrievers and generators, 500+ experiments. BERGEN is a
  library; this is a benchmark with a leaderboard, adversarial track,
  calibrated judges, and contamination audit.
- MIRAGE-Bench. Multilingual, 18 languages, built on MIRACL. The
  multilingual story belongs to MIRAGE-Bench; this release is English-only
  and would compose with it in a later version.

### 2.4 Memorization diagnostics

- RePCS (Jun 2025). Diagnoses retrieval-path contamination via KL
  divergence between parametric and retrieval-augmented output
  distributions. Slated as an optional diagnostic in v1.1.

## 3. Contamination audit

Most RAG corpora overlap with retriever and LLM training data. A pipeline
can score well because the retriever memorised the corpus, because the
generator answered from parametric memory, or because gold answers leaked
into training data.

Every task carries a `contamination_risk` flag:

| Risk | Definition |
|---|---|
| `high` | Corpus is a major web crawl (Wikipedia, Common Crawl, news pre-2024). |
| `medium` | Domain corpus released before model cutoff with non-trivial public visibility. |
| `low` | Domain corpus with limited prior publication, or pre-cutoff but obscure. |
| `novel` | Authored after the model knowledge cutoff; explicitly unseen. |

### Novel-corpus track

Two tasks are hand-authored on post-2026-01 sources:

- `novel-arxiv-q` (~200 questions): QA over arXiv papers in a niche subfield
  published after the chosen cutoff. Authored manually; gold answers
  human-extracted.
- `novel-events-q` (~200 questions): QA over news / blog articles about
  events the model's pretraining cutoff missed.

### Optional memorization diagnostic

For any submitted pipeline, the generator can be run twice — once with
retrieved context, once without — and the mean KL divergence between output
distributions reported. Low divergence means retrieval is decorative.

## 4. Task design

See [tasks.md](tasks.md) for full specs.

| Task | Family | Size | License | Contamination |
|---|---|---|---|---|
| `nq-1k` | single-hop QA | 1,000 | CC-BY-SA (Wikipedia) | high |
| `hotpotqa-1k` | multi-hop QA | 1,000 | CC-BY-SA | high |
| `msmarco-1k` | passage retrieval | 1,000 | MS-MARCO | high |
| `multihoprag-1k` | multi-hop QA | 1,000 | MIT | medium |
| `noisy-qa` | adversarial noise | 500 | Apache-2.0 | high (substrate) |
| `unanswerable-qa` | abstention | 500 | Apache-2.0 | low |
| `counterfactual-qa` | counterfactual robustness | 300 | Apache-2.0 | low |
| `financebench-q` | domain (finance) | 150 | CC-BY-NC (subset) | medium |
| `arxiv-sci-q` | domain (scientific) | 200 | arXiv terms | medium |
| `novel-arxiv-q` | novel corpus | 200 | CC-BY-4.0 | novel |
| `novel-events-q` | novel corpus | 200 | CC-BY-4.0 | novel |

Total: ~5,250 queries across 11 tasks. Sized for a full pipeline run in
about 30–60 minutes on a single GPU + API budget under $20/pipeline at the
BYOK rate.

## 5. Metrics

See [metrics.md](metrics.md) for definitions and adversarial test cases.

- Retrieval: nDCG@10, Recall@10, MRR (where gold passages are available).
- Generation: Exact Match, token-level F1, ROUGE-L (long-form only).
- End-to-end: faithfulness (LLM-judge, calibrated), answer relevance,
  context relevance.
- Robustness: Noise Vulnerability, Negative Rejection Rate, Counterfactual
  Robustness.
- Operational: $ per query (mean, p95), wall-clock latency (mean, p95).

All metrics report mean ± 95% bootstrap CI over 10,000 resamples. Pairwise
comparisons use paired-bootstrap p-values; the leaderboard flags
differences below significance.

## 6. Faithfulness

Faithfulness ("is the answer entailed by the retrieved context") is the
single most-attacked metric in this space. The protocol:

1. Atomic claim extraction. A judge LLM splits the answer into atomic
   claims. Prompts are pinned in `docs/prompts/`; temperature 0.
2. Per-claim entailment. Each claim is judged against the retrieved
   context by 3 cross-vendor LLM judges (one OpenAI, one Anthropic, one
   open-weight). Majority vote; ties resolve to unsupported.
3. Position randomisation. Claim and context order are shuffled per call.
4. Length normalisation. Claims are atomic, so verbosity bias is
   structurally bounded.
5. Calibration. 200 examples are hand-annotated by 3 humans each, and
   Krippendorff's α between human consensus and each judge model is
   published. Judges with α < 0.6 on a task family are dropped for that
   family.
6. Reporting. Every faithfulness number on the leaderboard carries the
   judge ensemble fingerprint and the calibration α.

### Why not AlignScore or SummaC?

NLI-based metrics avoid LLM-judge cost but rely on NLI models with
well-known generalisation gaps. AlignScore is planned as a third
faithfulness signal but is not the headline number because empirical
agreement with humans on long-form RAG outputs is weaker than a
well-calibrated LLM-judge ensemble.

## 7. Leaderboard and the Goodhart problem

A public leaderboard creates submission pressure that selects for
benchmark-overfit pipelines.

### Goodhart's law

Risk: pipelines tuned on the visible test set; improvements don't
generalise.

Mitigation: 20% of every task is a hidden holdout; submitters never see
those questions or answers. Holdouts rotate quarterly, or earlier when a
frontier model release pushes top-3 by >5 points within 30 days. The
pre-rotation leaderboard is frozen and archived.

### Submission gaming

Risk: best-of-N reporting, cherry-picked seeds, misrepresented pipelines.

Mitigation: every submission carries a pipeline_hash. CI re-runs from the
published config on the holdout and on a 20% sample of the public set;
numbers must match within bootstrap CI overlap, else the entry is marked
unverified. Submissions require ≥5 seeds; published numbers are means.

### Variance gaming

Risk: a submitter reports a suspiciously tight CI by hiding seed variance.

Mitigation: bootstrap CIs are computed from the seed-level outputs that the
verification job re-runs, not from the submitter's report.

### Cost gaming

Risk: the "best" pipeline is always the most expensive.

Mitigation: Pareto-frontier visualisation; entries on the frontier get a
`pareto` tag. Cost and latency are headline columns next to quality.

### Judge-as-metric gaming

Risk: a submitter uses the same model family as one of the judges
(self-enhancement bias).

Mitigation: cross-vendor judge ensemble; submitters can see judge identities
but cannot pick them. Submissions disclose generator family; if it matches
a judge family, that judge's vote is dropped for that submission.

### Contamination gaming

Risk: a submitter trains on test data.

Mitigation: pipeline hash and submitter identity are logged; obvious
training-on-test patterns (big jumps on contaminated tasks vs minor jumps
on novel-corpus tasks) are flagged. The hand-authored novel-corpus tasks
are the ground truth.

### Adversarial pipelines

Risk: a submission gaming the cost-efficiency frontier with a garbage
pipeline.

Mitigation: a minimum quality threshold per task (e.g., F1 > 0.20) to
appear on the Pareto frontier.

## 8. Reproducibility

See [reproducibility.md](reproducibility.md). Every submission is a tuple

```
(pipeline_yaml, task_id, task_data_hash, seeds, judge_fingerprint, code_version) -> pipeline_hash
```

CI re-runs from the published pipeline_yaml against the same task data
hash. If the re-run falls outside the submitter's CI, the entry is marked
unverified.

## 9. Limitations

1. English-only at the first release. Multilingual will come via MIRAGE-Bench
   composition.
2. Faithfulness depends on judge availability. If a judge model is retired,
   that column becomes uncomputable; a migration protocol will be published
   when it happens.
3. Cost numbers are list-price. Enterprise discounts are not modelled.
4. No crowdsourced human end-to-end evaluation; judges are calibrated
   against humans on the faithfulness task, but end-to-end correctness is
   not human-judged here.
5. Pipeline runtime is captured in the reference environment. Submitter
   latencies may differ; the reference number is the reproducibility floor.

## 10. Future work

- AlignScore / NLI-based faithfulness as a third signal.
- RePCS-style memorisation audit per pipeline.
- Multilingual track via MIRAGE-Bench composition.
- Medical and legal domain tracks.
- Hosted submission API if adoption warrants it.

## References

- Thakur et al., "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation
  of Information Retrieval Models", NeurIPS 2021.
- Es et al., "RAGAs: Automated Evaluation of Retrieval Augmented Generation",
  arXiv 2023 / EACL 2024.
- Saad-Falcon et al., "ARES: An Automated Evaluation Framework for
  Retrieval-Augmented Generation Systems", NAACL 2024.
- Chen et al., "Benchmarking Large Language Models in Retrieval-Augmented
  Generation" (RGB), AAAI 2024.
- "MIRAGE: A Metric-Intensive Benchmark for Retrieval-Augmented Generation
  Evaluation", arXiv 2504.17137 (Apr 2025).
- Tang & Yang, "MultiHop-RAG: Benchmarking Retrieval-Augmented Generation
  for Multi-Hop Queries", ICLR 2024.
- "CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented
  Generation", TOIS 2024.
- Friel et al., "RAGBench: Explainable Benchmark for Retrieval-Augmented
  Generation Systems", arXiv 2407.11005 (2024).
- Rau et al., "BERGEN: A Benchmarking Library for Retrieval-Augmented
  Generation", EMNLP Findings 2024 (arXiv 2407.01102).
- Adolphs et al., "MIRAGE-Bench: Multilingual RAG Evaluation Benchmark"
  (built on MIRACL).
- "RePCS: Diagnosing Data Memorization in LLM-Powered Retrieval-Augmented
  Generation", arXiv 2506.15513 (Jun 2025).
- "A survey on LLM-as-a-judge", ScienceDirect S2666675825004564 (2025).
