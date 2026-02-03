# Tasks

A task is a (corpus, queries, gold answers, evaluation protocol) tuple. The
11 first-release tasks span 5 families. [methodology.md §4](methodology.md#4-task-design)
covers the rationale.

## Task families

| Family | Tasks | What it measures |
|---|---|---|
| Single-hop QA | `nq-1k`, `msmarco-1k` | retrieval + extractive answer |
| Multi-hop QA | `hotpotqa-1k`, `multihoprag-1k` | cross-document reasoning |
| Adversarial | `noisy-qa`, `unanswerable-qa`, `counterfactual-qa` | RGB's 4 abilities |
| Domain | `financebench-q`, `arxiv-sci-q` | out-of-Wikipedia distribution |
| Novel corpus | `novel-arxiv-q`, `novel-events-q` | post-cutoff, uncontaminated |

## Task spec format

Every task ships a `task.yaml` with this shape:

```yaml
task:
  id: nq-1k
  name: Natural Questions (1K subset, public split)
  family: single-hop-qa
  size: 1000
  corpus_size: 1247239
  language: en
  source:
    upstream: https://ai.google.com/research/NaturalQuestions
    citation: kwiatkowski2019natural
    license: CC-BY-SA-3.0
    download: hf://datasets/google-research-datasets/natural_questions
  contamination_risk: high   # Wikipedia substrate
  splits:
    public: 800     # visible to submitters
    holdout: 200    # hidden; CI re-verifies on this
  evaluation:
    retrieval_metrics: [recall@10, ndcg@10, mrr@10]
    generation_metrics: [exact_match, token_f1]
    end_to_end_metrics: [answer_relevance, faithfulness]
    operational: [cost_per_query_usd, latency_p95_ms]
  prompt:
    template: default  # see docs/metrics.md
  notes: |
    Substrate is Wikipedia; treat as high-contamination. Use alongside
    novel-arxiv-q for an uncontaminated comparison point.
```

---

## v1 tasks

### `nq-1k` — Natural Questions (single-hop)
- **Source:** Google Natural Questions, 1K subset sampled with stratification on question length.
- **Corpus:** Wikipedia (English, snapshot pinned in `task.yaml`).
- **Family:** single-hop QA.
- **Size:** 1,000 queries (800 public / 200 holdout).
- **License:** CC-BY-SA-3.0.
- **Contamination:** `high`. Wikipedia is in nearly every retriever's training.
- **Notes:** canonical single-hop; dense-retriever recall is inflated by Wikipedia overlap with pretraining.

### `hotpotqa-1k` — HotpotQA (multi-hop)
- **Source:** HotpotQA distractor setting, 1K subset.
- **Corpus:** Wikipedia.
- **Family:** multi-hop QA.
- **Size:** 1,000 (800/200).
- **License:** CC-BY-SA-4.0.
- **Contamination:** `high`.
- **Notes:** Industry-standard multi-hop. Bridges to MultiHop-RAG.

### `msmarco-1k` — MS MARCO Passage (retrieval-heavy)
- **Source:** MS MARCO dev, 1K subset.
- **Corpus:** MS MARCO Passage collection.
- **Family:** single-hop, retrieval-heavy.
- **Size:** 1,000 (800/200).
- **License:** MS-MARCO non-commercial.
- **Contamination:** `high`. MS-MARCO is in nearly every retriever's training.
- **Notes:** Retrieval-quality signal at scale; complements NQ.

### `multihoprag-1k` — MultiHop-RAG
- **Source:** Tang & Yang 2024.
- **Corpus:** News articles (provided by authors).
- **Family:** multi-hop QA.
- **Size:** 1,000 (800/200).
- **License:** MIT.
- **Contamination:** `medium`. News corpus, partial overlap with crawl-trained models.
- **Notes:** Explicitly designed to expose multi-hop failure modes.

### `noisy-qa` — RGB noise-robustness (rag-bench-authored)
- **Source:** rag-bench-authored, gold answers via NQ; distractors drawn from related Wikipedia categories using BM25-near-but-wrong-entity.
- **Family:** adversarial (noise robustness).
- **Size:** 500 (400/100).
- **License:** Apache-2.0 (authoring); CC-BY-SA-3.0 (Wikipedia substrate).
- **Contamination:** `high` substrate, but distractor injection prevents memorization-based shortcuts.
- **Notes:** Maps to RGB ability #1. Discriminates retrievers that confuse near-neighbors from those that don't.

### `unanswerable-qa` — Negative rejection (rag-bench-authored)
- **Source:** rag-bench-authored. Each item is a question + a corpus that *does not* contain the answer.
- **Family:** adversarial (negative rejection).
- **Size:** 500 (400/100).
- **License:** Apache-2.0.
- **Contamination:** `low`. Synthetic questions over diverse synthetic corpora.
- **Evaluation:** Abstention precision (when model abstains, was the answer truly unavailable?), abstention recall (when answer was unavailable, did the model abstain?).
- **Notes:** Maps to RGB ability #2 (negative rejection); the most production-relevant failure mode.

### `counterfactual-qa` — Counterfactual robustness (rag-bench-authored)
- **Source:** rag-bench-authored. Questions with well-known parametric answers; retrieved corpus contains a *plausibly-wrong* fact that contradicts the parametric answer.
- **Family:** adversarial (counterfactual robustness).
- **Size:** 300 (240/60).
- **License:** Apache-2.0.
- **Contamination:** `low` for the planted facts; questions themselves draw from well-known parametric knowledge.
- **Evaluation:** Two metrics: `counterfactual_compliance` (does the model trust the retrieved counter-fact?) and `counterfactual_resistance` (does the model resist when the counter-fact is implausible?). The healthy pipeline is *high compliance* on plausible counters and *low compliance* on implausible ones.
- **Notes:** Maps to RGB ability #4. Discriminates pipelines that blindly trust retrieval from those that reason about consistency.

### `financebench-q` — FinanceBench subset (domain)
- **Source:** FinanceBench (Islam et al. 2023), subset.
- **Corpus:** SEC 10-K filings.
- **Family:** domain QA (finance).
- **Size:** 150 (120/30).
- **License:** CC-BY-NC-4.0 (FinanceBench).
- **Contamination:** `medium`. SEC filings are public; model coverage varies.
- **Notes:** Out-of-Wikipedia distribution; numeric-heavy; tests pipeline behavior on long documents.

### `arxiv-sci-q` — Scientific QA (rag-bench-authored)
- **Source:** rag-bench-authored questions over arXiv abstracts in NLP / IR (pre-2026-01 cutoff window).
- **Corpus:** ~10K arXiv abstracts.
- **Family:** domain QA (scientific).
- **Size:** 200 (160/40).
- **License:** arXiv perpetual non-exclusive (abstracts); Apache-2.0 (questions).
- **Contamination:** `medium`. ArXiv is in most pretrains.
- **Notes:** Tests retrieval behavior on technical jargon-heavy text.

### `novel-arxiv-q` — Post-cutoff arXiv (rag-bench-authored, novel)
- **Source:** rag-bench-authored. 50 arXiv papers in a niche subfield published 2026-02 to 2026-04.
- **Corpus:** Full text of those 50 papers, chunked.
- **Family:** novel-corpus.
- **Size:** 200 (160/40).
- **License:** CC-BY-4.0 (questions); arXiv license for paper text.
- **Contamination:** `novel`. The paper text is plausibly outside any current model's pretraining cutoff and unlikely to be retriever pretraining.
- **Notes:** Uncontaminated comparison point. If a pipeline's nq-1k F1 is much higher than novel-arxiv-q F1, parametric memorization may be doing the work.

### `novel-events-q` — Post-cutoff events (rag-bench-authored, novel)
- **Source:** rag-bench-authored. News/blog articles about events in Feb–Apr 2026.
- **Corpus:** ~500 articles.
- **Family:** novel-corpus.
- **Size:** 200 (160/40).
- **License:** CC-BY-4.0 (questions); fair use (snippets) + URLs for sources.
- **Contamination:** `novel`.
- **Notes:** Second uncontaminated comparison; tests event-aware reasoning rather than entity recall.

---

## Adding new tasks (post-v1)

Submit a PR with:
1. `tasks-data/<task-id>/task.yaml` matching the spec format
2. Loader at `src/rag_bench/tasks/<task_id>.py`
3. ≥3 representative example items committed to `tasks-data/<task-id>/examples.jsonl`
4. Documentation update to this file
5. CI confirms a BM25 baseline runs end-to-end on the task

We will not accept tasks where:
- The corpus license forbids redistribution and the corpus is unobtainable without a paid agreement
- The contamination risk cannot be assessed
- The gold answers were generated by an LLM without human review

See [submitting.md](submitting.md) for the full process.
