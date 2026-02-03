# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). SemVer.
Pre-1.0 minor bumps may include methodology-affecting changes; the
pipeline_hash contract makes those visible.

## Unreleased

Toward v1.0:

- Run all 8 reference pipelines on the full v1.0 task suite and seed the
  launch leaderboard.
- Grow the bootstrap faithfulness calibration set (50 items) to 200 with
  3-rater human annotation; publish per-task-family Krippendorff α.
- Grow the adversarial seed sets toward the docs/tasks.md targets
  (500 / 500 / 300).
- Deploy the static-export frontend; PyPI publish.

## 0.1.0 — 2026-05-13

First public-prep release.

### Added

Pipeline composer + 3 baselines per stage. Chunkers: `recursive`, `fixed`,
`sentence`. Retrievers: `bm25` (pure-Python), `dense` (sentence-transformers,
lazy), `hybrid` (RRF). Rerankers: `identity`, `lexical-overlap`,
`cross-encoder` (lazy). Generators: `echo`, `extractive`, `anthropic`,
`openai` (last two lazy).

Tasks:

- `synthetic-10`, offline; used by the CI smoke run.
- `nq-1k`, `hotpotqa-1k`, `msmarco-1k`: HF Datasets loaders with parquet
  cache and a deterministic 80/20 public/holdout split.
- `noisy-qa`, `unanswerable-qa`, `counterfactual-qa`: adversarial seeds
  under RGB's 4-ability taxonomy.

Metrics:

- Retrieval: Recall@k, nDCG@k, MRR@k with chunk-id / doc-id / substring
  matching.
- Generation: ExactMatch, TokenF1 (SQuAD-style normalizer), LengthRatio
  (sanity flag).
- End-to-end: Faithfulness — atomic-claim extraction, N=3 cross-vendor
  judges, majority vote, position-randomized contexts, self-enhancement
  guard.
- Adversarial: AbstentionRecall / AbstentionPrecision /
  NegativeRejectionRate / PlausibleCompliance / ImplausibleResistance.
- Bootstrap CIs (percentile + BCa); paired-bootstrap p-value.

Judges + calibration:

- `Judge` ABC; `DummyJudge` (offline), `AnthropicJudge`, `OpenAIJudge`.
- `run_calibration` + Krippendorff α (nominal, 2-rater).
- 50-item bootstrap calibration set across 5 task families.

Reproducibility + submissions:

- `canonicalize_yaml`, `judge_ensemble_fingerprint`, `pipeline_hash`,
  `verify_run`.
- `Submission` schema, `build_submission`, `validate_submission`.
- `Runner` produces a `RunRecord` with per-task MetricResults + cost
  (mean, p95) + latency (mean, p95).

Leaderboard + frontend:

- `generate_leaderboard` aggregates submissions into the frontend data
  file; Pareto and `verbose` tagging.
- Next.js static-export site with a sortable leaderboard and a log-cost ×
  quality scatter.
- 3 offline-reproducible demo entries seeded.

CLI: `eval`, `show`, `submit`, `verify`, `tasks ls`/`show`,
`components ls`, `leaderboard regenerate`.

Documentation: methodology survey, task specs, metric definitions,
adversarial-track design, reproducibility protocol, quickstart, submission
guide, FAQ, architecture overview, canonical judge / generator prompts.

Repo hygiene: Apache 2.0 LICENSE, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY,
issue + PR templates, ruff + pytest CI on Python 3.11 / 3.12.

## 0.0.1 — 2026-05-13

Initial repository skeleton.
