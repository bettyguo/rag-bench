# Submitting to the Leaderboard

## TL;DR

```bash
rag-bench eval my-pipeline.yaml --tasks v1.0-suite --seeds 5 --out ./result.json
rag-bench submit ./result.json
# Opens a PR. CI re-verifies. Entry appears (or doesn't) within ~24 hours.
```

## Eligibility

A submission must:
1. Use a `pipeline.yaml` config that `rag-bench` can re-run end-to-end without manual intervention (no notebook glue, no human-in-the-loop).
2. Disclose API providers and model names. Closed/proprietary models are allowed; opaque "model X via partner Y" pipelines must accept the `unverified-external` badge if we can't re-call.
3. Provide ≥5 seeds.
4. Use `temperature: 0` for the generator, OR justify temperature > 0 in the submission notes (we won't reject, but the leaderboard will display a `stochastic` tag).
5. Cover the v1.0 task suite at minimum: `nq-1k`, `hotpotqa-1k`, `noisy-qa`, `unanswerable-qa`. Optional: `multihoprag-1k`, `counterfactual-qa`, `financebench-q`, `arxiv-sci-q`, `novel-arxiv-q`, `novel-events-q`. Faithfulness reporting is optional.

## What we reject

- Pipelines that use test set as training data (will be flagged via contamination-pattern detection: anomalous gap between contaminated and novel-corpus tasks)
- Pipelines that fail re-verification (numbers don't overlap with submitter-reported CI at 95%)
- Submissions with obvious metric gaming (e.g. a "pipeline" that hardcodes gold answers; detected via `task_data_hash` mismatch and code review)
- Pipelines that require >24 hr wall-clock to re-run on the reference environment (we'll work with you to subset)

## Submission lifecycle

```
[ User opens PR ]
       │
       ▼
[ schema check ]           ← rejects on malformed result.json
       │
       ▼
[ pipeline_hash recompute ] ← rejects if submitter's hash doesn't match canonical
       │
       ▼
[ holdout re-run ]          ← ~$0.10–$3.00 per submission; budgeted
       │
       ▼
[ CI overlap check ]        ← passes → merge; fails → reject with `not-reproducible`
       │
       ▼
[ leaderboard regeneration ]
       │
       ▼
[ entry visible ]
```

Typical end-to-end time: a few hours up to 24 hr depending on holdout re-run cost.

## Submission frequency

- We accept at most 3 verified submissions per submitter per quarter (rate-limit to keep verification budget sane).
- "Updates" to an existing pipeline (e.g., changed reranker top_k) count as new submissions.
- If you maintain a major OSS project, your submissions are accepted on different rate-limit terms; open an issue first.

## Submitter identity

- Anonymous submissions are allowed and welcomed.
- For verified submissions, we need a contact (email or GitHub handle) so we can reach you if the holdout re-run reveals a discrepancy.
- We do NOT publish your contact on the leaderboard unless you opt in.

## Removing a submission

You can request your own submission removed at any time. We will preserve the public commit history (it's a public PR) but redact the leaderboard entry.

## Adding new tasks

See [tasks.md §Adding new tasks](tasks.md#adding-new-tasks-post-v1). Different process; goes through community review.
