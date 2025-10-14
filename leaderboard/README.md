# `leaderboard/`

Authoritative leaderboard storage. Append-only via PRs.

## Layout

```
leaderboard/
├── submissions/
│   └── <pipeline_hash>.json    # one file per submission; never edited in place
├── verified/
│   └── verified.json           # set of verified pipeline_hashes (CI-managed)
└── README.md                   # this file
```

## Submission flow

1. Submitter runs `rag-bench eval ... --out result.json`.
2. Submitter runs `rag-bench submit ./result.json` (opens a PR adding `leaderboard/submissions/<hash>.json`).
3. CI (.github/workflows/leaderboard.yml):
   - Validates the file against the submission schema
   - Recomputes pipeline_hash and confirms it matches
   - Re-runs the pipeline on the hidden holdout split + a 20% sample of the public split
   - Calls `rag_bench.repro.verify_run` to check submitter CIs overlap re-run means
   - If verified: writes the hash to `verified/verified.json`, regenerates `frontend/data/leaderboard.json`
4. PR is auto-labeled `verified` or `not-reproducible`; merge requires `verified` label.

## Why this layout

- Submissions are content-addressed (`<pipeline_hash>.json`); collisions impossible without identical YAML + tasks + seeds.
- Verified set is a separate file so re-verifications (after holdout rotation, after metric-breaking version bumps) can update without rewriting submissions.
- Frontend reads only `frontend/data/leaderboard.json`; the directory above is the source of truth.

## Holdout rotation

Default quarterly cadence; early rotation when frontier
models lift top-3 by >5 points within 30 days. Rotation:

1. Freeze current leaderboard with a date-stamped column.
2. Generate a new holdout sample (deterministic, seeded).
3. Re-verify all entries against the new holdout.
4. Publish the retired holdout (`tasks-data/<task>/retired-holdouts/<date>.jsonl`).
