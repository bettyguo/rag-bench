# Reproducibility Protocol

> Every leaderboard entry is a content-addressed tuple. Re-running the published config must reproduce the published numbers within bootstrap CI overlap, or the entry is flagged `unverified`.

## 1. The pipeline hash

```
pipeline_hash = sha256(
    canonical(pipeline_yaml)
    ‖ task_data_hash
    ‖ judge_ensemble_fingerprint   (if faithfulness reported)
    ‖ rag_bench_version
    ‖ seed_list (sorted)
)
```

- `canonical(pipeline_yaml)`: pipeline config serialized with sorted keys and stripped comments. Implemented in `rag_bench.repro.canonicalize`.
- `task_data_hash`: SHA-256 over the materialized task JSONL (queries + gold + corpus chunk hashes). Pinned per task in `tasks-data/<task>/data.sha256`.
- `judge_ensemble_fingerprint`: per-judge `(model_id, model_version, prompt_hash, temperature)` for each of the N=3 judges. Stable across runs of the same judges.
- `rag_bench_version`: from `pyproject.toml`. Bumped on breaking changes.
- `seed_list`: e.g. `[0, 1, 2, 3, 4]`.

The pipeline hash uniquely identifies a submission. Two submissions with the same hash MUST produce numbers within bootstrap CI overlap (else our verification is broken).

## 2. Submission flow

1. User runs `rag-bench eval <pipeline.yaml> --tasks <task-ids> --seeds 5`. Output: `result.json` containing per-query scores, summary stats, pipeline_hash.
2. User runs `rag-bench submit ./result.json`. The submission opens a PR against `leaderboard/submissions/<pipeline_hash>.json`.
3. CI checks:
   - `result.json` schema validates.
   - `pipeline_hash` matches our canonical recomputation.
   - `task_data_hash` matches our pinned hashes.
4. CI re-runs the pipeline on the **hidden holdout (200 items per task)** and a **20% sample of the public set**.
5. Re-run scores must overlap with submitter's reported CIs at the 95% level. If they do not, the submission is closed with `not-reproducible` label.
6. If they do, the submission merges; the leaderboard regenerates; the entry appears with a `verified` badge.

## 3. The `result.json` schema

```jsonc
{
  "rag_bench_version": "0.1.1",
  "pipeline_hash": "sha256:…",
  "pipeline_yaml": "…",              // verbatim, comments stripped
  "submitter": {
    "name": "…",                     // anonymous allowed
    "contact": "…"
  },
  "judge_ensemble": {                // null if no faithfulness
    "judges": [
      { "name": "anthropic-A", "model": "claude-haiku-4-5", "prompt_hash": "…" },
      { "name": "openai-B",    "model": "gpt-5.1",          "prompt_hash": "…" },
      { "name": "openweight-C","model": "Qwen3.5-72B-Instruct", "prompt_hash": "…" }
    ],
    "fingerprint": "sha256:…"
  },
  "tasks": {
    "nq-1k": {
      "task_data_hash": "sha256:…",
      "seeds": [0, 1, 2, 3, 4],
      "metrics": {
        "recall@10": { "mean": 0.71, "ci_95": [0.68, 0.74] },
        "ndcg@10":   { "mean": 0.62, "ci_95": [0.59, 0.65] },
        "exact_match": { "mean": 0.43, "ci_95": [0.40, 0.46] },
        "token_f1":  { "mean": 0.55, "ci_95": [0.52, 0.58] },
        "answer_relevance": { "mean": 0.87, "ci_95": [0.85, 0.89] },
        "faithfulness": { "mean": 0.81, "ci_95": [0.78, 0.84] },
        "cost_per_query_usd": { "mean": 0.0024, "p95": 0.0041 },
        "latency_p95_ms": 2103
      },
      "per_query": "scores/nq-1k.jsonl"   // referenced; CI can spot-check
    },
    // … other tasks
  }
}
```

## 4. What re-verification does and doesn't catch

**Catches:**
- Pipeline that hardcoded answers
- Pipeline that uses a different generator from the one declared
- Pipeline that secretly used additional information
- Pipeline whose nominal CI is much tighter than reality (we recompute CI from re-run data)

**Does not catch:**
- Pipeline that calls an external service the verifier can't reach (we flag and mark `unverified-external`)
- Pipeline whose stochasticity is dominated by something other than seed (e.g., temperature 1.0 with no seed support); we encourage temperature 0 unless the pipeline owner has a reason not to.

## 5. The `unverified` badge

Submissions that can't be re-run (private API, gated model, expired credentials) can still appear on the leaderboard but carry the `unverified` badge. The default leaderboard view hides `unverified` entries; users can opt in. We do not claim parity between verified and unverified entries.

## 6. Re-verification cost

CI re-runs cost real money (we're calling the same APIs the submitter did). Policy:
- We re-run on holdout (200 items × N tasks) + 20% public sample
- Estimated cost per submission: $0.10–$3.00 depending on tasks and judge usage
- This cost is borne by the `rag-bench` maintainers from a community-supported budget
- If submission volume exceeds budget, we deprioritize submissions that are minor variations of existing pipelines

## 7. Holdout rotation

- Holdout rotates quarterly by default.
- Early rotation is triggered if a frontier-model release pushes top-3 leaderboard scores by >5 points within 30 days.
- The pre-rotation leaderboard is frozen with a date stamp; the post-rotation column is new.
- The retired holdout is published so the community can analyse what worked and what overfit.

## 8. Code version pinning

Submissions pin a `rag_bench_version`. If we ship a metric-breaking change, we bump the major version and re-run all `verified` submissions at the new version. Numbers that change >2 absolute points get a notation on the leaderboard pointing to the changelog.

We commit to no metric-breaking change in a minor version bump.
