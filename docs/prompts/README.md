# Canonical prompts

Every prompt that enters a leaderboard number is content-addressed via SHA-256
into the `pipeline_hash`. The prompts themselves are reproduced here so
submitters and reviewers can audit them without running the code.

| File | Used by | Hashed into |
| --- | --- | --- |
| `judge-claim.md` | Faithfulness LLM-judge | `judge_ensemble_fingerprint` → `pipeline_hash` |
| `generator-default.md` | Default generator prompt template | (carried inside `pipeline_yaml`) |

## Why publish prompts

LLM-as-judge metrics have a known sensitivity to prompt wording. A judge prompt
change can shift faithfulness numbers by 5+ absolute points across a leaderboard.
By publishing the canonical prompts:

1. Submitters can reproduce our judge calls exactly.
2. Reviewers can verify our claims about position-randomization and length-normalization.
3. When we change a prompt, the SHA-256 changes, the pipeline_hash changes, and all prior submissions are mechanically marked as "not comparable" until re-verified.

## Modifying these prompts

A prompt change is a methodology change. All prior verified leaderboard
entries are re-run at maintainer cost. The previous prompt and its hash
are preserved under `archive/`.

## What's NOT here

- Pipeline-side prompts. Those live in your `pipeline.yaml`'s `generator.prompt_template`.
- Per-submitter judge customizations. Submitters cannot customize judge prompts; that's by design.
- The atomic-claim extraction prompt (v0.0 uses sentence-boundary rules; an LLM-driven splitter will be added in v1.1 and prompt-hashed at that time).
