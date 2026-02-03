# FAQ

### Why not just use RAGAS / TruLens / DeepEval?

Those are metric libraries, not benchmarks. They give you scoring functions
but no standardised tasks, no public leaderboard, and no re-verification.
The harness here composes with them; its faithfulness metric is shaped
like RAGAS' but calibrated and bias-guarded. See
[metrics.md](metrics.md#faithfulness).

### How is this different from BERGEN (Naver, EMNLP'24)?

BERGEN is a benchmarking library: YAML-configurable pipelines, many
retrievers and generators, reproducibility-focused. It's not a benchmark
with a leaderboard, doesn't ship an adversarial track, doesn't calibrate
LLM judges, and doesn't audit for retriever-corpus contamination. The
harness here adds those four.

### Why not BEIR?

BEIR is retrieval-only. It stops where RAG starts. This harness measures
retrieval, generation, and end-to-end on the same items. See
[methodology.md §2.1](methodology.md#21-retrieval-only).

### Aren't LLM-as-judge metrics unreliable?

Uncalibrated, yes. The protocol uses three cross-vendor judges with
majority voting, position-randomised contexts, atomic claims (so verbosity
bias is structurally bounded), and a self-enhancement guard that drops the
judge whose family matches the generator. Krippendorff's α against ≥200
human-annotated examples is published; judges with α < 0.6 on a task
family are dropped for that family. F1 and EM are always shown alongside;
faithfulness is never the sole quality number.

### What about your task corpora being in retriever training data?

Every task carries a `contamination_risk` rating (`high` / `medium` / `low`
/ `novel`) and the repo ships two `novel`-tagged tasks hand-authored on
post-2026-01 sources for an uncontaminated comparison point. See
[methodology.md §3](methodology.md#3-contamination-audit).

### Can I tune my pipeline on the public split?

You can, but the leaderboard's re-verification job runs on a hidden
200-item-per-task holdout. Submissions that overfit show up as
not-reproducible (the verification mean falls outside the submitter's 95%
CI). Holdouts rotate quarterly, with an earlier rotation triggered when a
frontier-model release pushes top-3 by more than 5 points within 30 days.

### Why bootstrap CIs instead of standard errors?

Token-F1, faithfulness fractions, and ratios are non-Gaussian on small
task sizes. Bootstrap CIs are non-parametric and well-behaved for these
distributions; BCa is available for the more skewed metrics. See
[metrics.md §0](metrics.md#0-statistical-protocol).

### Why is the leaderboard PR-based instead of a hosted API?

A PR audit trail is operationally simple. A hosted submission API would
add abuse-handling and rate-limit work that doesn't earn its keep until
submission volume is real.

### Can I submit a closed / proprietary model?

Yes. If the model can't be re-called for verification, the entry gets an
`unverified-external` badge and is hidden from the default leaderboard
view but discoverable on opt-in. See [submitting.md](submitting.md).

### Why don't you support streaming / multi-turn / multilingual / medical / legal?

First-release scope. Multilingual will come via
[MIRAGE-Bench](https://mirage-bench.github.io/) composition; medical and
legal have licensing complexity that wasn't worth eating early. Streaming
and multi-turn are out of single-turn QA scope.

### What's the budget for faithfulness verification?

About $150 one-time to call all judges on the 200-item calibration set,
plus $0.10–$3.00 per submission for holdout re-verification. Submitters
fund their own pipeline runs (BYOK).

### Will faithfulness numbers be comparable if you change the judge ensemble?

Yes and no. The judge ensemble fingerprint is part of the pipeline_hash,
so a submission with judge ensemble A and a submission with judge ensemble
B have different hashes. On the leaderboard, the ensemble identity is
shown next to the faithfulness column and the calibration α is linkable.
A material ensemble change triggers re-running prior numbers.

### Who's behind this?

The `git log`. Anonymous contributions are welcome.

### Why isn't task X on the leaderboard?

If it's a published benchmark, it probably hasn't been wrapped yet; open
an issue tagged `task-proposal`. If it's private, we won't host it; a
task loader that pulls from your URL is the path.

### I disagree with metric X.

Open a methodology issue.
