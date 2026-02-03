# Contributing

Thanks for considering a contribution.

## Where things land

- Bug fix: PR with a regression test.
- New component (chunker, retriever, reranker, generator): PR adding a
  class, a `ComponentConfig` subclass, a `@register(...)` decorator, and
  at least one test.
- New task: see [docs/tasks.md](docs/tasks.md). Multi-step; community
  review before merge.
- New metric: open an issue first. Metric design is a methodology
  change.
- Methodology pushback: open an issue labelled `methodology`. Bring
  evidence; expect back-and-forth.
- Leaderboard submission: PR adding `leaderboard/submissions/<hash>.json`.
  See [docs/submitting.md](docs/submitting.md).

## Dev loop

```bash
git clone https://github.com/airine/rag-bench
cd rag-bench
pip install -e ".[dev]"
pytest tests/ -q
ruff check src/ tests/
```

CI runs the same two commands. If they pass locally, CI passes.

## Style

- Python 3.11+; `from __future__ import annotations` throughout.
- Type hints on public surfaces.
- Pydantic v2 for config; `extra="forbid"` on every `ComponentConfig`.
- Dataclasses (frozen where useful) for runtime types.
- No magic env-var configuration; the pipeline YAML is the contract.

## Commit style

One topic per PR. Don't bundle a new component with an unrelated metric
rename. Plain commit messages are fine; no decorative emoji.

## What's not accepted

- Submissions whose gold answers were LLM-generated without human review.
- Tasks whose corpus licence forbids redistribution and that aren't
  obtainable without a paid agreement.
- Pipeline_hash workarounds that obscure differences (e.g. a runtime
  prompt tweak that doesn't enter the hash). Reproducibility is the
  load-bearing invariant; see [docs/reproducibility.md](docs/reproducibility.md).
- Judge prompt or model changes that bypass the prompt_hash. If you
  change a judge, the prompt_hash changes; that's by design.
- Drive-by methodology objections without a proposal: "this metric is
  bad" is less useful than "this metric is bad and here's the alternative
  plus the experiment that distinguishes them".

## Especially useful right now

- End-to-end runs on the HF-backed tasks (NQ, HotpotQA, MS-MARCO) against
  current dataset revisions; the loaders are tested offline.
- Expanding the 200-item faithfulness calibration set; see
  [docs/metrics.md](docs/metrics.md).
- Second-rater pass on the hand-authored adversarial seed sets.
- Reproducing published numbers (BM25 baseline F1 on NQ, etc.).
  Discrepancies are bugs.

## Licence

Contributions are Apache-2.0 (code) or CC-BY-4.0 (data), matching the
project licence.
