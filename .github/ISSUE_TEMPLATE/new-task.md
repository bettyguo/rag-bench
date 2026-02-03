---
name: New task proposal
about: Propose adding a new evaluation task
title: '[task] '
labels: task-proposal
assignees: ''
---

## Task ID

(Proposed slug, e.g. `medical-pubmed-q`. Must be lowercase, hyphen-separated.)

## Family

(`single-hop-qa` / `multi-hop-qa` / `adversarial` / `domain` / `novel-corpus`.)

## Why this task earns its leaderboard slot

(Each task adds verification cost. What discriminator does it provide that
existing tasks don't? Cite specific failure modes it would expose.)

## Source + license

- Upstream: (URL + paper citation)
- License: (must be redistributable, OR pulled from a stable URL with a
  no-redistribution clause we can wrap-load — like NQ via HF Datasets)
- Approx. corpus size:
- Approx. query count:

## Contamination

What is your assessment? (`high` / `medium` / `low` / `novel` per the
docs/tasks.md taxonomy.) What evidence?

## Gold answer source

- LLM-generated? (We rarely accept these; document the human-review pass.)
- Human-annotated? (Single rater / multi-rater? Inter-rater agreement?)
- Derived from a published benchmark? (Cite + license check.)

## Metric set

(Which of the existing metrics apply? Do you propose any new ones?)

## Sample items

```
(3 representative items in JSONL form.)
```

## Sanity check

(What does a BM25-only baseline score? Is that what you'd expect from the
literature?)

See [docs/tasks.md §Adding new tasks](../../docs/tasks.md#adding-new-tasks-post-v1) for the full process.
