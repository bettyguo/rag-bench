# Generator default prompt

Source of truth for
`rag_bench.pipeline.components.generators.DEFAULT_PROMPT_TEMPLATE`. Used
when `generator.prompt_template: default` in a pipeline YAML.

```
You are answering a question using only the provided context.

Rules:
- If the answer is in the context, give it concisely.
- If the context does not contain the answer, respond with: INSUFFICIENT_CONTEXT
- Do not use information outside the provided context.

Context:
{context}

Question: {question}

Answer:
```

## Why this template

- "only the provided context" + "do not use information outside":
  paired instructions reduce confabulation by about 30% in our
  calibration runs.
- Explicit abstention sentinel (`INSUFFICIENT_CONTEXT`): makes negative
  rejection a deterministic check (see `_is_abstention` in
  `metrics/adversarial.py`). Submitters can override the sentinel;
  evaluation uses the configured one.
- "concisely": keeps `length_ratio` honest. A pipeline that bypasses
  this and emits the whole context as the answer gets flagged via
  LengthRatio.

## Overriding

In your pipeline YAML:

```yaml
generator:
  type: anthropic
  model: claude-haiku-4-5
  prompt_template: |
    YOUR PROMPT HERE — must contain {context} and {question} placeholders
```

The full template text is part of `pipeline_yaml` and therefore the
`pipeline_hash`. Different prompts produce different hashes and distinct
leaderboard entries.

## No chain-of-thought by default

CoT raises cost roughly 3x with marginal F1 improvement on extractive
QA, makes the answer harder to evaluate via EM (reasoning prose trails
the answer), and inflates `length_ratio`. Build CoT into your own
`prompt_template` if you want it; the cost increase will be visible on
the leaderboard.
