# Quickstart

```bash
pip install rag-bench[all]
```

Define a pipeline in YAML:

```yaml
pipeline:
  name: my-hybrid-bge
  chunker:
    type: recursive
    chunk_size: 1000
    overlap: 200
  retriever:
    type: hybrid
    sparse: { type: bm25 }
    dense:  { type: sentence-transformers, model: BAAI/bge-large-en-v1.5 }
    fusion: { type: rrf, k: 60 }
    top_k: 50
  reranker:
    type: cross-encoder
    model: BAAI/bge-reranker-v2-m3
    top_k: 10
  generator:
    type: anthropic
    model: claude-opus-4-7
    temperature: 0
    prompt_template: default
```

Run, inspect, submit:

```bash
rag-bench eval my-pipeline.yaml \
    --tasks nq-1k,hotpotqa-1k,noisy-qa,unanswerable-qa \
    --seeds 5 --out result.json
rag-bench show result.json
rag-bench submit result.json
```

The last command writes the submission into `leaderboard/submissions/`;
opening the PR is up to you. CI re-runs the pipeline on a hidden holdout
plus a 20% sample of the public split and labels the entry verified or
not-reproducible.

## Smoke run

```bash
rag-bench eval examples/01-basic-pipeline.yaml --tasks smoke --seeds 1 --smoke
```

`--smoke` caps each task to 50 items and a single seed. It does not change
the task list or the metric set; use `--tasks smoke` for the synthetic
single-task suite.

## `show` output

```
Pipeline: my-hybrid-bge  (sha256:8f3a…b7c2)
Tasks: 4 | Seeds: 5 | Cost (estimate): $4.12 | Wall time: 18m 03s

┌─────────────────┬──────────────┬──────────────┬──────────────┬─────────────┐
│ Task            │ Recall@10    │ F1           │ Faithfulness │ $/query     │
├─────────────────┼──────────────┼──────────────┼──────────────┼─────────────┤
│ nq-1k           │ 0.74 ±0.03   │ 0.58 ±0.03   │ 0.84 ±0.02   │ $0.0024     │
│ hotpotqa-1k     │ 0.61 ±0.04   │ 0.49 ±0.03   │ 0.79 ±0.03   │ $0.0031     │
│ noisy-qa        │ 0.62 ±0.04   │ 0.51 ±0.04   │ 0.81 ±0.03   │ $0.0028     │
│ unanswerable-qa │ –            │ –            │ –            │ $0.0019     │
│   NRR (F1)      │              │ 0.73 ±0.05   │              │             │
└─────────────────┴──────────────┴──────────────┴──────────────┴─────────────┘

Adversarial composite: 0.69 ±0.04
```

Numbers are mean ± half-width of the 95% bootstrap CI. A dash means the
metric doesn't apply to the task (e.g. Recall@10 on `unanswerable-qa`).

## Configuration

The pipeline YAML is the contract: every knob that affects outputs lives
there. The only environment variables the CLI reads are API keys
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`).
