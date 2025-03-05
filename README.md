# rag-bench

A reproducible benchmark for end-to-end RAG pipelines. Plug in your chunker,
retriever, reranker and generator; run against a curated, contamination-flagged
task suite; get comparable numbers with content-addressed pipeline hashes.

Apache 2.0. Pre-1.0 alpha.

## Quickstart

```bash
pip install -e .
rag-bench eval examples/01-basic-pipeline.yaml \
    --tasks synthetic-10 --seeds 1 --out result.json
rag-bench show result.json
```

```
                bm25-echo · sha256:00232cc41bb057d3…
┌──────────────┬──────────────┬────────┬───────────────────┬────┐
│ Task         │ Metric       │   Mean │            95% CI │  n │
├──────────────┼──────────────┼────────┼───────────────────┼────┤
│ synthetic-10 │ recall@10    │ 1.0000 │  [1.0000, 1.0000] │ 10 │
│ synthetic-10 │ token_f1     │ 0.2205 │  [0.1691, 0.2700] │ 10 │
│ synthetic-10 │ length_ratio │ 9.9833 │ [6.9333, 13.7675] │ 10 │
└──────────────┴──────────────┴────────┴───────────────────┴────┘
```

BM25 retrieves the right document every time; the echo "generator" hands the
whole chunk back, so token_f1 is low and `length_ratio` flags the answer as
verbose. See [docs/quickstart.md](docs/quickstart.md) for the full demo.

## What's measured

| Family | Tasks | Metrics |
| --- | --- | --- |
| Single-hop QA | `nq-1k`, `msmarco-1k` | Recall@k, nDCG@k, MRR@k, F1, EM |
| Multi-hop QA | `hotpotqa-1k`, `multihoprag-1k` | Recall@k, F1, EM |
| Noise robustness | `noisy-qa` | Noise Vulnerability |
| Abstention | `unanswerable-qa` | NRR-F1, abstention precision/recall |
| Counterfactuals | `counterfactual-qa` | Plausible compliance, implausible resistance |
| Domain QA | `financebench-q`, `arxiv-sci-q` | F1, EM |
| Novel corpus | `novel-arxiv-q`, `novel-events-q` | F1, EM (uncontaminated) |
| Faithfulness (optional) | all of the above | Calibrated multi-judge LLM-as-judge |
| Operational | all tasks | $/query, latency p95 |

Every metric reports mean ± 95% bootstrap CI. Leaderboard submissions require
≥5 seeds. See [docs/metrics.md](docs/metrics.md).

## What's NOT measured

- Retrieval-only — use [BEIR](https://github.com/beir-cellar/beir).
- LLM-as-judge as the headline number. Faithfulness is supplementary; F1/EM
  are always shown alongside.
- A single "overall RAG score" that hides per-task behaviour.
- Submissions without re-verification.

[docs/methodology.md](docs/methodology.md) lists the methodological gaps in
the existing artefacts and how the harness addresses each.

## Related work

We compose with rather than replace:
[BEIR](https://github.com/beir-cellar/beir) (retrieval-only),
[RAGAS](https://github.com/explodinggradients/ragas) /
[ARES](https://github.com/stanford-futuredata/ARES) (metric libraries),
[BERGEN](https://github.com/naver/bergen) (a benchmarking library, not a
leaderboard), RGB (AAAI'24, 4-ability adversarial taxonomy adopted here),
and MIRAGE / MIRAGE-Bench (noise-vulnerability vocabulary, multilingual scope
for a future release).

## Layout

```
docs/                methodology, tasks, metrics, adversarial, repro, quickstart, architecture
src/rag_bench/       harness
  pipeline/          Chunker / Retriever / Reranker / Generator ABCs + baselines
  tasks/             base + synthetic + NQ/HotpotQA/MSMARCO + adversarial seeds
  metrics/           retrieval, generation, faithfulness, adversarial
  judges.py          Judge ABC + Anthropic / OpenAI / Dummy
  calibration.py     Krippendorff α
  repro.py           canonical YAML + pipeline_hash + verify_run
  submission.py      result.json schema
  leaderboard.py     submissions → frontend data
  runner.py          Pipeline × Task × seeds → MetricResults
  cli.py
tests/               offline test suite
frontend/            Next.js static export
leaderboard/         submissions/, verified/
```

## Submitting

PR-based. See [docs/submitting.md](docs/submitting.md). CI re-runs the
pipeline on a hidden holdout + a 20% sample of the public split and either
verifies the entry or rejects it as not-reproducible.

## License

Apache 2.0 for the code. CC-BY-4.0 for the task data authored in this repo.
Wrapped datasets keep their upstream licences; see each task's spec.
