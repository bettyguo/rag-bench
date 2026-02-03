# Architecture

The runtime shape of the harness. See [methodology.md](methodology.md)
for the rationale and [reproducibility.md](reproducibility.md) for the
hash contract.

## 1. The data flow

```
                       ┌──────────────────────────┐
                       │  pipeline.yaml (user)    │
                       └──────────────┬───────────┘
                                      │ load + validate
                                      ▼
                  ┌───────────────────────────────────┐
                  │  PipelineConfig (pydantic)        │
                  └──────────────┬────────────────────┘
                                 │ compose
                                 ▼
              ┌────────────────────────────────────────┐
              │  Pipeline                              │
              │   ├── Chunker                          │
              │   ├── Retriever                        │
              │   ├── Reranker                         │
              │   └── Generator                        │
              └──────────────┬─────────────────────────┘
                             │  Task.iter_items()
                             │  Pipeline.answer(item)
                             ▼
                  ┌──────────────────────────┐
                  │  Runner                  │
                  │   - per-query records    │
                  │   - per-seed re-runs     │
                  │   - cost + latency       │
                  └──────────────┬───────────┘
                                 │ Metrics
                                 ▼
                  ┌──────────────────────────┐
                  │  result.json             │
                  │   + pipeline_hash        │
                  └──────────────────────────┘
```

## 2. The five component abstractions

Every component implements one of five base classes in `rag_bench.pipeline.base`:

| Base class | Inputs | Outputs |
|---|---|---|
| `Chunker` | `list[Document]` | `list[Chunk]` |
| `Retriever` | `Query` (+ chunks from indexing) | `list[RetrievalResult]` |
| `Reranker` | `Query, list[RetrievalResult]` | `list[RetrievalResult]` (re-ordered) |
| `Generator` | `Query, list[RetrievalResult]` | `GenerationResult` (text + token usage) |
| `Pipeline` | (composes the above) | `PipelineResult` per `Query` |

A new component implementation = subclass + `@register("name")` decorator. The pipeline composer in `rag_bench.pipeline.compose` reads YAML, looks up registered names, and instantiates with validated configs.

## 3. Tasks

A `Task` (in `rag_bench.tasks.base`) emits an iterable of `TaskItem`s. Each `TaskItem` carries:
- `query: Query`
- `gold_answers: list[str]`
- `gold_passages: list[str] | None` (for retrieval metric tasks)
- `corpus: Corpus | str` (path or in-memory; a corpus_ref hash always available)
- `task_id`, `item_id`
- `metadata: dict[str, Any]`

Tasks are responsible for materialization (download from HF datasets, license check, contamination flag) and split selection (public vs holdout).

## 4. Metrics

Metrics in `rag_bench.metrics.*` are pure functions over `PipelineResult` + `TaskItem`. Each metric returns a `MetricResult` with `mean`, `ci_95`, `per_query: list[float]`.

The runner collects per-query metric outputs into a DuckDB table; the final aggregation computes summary statistics.

## 5. The runner

```python
runner = Runner(pipeline, task, seeds=[0,1,2,3,4])
result = runner.run()  # → RunRecord
```

`RunRecord` carries pipeline_hash, task_data_hash, judge_fingerprint, per-query records, cost & latency timings.

## 6. Reproducibility hash

`rag_bench.repro` exposes:

- `canonicalize_yaml(yaml_str)`: sort keys, strip comments, normalize.
- `pipeline_hash(pipeline_yaml, task_data_hash, judge_fingerprint, version, seeds)`.
- `verify_run(submitted, rerun)`: returns a `VerificationOutcome`.

## 7. CLI

Single entrypoint `rag-bench`:

- `eval <pipeline.yaml> --tasks X,Y --seeds N --out result.json`
- `show result.json`
- `submit result.json`
- `verify result.json`
- `tasks ls` / `tasks show <id>`
- `components ls`
- `leaderboard regenerate`

Click-based; Rich-formatted output.

## 8. Frontend

Next.js static export. Reads `frontend/data/leaderboard.json` regenerated
by CI on every accepted submission. Pages:

- `/`: sortable, filterable overall leaderboard.
- `/pareto`: quality × cost scatter.
- `/task/<task-id>` and `/pipeline/<hash>`: planned.

## 9. Out of scope

- Distributed / GPU-cluster execution. Single-node only.
- Streaming generation. Full responses are collected.
- Online retrieval indexes. Indexes are built once per task, cached, and
  re-used across pipelines that share the retriever config.

## 10. Types and validation

Pydantic v2 throughout for config validation. Component configs are
typed (e.g. `RecursiveChunkerConfig`, `BM25Config`); the composer
dispatches on the `type` field. The `extra="forbid"` setting catches
typo'd YAML keys immediately.
