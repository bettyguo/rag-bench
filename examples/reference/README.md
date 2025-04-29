# Reference pipelines

Eight pipeline configurations spanning the common RAG design space. Each
isolates one axis from the next, so per-axis quality contributions are
visible on the leaderboard:

| # | YAML | Family | Cost | API keys |
| --- | --- | --- | --- | --- |
| 1 | `01-bm25-only.yaml` | sparse-only | free | none |
| 2 | `02-dense-bge-small.yaml` | dense-only | low | none (local model) |
| 3 | `03-hybrid-bm25-bge.yaml` | hybrid | low | none |
| 4 | `04-hybrid-rerank.yaml` | hybrid + cross-encoder | low | none |
| 5 | `05-hybrid-rerank-claude-haiku.yaml` | full RAG (cheap) | medium | `ANTHROPIC_API_KEY` |
| 6 | `06-hybrid-rerank-claude-opus.yaml` | full RAG | high | `ANTHROPIC_API_KEY` |
| 7 | `07-hybrid-rerank-gpt5.yaml` | full RAG | medium | `OPENAI_API_KEY` |
| 8 | `08-hybrid-rerank-local-qwen.yaml` | full RAG (open-weight) | low | local vLLM |

The extractive generator in 1–4 measures the retrieval ceiling cheaply.
The LLM-generator variants (5–8) measure what an LLM adds, or removes, on
top of the same retriever and reranker.

## Axis differences

- 1 → 2: sparse vs dense (same generator).
- 2 → 3: dense vs hybrid.
- 3 → 4: hybrid vs hybrid + rerank.
- 4 → 5: extractive vs LLM generator (cheap).
- 5 → 6: cheap LLM vs expensive LLM (same retriever).
- 5 → 7: Anthropic vs OpenAI at matched cost.
- 5 → 8: API vs local open-weight.

## Running them

```bash
for f in examples/reference/*.yaml; do
  rag-bench eval "$f" \
    --tasks nq-1k,hotpotqa-1k,noisy-qa,unanswerable-qa,counterfactual-qa \
    --seeds 5 \
    --out "benchmarks/reference-runs/$(basename "$f" .yaml).json"
done
```
