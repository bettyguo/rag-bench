# Faithfulness calibration set

Human-annotated examples used to calibrate the multi-judge faithfulness
metric. For each judge family, Krippendorff's α vs the human consensus
is published per task family; judges with α < 0.6 on a family are dropped
for that family.

## Files

- `bootstrap-v0.jsonl` — 50-item author-labelled bootstrap set. The
  `gold_verdict` values are a working draft, not the calibration ground
  truth; they're the input that the human annotation pass refines.
- `human-pass-v1.jsonl` (to be added) — same 200 items (extended from
  the bootstrap), each labelled by ≥3 independent annotators. The
  majority-vote consensus becomes the published `gold_verdict`. Raw
  per-annotator labels are archived for inter-annotator-agreement
  analysis.

## Item schema

```jsonc
{
  "item_id": "cf-001",
  "task_family": "counterfactual",   // single-hop | multi-hop | adversarial-noise | adversarial-abstention | counterfactual
  "claim": "Paris is the capital of France.",
  "context": "Paris is the capital city of France ...",
  "gold_verdict": "supported",        // supported | refuted | neutral
  "rationale": "Direct entailment in the first sentence.",
  "notes": "optional; flag ambiguity here"
}
```

## Authoring rules

1. Atomic claims only. One subject-verb-object relation per claim.
2. Stratify across task families. Aim for about 40 items per family in
   the full 200-item set.
3. Stratify across verdicts: roughly 33% supported, 33% refuted, 33%
   neutral. Real RAG distributions skew supported; the calibration set
   balances to detect bias across verdict types.
4. Span context lengths from one sentence to multiple paragraphs.
5. Include borderline cases. A calibration set of obvious items leaves
   no signal where signal matters.
6. Don't reach for claim shapes that match the chosen judges; aim for
   shapes you find confusing.

## Running calibration

```python
import json
from rag_bench.calibration import CalibrationItem, run_calibration
from rag_bench.judges import AnthropicJudge, OpenAIJudge

items = []
with open("benchmarks/faithfulness-calibration/bootstrap-v0.jsonl") as f:
    for line in f:
        d = json.loads(line)
        items.append(CalibrationItem(
            item_id=d["item_id"],
            claim=d["claim"],
            context=d["context"],
            gold_verdict=d["gold_verdict"],
            task_family=d["task_family"],
        ))

judges = [
    AnthropicJudge(model="claude-haiku-4-5"),
    OpenAIJudge(model="gpt-5-mini"),
    # plus a third judge from an open-weight family
]

report = run_calibration(judges, items)
print(report.to_jsonable())
```

The output is a `CalibrationReport` with per-judge α overall and per
task family. Save as `bootstrap-v0-calibration-report.json` and publish
alongside any leaderboard column that uses faithfulness.

## Human annotation protocol

1. Three annotators per item, blind to each other's labels.
2. Annotators see the same prompt presentation a judge sees.
3. Annotators may flag "ambiguous" as a fourth verdict slot; those
   items become the inter-annotator-disagreement target set.
4. Gold verdict = majority of the three labels; ties resolve to
   `neutral`.
5. Per-pair Krippendorff's α (annotator vs annotator) is reported
   alongside the judge α to establish the human upper bound.

## Budget

About $150 in judge API (3 judges × 200 items × ~$0.001 per call) and
20 hr of human annotation total (3 annotators × ~7 hr each, paced at
about 30 items per hour).
