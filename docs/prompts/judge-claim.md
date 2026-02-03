# Judge claim prompt

Source of truth for `rag_bench.judges.CLAIM_PROMPT`. The SHA-256 of the
text below (computed in `judges.py`) is the `prompt_hash` that enters
the pipeline_hash for any submission reporting faithfulness.

```
You are checking whether a single atomic claim is entailed by a provided context.

CONTEXT:
{context}

CLAIM:
{claim}

Respond with EXACTLY ONE WORD from: supported, refuted, neutral
- "supported" if the context entails the claim.
- "refuted"   if the context contradicts the claim.
- "neutral"   if the context neither supports nor refutes the claim.

VERDICT:
```

## Design notes

- Single-word answer: trivial to parse, rules out judge-side reasoning
  spillover into the answer slot.
- Three-way verdict: cleaner than binary. "refuted" matters for
  counterfactual tasks; "neutral" is the right default when the context
  doesn't speak to the claim.
- No CoT (for cost). The N=3 cross-vendor majority vote compensates.
  Worth revisiting if a calibration study shows CoT raises α > 0.10
  against humans at acceptable extra cost.
- No few-shot examples. They'd couple the judge to in-distribution
  cases; the adversarial claim shapes are diverse enough that few-shot
  risks hurting transfer.

## Known limitations

- Long contexts (>8K tokens) may stress some judge models' attention. We
  truncate context at the format-time level; this is documented and the choice
  enters the hash.
- The "supported" majority rule treats a 1-supported / 1-refuted / 1-neutral
  vote as **unsupported**, which under-counts borderline cases. This is the
  conservative direction.
- Position bias is partially defanged by per-judge per-claim shuffling; it is
  not fully eliminated. Calibration against humans quantifies the residual.
