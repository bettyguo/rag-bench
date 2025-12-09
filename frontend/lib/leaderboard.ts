// Frontend types matching src/rag_bench/leaderboard.py output.

export interface MetricBucket {
  [metricName: string]: number | number[];
  n_items: number;
}

export interface Operational {
  cost_per_query_usd_mean: number;
  latency_ms_p95: number;
}

export interface Entry {
  pipeline_hash: string;
  pipeline_name: string;
  submitter: { name: string; contact: string };
  verified: boolean;
  submitted_at: string | null;
  metrics: { [taskId: string]: MetricBucket };
  operational: Operational;
  judge_fingerprint: string;
  tags: string[];
}

export interface LeaderboardData {
  generated_at: string;
  rag_bench_version: string;
  tasks: string[];
  entries: Entry[];
}

export function meanF1(entry: Entry): number {
  const values: number[] = [];
  for (const tid of Object.keys(entry.metrics)) {
    const m = entry.metrics[tid];
    const v = m["token_f1"];
    if (typeof v === "number") values.push(v);
  }
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
}

export function isPareto(entry: Entry): boolean {
  return entry.tags.includes("pareto");
}
