// Default leaderboard page. Renders the verified-by-default table with sort + filter affordances.
// Reads data at build time from ../data/leaderboard.json.

import type { GetStaticProps, NextPage } from "next";
import Head from "next/head";
import { useMemo, useState } from "react";
import { Entry, LeaderboardData, meanF1 } from "../lib/leaderboard";

interface Props {
  data: LeaderboardData;
}

const LeaderboardPage: NextPage<Props> = ({ data }) => {
  const [showUnverified, setShowUnverified] = useState(false);
  const [sortKey, setSortKey] = useState<"overall" | "cost" | "latency">("overall");

  const entries = useMemo(() => {
    const filtered = showUnverified ? data.entries : data.entries.filter((e) => e.verified);
    const sorted = [...filtered];
    if (sortKey === "overall") {
      sorted.sort((a, b) => meanF1(b) - meanF1(a));
    } else if (sortKey === "cost") {
      sorted.sort(
        (a, b) => a.operational.cost_per_query_usd_mean - b.operational.cost_per_query_usd_mean
      );
    } else {
      sorted.sort((a, b) => a.operational.latency_ms_p95 - b.operational.latency_ms_p95);
    }
    return sorted;
  }, [data.entries, showUnverified, sortKey]);

  return (
    <>
      <Head>
        <title>rag-bench leaderboard</title>
        <meta
          name="description"
          content="Reproducible RAG benchmark. Plug in your pipeline; get honest, comparable numbers."
        />
      </Head>
      <main style={{ fontFamily: "ui-sans-serif, system-ui", maxWidth: 1200, margin: "2rem auto", padding: "0 1rem" }}>
        <header style={{ marginBottom: "2rem" }}>
          <h1 style={{ margin: 0 }}>rag-bench leaderboard</h1>
          <p style={{ color: "#555" }}>
            Reproducible RAG evaluation. {data.entries.length} entries · generated {data.generated_at}.
            <br />
            <a href="https://github.com/airine/rag-bench/blob/main/docs/methodology.md">Methodology</a> ·{" "}
            <a href="https://github.com/airine/rag-bench/blob/main/docs/submitting.md">Submit your pipeline</a> ·{" "}
            <a href="/pareto/">Pareto frontier</a>
          </p>
        </header>

        <div style={{ display: "flex", gap: "1rem", alignItems: "center", marginBottom: "1rem" }}>
          <label>
            <input
              type="checkbox"
              checked={showUnverified}
              onChange={(e) => setShowUnverified(e.target.checked)}
            />{" "}
            Show unverified
          </label>
          <label>
            Sort:{" "}
            <select value={sortKey} onChange={(e) => setSortKey(e.target.value as any)}>
              <option value="overall">Overall (mean F1)</option>
              <option value="cost">Cost (per query)</option>
              <option value="latency">Latency (p95)</option>
            </select>
          </label>
        </div>

        <div style={{ overflowX: "auto" }}>
          <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 14 }}>
            <thead>
              <tr style={{ background: "#f4f4f4" }}>
                <th style={th}>Pipeline</th>
                <th style={th}>Mean F1</th>
                {data.tasks.map((t) => (
                  <th style={th} key={t}>{t}</th>
                ))}
                <th style={th}>Cost / q</th>
                <th style={th}>Latency p95</th>
                <th style={th}>Tags</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((e) => (
                <Row key={e.pipeline_hash} entry={e} tasks={data.tasks} />
              ))}
              {entries.length === 0 && (
                <tr>
                  <td colSpan={data.tasks.length + 4} style={{ padding: "2rem", textAlign: "center", color: "#777" }}>
                    No entries yet. Open a PR to submit; see
                    {" "}<a href="https://github.com/airine/rag-bench/blob/main/docs/submitting.md">docs/submitting.md</a>.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </main>
    </>
  );
};

const th: React.CSSProperties = {
  textAlign: "left",
  padding: "0.5rem",
  borderBottom: "1px solid #ddd",
};

function Row({ entry, tasks }: { entry: Entry; tasks: string[] }) {
  const f1 = meanF1(entry);
  return (
    <tr style={{ borderBottom: "1px solid #eee" }}>
      <td style={td}>
        <strong>{entry.pipeline_name}</strong>{" "}
        <span style={{ color: "#999", fontSize: 11 }} title={entry.pipeline_hash}>
          {entry.pipeline_hash.slice(7, 15)}…
        </span>
        <br />
        <span style={{ fontSize: 12, color: "#777" }}>by {entry.submitter.name}</span>
      </td>
      <td style={td}><strong>{f1.toFixed(3)}</strong></td>
      {tasks.map((t) => {
        const m = entry.metrics[t];
        const f1 = m?.["token_f1"];
        return (
          <td style={td} key={t}>
            {typeof f1 === "number" ? f1.toFixed(3) : "—"}
          </td>
        );
      })}
      <td style={td}>${entry.operational.cost_per_query_usd_mean.toFixed(4)}</td>
      <td style={td}>{Math.round(entry.operational.latency_ms_p95)} ms</td>
      <td style={td}>
        {entry.verified && <span style={{ ...badge, background: "#0a7" }}>verified</span>}
        {!entry.verified && <span style={{ ...badge, background: "#999" }}>unverified</span>}
        {entry.tags.map((t) => (
          <span key={t} style={{ ...badge, background: "#36c" }}>{t}</span>
        ))}
      </td>
    </tr>
  );
}

const td: React.CSSProperties = {
  padding: "0.5rem",
  verticalAlign: "top",
};

const badge: React.CSSProperties = {
  display: "inline-block",
  padding: "0.1rem 0.4rem",
  marginRight: 4,
  borderRadius: 4,
  color: "#fff",
  fontSize: 11,
};

export const getStaticProps: GetStaticProps<Props> = async () => {
  // Build-time: read frontend/data/leaderboard.json (regenerated by CI).
  const fs = await import("node:fs/promises");
  const path = await import("node:path");
  const file = path.join(process.cwd(), "data", "leaderboard.json");
  let data: LeaderboardData;
  try {
    const raw = await fs.readFile(file, "utf-8");
    data = JSON.parse(raw);
  } catch {
    data = {
      generated_at: new Date().toISOString(),
      rag_bench_version: "0.0.1",
      tasks: [],
      entries: [],
    };
  }
  return { props: { data } };
};

export default LeaderboardPage;
