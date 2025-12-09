// Pareto frontier (quality × cost) page. Renders a simple ASCII-like scatter
// using inline SVG; no charting deps. Pareto-tagged entries get the gold dot.

import type { GetStaticProps, NextPage } from "next";
import Head from "next/head";
import { Entry, LeaderboardData, meanF1 } from "../lib/leaderboard";

const ParetoPage: NextPage<{ data: LeaderboardData }> = ({ data }) => {
  const verified = data.entries.filter((e) => e.verified);
  const points = verified.map((e) => ({
    entry: e,
    quality: meanF1(e),
    cost: Math.max(e.operational.cost_per_query_usd_mean, 1e-5),
  }));
  const maxQ = Math.max(0.01, ...points.map((p) => p.quality));
  const maxCost = Math.max(0.01, ...points.map((p) => p.cost));
  const W = 900;
  const H = 480;
  const PAD = 60;

  function xx(c: number) {
    return PAD + (Math.log10(c + 1e-6) - Math.log10(1e-5)) * ((W - 2 * PAD) / (Math.log10(maxCost + 1e-6) - Math.log10(1e-5)));
  }
  function yy(q: number) {
    return H - PAD - (q / maxQ) * (H - 2 * PAD);
  }

  return (
    <>
      <Head>
        <title>Pareto frontier — rag-bench</title>
      </Head>
      <main style={{ fontFamily: "ui-sans-serif, system-ui", maxWidth: 1100, margin: "2rem auto", padding: "0 1rem" }}>
        <header>
          <h1>Pareto frontier · quality × cost</h1>
          <p style={{ color: "#555" }}>
            Y: mean F1 across reported tasks. X: cost per query (log scale, USD).
            Gold dots are on the Pareto frontier (no other verified entry dominates them on both axes).
          </p>
          <p><a href="/">← back to leaderboard</a></p>
        </header>

        <svg width={W} height={H} style={{ border: "1px solid #ddd", background: "#fff" }}>
          {/* Axes */}
          <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#333" />
          <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="#333" />
          <text x={W / 2} y={H - 15} textAnchor="middle" style={{ fontSize: 12 }}>cost per query (USD, log scale)</text>
          <text x={20} y={H / 2} textAnchor="middle" transform={`rotate(-90 20 ${H / 2})`} style={{ fontSize: 12 }}>
            quality (mean F1)
          </text>

          {points.map((p, i) => (
            <g key={p.entry.pipeline_hash}>
              <circle
                cx={xx(p.cost)}
                cy={yy(p.quality)}
                r={6}
                fill={p.entry.tags.includes("pareto") ? "#daa520" : "#36c"}
                opacity={0.85}
              />
              <text x={xx(p.cost) + 8} y={yy(p.quality) - 6} style={{ fontSize: 10 }}>
                {p.entry.pipeline_name}
              </text>
            </g>
          ))}

          {points.length === 0 && (
            <text x={W / 2} y={H / 2} textAnchor="middle" style={{ fontSize: 14, fill: "#888" }}>
              No verified entries yet.
            </text>
          )}
        </svg>
      </main>
    </>
  );
};

export const getStaticProps: GetStaticProps<{ data: LeaderboardData }> = async () => {
  const fs = await import("node:fs/promises");
  const path = await import("node:path");
  const file = path.join(process.cwd(), "data", "leaderboard.json");
  let data: LeaderboardData;
  try {
    const raw = await fs.readFile(file, "utf-8");
    data = JSON.parse(raw);
  } catch {
    data = { generated_at: new Date().toISOString(), rag_bench_version: "0.0.1", tasks: [], entries: [] };
  }
  return { props: { data } };
};

export default ParetoPage;
