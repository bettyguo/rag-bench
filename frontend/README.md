# rag-bench frontend

Next.js static-export site for the rag-bench leaderboard.

## Local dev

```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

## Data refresh

The frontend reads `frontend/data/leaderboard.json` at build time. To regenerate
from current `leaderboard/submissions/*.json` files:

```bash
# From repo root:
python -m rag_bench.cli leaderboard regenerate \
    --submissions-dir leaderboard/submissions \
    --out frontend/data/leaderboard.json
```

CI does this automatically on every accepted submission PR (see [.github/workflows/leaderboard.yml]).

## Deploy

`next.config.mjs` is set to `output: 'export'` so `npm run build` emits a static
site under `frontend/out/`. Deploy via:

- Vercel (auto-detected)
- GitHub Pages (deploy `out/`)
- Any static host

## Pages

| Route | File | Purpose |
| --- | --- | --- |
| `/` | `pages/index.tsx` | Sortable, filterable leaderboard table |
| `/pareto/` | `pages/pareto.tsx` | Pareto frontier scatter (quality × cost) |
| `/task/<task-id>/` | TODO | Per-task breakdown |
| `/pipeline/<hash>/` | TODO | Pipeline detail page |
