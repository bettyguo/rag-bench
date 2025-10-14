"""rag-bench CLI.

Surfaces:
  rag-bench eval <pipeline.yaml> --tasks X,Y --seeds N --out result.json
  rag-bench show <result.json>
  rag-bench submit <result.json> [--submitter NAME] [--leaderboard-dir DIR]
  rag-bench verify <result.json>
  rag-bench tasks ls
  rag-bench components ls
  rag-bench leaderboard regenerate --submissions-dir D --out F
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from rag_bench import __version__
from rag_bench.leaderboard import generate_leaderboard
from rag_bench.metrics.adversarial import (
    AbstentionPrecision,
    AbstentionRecall,
    ImplausibleResistance,
    NegativeRejectionRate,
    PlausibleCompliance,
)
from rag_bench.metrics.base import Metric
from rag_bench.metrics.generation import ExactMatch, LengthRatio, TokenF1
from rag_bench.metrics.retrieval import MRRAtK, NDCGAtK, RecallAtK
from rag_bench.pipeline.base import registered_components
from rag_bench.pipeline.compose import compose_from_path
from rag_bench.runner import Runner
from rag_bench.submission import (
    Submission,
    Submitter,
    build_submission,
    validate_submission,
)

# Trigger task registration by importing the modules. The compose module
# already imports the component modules, so chunkers/retrievers/etc. are
# auto-registered via that path.
from rag_bench.tasks import counterfactual_qa as _t_cf  # noqa: F401
from rag_bench.tasks import hotpotqa as _t_hp  # noqa: F401
from rag_bench.tasks import msmarco as _t_ms  # noqa: F401
from rag_bench.tasks import noisy_qa as _t_nq  # noqa: F401
from rag_bench.tasks import nq as _t_nq2  # noqa: F401
from rag_bench.tasks import synthetic as _t_syn  # noqa: F401
from rag_bench.tasks import unanswerable_qa as _t_ua  # noqa: F401
from rag_bench.tasks.base import Task, get_task_cls, list_tasks

console = Console()


#
# Documented in docs/submitting.md "Eligibility" and docs/quickstart.md as
# `--tasks v1.0-suite`. Keeping the mapping here so it's one place to update
# when the v1 suite changes.

TASK_SUITES: dict[str, tuple[str, ...]] = {
    # v1.0-suite mirrors submission.REQUIRED_TASKS_V1_0 and docs/submitting.md §5
    # "minimum suite"; counterfactual-qa is OPTIONAL.
    "v1.0-suite": ("nq-1k", "hotpotqa-1k", "noisy-qa", "unanswerable-qa"),
    # v1.0-all adds the optional tasks for submitters who want fuller coverage.
    "v1.0-all": (
        "nq-1k", "hotpotqa-1k", "msmarco-1k",
        "noisy-qa", "unanswerable-qa", "counterfactual-qa",
    ),
    "adversarial": ("noisy-qa", "unanswerable-qa", "counterfactual-qa"),
    "smoke": ("synthetic-10",),
}


def _expand_task_csv(tasks_csv: str) -> list[str]:
    """Expand suite shortcuts to concrete task ids; preserve order; de-dup."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in tasks_csv.split(","):
        token = raw.strip()
        if not token:
            continue
        expansion = TASK_SUITES.get(token, (token,))
        for tid in expansion:
            if tid not in seen:
                out.append(tid)
                seen.add(tid)
    return out


_DEFAULT_GEN_METRICS = (TokenF1, ExactMatch, LengthRatio)
_DEFAULT_RET_METRICS = (RecallAtK, NDCGAtK, MRRAtK)


def _metrics_for_task(task: Task) -> list[Metric]:
    out: list[Metric] = []
    if task.spec.metrics_retrieval:
        out.extend([RecallAtK(k=10), NDCGAtK(k=10), MRRAtK(k=10)])
    if task.spec.metrics_generation:
        out.extend([TokenF1(), ExactMatch(), LengthRatio()])
    # Adversarial metrics depend on the task family
    if task.task_id == "unanswerable-qa":
        out.extend([AbstentionRecall(), AbstentionPrecision(), NegativeRejectionRate()])
    if task.task_id == "counterfactual-qa":
        out.extend([PlausibleCompliance(), ImplausibleResistance()])
    return out


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="rag-bench")
def main() -> None:
    """rag-bench — reproducible RAG benchmark."""


@main.command()
@click.argument("pipeline_yaml", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--tasks",
    "tasks_csv",
    required=True,
    help="Comma-separated task ids (e.g. nq-1k,hotpotqa-1k,noisy-qa).",
)
@click.option("--seeds", type=int, default=5, show_default=True, help="Number of seeds.")
@click.option(
    "--split",
    type=click.Choice(["public", "holdout", "all"]),
    default="public",
    show_default=True,
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False),
    default="result.json",
    show_default=True,
    help="Output result.json path.",
)
@click.option("--max-items", type=int, default=None, help="Cap per task (for smoke runs).")
@click.option(
    "--smoke",
    is_flag=True,
    help="Smoke run: cap to 50 items per task and 1 seed. Use --tasks smoke "
    "for the synthetic-10 single-task suite; this flag does not change the "
    "task list or the metric set.",
)
def eval(pipeline_yaml: str, tasks_csv: str, seeds: int, split: str, out: str, max_items: int | None, smoke: bool) -> None:
    """Run a pipeline across tasks and emit result.json."""
    if smoke:
        max_items = max_items or 50
        seeds = min(seeds, 1)
    task_ids = _expand_task_csv(tasks_csv)
    pipe = compose_from_path(pipeline_yaml)
    seeds_tuple = tuple(range(seeds))
    runner = Runner(pipe, seeds=seeds_tuple, split=split)
    tasks: list[Task] = []
    import inspect

    with console.status(f"Loading {len(task_ids)} task(s)..."):
        for tid in task_ids:
            cls = get_task_cls(tid)
            try:
                sig = inspect.signature(cls)
                if max_items is not None and "max_items" in sig.parameters:
                    t = cls(max_items=max_items)
                else:
                    t = cls()
            except (TypeError, ValueError):
                t = cls()
            tasks.append(t)

    console.print(f"[bold]Pipeline:[/] {pipe.name}")
    console.print(f"[bold]Tasks:[/] {', '.join(task_ids)} · seeds={seeds_tuple} · split={split}")

    record = runner.run((t, _metrics_for_task(t)) for t in tasks)

    sub = build_submission(
        pipeline_yaml=Path(pipeline_yaml).read_text(encoding="utf-8"),
        run_record=record,
        tasks=tasks,
    )
    sub.save(out)
    console.print(f"[green]Wrote[/] {out}")
    _render_summary(sub)


def _render_summary(sub: Submission) -> None:
    table = Table(title=f"{sub.pipeline_name} · {sub.pipeline_hash[:23]}…")
    table.add_column("Task", style="cyan", no_wrap=True)
    table.add_column("Metric")
    table.add_column("Mean", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("n", justify="right")
    for tid, payload in sub.tasks.items():
        for name, m in payload.get("metrics", {}).items():
            lo, hi = m["ci_95"]
            table.add_row(
                tid,
                name,
                f"{m['mean']:.4f}",
                f"[{lo:.4f}, {hi:.4f}]",
                str(m.get("n", "?")),
            )
    console.print(table)


@main.command()
@click.argument("result_json", type=click.Path(exists=True, dir_okay=False))
def show(result_json: str) -> None:
    """Render a result.json as a Rich table."""
    sub = Submission.load(result_json)
    console.print(f"[bold]pipeline_hash:[/] {sub.pipeline_hash}")
    console.print(f"[bold]submitter:[/]    {sub.submitter.name}")
    console.print(f"[bold]seeds:[/]        {sub.seeds}")
    console.print(f"[bold]judges:[/]       {sub.judge_ensemble.fingerprint if sub.judge_ensemble else 'none'}")
    console.print(f"[bold]wall_time_s:[/]  {sub.total_wall_time_s:.2f}")
    _render_summary(sub)


@main.command()
@click.argument("result_json", type=click.Path(exists=True, dir_okay=False))
@click.option("--submitter", default="anonymous", help="Submitter name.")
@click.option("--contact", default="", help="Submitter contact (optional).")
@click.option(
    "--leaderboard-dir",
    type=click.Path(file_okay=False),
    default="leaderboard/submissions",
    show_default=True,
)
@click.option("--require-v1-suite/--no-require-v1-suite", default=False, show_default=True)
def submit(result_json: str, submitter: str, contact: str, leaderboard_dir: str, require_v1_suite: bool) -> None:
    """Copy a result.json into leaderboard/submissions/ for PR submission."""
    from datetime import UTC, datetime

    sub = Submission.load(result_json)
    sub.submitter = Submitter(name=submitter, contact=contact)
    # Stamp a stable submitted_at so the leaderboard display survives
    # git clones; file mtime is unreliable for this.
    if not sub.submitted_at:
        sub.submitted_at = datetime.now(UTC).isoformat()
    outcome = validate_submission(sub, require_v1_suite=require_v1_suite)
    for w in outcome.warnings:
        console.print(f"[yellow]warning:[/] {w}")
    if not outcome.ok:
        for e in outcome.errors:
            console.print(f"[red]error:[/] {e}")
        raise click.ClickException("submission failed validation")
    dest = Path(leaderboard_dir) / f"{sub.pipeline_hash.replace('sha256:', '')}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    sub.save(dest)
    console.print(f"[green]Wrote[/] {dest}")
    console.print("Next: open a PR. The leaderboard CI will validate + re-verify.")


@main.command()
@click.argument("result_json", type=click.Path(exists=True, dir_okay=False))
def verify(result_json: str) -> None:
    """Recompute pipeline_hash and run validation; offline sanity check.

    Does NOT re-run the pipeline (that's the CI's job; requires API keys + budget).
    """
    sub = Submission.load(result_json)
    from rag_bench.repro import pipeline_hash

    # Cross-check the declared rag_bench_version against the runtime version.
    # The hash recomputation uses sub.rag_bench_version, so without this
    # guard a submitter can forge the version and self-validate.
    if sub.rag_bench_version != __version__:
        console.print("[red]rag_bench_version mismatch[/]")
        console.print(f"  declared: {sub.rag_bench_version}")
        console.print(f"  runtime:  {__version__}")
        console.print("  Re-run `rag-bench eval` under the current version to re-verify.")
        sys.exit(1)

    tdh = {t: payload["task_data_hash"] for t, payload in sub.tasks.items()}
    recomputed = pipeline_hash(
        sub.pipeline_yaml,
        task_data_hashes=tdh,
        judge_fingerprint=sub.judge_ensemble.fingerprint if sub.judge_ensemble else "none",
        seeds=sub.seeds,
        version=sub.rag_bench_version,
    )
    if recomputed != sub.pipeline_hash:
        console.print("[red]pipeline_hash mismatch[/]")
        console.print(f"  declared:   {sub.pipeline_hash}")
        console.print(f"  recomputed: {recomputed}")
        sys.exit(1)
    outcome = validate_submission(sub, require_v1_suite=False)
    for w in outcome.warnings:
        console.print(f"[yellow]warning:[/] {w}")
    if not outcome.ok:
        for e in outcome.errors:
            console.print(f"[red]error:[/] {e}")
        sys.exit(1)
    console.print("[green]OK[/] — pipeline_hash matches; schema valid.")


@main.group()
def tasks() -> None:
    """Task management."""


@tasks.command("ls")
def tasks_ls() -> None:
    """List all registered tasks."""
    table = Table(title="Registered tasks")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Family")
    table.add_column("Size", justify="right")
    table.add_column("Contamination")
    for tid in list_tasks():
        cls = get_task_cls(tid)
        spec = cls.spec
        table.add_row(tid, spec.name, spec.family, str(spec.size), spec.contamination_risk)
    console.print(table)
    console.print()
    console.print("[bold]Task suites:[/]")
    for suite_name, members in TASK_SUITES.items():
        console.print(f"  [cyan]{suite_name}[/]: {', '.join(members)}")


@tasks.command("show")
@click.argument("task_id")
def tasks_show(task_id: str) -> None:
    """Inspect a single task's spec."""
    cls = get_task_cls(task_id)
    spec = cls.spec
    console.print(f"[bold cyan]{spec.id}[/] · {spec.name}")
    console.print(f"  family:             {spec.family}")
    console.print(f"  language:           {spec.language}")
    console.print(f"  size:               {spec.size}")
    console.print(f"  contamination:      {spec.contamination_risk}")
    console.print(f"  license:            {spec.license}")
    console.print(f"  upstream:           {spec.upstream_url}")
    if spec.metrics_retrieval:
        console.print(f"  metrics_retrieval:  {', '.join(spec.metrics_retrieval)}")
    if spec.metrics_generation:
        console.print(f"  metrics_generation: {', '.join(spec.metrics_generation)}")
    if spec.metrics_end_to_end:
        console.print(f"  metrics_end_to_end: {', '.join(spec.metrics_end_to_end)}")


@main.group()
def components() -> None:
    """Component management."""


@components.command("ls")
def components_ls() -> None:
    """List all registered components grouped by stage."""
    reg = registered_components()
    for stage, names in reg.items():
        console.print(f"[bold cyan]{stage}[/]: {', '.join(names) if names else '(none)'}")


@main.group()
def leaderboard() -> None:
    """Leaderboard data management."""


@leaderboard.command("regenerate")
@click.option(
    "--submissions-dir",
    type=click.Path(file_okay=False),
    default="leaderboard/submissions",
    show_default=True,
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False),
    default="frontend/data/leaderboard.json",
    show_default=True,
)
def leaderboard_regenerate(submissions_dir: str, out: str) -> None:
    """Aggregate leaderboard/submissions/*.json into frontend/data/leaderboard.json."""
    data = generate_leaderboard(submissions_dir, out)
    console.print(f"[green]Wrote[/] {out} ({len(data['entries'])} entries, {len(data['tasks'])} tasks)")


if __name__ == "__main__":
    main()
