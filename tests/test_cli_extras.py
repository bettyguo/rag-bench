"""CLI tests for the suite shortcuts + tasks show command."""

from __future__ import annotations

from click.testing import CliRunner

from rag_bench.cli import TASK_SUITES, _expand_task_csv, main
from rag_bench.submission import REQUIRED_TASKS_V1_0


def test_v1_suite_matches_submission_validator():
    # The CLI shortcut and the validator must agree on what counts as the
    # minimum v1.0 suite, or submitters over- or under-run.
    assert set(TASK_SUITES["v1.0-suite"]) == set(REQUIRED_TASKS_V1_0)


def test_v1_all_is_a_superset_of_v1_suite():
    assert set(TASK_SUITES["v1.0-suite"]).issubset(set(TASK_SUITES["v1.0-all"]))
    assert "counterfactual-qa" in TASK_SUITES["v1.0-all"]


def test_expand_v1_suite():
    out = _expand_task_csv("v1.0-suite")
    assert "nq-1k" in out
    assert "hotpotqa-1k" in out
    assert "noisy-qa" in out
    assert "unanswerable-qa" in out
    # counterfactual is optional per docs/submitting.md; only in v1.0-all
    assert "counterfactual-qa" not in out
    assert len(out) == 4


def test_expand_adversarial_suite():
    out = _expand_task_csv("adversarial")
    assert out == ["noisy-qa", "unanswerable-qa", "counterfactual-qa"]


def test_expand_mixed_suite_and_concrete():
    out = _expand_task_csv("adversarial,nq-1k")
    assert out == ["noisy-qa", "unanswerable-qa", "counterfactual-qa", "nq-1k"]


def test_expand_dedups():
    out = _expand_task_csv("noisy-qa,adversarial,noisy-qa")
    # noisy-qa appears in adversarial AND as concrete; should only appear once
    assert out.count("noisy-qa") == 1


def test_expand_unknown_passes_through():
    """Unknown tokens are treated as concrete task ids — the registry lookup catches them later."""
    out = _expand_task_csv("not-a-real-task")
    assert out == ["not-a-real-task"]


def test_suite_registry_covers_documented_names():
    assert "v1.0-suite" in TASK_SUITES
    assert "adversarial" in TASK_SUITES
    assert "smoke" in TASK_SUITES


def test_cli_tasks_show_synthetic():
    r = CliRunner().invoke(main, ["tasks", "show", "synthetic-10"])
    assert r.exit_code == 0, r.output
    assert "synthetic-10" in r.output
    assert "single-hop-qa" in r.output
    assert "novel" in r.output  # contamination_risk


def test_cli_tasks_show_unknown_errors():
    r = CliRunner().invoke(main, ["tasks", "show", "not-a-real-task"])
    assert r.exit_code != 0


def test_cli_tasks_ls_includes_suites():
    r = CliRunner().invoke(main, ["tasks", "ls"])
    assert r.exit_code == 0, r.output
    assert "v1.0-suite" in r.output
    assert "adversarial" in r.output


def test_smoke_helptext_does_not_claim_unimplemented_behavior():
    # helptext used to claim --smoke restricts to nq-1k and skips
    # faithfulness; neither was true. Keep the false claims out.
    r = CliRunner().invoke(main, ["eval", "--help"])
    assert r.exit_code == 0, r.output
    full_help = r.output
    assert "nq-1k only" not in full_help
    assert "skips faithfulness" not in full_help
