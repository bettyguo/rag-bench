# Security Policy

## Supported versions

`rag-bench` is pre-1.0. Security fixes target the `main` branch; backports to
prior tags are best-effort.

| Version | Supported |
| --- | --- |
| `0.0.x` | ✅ (current) |

## Reporting a vulnerability

If you discover a vulnerability, please **do not open a public issue**.

1. Open a private [GitHub Security Advisory](https://github.com/airine/rag-bench/security/advisories/new) on the repo.
2. Include: a description, reproducer, affected versions, and (if known) suggested fix.
3. We will acknowledge within 5 working days and aim for a patch + advisory within 30 days for high-severity items.

We will credit you in the advisory unless you ask us not to.

## What counts as a security issue

- Code execution via crafted pipeline YAML, task data, submission JSON, or judge prompt.
- Privilege escalation in CI workflows.
- Leakage of API keys or other secrets via logs or result.json.
- Path traversal in CLI commands that take filesystem paths.

## What we treat as standard bugs, not security issues

- A pipeline that produces wrong metrics (file as a regular bug).
- A submission that passes validation but shouldn't (file as a methodology issue).
- LLM-judge disagreement with humans (file as a calibration concern).
- A pipeline-hash collision attack via a YAML-comment-injection trick — file privately; we'll classify with you.

## API-key handling

`rag-bench` reads API keys from environment variables (`ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `OPENAI_BASE_URL`). We do not write them to disk, do not log
them, and the only YAML field that can affect API behavior is the model name.

The submission `result.json` does **not** contain API keys, judge responses
verbatim, or any submitter-private information beyond what `submitter`
explicitly declared.
