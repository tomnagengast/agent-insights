# Agent Instructions

## Project Context

`agent-insights` is a Python CLI that turns local AI-agent session logs into usage
reports. The package entry point is `agent-insights`, and the main implementation
lives in `src/agent_insights/insights.py`.

The CLI reads source agent data from local config directories such as
`~/.claude`, `~/.codex`, `~/.cursor`, and `~/.gemini`. Treat those source logs as
read-only. Generated artifacts belong under `./insights-output/`.

## Python Workflow

Use `uv` for all Python operations in this repo. Do not run bare `python`,
`python3`, or `pip` commands unless the user explicitly asks for that.

Preferred commands:

```sh
uv run agent-insights --help
uv run agent-insights report --dry-run --skip-facets
uv run agent-insights report --dry-run --skip-facets --agent claude --agent codex
uv run python -m compileall src
uv run python -m agent_insights.cli report --dry-run --skip-facets
uv run --extra release pyinstaller --version
```

When running from outside this repository, pass the project explicitly:

```sh
uv run --project /Users/tom/repos/tomnagengast/agent-insights agent-insights --help
```

## Development Notes

Keep changes scoped. This project intentionally has a small dependency surface,
so do not add dependencies unless they are needed for the requested behavior.

Preserve the existing CLI behavior and output layout unless the task is
specifically about changing them. Default Claude reports write directly under
`insights-output/`; explicit agent reports write under
`insights-output/<agent>/`.

For multi-agent reports, preserve live progress output from child processes. The
parent process should keep the final JSON summary on stdout.

## Verification

For code changes, run the smallest useful `uv`-based verification. Start with:

```sh
uv run python -m compileall src
uv run agent-insights report --dry-run --skip-facets --agent claude --agent codex
```

For documentation-only changes, inspect the rendered Markdown or at least read
the changed file back before finishing.
