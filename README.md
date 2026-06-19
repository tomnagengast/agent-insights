# agent-insights

`agent-insights` turns local AI-agent session logs into usage reports.

It currently supports Claude Code, Codex CLI, Cursor CLI, and Gemini CLI logs.
By default it analyzes Claude Code sessions. Pass `--agent` one or more times to
analyze other harnesses.

```sh
agent-insights report
agent-insights report --agent codex
agent-insights report --agent claude --agent codex --agent cursor --agent gemini
```

The report pipeline writes JSON and HTML artifacts under `./insights-output/`.
Facet and report generation shell out to authenticated `claude -p`, matching the
original Claude Code `/insights` behavior.

## Install

Tagged releases publish a Homebrew cask:

```sh
brew tap tomnagengast/tap
brew install --cask tomnagengast/tap/agent-insights-cli
```

For local development:

```sh
python -m pip install -e .
agent-insights --help
```

## Quickstart

Run a full report for the default agent:

```sh
agent-insights report
open ./insights-output/report.html
```

Run a report for a specific project:

```sh
agent-insights report --project /path/to/project
```

Run reports for multiple agents in parallel:

```sh
agent-insights report --agent claude --agent codex --agent cursor --agent gemini
```

Use `--dry-run` to inspect what would happen without making LLM calls:

```sh
agent-insights report --dry-run --agent codex
```

## Docs

- [Usage guide](docs/usage.md): commands, workflows, outputs, and when to use
  reports, facets, or corrections.
- [Agent sources](docs/agent-sources.md): supported agents, transcript
  locations, config environment variables, and output directories.
- [Release guide](docs/release.md): tag-driven release and Homebrew publishing
  steps.

## Development Checks

```sh
python -m compileall src
PYTHONPATH=src python -m agent_insights.cli report --dry-run --skip-facets
```
