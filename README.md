# agent-insights

**Turn local agent session logs into evidence-backed reports.** `agent-insights`
shows where coding agents help, where they stall, and which project
instructions might reduce repeated steering.

[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](#roadmap)
[![CI](https://github.com/tomnagengast/agent-insights/actions/workflows/ci.yml/badge.svg)](https://github.com/tomnagengast/agent-insights/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

It currently supports Claude Code, Codex CLI, Cursor CLI, and Gemini CLI logs.
By default it analyzes Claude Code sessions. Pass `--agent` one or more times to
analyze other harnesses.

```sh
agent-insights report
agent-insights report --agent codex
agent-insights report --agent claude --agent codex --agent cursor --agent gemini
```

The report pipeline writes JSON and HTML artifacts under `./insights-output/`
by default. Pass `--output <dir>` to use a dedicated output directory for a
run. Facet and report generation shell out to authenticated `claude -p`.

## Install

Tagged releases publish a Homebrew cask:

```sh
brew tap tomnagengast/tap
brew install --cask tomnagengast/tap/agent-insights-cli
```

For local development, use `uv`:

```sh
uv run agent-insights --help
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

Write a run to a dedicated output directory:

```sh
agent-insights report --output ./tmp/insights-run
```

Use `--dry-run` to inspect what would happen without making LLM calls:

```sh
agent-insights report --dry-run --agent codex
```

## Why agent-insights

Raw transcripts are too detailed to review by hand once you use agents every day.
`agent-insights` turns those sessions into directional reports: what work you
delegate, which workflows are effective, where agents get stuck, and which
instructions are worth making durable.

The reports are not ledgers. They are pattern-finding tools built from model
judgments over local transcripts. Validate important findings against example
sessions before changing standing instructions.

## Docs

- [Docs index](docs/README.md): where to start and how the guides fit together.
- [Install](docs/install.md): Homebrew, local development, and release assets.
- [Getting started](docs/getting-started.md): first reports, scoped runs, and dry
  runs.
- [Usage guide](docs/usage.md): commands, workflows, outputs, and when to use
  reports, facets, or corrections.
- [Agent sources](docs/agent-sources.md): supported agents, transcript
  locations, config environment variables, and output directories.
- [Architecture](docs/architecture.md): transcript discovery, facets,
  aggregation, corrections, and report output.
- [Release guide](docs/release.md): tag-driven release and Homebrew publishing
  steps.
- [Contributing](docs/contributing.md): development loop and repo conventions.

## Related tools

- [`scout`](https://github.com/tomnagengast/scout) maps docs and repos so agents
  load the right context.
- [`agent-memoryd`](https://github.com/tomnagengast/agent-memoryd) stores durable
  local memories and exposes them to agents over MCP.

## Roadmap

- [ ] First-class CI and release checks for every supported package path
- [ ] Clearer example reports and fixtures
- [ ] More agent-specific transcript parsers
- [ ] Better correction-to-instruction review workflow

## Development Checks

```sh
uv run python -m compileall src
uv run agent-insights report --dry-run --skip-facets --agent claude --agent codex
```
