# Getting Started

Run a dry run first to see what would happen without making LLM calls:

```sh
uv run agent-insights report --dry-run --skip-facets
```

Run a full report for the default Claude Code source:

```sh
uv run agent-insights report
open ./insights-output/report.html
```

Scope a report to one project:

```sh
uv run agent-insights report --project /path/to/project
```

Run explicit agent reports in parallel:

```sh
uv run agent-insights report --agent claude --agent codex --agent cursor --agent gemini
open ./insights-output/codex/report.html
```

Reports are directional. Use them to find patterns in agent usage, friction, and instruction opportunities; validate important claims against example sessions before changing durable project instructions.
