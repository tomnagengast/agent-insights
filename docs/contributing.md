# Contributing

`agent-insights` is intended to stay small, local-first, and easy to run from a checkout.

## Development Loop

Use `uv` for Python operations:

```sh
uv run agent-insights --help
uv run python -m compileall src
uv run agent-insights report --dry-run --skip-facets --agent claude --agent codex
```

Release packaging uses PyInstaller:

```sh
uv run --extra release pyinstaller --version
```

## Project Shape

The package entry point is `agent-insights`.

The main implementation lives in `src/agent_insights/insights.py`.

Release packaging lives in `scripts/` and `packaging/pyinstaller/`.

Docs live in `docs/`.

Generated reports belong under `insights-output/` and should not be committed.

## Documentation

Update `README.md` and the relevant docs page when changing commands, supported agents, output layout, report behavior, release steps, or local development workflow.
