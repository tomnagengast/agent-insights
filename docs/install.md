# Install

`agent-insights` is a Python CLI. Tagged releases publish standalone macOS archives through a Homebrew cask, and local development uses `uv`.

## Homebrew

```sh
brew tap tomnagengast/tap
brew install --cask tomnagengast/tap/agent-insights-cli
agent-insights --version
```

Release archives currently target macOS arm64 and Intel. Other platforms should use a source checkout.

## Local Development

Install `uv`, then run commands from the repository root:

```sh
uv run agent-insights --help
uv run agent-insights --version
```

The project intentionally keeps runtime dependencies small. Use `uv` for Python operations in this repo.

## LLM Setup

Facet and report generation call authenticated `claude -p`. Dry runs do not make LLM calls:

```sh
uv run agent-insights report --dry-run --skip-facets
```

Before running a full report, make sure `claude` is on `PATH` and authenticated.
