# agent-insights

`agent-insights` turns the bundled `reflect` skill into an installable CLI.
It reads Claude Code session data from `~/.claude/`, writes all artifacts to
`./insights-output/`, and preserves the skill's main commands:

```bash
agent-insights report --dry-run
agent-insights report
agent-insights facet ~/.claude/projects/<project>/<session>.jsonl --save
agent-insights facets --project /path/to/project
agent-insights corrections --project /path/to/project --max-sessions 50
```

Facet and report generation shell out to authenticated `claude -p`, matching
the original skill behavior.

## Homebrew

Tagged releases publish a Homebrew cask to `tomnagengast/homebrew-tap`:

```bash
brew tap tomnagengast/tap
brew install --cask tomnagengast/tap/agent-insights-cli
```

See [docs/release.md](docs/release.md) for release steps.
