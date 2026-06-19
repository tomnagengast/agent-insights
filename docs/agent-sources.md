# Agent Sources

`agent-insights report` accepts a repeatable `--agent` option:

```sh
agent-insights report
agent-insights report --agent codex
agent-insights report --agent claude --agent codex --agent cursor --agent gemini
```

When no agent is specified, `report` behaves like the original Claude-only tool
and writes to `./insights-output/`. When an agent is specified explicitly, output
is isolated under `./insights-output/<agent>/`.

## Supported Agents

| Agent | Config root | Sessions scanned | Instruction file referenced in prompts |
| --- | --- | --- | --- |
| `claude` | `$CLAUDE_CONFIG_DIR` or `~/.claude` | `<root>/projects/**/*.jsonl` | `CLAUDE.md` |
| `codex` | `$CODEX_HOME` or `~/.codex` | `<root>/sessions/**/*.jsonl` | `AGENTS.md` |
| `cursor` | `$CURSOR_CONFIG_DIR` or `~/.cursor` | `<root>/projects/**/*.jsonl` | `.cursor/rules` |
| `gemini` | `$GEMINI_CONFIG_DIR` or `~/.gemini` | `<root>/tmp/**/*.{json,jsonl}` | `GEMINI.md` |

## Output Layout

Default Claude run:

```text
insights-output/
  report.html
  report.json
  facets/
  session-meta/
```

Explicit agent run:

```text
insights-output/
  codex/
    report.html
    report.json
    facets/
    session-meta/
```

Multi-agent runs launch one subprocess per selected agent and wait for all of
them to finish. The parent process prints a small JSON summary with each agent's
return code and output paths.

## Parser Notes

Claude sessions use the original Claude Code JSONL parser.

Codex sessions are JSONL records with useful message data nested under
`payload`. The parser unwraps those records and filters harness-injected setup
messages, such as `AGENTS.md` context and permission metadata, so user-message
counts and first prompts reflect actual user requests.

Cursor support currently scans JSONL transcript files under `~/.cursor/projects`
and skips unrelated MCP package metadata. Cursor local data can include other
project-scoped files that are not transcripts.

Gemini support scans JSON and JSONL files under `~/.gemini/tmp`. The parser is
generic because Gemini's local session shape is less specialized in this tool
today.

## Agent-Specific Analysis

Different harnesses have different context systems, so report prompts are
specialized per agent:

- Claude Code reports refer to `CLAUDE.md`, skills, hooks, MCP, task agents, and
  headless mode.
- Codex reports refer to `AGENTS.md`, resume/fork workflows, `codex exec`,
  sandbox and permission settings, MCP, and plugins.
- Cursor reports refer to `.cursor/rules`, `cursor-agent resume`, IDE handoff,
  MCP, and reusable prompt scaffolds.
- Gemini reports refer to `GEMINI.md`, chat save/resume, MCP, grounded research,
  and migration paths for newer Gemini-family workflows.

Facet JSON keeps the existing `claude_helpfulness` key for compatibility, even
when the analyzed session comes from another agent.
