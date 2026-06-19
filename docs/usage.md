# Usage

`agent-insights` has two analysis layers:

- Facets turn individual sessions into structured analytics.
- Corrections turn repeated user steering into candidate project instructions.

The main command is `report`. Most users should start there and only reach for
the lower-level commands when debugging or preparing instruction changes.

## Commands

### `report`

`report` discovers sessions, loads cached metadata and facets, generates missing
facets, aggregates the structured data, and writes a report.

```sh
agent-insights report
agent-insights report --project /path/to/project
agent-insights report --project /path/to/project --skip-facets
agent-insights report --agent codex
agent-insights report --agent claude --agent codex --agent gemini
agent-insights report --output ./tmp/insights-run
```

Outputs:

- `./insights-output/report.json` for the default Claude report
- `./insights-output/report.html` for the default Claude report
- `./insights-output/<agent>/report.json` for explicit agent runs
- `./insights-output/<agent>/report.html` for explicit agent runs
- cached facets under each output directory's `facets/` folder

Pass `--output <dir>` to replace `./insights-output/` with a dedicated output
directory. Explicit agent runs still write under `<dir>/<agent>/`.

Use `report` when you want to understand overall agent usage: what kinds of work
you delegate, where the agent helps, where it creates friction, and which
workflow changes might be worth trying.

The report is directional rather than a ledger. It is built from model judgments
over transcripts, so use it as a pattern-finding review and validate important
claims against example sessions before changing durable instructions.

### `facets`

`facets` scans known Claude sessions and generates any missing cached facets
without producing the final report.

```sh
agent-insights facets --project /path/to/project
```

Use this when you specifically care about warming or refreshing the cache. If
your real goal is to learn something from the data, run `report` instead and let
it manage facets.

### `facet`

`facet` analyzes one Claude session JSONL file.

```sh
agent-insights facet ~/.claude/projects/<project>/<session>.jsonl
agent-insights facet ~/.claude/projects/<project>/<session>.jsonl --save
```

Use this to debug a single transcript's classification, such as why a session
counted as friction or why it was categorized as exploration.

With `--save`, the facet is written to `./insights-output/facets/`, or to the
matching `facets/` directory under `--output`.

### `corrections`

`corrections` extracts moments where the user corrected, redirected, interrupted,
or overrode Claude Code, then synthesizes candidate `CLAUDE.md` rules.

```sh
agent-insights corrections --project /path/to/project --max-sessions 50
agent-insights corrections --project /path/to/project --claude-md /path/to/CLAUDE.md
```

Outputs:

- `./insights-output/corrections.json`
- `./insights-output/rules.json`

Pass `--output <dir>` to write these files under a dedicated output directory.

Use corrections when you notice repeated steering: the agent picks the wrong
workflow, edits the wrong file, over-scopes a change, or makes you restate the
same convention. Keep only rules that would have prevented multiple sessions of
friction.

## Facets

A facet is one structured JSON summary for one session. It captures:

- the user's underlying goal
- explicit goal categories
- outcome
- explicit satisfaction signals
- helpfulness
- session type
- friction categories and details
- primary success mode
- one concise session summary

Facets let the report compare many sessions without re-reading every transcript
every time. Once a session has a cached facet, future reports can aggregate it
quickly.

## Recommended Workflows

For a normal project review:

```sh
agent-insights report --project /path/to/project
open ./insights-output/report.html
```

Read the friction and usage-pattern sections first. If they show recurring
behavior you want to change, follow up with:

```sh
agent-insights corrections --project /path/to/project --max-sessions 50
```

Review `./insights-output/rules.json`, keep only rules that are true and
reusable, then update `CLAUDE.md` in your own voice.

For a multi-agent review:

```sh
agent-insights report --output ./tmp/agent-review --agent claude --agent codex --agent cursor --agent gemini
open ./tmp/agent-review/codex/report.html
```

Each selected agent runs in its own subprocess and writes to its own output
directory, so reports do not overwrite each other.

For debugging a strange report result:

```sh
agent-insights facet ~/.claude/projects/<project>/<session>.jsonl
```

If the single-session facet is wrong, inspect transcript parsing or the facet
prompt. If the facet is right but the report is wrong, inspect aggregation or
report generation.

## Choosing A Command

| Goal | Command | Why |
| --- | --- | --- |
| Understand overall usage | `report` | Produces the HTML and JSON report from aggregated facets. |
| Refresh cached analytics | `facets` | Generates missing per-session facets without the report step. |
| Debug one session | `facet` | Shows exactly how one transcript is classified. |
| Improve project instructions | `corrections` | Finds repeated steering and turns it into candidate rules. |

If unsure, run:

```sh
agent-insights report --project /path/to/project
```
