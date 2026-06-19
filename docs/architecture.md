# Architecture

`agent-insights` turns local agent transcripts into cached per-session facets, then aggregates those facets into JSON and HTML reports.

## Layers

Agent sources describe where each harness stores transcripts and which instruction system matters for report prompts. Claude Code, Codex CLI, Cursor CLI, and Gemini CLI each have different local roots and transcript shapes.

Discovery scans the configured source roots and builds lightweight session metadata. Project scoping filters sessions to a requested project path when possible.

Facet generation turns one session into structured analytics: goal, categories, outcome, helpfulness, friction, success mode, and a concise summary. Facets are cached under the output directory, `insights-output/` by default, so repeated reports do not need to reread every transcript.

Report aggregation combines cached facets, asks the LLM for higher-level findings, and writes `report.json` plus `report.html`.

Corrections extract user steering moments and synthesize candidate project instructions. Treat these as suggestions to review, not rules to apply blindly.

## Output Layout

Default Claude runs write to:

```text
insights-output/
  report.html
  report.json
  facets/
  session-meta/
```

Explicit agent runs write to:

```text
insights-output/<agent>/
  report.html
  report.json
  facets/
  session-meta/
```

`insights-output/` is generated local output and is ignored by git.

`--output <dir>` swaps the base output directory while preserving the same
default and explicit-agent layout.

## Current Boundaries

LLM calls are made through `claude -p`. Dry-run mode avoids those calls. Report findings are model judgments over local transcripts, so use them for pattern finding and validate important findings before changing standing instructions.
