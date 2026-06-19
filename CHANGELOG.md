# Changelog

All notable changes to `agent-insights` are tracked here.

## v0.1.4 - 2026-06-19

- Fixes `agent-insights -v` and `agent-insights --version` to print the Homebrew-style semver output, such as `agent-insights v0.1.4`.
- Adds a release archive guard so packaged binaries must match the release tag version.

## v0.1.3 - 2026-06-19

- Adds CI, Dependabot, issue templates, a pull request template, CODEOWNERS, license, security policy, and changelog.
- Adds docs for installation, getting started, architecture, and contributing.
- Adds a richer CLI output renderer and a configurable output directory.
- Updates repository docs and release workflows to use `uv`.

## v0.1.2 - 2026-06-19

- Adds short `-v` / `--version` support.
- Adds multi-agent report support for Claude Code, Codex CLI, Cursor CLI, and Gemini CLI.
- Publishes standalone macOS release archives and the `agent-insights-cli` Homebrew cask.

## v0.1.0 - 2026-06-19

- Initial public release.
- Adds local session discovery, facet generation, corrections, report aggregation, and HTML/JSON report output.
