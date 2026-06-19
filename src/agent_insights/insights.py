#!/usr/bin/env python3
"""
Generate AI agent session insights reports.

Reverse-engineered from the claude binary (v2.1.63) on 2026-02-28.
Traces every API call from /insights execution through report.html generation.

Usage:
    # Generate facet for a single session
    agent-insights facet <session-jsonl-path>

    # Generate facets for all sessions missing them
    agent-insights facets

    # Generate the full report (facets + report)
    agent-insights report

    # Generate a report for one non-default agent
    agent-insights report --agent codex

    # Generate reports for several agents in parallel
    agent-insights report --agent claude --agent codex --agent gemini

    # Dry run — show what would happen without making API calls
    agent-insights report --dry-run

Uses `claude -p` (headless mode) for LLM calls — works with Max plans,
no ANTHROPIC_API_KEY needed. Just needs `claude` on PATH and authenticated.
"""

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from . import __version__
from .console import ConsoleRenderer
from .console import format_elapsed
from .console import stderr_console

# ---------------------------------------------------------------------------
# Constants (extracted from claude binary)
# ---------------------------------------------------------------------------

# Source dirs (read-only — never written to)
CLAUDE_DIR = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude").expanduser()
CLAUDE_USAGE_DIR = CLAUDE_DIR / "usage-data"
CLAUDE_FACETS_DIR = CLAUDE_USAGE_DIR / "facets"
CLAUDE_META_DIR = CLAUDE_USAGE_DIR / "session-meta"
PROJECTS_DIR = CLAUDE_DIR / "projects"
CODEX_DIR = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser()
CURSOR_DIR = Path(os.environ.get("CURSOR_CONFIG_DIR") or Path.home() / ".cursor").expanduser()
GEMINI_DIR = Path(os.environ.get("GEMINI_CONFIG_DIR") or Path.home() / ".gemini").expanduser()

# Output dir (all writes go here)
OUTPUT_DIR = Path.cwd() / "insights-output"
OUT_FACETS_DIR = OUTPUT_DIR / "facets"
OUT_META_DIR = OUTPUT_DIR / "session-meta"

# Both facet and report generation use Opus in the official pipeline.
# Override with env vars if desired (e.g. to use Haiku for cheaper testing).
FACET_MODEL = os.environ.get("INSIGHTS_FACET_MODEL", "claude-opus-4-6")
REPORT_MODEL = os.environ.get("INSIGHTS_REPORT_MODEL", "claude-opus-4-6")

# Pipeline limits (from kb8 function)
SESSION_BATCH_SIZE = 50
MAX_NEW_FACETS = 200
JSONL_PARSE_BATCH = 10
FACET_PARALLEL_BATCH = 50
MAX_TRANSCRIPT_LEN = 30_000
TRANSCRIPT_CHUNK_SIZE = 25_000


@dataclass(frozen=True)
class AgentSpec:
    name: str
    display_name: str
    instruction_file: str
    config_dir: Path
    output_dir: Path
    source_facets_dir: Path | None = None
    source_meta_dir: Path | None = None

    @property
    def out_facets_dir(self) -> Path:
        return self.output_dir / "facets"

    @property
    def out_meta_dir(self) -> Path:
        return self.output_dir / "session-meta"


AGENT_CHOICES = ("claude", "codex", "cursor", "gemini")


def get_agent_spec(agent_name: str, *, isolated_output: bool = False) -> AgentSpec:
    output_dir = OUTPUT_DIR if agent_name == "claude" and not isolated_output else OUTPUT_DIR / agent_name
    if agent_name == "claude":
        return AgentSpec(
            name="claude",
            display_name="Claude Code",
            instruction_file="CLAUDE.md",
            config_dir=CLAUDE_DIR,
            output_dir=output_dir,
            source_facets_dir=CLAUDE_FACETS_DIR,
            source_meta_dir=CLAUDE_META_DIR,
        )
    if agent_name == "codex":
        return AgentSpec(
            name="codex",
            display_name="Codex CLI",
            instruction_file="AGENTS.md",
            config_dir=CODEX_DIR,
            output_dir=output_dir,
        )
    if agent_name == "cursor":
        return AgentSpec(
            name="cursor",
            display_name="Cursor CLI",
            instruction_file=".cursor/rules",
            config_dir=CURSOR_DIR,
            output_dir=output_dir,
        )
    if agent_name == "gemini":
        return AgentSpec(
            name="gemini",
            display_name="Gemini CLI",
            instruction_file="GEMINI.md",
            config_dir=GEMINI_DIR,
            output_dir=output_dir,
        )
    raise ValueError(f"Unsupported agent: {agent_name}")

# ---------------------------------------------------------------------------
# Prompts (extracted verbatim from binary)
# ---------------------------------------------------------------------------

FACET_SYSTEM_PROMPT = ""  # u0([]) = base system prompt = empty for insights

FACET_PROMPT_PREFIX = """Analyze this Claude Code session and extract structured facets.
CRITICAL GUIDELINES:
1. **goal_categories**: Count ONLY what the USER explicitly asked for.
   - DO NOT count Claude's autonomous codebase exploration
   - DO NOT count work Claude decided to do on its own
   - ONLY count when user says "can you...", "please...", "I need...", "let's..."
2. **user_satisfaction_counts**: Base ONLY on explicit user signals.
   - "Yay!", "great!", "perfect!" → happy
   - "thanks", "looks good", "that works" → satisfied
   - "ok, now let's..." (continuing without complaint) → likely_satisfied
   - "that's not right", "try again" → dissatisfied
   - "this is broken", "I give up" → frustrated
3. **friction_counts**: Be specific about what went wrong.
   - misunderstood_request: Claude interpreted incorrectly
   - wrong_approach: Right goal, wrong solution method
   - buggy_code: Code didn't work correctly
   - user_rejected_action: User said no/stop to a tool call
   - excessive_changes: Over-engineered or changed too much
4. If very short or just warmup, use warmup_minimal for goal_category
SESSION:
"""

FACET_SCHEMA_SUFFIX = """
RESPOND WITH ONLY A VALID JSON OBJECT matching this schema:
  "underlying_goal": "What the user fundamentally wanted to achieve",
  "goal_categories": {"category_name": count, ...},
  "outcome": "fully_achieved|mostly_achieved|partially_achieved|not_achieved|unclear_from_transcript",
  "user_satisfaction_counts": {"level": count, ...},
  "claude_helpfulness": "unhelpful|slightly_helpful|moderately_helpful|very_helpful|essential",
  "session_type": "single_task|multi_task|iterative_refinement|exploration|quick_question",
  "friction_counts": {"friction_type": count, ...},
  "friction_detail": "One sentence describing friction or empty",
  "primary_success": "none|fast_accurate_search|correct_code_edits|good_explanations|proactive_help|multi_file_changes|good_debugging",
  "brief_summary": "One sentence: what user wanted and whether they got it"
}"""


def agent_text(text: str, spec: AgentSpec) -> str:
    return (
        text
        .replace("CLAUDE.md", spec.instruction_file)
        .replace("Claude Code", spec.display_name)
        .replace("Claude's", f"{spec.display_name}'s")
        .replace("Claude", spec.display_name)
        .replace(" CC ", f" {spec.display_name} ")
    )


def agent_schema_suffix(spec: AgentSpec) -> str:
    suffix = agent_text(FACET_SCHEMA_SUFFIX, spec)
    if spec.name != "claude":
        suffix += (
            f"\n\nNOTE: keep the key `claude_helpfulness` for compatibility, "
            f"but evaluate {spec.display_name}'s helpfulness."
        )
    return suffix


def facet_prompt_prefix(spec: AgentSpec) -> str:
    return f"""Analyze this {spec.display_name} session and extract structured facets.
CRITICAL GUIDELINES:
1. **goal_categories**: Count ONLY what the USER explicitly asked for.
   - DO NOT count {spec.display_name}'s autonomous codebase exploration
   - DO NOT count work {spec.display_name} decided to do on its own
   - ONLY count when user says "can you...", "please...", "I need...", "let's..."
2. **user_satisfaction_counts**: Base ONLY on explicit user signals.
   - "Yay!", "great!", "perfect!" -> happy
   - "thanks", "looks good", "that works" -> satisfied
   - "ok, now let's..." (continuing without complaint) -> likely_satisfied
   - "that's not right", "try again" -> dissatisfied
   - "this is broken", "I give up" -> frustrated
3. **friction_counts**: Be specific about what went wrong.
   - misunderstood_request: {spec.display_name} interpreted incorrectly
   - wrong_approach: Right goal, wrong solution method
   - buggy_code: Code didn't work correctly
   - user_rejected_action: User said no/stop to a tool call
   - excessive_changes: Over-engineered or changed too much
4. The `claude_helpfulness` field means how helpful this {spec.display_name} session was.
5. If very short or just warmup, use warmup_minimal for goal_category
SESSION:
"""


def feature_reference(spec: AgentSpec) -> str:
    if spec.name == "codex":
        return """## Codex CLI FEATURES REFERENCE (pick from these for features_to_try):
1. **AGENTS.md**: Project instructions that shape Codex behavior.
   - How to use: Add or refine AGENTS.md at the repo root
   - Good for: standing repo conventions, test commands, review preferences
2. **Session Resume/Fork**: Continue, fork, archive, or delete local sessions.
   - How to use: `codex resume`, `codex resume --last`, or `codex fork`
   - Good for: long-running tasks, parallel alternatives, recovering context
3. **Non-interactive Mode**: Run Codex from scripts and automation.
   - How to use: `codex exec "fix the failing tests"`
   - Good for: repeatable maintenance, CI assistance, scripted code review
4. **Sandbox and Permissions**: Control what commands and file edits are allowed.
   - How to use: tune Codex config and approval modes
   - Good for: safer autonomy on large or sensitive repos
5. **MCP and Plugins**: Connect Codex to external tools and reusable workflows.
   - How to use: configure MCP servers or install plugins
   - Good for: GitHub, Linear, browser testing, docs, and internal systems"""
    if spec.name == "cursor":
        return """## Cursor CLI FEATURES REFERENCE (pick from these for features_to_try):
1. **Cursor Rules**: Repo or user rules that steer Cursor's agent.
   - How to use: Add rules under `.cursor/rules`
   - Good for: coding style, framework conventions, common commands
2. **Session Resume**: Resume recent Cursor agent sessions.
   - How to use: `cursor-agent resume` or `cursor-agent --resume <thread-id>`
   - Good for: returning to interrupted CLI work
3. **IDE Handoff**: Use the CLI with Cursor's editor context.
   - How to use: start in the CLI, then inspect/refine changes in Cursor
   - Good for: reviewing diffs and continuing implementation visually
4. **MCP Servers**: Connect Cursor to external tools.
   - How to use: configure MCP servers in Cursor settings
   - Good for: docs, issue trackers, browser tools, databases
5. **Prompt Scaffolds**: Reusable prompts for repeatable repo tasks.
   - How to use: save task-specific prompts in project docs or rules
   - Good for: reviews, migrations, test-fix loops, release chores"""
    if spec.name == "gemini":
        return """## Gemini CLI FEATURES REFERENCE (pick from these for features_to_try):
1. **GEMINI.md**: Project guidance loaded into Gemini CLI context.
   - How to use: Add or refine GEMINI.md at the repo root
   - Good for: conventions, commands, architecture notes
2. **Chat Save/Resume**: Save and resume conversation checkpoints.
   - How to use: `/chat save <tag>` and `/chat resume <tag>`
   - Good for: branching investigation and returning to known context
3. **MCP Servers**: Connect Gemini CLI to external tools.
   - How to use: configure MCP servers for repo or user workflows
   - Good for: docs, tickets, web context, internal APIs
4. **Grounded Research**: Use Gemini's search and large-context strengths.
   - How to use: ask it to verify current docs before implementation
   - Good for: API/library migrations and unfamiliar systems
5. **Antigravity Migration**: Move newer workflows to Antigravity CLI where applicable.
   - How to use: map Gemini skills, hooks, and extensions to Antigravity plugins
   - Good for: future-proofing Gemini CLI workflows"""
    return """## CC FEATURES REFERENCE (pick from these for features_to_try):
1. **MCP Servers**: Connect Claude to external tools, databases, and APIs via Model Context Protocol.
   - How to use: Run `claude mcp add <server-name> -- <command>`
   - Good for: database queries, Slack integration, GitHub issue lookup, connecting to internal APIs
2. **Custom Skills**: Reusable prompts you define as markdown files that run with a single /command.
   - How to use: Create `.claude/skills/commit/SKILL.md` with instructions. Then type `/commit` to run it.
   - Good for: repetitive workflows - /commit, /review, /test, /deploy, /pr, or complex multi-step workflows
3. **Hooks**: Shell commands that auto-run at specific lifecycle events.
   - How to use: Add to `.claude/settings.json` under "hooks" key.
   - Good for: auto-formatting code, running type checks, enforcing conventions
4. **Headless Mode**: Run Claude non-interactively from scripts and CI/CD.
   - How to use: `claude -p "fix lint errors" --allowedTools "Edit,Read,Bash"`
   - Good for: CI/CD integration, batch code fixes, automated reviews
5. **Task Agents**: Claude spawns focused sub-agents for complex exploration or parallel work.
   - How to use: Claude auto-invokes when helpful, or ask "use an agent to explore X"
   - Good for: codebase exploration, understanding complex systems"""


def report_prompts(spec: AgentSpec) -> list[dict]:
    prompts = []
    for prompt_def in REPORT_PROMPTS:
        prompt_def = copy.deepcopy(prompt_def)
        prompt = agent_text(prompt_def["prompt"], spec)
        if spec.name != "claude":
            prompt = prompt.replace(
                '"claude_md_additions"',
                f'"{spec.name}_instruction_additions"',
            )
        if prompt_def["name"] == "suggestions":
            prompt = re.sub(
                r"## .*?FEATURES REFERENCE[\s\S]*?RESPOND WITH ONLY A VALID JSON OBJECT:",
                feature_reference(spec) + "\nRESPOND WITH ONLY A VALID JSON OBJECT:",
                prompt,
                count=1,
            )
        prompt_def["prompt"] = prompt
        prompts.append(prompt_def)
    return prompts

CHUNK_SUMMARIZE_PROMPT = """Summarize this portion of a Claude Code session transcript. Focus on:
1. What the user asked for
2. What Claude did (tools used, files modified)
3. Any friction or issues
4. The outcome
Keep it concise - 3-5 sentences. Preserve specific details like file names, error messages, and user feedback.
TRANSCRIPT CHUNK:
"""

# 7 report section prompts (from Ub8 array) + at_a_glance (generated after)
REPORT_PROMPTS = [
    {
        "name": "project_areas",
        "prompt": """Analyze this Claude Code usage data and identify project areas.
RESPOND WITH ONLY A VALID JSON OBJECT:
  "areas": [
    {"name": "Area name", "session_count": N, "description": "2-3 sentences about what was worked on and how Claude Code was used."}
Include 4-5 areas. Skip internal CC operations.""",
        "max_tokens": 8192,
    },
    {
        "name": "interaction_style",
        "prompt": """Analyze this Claude Code usage data and describe the user's interaction style.
RESPOND WITH ONLY A VALID JSON OBJECT:
  "narrative": "2-3 paragraphs analyzing HOW the user interacts with Claude Code. Use second person 'you'. Describe patterns: iterate quickly vs detailed upfront specs? Interrupt often or let Claude run? Include specific examples. Use **bold** for key insights.",
  "key_pattern": "One sentence summary of most distinctive interaction style"
}""",
        "max_tokens": 8192,
    },
    {
        "name": "what_works",
        "prompt": """Analyze this Claude Code usage data and identify what's working well for this user. Use second person ("you").
RESPOND WITH ONLY A VALID JSON OBJECT:
  "intro": "1 sentence of context",
  "impressive_workflows": [
    {"title": "Short title (3-6 words)", "description": "2-3 sentences describing the impressive workflow or approach. Use 'you' not 'the user'."}
Include 3 impressive workflows.""",
        "max_tokens": 8192,
    },
    {
        "name": "friction_analysis",
        "prompt": """Analyze this Claude Code usage data and identify friction points for this user. Use second person ("you").
RESPOND WITH ONLY A VALID JSON OBJECT:
  "intro": "1 sentence summarizing friction patterns",
  "categories": [
    {"category": "Concrete category name", "description": "1-2 sentences explaining this category and what could be done differently. Use 'you' not 'the user'.", "examples": ["Specific example with consequence", "Another example"]}
Include 3 friction categories with 2 examples each.""",
        "max_tokens": 8192,
    },
    {
        "name": "suggestions",
        "prompt": """Analyze this Claude Code usage data and suggest improvements.
## CC FEATURES REFERENCE (pick from these for features_to_try):
1. **MCP Servers**: Connect Claude to external tools, databases, and APIs via Model Context Protocol.
   - How to use: Run `claude mcp add <server-name> -- <command>`
   - Good for: database queries, Slack integration, GitHub issue lookup, connecting to internal APIs
2. **Custom Skills**: Reusable prompts you define as markdown files that run with a single /command.
   - How to use: Create `.claude/skills/commit/SKILL.md` with instructions. Then type `/commit` to run it.
   - Good for: repetitive workflows - /commit, /review, /test, /deploy, /pr, or complex multi-step workflows
3. **Hooks**: Shell commands that auto-run at specific lifecycle events.
   - How to use: Add to `.claude/settings.json` under "hooks" key.
   - Good for: auto-formatting code, running type checks, enforcing conventions
4. **Headless Mode**: Run Claude non-interactively from scripts and CI/CD.
   - How to use: `claude -p "fix lint errors" --allowedTools "Edit,Read,Bash"`
   - Good for: CI/CD integration, batch code fixes, automated reviews
5. **Task Agents**: Claude spawns focused sub-agents for complex exploration or parallel work.
   - How to use: Claude auto-invokes when helpful, or ask "use an agent to explore X"
   - Good for: codebase exploration, understanding complex systems
RESPOND WITH ONLY A VALID JSON OBJECT:
  "claude_md_additions": [
    {"addition": "A specific line or block to add to CLAUDE.md based on workflow patterns. E.g., 'Always run tests after modifying auth-related files'", "why": "1 sentence explaining why this would help based on actual sessions", "prompt_scaffold": "Instructions for where to add this in CLAUDE.md. E.g., 'Add under ## Testing section'"}
  ],
  "features_to_try": [
    {"feature": "Feature name from CC FEATURES REFERENCE above", "one_liner": "What it does", "why_for_you": "Why this would help YOU based on your sessions", "example_code": "Actual command or config to copy"}
  ],
  "usage_patterns": [
    {"title": "Short title", "suggestion": "1-2 sentence summary", "detail": "3-4 sentences explaining how this applies to YOUR work", "copyable_prompt": "A specific prompt to copy and try"}
IMPORTANT for claude_md_additions: PRIORITIZE instructions that appear MULTIPLE TIMES in the user data. If user told Claude the same thing in 2+ sessions (e.g., 'always run tests', 'use TypeScript'), that's a PRIME candidate - they shouldn't have to repeat themselves.
IMPORTANT for features_to_try: Pick 2-3 from the CC FEATURES REFERENCE above. Include 2-3 items for each category.""",
        "max_tokens": 8192,
    },
    {
        "name": "on_the_horizon",
        "prompt": """Analyze this Claude Code usage data and identify future opportunities.
RESPOND WITH ONLY A VALID JSON OBJECT:
  "intro": "1 sentence about evolving AI-assisted development",
  "opportunities": [
    {"title": "Short title (4-8 words)", "whats_possible": "2-3 ambitious sentences about autonomous workflows", "how_to_try": "1-2 sentences mentioning relevant tooling", "copyable_prompt": "Detailed prompt to try"}
Include 3 opportunities. Think BIG - autonomous workflows, parallel agents, iterating against tests.""",
        "max_tokens": 8192,
    },
    {
        "name": "fun_ending",
        "prompt": """Analyze this Claude Code usage data and find a memorable moment.
RESPOND WITH ONLY A VALID JSON OBJECT:
  "headline": "A memorable QUALITATIVE moment from the transcripts - not a statistic. Something human, funny, or surprising.",
  "detail": "Brief context about when/where this happened"
Find something genuinely interesting or amusing from the session summaries.""",
        "max_tokens": 8192,
    },
]

# Allowed enum values (from qb8)
ENUMS = {
    "goal_categories": [
        "debug_investigate", "implement_feature", "fix_bug", "write_script_tool",
        "refactor_code", "configure_system", "create_pr_commit", "analyze_data",
        "understand_codebase", "write_tests", "write_docs", "deploy_infra",
        "warmup_minimal",
    ],
    "friction_types": [
        "misunderstood_request", "wrong_approach", "buggy_code",
        "user_rejected_action", "claude_got_blocked", "user_stopped_early",
        "wrong_file_or_location", "excessive_changes", "slow_or_verbose",
        "tool_failed", "user_unclear", "external_issue",
    ],
    "satisfaction_levels": [
        "frustrated", "dissatisfied", "likely_satisfied", "satisfied",
        "happy", "unsure", "neutral", "delighted",
    ],
    "outcomes": [
        "fully_achieved", "mostly_achieved", "partially_achieved",
        "not_achieved", "unclear_from_transcript",
    ],
    "helpfulness": [
        "unhelpful", "slightly_helpful", "moderately_helpful",
        "very_helpful", "essential",
    ],
    "session_types": [
        "single_task", "multi_task", "iterative_refinement",
        "exploration", "quick_question",
    ],
    "success_types": [
        "none", "fast_accurate_search", "correct_code_edits",
        "good_explanations", "proactive_help", "multi_file_changes",
        "handled_complexity", "good_debugging",
    ],
}

DISPLAY_NAMES = {
    "debug_investigate": "Debug/Investigate",
    "implement_feature": "Implement Feature",
    "fix_bug": "Fix Bug",
    "write_script_tool": "Write Script/Tool",
    "refactor_code": "Refactor Code",
    "configure_system": "Configure System",
    "create_pr_commit": "Create PR/Commit",
    "analyze_data": "Analyze Data",
    "understand_codebase": "Understand Codebase",
    "write_tests": "Write Tests",
    "write_docs": "Write Docs",
    "deploy_infra": "Deploy/Infra",
    "warmup_minimal": "Cache Warmup",
    "fast_accurate_search": "Fast/Accurate Search",
    "correct_code_edits": "Correct Code Edits",
    "good_explanations": "Good Explanations",
    "proactive_help": "Proactive Help",
    "multi_file_changes": "Multi-file Changes",
    "handled_complexity": "Multi-file Changes",
    "good_debugging": "Good Debugging",
    "misunderstood_request": "Misunderstood Request",
    "wrong_approach": "Wrong Approach",
    "buggy_code": "Buggy Code",
    "user_rejected_action": "User Rejected Action",
    "claude_got_blocked": "Claude Got Blocked",
    "user_stopped_early": "User Stopped Early",
    "wrong_file_or_location": "Wrong File/Location",
    "excessive_changes": "Excessive Changes",
    "slow_or_verbose": "Slow/Verbose",
    "tool_failed": "Tool Failed",
    "user_unclear": "User Unclear",
    "external_issue": "External Issue",
}

# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------

@dataclass
class ParsedSession:
    session_id: str
    project_path: str
    created: datetime
    modified: datetime
    messages: list  # [{type, message?, ...}]
    first_prompt: str = ""
    agent: str = "claude"
    agent_display: str = "Claude Code"

    @property
    def start_time(self) -> str:
        return self.created.isoformat()

    @property
    def duration_minutes(self) -> int:
        return max(1, round((self.modified - self.created).total_seconds() / 60))


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        timestamp = value / 1000 if value > 10_000_000_000 else value
        try:
            return datetime.fromtimestamp(timestamp)
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _first_str(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return ""


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float, bool)):
        return str(content)
    if isinstance(content, list):
        parts = [_content_to_text(item) for item in content]
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if content.get("type") == "tool_use" and content.get("name"):
            return f"[Tool: {content['name']}]"
        for key in ("text", "content", "message", "output", "summary", "value"):
            text = _content_to_text(content.get(key))
            if text:
                return text
        if "parts" in content:
            return _content_to_text(content["parts"])
    return ""


def _infer_role(obj: dict) -> str:
    role = _first_str(
        obj.get("role"),
        obj.get("type"),
        obj.get("author"),
        obj.get("speaker"),
        obj.get("sender"),
    ).lower()
    message = obj.get("message")
    if isinstance(message, dict):
        role = role or _first_str(message.get("role"), message.get("type")).lower()
    if role in ("human", "prompt", "input"):
        return "user"
    if role in ("model", "ai", "bot", "agent"):
        return "assistant"
    if role in ("user", "assistant"):
        return role
    return ""


def _extract_tool_name(obj: dict) -> str:
    for key in ("tool", "tool_name", "name", "command", "cmd"):
        value = obj.get(key)
        if isinstance(value, str) and value:
            return value
    for key in ("function", "call"):
        value = obj.get(key)
        if isinstance(value, dict):
            name = value.get("name")
            if isinstance(name, str) and name:
                return name
    return ""


def _record_payload(obj: dict) -> dict:
    payload = obj.get("payload")
    if not isinstance(payload, dict):
        return obj
    record = dict(payload)
    record.setdefault("timestamp", obj.get("timestamp"))
    outer_type = obj.get("type")
    if outer_type == "response_item":
        record.setdefault("event", record.get("type"))
    elif outer_type == "event_msg":
        payload_type = record.get("type")
        if payload_type == "user_message":
            record["role"] = "user"
            record["content"] = record.get("message")
        elif payload_type == "agent_message":
            record["role"] = "assistant"
            record["content"] = record.get("message")
        else:
            record.setdefault("event", payload_type)
    elif outer_type in ("session_meta", "turn_context"):
        record.setdefault("event", outer_type)
    return record


def _is_context_injection(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith((
        "# AGENTS.md instructions",
        "<permissions instructions>",
        "<environment_context>",
        "<app-context>",
        "<collaboration_mode>",
        "<skills_instructions>",
        "<plugins_instructions>",
    ))


def _normalize_generic_message(obj: dict) -> dict | None:
    obj = _record_payload(obj)
    typ = _first_str(obj.get("type"), obj.get("event"), obj.get("kind")).lower()
    if "tool" in typ or typ in ("function_call", "function-call", "exec", "command"):
        name = _extract_tool_name(obj) or typ
        return {
            "type": "assistant",
            "message": {"content": [{"type": "tool_use", "name": name}]},
            "timestamp": obj.get("timestamp") or obj.get("time") or obj.get("created_at"),
            "cwd": obj.get("cwd") or obj.get("project_path"),
        }

    role = _infer_role(obj)
    if not role:
        return None

    message = obj.get("message")
    content = None
    if isinstance(message, dict):
        content = message.get("content") or message.get("text") or message.get("parts")
    elif message is not None:
        content = message
    if content is None:
        content = obj.get("content") or obj.get("text") or obj.get("parts") or obj.get("output")

    text = _content_to_text(content)
    if not text:
        return None
    if role == "user" and _is_context_injection(text):
        return None

    return {
        "type": role,
        "message": {"content": text},
        "timestamp": obj.get("timestamp") or obj.get("time") or obj.get("created_at") or obj.get("createdAt"),
        "cwd": obj.get("cwd") or obj.get("project_path") or obj.get("workspace") or obj.get("root"),
    }


def _iter_message_candidates(value: Any):
    if isinstance(value, list):
        for item in value:
            yield from _iter_message_candidates(item)
    elif isinstance(value, dict):
        record = _record_payload(value)
        record_type = _first_str(record.get("type"), record.get("event"), record.get("kind")).lower()
        if _infer_role(record) or "tool" in record_type or record_type in ("function_call", "function-call"):
            yield value
            return
        for key in ("messages", "history", "turns", "conversation", "entries", "items", "events"):
            if key in value:
                yield from _iter_message_candidates(value[key])


def _build_parsed_session(
    *,
    path: Path,
    spec: AgentSpec,
    raw_objects: list[dict],
    normalized_messages: list[dict],
) -> ParsedSession | None:
    session_id = None
    project_path = ""
    first_prompt = ""
    timestamps = []

    for obj in raw_objects + normalized_messages:
        record = _record_payload(obj) if isinstance(obj, dict) else obj
        if not session_id:
            session_id = _first_str(
                record.get("sessionId") if isinstance(record, dict) else None,
                record.get("session_id") if isinstance(record, dict) else None,
                record.get("sessionID") if isinstance(record, dict) else None,
                record.get("conversation_id") if isinstance(record, dict) else None,
                record.get("id") if isinstance(record, dict) else None,
            )
        if not project_path and isinstance(record, dict):
            project_path = _first_str(
                record.get("cwd"),
                record.get("project_path"),
                record.get("workspace"),
                record.get("root"),
                record.get("directory"),
            )
            metadata = record.get("metadata")
            if not project_path and isinstance(metadata, dict):
                project_path = _first_str(metadata.get("cwd"), metadata.get("project_path"), metadata.get("workspace"))
        ts = _parse_timestamp(record.get("timestamp") or record.get("time") or record.get("created_at") or record.get("createdAt")) if isinstance(record, dict) else None
        if ts:
            timestamps.append(ts)

    for msg in normalized_messages:
        if msg.get("type") == "user" and not first_prompt:
            first_prompt = _content_to_text(msg.get("message", {}).get("content"))[:200]
            break

    stat = path.stat()
    created = timestamps[0] if timestamps else datetime.fromtimestamp(stat.st_birthtime if hasattr(stat, 'st_birthtime') else stat.st_ctime)
    modified = timestamps[-1] if timestamps else datetime.fromtimestamp(stat.st_mtime)

    return ParsedSession(
        session_id=session_id or path.stem,
        project_path=project_path,
        created=created,
        modified=modified,
        messages=normalized_messages,
        first_prompt=first_prompt,
        agent=spec.name,
        agent_display=spec.display_name,
    )


def parse_jsonl(path: str | Path, spec: AgentSpec | None = None) -> ParsedSession | None:
    """Parse a session JSONL file into a structured session object."""
    spec = spec or get_agent_spec("claude")
    path = Path(path)
    messages = []
    session_id = None
    project_path = ""
    first_prompt = ""
    timestamps = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not session_id and obj.get("sessionId"):
                session_id = obj["sessionId"]
            if not project_path and obj.get("cwd"):
                project_path = obj["cwd"]

            msg_type = obj.get("type")
            if msg_type in ("user", "assistant"):
                messages.append(obj)
                ts = obj.get("timestamp")
                if ts:
                    try:
                        timestamps.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
                    except (ValueError, TypeError):
                        pass

                # Capture first user prompt
                if msg_type == "user" and not first_prompt:
                    content = obj.get("message", {}).get("content", "")
                    if isinstance(content, str):
                        first_prompt = content[:200]
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                first_prompt = block.get("text", "")[:200]
                                break

    if not session_id:
        session_id = path.stem

    # Derive timestamps from file stat if not found in content
    stat = path.stat()
    created = timestamps[0] if timestamps else datetime.fromtimestamp(stat.st_birthtime if hasattr(stat, 'st_birthtime') else stat.st_ctime)
    modified = timestamps[-1] if timestamps else datetime.fromtimestamp(stat.st_mtime)

    return ParsedSession(
        session_id=session_id,
        project_path=project_path,
        created=created,
        modified=modified,
        messages=messages,
        first_prompt=first_prompt,
        agent=spec.name,
        agent_display=spec.display_name,
    )


def parse_generic_session(path: str | Path, spec: AgentSpec) -> ParsedSession | None:
    path = Path(path)
    raw_objects: list[dict] = []
    messages: list[dict] = []

    if path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    raw_objects.append(obj)
                    normalized = _normalize_generic_message(obj)
                    if normalized:
                        messages.append(normalized)
                    else:
                        for candidate in _iter_message_candidates(obj):
                            normalized = _normalize_generic_message(candidate)
                            if normalized:
                                messages.append(normalized)
    else:
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if isinstance(data, dict):
            raw_objects.append(data)
        for candidate in _iter_message_candidates(data):
            if isinstance(candidate, dict):
                raw_objects.append(candidate)
                normalized = _normalize_generic_message(candidate)
                if normalized:
                    messages.append(normalized)

    if not messages:
        return None

    return _build_parsed_session(
        path=path,
        spec=spec,
        raw_objects=raw_objects,
        normalized_messages=messages,
    )


def parse_agent_session(path: str | Path, spec: AgentSpec) -> ParsedSession | None:
    if spec.name == "claude":
        return parse_jsonl(path, spec)
    return parse_generic_session(path, spec)


# ---------------------------------------------------------------------------
# Transcript formatting (replicates Wb8)
# ---------------------------------------------------------------------------

def format_transcript(session: ParsedSession) -> str:
    """Format a parsed session into the text transcript sent to the LLM."""
    lines = [
        f"Agent: {session.agent_display}",
        f"Session: {session.session_id[:8]}",
        f"Date: {session.start_time}",
        f"Project: {session.project_path}",
        f"Duration: {session.duration_minutes} min",
        "",
    ]

    for msg in session.messages:
        if msg.get("type") == "user" and msg.get("message"):
            content = msg["message"].get("content", "")
            if isinstance(content, str):
                lines.append(f"[User]: {content[:500]}")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                        lines.append(f"[User]: {block['text'][:500]}")
        elif msg.get("type") == "assistant" and msg.get("message"):
            content = msg["message"].get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text" and "text" in block:
                            lines.append(f"[Assistant]: {block['text'][:300]}")
                        elif block.get("type") == "tool_use" and "name" in block:
                            lines.append(f"[Tool: {block['name']}]")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session-meta extraction (replicates npA + Jb8)
# ---------------------------------------------------------------------------

LANG_EXTENSIONS = {
    ".ts": "TypeScript", ".tsx": "TypeScript", ".js": "JavaScript",
    ".jsx": "JavaScript", ".py": "Python", ".rb": "Ruby", ".go": "Go",
    ".rs": "Rust", ".java": "Java", ".md": "Markdown", ".json": "JSON",
    ".yaml": "YAML", ".yml": "YAML", ".sh": "Shell", ".css": "CSS",
    ".html": "HTML",
}


def extract_session_meta(session: ParsedSession) -> dict:
    """Mechanical telemetry extraction from JSONL (no LLM needed)."""
    tool_counts: dict[str, int] = {}
    languages: dict[str, int] = {}
    git_commits = 0
    git_pushes = 0
    input_tokens = 0
    output_tokens = 0
    user_msg_count = 0
    asst_msg_count = 0
    user_interruptions = 0
    tool_errors = 0
    tool_error_categories: dict[str, int] = {}
    lines_added = 0
    lines_removed = 0
    files_modified: set[str] = set()
    message_hours: list[int] = []
    user_timestamps: list[str] = []

    for msg in session.messages:
        if msg.get("type") == "assistant":
            asst_msg_count += 1
            content = msg.get("message", {}).get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        name = block.get("name", "Unknown")
                        tool_counts[name] = tool_counts.get(name, 0) + 1

        elif msg.get("type") == "user":
            content = msg.get("message", {}).get("content", "")
            has_text = False
            if isinstance(content, str) and content.strip():
                has_text = True
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        has_text = True
                        break
            if has_text:
                user_msg_count += 1

    return {
        "session_id": session.session_id,
        "project_path": session.project_path,
        "start_time": session.start_time,
        "duration_minutes": session.duration_minutes,
        "user_message_count": user_msg_count,
        "assistant_message_count": asst_msg_count,
        "tool_counts": tool_counts,
        "languages": languages,
        "git_commits": git_commits,
        "git_pushes": git_pushes,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "first_prompt": session.first_prompt,
        "user_interruptions": user_interruptions,
        "user_response_times": [],
        "tool_errors": tool_errors,
        "tool_error_categories": tool_error_categories,
        "uses_task_agent": False,
        "uses_mcp": False,
        "uses_web_search": False,
        "uses_web_fetch": False,
        "lines_added": lines_added,
        "lines_removed": lines_removed,
        "files_modified": len(files_modified),
        "message_hours": message_hours,
        "user_message_timestamps": user_timestamps,
    }


# ---------------------------------------------------------------------------
# Facet validation (replicates I_0)
# ---------------------------------------------------------------------------

def validate_facet(facet: dict) -> bool:
    """Check that a facet has all required fields with correct types."""
    return (
        isinstance(facet.get("underlying_goal"), str)
        and isinstance(facet.get("outcome"), str)
        and isinstance(facet.get("brief_summary"), str)
        and isinstance(facet.get("goal_categories"), dict)
        and isinstance(facet.get("user_satisfaction_counts"), dict)
        and isinstance(facet.get("friction_counts"), dict)
    )


# ---------------------------------------------------------------------------
# Session filters (replicates pipeline logic)
# ---------------------------------------------------------------------------

def is_insights_session(session: ParsedSession) -> bool:
    """Filter out sessions that are themselves /insights runs."""
    for msg in session.messages[:5]:
        if msg.get("type") == "user" and msg.get("message"):
            content = msg["message"].get("content", "")
            if isinstance(content, str):
                if "RESPOND WITH ONLY A VALID JSON OBJECT" in content:
                    return True
                if "record_facets" in content:
                    return True
    return False


def meets_minimum_criteria(meta: dict) -> bool:
    """Session must have >=2 user messages AND >=1 minute duration."""
    return meta["user_message_count"] >= 2 and meta["duration_minutes"] >= 1


def is_warmup_only(facet: dict) -> bool:
    """Filter out sessions that are just cache warmups."""
    cats = facet.get("goal_categories", {})
    active = [k for k, v in cats.items() if (v or 0) > 0]
    return len(active) == 1 and active[0] == "warmup_minimal"


# ---------------------------------------------------------------------------
# LLM call via `claude -p` (headless mode)
# ---------------------------------------------------------------------------


def call_api(
    *,
    model: str,
    system: str,
    user_prompt: str,
    max_tokens: int,
    label: str = "",
) -> str | None:
    """Call the LLM via `claude -p` headless mode.

    Uses your authenticated Claude Code session (Max plan, API key, etc.)
    instead of requiring a separate ANTHROPIC_API_KEY.
    """
    console = stderr_console()
    start = time.time()
    console.progress(label, console.join(["calling claude -p", f"model {model}", f"max tokens {max_tokens}"]))

    # Write prompt to a temp file to avoid shell escaping issues with large prompts
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(user_prompt)
        prompt_file = f.name

    try:
        # Build command — claude -p reads prompt from stdin or argument
        cmd = [
            "claude", "-p",
            "--model", model,
            "--max-turns", "1",
        ]

        # Env: unset CLAUDECODE to allow nested invocation
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

        # Pass prompt via stdin from file
        with open(prompt_file) as pf:
            result = subprocess.run(
                cmd,
                stdin=pf,
                capture_output=True,
                text=True,
                env=env,
                timeout=300,
            )

        elapsed = time.time() - start

        if result.returncode != 0:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            detail = stderr or stdout  # claude -p sometimes writes errors to stdout
            console.error(
                console.join([
                    f"{label or 'claude'} failed in {format_elapsed(elapsed)}",
                    f"exit {result.returncode}: {detail[:300]}",
                ])
            )
            return None

        text = result.stdout.strip()
        console.success(
            console.join([
                f"{label or 'claude'} done in {format_elapsed(elapsed)}",
                f"{len(text)} chars response",
            ])
        )
        return text

    finally:
        os.unlink(prompt_file)


def extract_json(text: str) -> dict | None:
    """Extract the first JSON object from LLM response text."""
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Phase 1-3: Session discovery & session-meta
# ---------------------------------------------------------------------------

def _discover_claude_sessions(project_path: str | None, spec: AgentSpec) -> list[dict]:
    projects_dir = spec.config_dir / "projects"
    sessions = []
    if not projects_dir.exists():
        return sessions

    project_dirs = None
    if project_path:
        encoded = project_path.replace("/", "-")
        project_dirs = []
        for d in projects_dir.iterdir():
            if d.is_dir() and (d.name == encoded or d.name.startswith(encoded + "-")):
                project_dirs.append(d)
        if not project_dirs:
            console = stderr_console()
            console.warning(f"no project dir found matching {project_path}")
            console.detail(f"looking for {encoded}*")

    scan_dirs = project_dirs if project_dirs else [
        d for d in projects_dir.iterdir() if d.is_dir()
    ]

    for project_dir in scan_dirs:
        for jsonl_file in project_dir.glob("*.jsonl"):
            stat = jsonl_file.stat()
            sessions.append({
                "session_id": jsonl_file.stem,
                "path": str(jsonl_file),
                "mtime": stat.st_mtime,
                "size": stat.st_size,
            })
    return sessions


def _discover_generic_sessions(project_path: str | None, spec: AgentSpec) -> list[dict]:
    roots = {
        "codex": [spec.config_dir / "sessions"],
        "cursor": [spec.config_dir / "projects"],
        "gemini": [spec.config_dir / "tmp"],
    }[spec.name]
    suffixes = {".jsonl"} if spec.name == "cursor" else {".jsonl", ".json"}
    sessions = []

    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            if spec.name == "cursor" and "/mcps/" in path.as_posix():
                continue
            if project_path and project_path not in str(path):
                parsed = parse_agent_session(path, spec)
                if not parsed or parsed.project_path != project_path:
                    continue
            stat = path.stat()
            sessions.append({
                "session_id": path.stem,
                "path": str(path),
                "mtime": stat.st_mtime,
                "size": stat.st_size,
            })
    return sessions


def discover_sessions(project_path: str | None = None, spec: AgentSpec | None = None) -> list[dict]:
    """Find session JSONL files, optionally scoped to a single project.

    If project_path is given (e.g. "/Users/you/myproject"),
    only returns sessions from matching project dirs (including worktrees).
    """
    spec = spec or get_agent_spec("claude")
    if spec.name == "claude":
        sessions = _discover_claude_sessions(project_path, spec)
    else:
        sessions = _discover_generic_sessions(project_path, spec)

    sessions.sort(key=lambda s: s["mtime"], reverse=True)
    return sessions


def load_cached_session_meta(session_id: str, spec: AgentSpec | None = None) -> dict | None:
    """Load existing session-meta from the agent cache when available."""
    spec = spec or get_agent_spec("claude")
    for d in (spec.out_meta_dir, spec.source_meta_dir):
        if not d:
            continue
        path = d / f"{session_id}.json"
        if path.exists():
            return json.loads(path.read_text())
    return None


def load_cached_facet(session_id: str, spec: AgentSpec | None = None) -> dict | None:
    """Load existing facet — checks local output dir first, then agent cache."""
    spec = spec or get_agent_spec("claude")
    for d in (spec.out_facets_dir, spec.source_facets_dir):
        if not d:
            continue
        path = d / f"{session_id}.json"
        if path.exists():
            facet = json.loads(path.read_text())
            if validate_facet(facet):
                return facet
    return None


def save_facet(facet: dict, spec: AgentSpec | None = None) -> None:
    """Write facet JSON to local output dir."""
    spec = spec or get_agent_spec("claude")
    spec.out_facets_dir.mkdir(parents=True, exist_ok=True)
    path = spec.out_facets_dir / f"{facet['session_id']}.json"
    path.write_text(json.dumps(facet, indent=2))


# ---------------------------------------------------------------------------
# Phase 4: Facet generation (LLM call #1 per session)
# ---------------------------------------------------------------------------

def summarize_chunk(chunk: str, spec: AgentSpec | None = None) -> str:
    """Summarize a long transcript chunk via LLM (for sessions > 30k chars).

    API call: Opus, max_tokens=500, prompt=CHUNK_SUMMARIZE_PROMPT + chunk
    """
    spec = spec or get_agent_spec("claude")
    text = call_api(
        model=FACET_MODEL,
        system="",
        user_prompt=agent_text(CHUNK_SUMMARIZE_PROMPT, spec) + chunk,
        max_tokens=500,
        label="chunk-summarize",
    )
    return text or chunk[:2000]


def prepare_transcript(session: ParsedSession, spec: AgentSpec | None = None) -> str:
    """Format transcript, summarizing chunks if > 30k chars (replicates hb8)."""
    spec = spec or get_agent_spec(session.agent)
    transcript = format_transcript(session)

    if len(transcript) <= MAX_TRANSCRIPT_LEN:
        return transcript

    # Split into chunks and summarize each
    chunks = []
    for i in range(0, len(transcript), TRANSCRIPT_CHUNK_SIZE):
        chunks.append(transcript[i : i + TRANSCRIPT_CHUNK_SIZE])

    console = stderr_console()
    console.detail(console.join([
        "long session",
        f"{len(transcript)} chars",
        f"summarizing {len(chunks)} chunks",
    ]))
    summaries = [summarize_chunk(c, spec) for c in chunks]

    meta = extract_session_meta(session)
    header = "\n".join([
        f"Agent: {session.agent_display}",
        f"Session: {meta['session_id'][:8]}",
        f"Date: {meta['start_time']}",
        f"Project: {meta['project_path']}",
        f"Duration: {meta['duration_minutes']} min",
        f"[Long session - {len(chunks)} parts summarized]",
        "",
    ])
    return header + "\n".join(summaries)


def generate_facet(session: ParsedSession, spec: AgentSpec | None = None, dry_run: bool = False) -> dict | None:
    """Generate a facet for a single session.

    API call: Opus, max_tokens=4096
    System: empty
    User prompt: FACET_PROMPT_PREFIX + transcript + FACET_SCHEMA_SUFFIX
    """
    spec = spec or get_agent_spec(session.agent)
    transcript = format_transcript(session) if dry_run else prepare_transcript(session, spec)
    user_prompt = facet_prompt_prefix(spec) + transcript + agent_schema_suffix(spec)

    if dry_run:
        console = stderr_console()
        console.skip(console.join(["dry run", f"would generate facet for {session.session_id[:8]}"]))
        console.detail(f"transcript length {len(transcript)} chars")
        console.detail(f"prompt length {len(user_prompt)} chars")
        return None

    text = call_api(
        model=FACET_MODEL,
        system="",
        user_prompt=user_prompt,
        max_tokens=4096,
        label=f"facet:{session.session_id[:8]}",
    )
    if not text:
        return None

    facet = extract_json(text)
    if not facet or not validate_facet(facet):
        stderr_console().warning(f"invalid facet for {session.session_id[:8]}")
        return None

    facet["session_id"] = session.session_id
    facet["agent"] = spec.name
    return facet


# ---------------------------------------------------------------------------
# Project-level: Correction extraction
# ---------------------------------------------------------------------------

CORRECTION_PROMPT = """Analyze this Claude Code session transcript and extract every moment where the user corrected, redirected, or overrode Claude's behavior.

For each correction, extract:
- **what_claude_did**: What Claude did or was about to do (1 sentence)
- **what_user_wanted**: What the user actually wanted instead (1 sentence)
- **category**: One of: wrong_approach, wrong_file, over_engineered, under_engineered, wrong_tool, misunderstood_intent, style_violation, missing_context, entered_plan_mode, ignored_instruction
- **verbatim_quote**: The user's exact words (or closest approximation from transcript), max 100 chars
- **claude_md_rule**: A concrete, actionable rule for CLAUDE.md that would prevent this from happening again. Write it as an imperative instruction to Claude. Be specific to this project, not generic. If no rule makes sense (one-off mistake), use null.
- **severity**: "blocking" (user couldn't proceed), "annoying" (user had to repeat themselves), "minor" (small correction)

Also extract:
- **repeated_instructions**: Things the user told Claude that they've likely said before in other sessions — standing preferences, project conventions, workflow rules. These are PRIME candidates for CLAUDE.md rules.

IMPORTANT:
- Only include genuine corrections, not normal back-and-forth collaboration
- If the user says "no", "stop", "that's not what I meant", "just do X", "don't do Y" — that's a correction
- If the user interrupts Claude mid-execution — figure out why and include it
- If the session has no corrections, return empty arrays

SESSION:
"""

CORRECTION_SCHEMA = """
RESPOND WITH ONLY A VALID JSON OBJECT:
{
  "corrections": [
    {
      "what_claude_did": "...",
      "what_user_wanted": "...",
      "category": "wrong_approach|wrong_file|over_engineered|under_engineered|wrong_tool|misunderstood_intent|style_violation|missing_context|entered_plan_mode|ignored_instruction",
      "verbatim_quote": "user's exact words",
      "claude_md_rule": "rule text or null",
      "severity": "blocking|annoying|minor"
    }
  ],
  "repeated_instructions": [
    {
      "instruction": "What the user told Claude",
      "verbatim_quote": "user's exact words",
      "claude_md_rule": "Suggested CLAUDE.md rule"
    }
  ],
  "session_id": "will be filled in"
}
"""


def extract_corrections(session: ParsedSession, dry_run: bool = False) -> dict | None:
    """Extract user corrections from a single session."""
    transcript = prepare_transcript(session)
    user_prompt = CORRECTION_PROMPT + transcript + CORRECTION_SCHEMA

    if dry_run:
        console = stderr_console()
        console.skip(console.join(["dry run", f"would extract corrections for {session.session_id[:8]}"]))
        return None

    text = call_api(
        model=FACET_MODEL,
        system="",
        user_prompt=user_prompt,
        max_tokens=4096,
        label=f"corrections:{session.session_id[:8]}",
    )
    if not text:
        return None

    result = extract_json(text)
    if not result:
        return None

    result["session_id"] = session.session_id
    return result


RULES_SYNTHESIS_PROMPT = """You are analyzing corrections extracted from multiple Claude Code sessions for a specific project. Your job is to synthesize these into concrete CLAUDE.md rules.

You will receive:
1. The current CLAUDE.md file for this project
2. A list of corrections from multiple sessions, each with a suggested rule

Your task:
- Group corrections that point to the same underlying issue
- Prioritize rules that appear across MULTIPLE sessions (the user is repeating themselves)
- Merge similar rules into a single, well-worded instruction
- Drop one-off corrections that don't generalize
- Format rules as imperative instructions to Claude ("Always...", "Never...", "When X, do Y...")
- For each rule, note which sessions it came from and how many times

Output rules organized by where they'd go in CLAUDE.md (existing section or new section).

CURRENT CLAUDE.md:
"""


def synthesize_rules(
    corrections: list[dict],
    claude_md_content: str,
    dry_run: bool = False,
) -> dict | None:
    """Synthesize corrections into CLAUDE.md rule suggestions."""
    # Flatten corrections into a compact format
    all_corrections = []
    all_instructions = []
    for c in corrections:
        sid = c.get("session_id", "?")[:8]
        for corr in c.get("corrections", []):
            all_corrections.append({
                "session": sid,
                "category": corr.get("category"),
                "what_happened": corr.get("what_claude_did"),
                "what_wanted": corr.get("what_user_wanted"),
                "quote": corr.get("verbatim_quote"),
                "suggested_rule": corr.get("claude_md_rule"),
                "severity": corr.get("severity"),
            })
        for inst in c.get("repeated_instructions", []):
            all_instructions.append({
                "session": sid,
                "instruction": inst.get("instruction"),
                "quote": inst.get("verbatim_quote"),
                "suggested_rule": inst.get("claude_md_rule"),
            })

    if not all_corrections and not all_instructions:
        stderr_console().skip("no corrections found across sessions")
        return None

    corrections_text = json.dumps(all_corrections, indent=2)
    instructions_text = json.dumps(all_instructions, indent=2)

    user_prompt = (
        RULES_SYNTHESIS_PROMPT
        + claude_md_content
        + "\n\nCORRECTIONS FROM SESSIONS:\n"
        + corrections_text
        + "\n\nREPEATED INSTRUCTIONS:\n"
        + instructions_text
        + """

RESPOND WITH ONLY A VALID JSON OBJECT:
{
  "rules": [
    {
      "rule": "The imperative instruction for CLAUDE.md",
      "section": "Which CLAUDE.md section this belongs in (existing or new)",
      "sessions": ["abc123", "def456"],
      "frequency": 3,
      "severity": "blocking|annoying|minor",
      "evidence": "Brief summary of what happened"
    }
  ],
  "already_covered": [
    {
      "rule": "Rule that already exists in CLAUDE.md",
      "but_violated_in": ["session_ids"],
      "suggestion": "How to strengthen the existing rule, or null if it's fine"
    }
  ]
}
"""
    )

    if dry_run:
        console = stderr_console()
        console.skip(console.join([
            "dry run",
            f"would synthesize {len(all_corrections)} corrections + {len(all_instructions)} instructions",
        ]))
        return None

    text = call_api(
        model=REPORT_MODEL,
        system="",
        user_prompt=user_prompt,
        max_tokens=8192,
        label="synthesize-rules",
    )
    if not text:
        return None
    return extract_json(text)


# ---------------------------------------------------------------------------
# Phase 5: Stats aggregation (mechanical — no LLM)
# ---------------------------------------------------------------------------

def aggregate_stats(
    session_metas: list[dict],
    facets: dict[str, dict],
) -> dict:
    """Aggregate stats across all sessions (replicates Yb8)."""
    stats: dict[str, Any] = {
        "total_sessions": len(session_metas),
        "sessions_with_facets": len(facets),
        "date_range": {"start": "", "end": ""},
        "total_messages": 0,
        "total_duration_hours": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "tool_counts": {},
        "languages": {},
        "git_commits": 0,
        "git_pushes": 0,
        "projects": {},
        "goal_categories": {},
        "outcomes": {},
        "satisfaction": {},
        "helpfulness": {},
        "session_types": {},
        "friction": {},
        "success": {},
        "session_summaries": [],
        "total_interruptions": 0,
        "total_tool_errors": 0,
        "tool_error_categories": {},
        "sessions_using_task_agent": 0,
        "sessions_using_mcp": 0,
        "sessions_using_web_search": 0,
        "sessions_using_web_fetch": 0,
        "total_lines_added": 0,
        "total_lines_removed": 0,
        "total_files_modified": 0,
        "days_active": 0,
        "messages_per_day": 0,
    }

    start_times = []

    for meta in session_metas:
        start_times.append(meta["start_time"])
        stats["total_messages"] += meta["user_message_count"]
        stats["total_duration_hours"] += meta["duration_minutes"] / 60
        stats["total_input_tokens"] += meta.get("input_tokens", 0)
        stats["total_output_tokens"] += meta.get("output_tokens", 0)
        stats["git_commits"] += meta.get("git_commits", 0)
        stats["git_pushes"] += meta.get("git_pushes", 0)
        stats["total_interruptions"] += meta.get("user_interruptions", 0)
        stats["total_tool_errors"] += meta.get("tool_errors", 0)
        stats["total_lines_added"] += meta.get("lines_added", 0)
        stats["total_lines_removed"] += meta.get("lines_removed", 0)
        stats["total_files_modified"] += meta.get("files_modified", 0)

        for tool, count in meta.get("tool_counts", {}).items():
            stats["tool_counts"][tool] = stats["tool_counts"].get(tool, 0) + count
        for lang, count in meta.get("languages", {}).items():
            stats["languages"][lang] = stats["languages"].get(lang, 0) + count
        if meta.get("project_path"):
            stats["projects"][meta["project_path"]] = stats["projects"].get(meta["project_path"], 0) + 1

        if meta.get("uses_task_agent"): stats["sessions_using_task_agent"] += 1
        if meta.get("uses_mcp"): stats["sessions_using_mcp"] += 1
        if meta.get("uses_web_search"): stats["sessions_using_web_search"] += 1
        if meta.get("uses_web_fetch"): stats["sessions_using_web_fetch"] += 1

        facet = facets.get(meta["session_id"])
        if facet:
            for cat, count in (facet.get("goal_categories") or {}).items():
                if count and count > 0:
                    stats["goal_categories"][cat] = stats["goal_categories"].get(cat, 0) + count
            stats["outcomes"][facet["outcome"]] = stats["outcomes"].get(facet["outcome"], 0) + 1
            for level, count in (facet.get("user_satisfaction_counts") or {}).items():
                if count and count > 0:
                    stats["satisfaction"][level] = stats["satisfaction"].get(level, 0) + count
            h = facet.get("claude_helpfulness", "")
            if h:
                stats["helpfulness"][h] = stats["helpfulness"].get(h, 0) + 1
            st = facet.get("session_type", "")
            if st:
                stats["session_types"][st] = stats["session_types"].get(st, 0) + 1
            for ft, count in (facet.get("friction_counts") or {}).items():
                if count and count > 0:
                    stats["friction"][ft] = stats["friction"].get(ft, 0) + count
            ps = facet.get("primary_success", "none")
            if ps and ps != "none":
                stats["success"][ps] = stats["success"].get(ps, 0) + 1

        if len(stats["session_summaries"]) < 50:
            stats["session_summaries"].append({
                "id": meta["session_id"][:8],
                "date": meta["start_time"][:10],
                "summary": meta.get("summary") or meta.get("first_prompt", "")[:100],
                "goal": facet["underlying_goal"] if facet else None,
            })

    start_times.sort()
    if start_times:
        stats["date_range"]["start"] = start_times[0][:10]
        stats["date_range"]["end"] = start_times[-1][:10]

    days = set(t[:10] for t in start_times)
    stats["days_active"] = len(days)
    stats["messages_per_day"] = (
        round(stats["total_messages"] / len(days) * 10) / 10 if days else 0
    )

    return stats


# ---------------------------------------------------------------------------
# Phase 6: Report generation (7+1 parallel LLM calls)
# ---------------------------------------------------------------------------

def build_report_context(stats: dict, facets: dict[str, dict], spec: AgentSpec | None = None) -> str:
    """Build the context string sent to each report prompt (replicates fb8)."""
    spec = spec or get_agent_spec("claude")
    summaries = "\n".join(
        f"- {f['brief_summary']} ({f['outcome']}, {f.get('claude_helpfulness', 'unknown')})"
        for f in list(facets.values())[:50]
    )
    friction_details = "\n".join(
        f"- {f['friction_detail']}"
        for f in list(facets.values())
        if f.get("friction_detail")
    )[:20]

    compact_stats = json.dumps({
        "sessions": stats["total_sessions"],
        "analyzed": stats["sessions_with_facets"],
        "date_range": stats["date_range"],
        "messages": stats["total_messages"],
        "hours": round(stats["total_duration_hours"]),
        "commits": stats["git_commits"],
        "top_tools": sorted(stats["tool_counts"].items(), key=lambda x: -x[1])[:8],
        "top_goals": sorted(stats["goal_categories"].items(), key=lambda x: -x[1])[:8],
        "outcomes": stats["outcomes"],
        "satisfaction": stats["satisfaction"],
        "friction": stats["friction"],
        "success": stats["success"],
        "languages": stats["languages"],
    }, indent=2)

    return (
        f"AGENT: {spec.display_name}\n"
        + compact_stats
        + "\nSESSION SUMMARIES:\n" + summaries
        + "\nFRICTION DETAILS:\n" + friction_details
        + f"\nUSER INSTRUCTIONS TO {spec.display_name.upper()}:\nNone captured"
    )


def generate_report_section(
    prompt_def: dict,
    context: str,
    dry_run: bool = False,
) -> tuple[str, dict | None]:
    """Generate one report section.

    API call: Opus, max_tokens=8192
    System: empty
    User prompt: section_prompt + "\\nDATA:\\n" + context
    """
    name = prompt_def["name"]
    user_prompt = prompt_def["prompt"] + "\nDATA:\n" + context

    if dry_run:
        console = stderr_console()
        console.skip(console.join(["dry run", f"would generate report section {name}"]))
        console.detail(f"context length {len(context)} chars")
        return name, None

    text = call_api(
        model=REPORT_MODEL,
        system="",
        user_prompt=user_prompt,
        max_tokens=prompt_def["max_tokens"],
        label=f"report:{name}",
    )
    if not text:
        return name, None

    result = extract_json(text)
    return name, result


def generate_at_a_glance(
    context: str,
    sections: dict[str, dict],
    spec: AgentSpec | None = None,
    dry_run: bool = False,
) -> dict | None:
    """Generate the at_a_glance summary using results from other sections.

    API call #8: Opus, max_tokens=8192
    This runs AFTER the 7 parallel calls, using their results as additional context.
    """
    areas = "\n".join(
        f"- {a['name']}: {a['description']}"
        for a in (sections.get("project_areas", {}).get("areas") or [])
    )
    wins = "\n".join(
        f"- {w['title']}: {w['description']}"
        for w in (sections.get("what_works", {}).get("impressive_workflows") or [])
    )
    friction = "\n".join(
        f"- {c['category']}: {c['description']}"
        for c in (sections.get("friction_analysis", {}).get("categories") or [])
    )
    features = "\n".join(
        f"- {f['feature']}: {f['one_liner']}"
        for f in (sections.get("suggestions", {}).get("features_to_try") or [])
    )
    patterns = "\n".join(
        f"- {p['title']}: {p['suggestion']}"
        for p in (sections.get("suggestions", {}).get("usage_patterns") or [])
    )
    horizon = "\n".join(
        f"- {o['title']}: {o['whats_possible']}"
        for o in (sections.get("on_the_horizon", {}).get("opportunities") or [])
    )

    spec = spec or get_agent_spec("claude")
    prompt = f"""You're writing an "At a Glance" summary for a {spec.display_name} usage insights report for {spec.display_name} users. The goal is to help them understand their usage and improve how they can use {spec.display_name} better, especially as models improve.
Use this 4-part structure:
1. **What's working** - What is the user's unique style of interacting with {spec.display_name} and what are some impactful things they've done? You can include one or two details, but keep it high level since things might not be fresh in the user's memory. Don't be fluffy or overly complimentary. Also, don't focus on the tool calls they use.
2. **What's hindering you** - Split into (a) {spec.display_name}'s fault (misunderstandings, wrong approaches, bugs) and (b) user-side friction (not providing enough context, environment issues -- ideally more general than just one project). Be honest but constructive.
3. **Quick wins to try** - Specific {spec.display_name} features they could try from the examples below, or a workflow technique if you think it's really compelling. (Avoid stuff like "Ask the agent to confirm before taking actions" or "Type out more context up front" which are less compelling.)
4. **Ambitious workflows for better models** - As we move to much more capable models over the next 3-6 months, what should they prepare for? What workflows that seem impossible now will become possible? Draw from the appropriate section below.
Keep each section to 2-3 not-too-long sentences. Don't overwhelm the user. Don't mention specific numerical stats or underlined_categories from the session data below. Use a coaching tone.
RESPOND WITH ONLY A VALID JSON OBJECT:
  "whats_working": "(refer to instructions above)",
  "whats_hindering": "(refer to instructions above)",
  "quick_wins": "(refer to instructions above)",
  "ambitious_workflows": "(refer to instructions above)"
SESSION DATA:
{context}
## Project Areas (what user works on)
{areas}
## Big Wins (impressive accomplishments)
{wins}
## Friction Categories (where things go wrong)
{friction}
## Features to Try
{features}
## Usage Patterns to Adopt
{patterns}
## On the Horizon (ambitious workflows for better models)
{horizon}"""

    if dry_run:
        console = stderr_console()
        console.skip(console.join(["dry run", "would generate at_a_glance"]))
        return None

    text = call_api(
        model=REPORT_MODEL,
        system="",
        user_prompt=prompt,
        max_tokens=8192,
        label="report:at_a_glance",
    )
    if not text:
        return None
    return extract_json(text)


# ---------------------------------------------------------------------------
# Phase 7: HTML report generation (replicates yb8)
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    """HTML-escape text."""
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _bar_rows(data: dict | list, color: str, display_names: dict | None = None) -> str:
    """Generate horizontal bar chart rows from {label: count} or [(label, count)]."""
    if isinstance(data, dict):
        items = sorted(data.items(), key=lambda x: -x[1])
    else:
        items = list(data)
    if not items:
        return '<div class="empty">No data</div>'
    max_val = max(v for _, v in items) or 1
    rows = []
    for label, val in items[:6]:
        name = (display_names or DISPLAY_NAMES).get(label, label.replace("_", " ").title())
        pct = (val / max_val) * 100
        rows.append(
            f'<div class="bar-row">'
            f'<div class="bar-label">{_esc(name)}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div></div>'
            f'<div class="bar-value">{val}</div>'
            f'</div>'
        )
    return "\n".join(rows)


def build_report_html(stats: dict, sections: dict, spec: AgentSpec | None = None) -> str:
    """Build report.html from stats + section JSON (replicates yb8 template)."""
    spec = spec or get_agent_spec("claude")
    s = sections
    total_msgs = stats.get("total_messages", 0)
    total_sessions = stats.get("total_sessions", 0)
    analyzed = stats.get("sessions_with_facets", 0)
    scanned = stats.get("total_sessions_scanned", total_sessions)
    date_start = stats.get("date_range", {}).get("start", "?")
    date_end = stats.get("date_range", {}).get("end", "?")
    hours = round(stats.get("total_duration_hours", 0))
    commits = stats.get("git_commits", 0)

    subtitle = f"{total_msgs:,} messages across {analyzed} sessions ({scanned} total) | {date_start} to {date_end}"

    # At a glance
    aag = s.get("at_a_glance", {})
    at_a_glance_html = ""
    if aag:
        at_a_glance_html = f"""
    <div class="at-a-glance">
      <div class="glance-title">At a Glance</div>
      <div class="glance-sections">
        <div class="glance-section"><strong>What's working:</strong> {_esc(aag.get('whats_working', ''))} <a href="#section-wins" class="see-more">Impressive Things You Did &rarr;</a></div>
        <div class="glance-section"><strong>What's hindering you:</strong> {_esc(aag.get('whats_hindering', ''))} <a href="#section-friction" class="see-more">Where Things Go Wrong &rarr;</a></div>
        <div class="glance-section"><strong>Quick wins to try:</strong> {_esc(aag.get('quick_wins', ''))} <a href="#section-features" class="see-more">Features to Try &rarr;</a></div>
        <div class="glance-section"><strong>Ambitious workflows:</strong> {_esc(aag.get('ambitious_workflows', ''))} <a href="#section-horizon" class="see-more">On the Horizon &rarr;</a></div>
      </div>
    </div>"""

    # Stats row
    stats_row = f"""
    <div class="stats-row">
      <div class="stat"><div class="stat-value">{total_msgs:,}</div><div class="stat-label">Messages</div></div>
      <div class="stat"><div class="stat-value">{analyzed}</div><div class="stat-label">Sessions</div></div>
      <div class="stat"><div class="stat-value">{hours}h</div><div class="stat-label">Duration</div></div>
      <div class="stat"><div class="stat-value">{commits}</div><div class="stat-label">Commits</div></div>
      <div class="stat"><div class="stat-value">{stats.get('days_active', 0)}</div><div class="stat-label">Days</div></div>
    </div>"""

    # Project areas
    areas_html = ""
    pa = s.get("project_areas", {})
    if pa.get("areas"):
        cards = []
        for a in pa["areas"]:
            cards.append(f"""
        <div class="project-area">
          <div class="area-header">
            <span class="area-name">{_esc(a.get('name', ''))}</span>
            <span class="area-count">~{a.get('session_count', '?')} sessions</span>
          </div>
          <div class="area-desc">{_esc(a.get('description', ''))}</div>
        </div>""")
        areas_html = f"""
    <h2 id="section-work">What You Work On</h2>
    <div class="project-areas">{''.join(cards)}
    </div>"""

    # Charts: goals + tools
    goals_chart = _bar_rows(stats.get("goal_categories", {}), "#2563eb")
    tools_chart = _bar_rows(stats.get("tool_counts", {}), "#0891b2", display_names={})
    langs_chart = _bar_rows(stats.get("languages", {}), "#10b981", display_names={})
    session_types_chart = _bar_rows(stats.get("session_types", {}), "#8b5cf6")

    charts_1 = f"""
    <div class="charts-row">
      <div class="chart-card">
        <div class="chart-title">What You Wanted</div>
        {goals_chart}
      </div>
      <div class="chart-card">
        <div class="chart-title">Top Tools Used</div>
        {tools_chart}
      </div>
    </div>
    <div class="charts-row">
      <div class="chart-card">
        <div class="chart-title">Languages</div>
        {langs_chart}
      </div>
      <div class="chart-card">
        <div class="chart-title">Session Types</div>
        {session_types_chart}
      </div>
    </div>"""

    # Interaction style
    style_html = ""
    ist = s.get("interaction_style", {})
    if ist:
        narrative = ist.get("narrative", "")
        paragraphs = "".join(f"<p>{_esc(p.strip())}</p>" for p in narrative.split("\n\n") if p.strip())
        key = ist.get("key_pattern", "")
        style_html = f"""
    <h2 id="section-usage">How You Use {_esc(spec.display_name)}</h2>
    <div class="narrative">
      {paragraphs}
      <div class="key-insight"><strong>Key pattern:</strong> {_esc(key)}</div>
    </div>"""

    # Big wins
    wins_html = ""
    ww = s.get("what_works", {})
    if ww.get("impressive_workflows"):
        cards = []
        for w in ww["impressive_workflows"]:
            cards.append(f"""
        <div class="big-win">
          <div class="big-win-title">{_esc(w.get('title', ''))}</div>
          <div class="big-win-desc">{_esc(w.get('description', ''))}</div>
        </div>""")
        intro = _esc(ww.get("intro", ""))
        wins_html = f"""
    <h2 id="section-wins">Impressive Things You Did</h2>
    <p class="section-intro">{intro}</p>
    <div class="big-wins">{''.join(cards)}
    </div>"""

    # Success + outcomes charts
    success_chart = _bar_rows(stats.get("success", {}), "#16a34a")
    outcomes_chart = _bar_rows(stats.get("outcomes", {}), "#8b5cf6")
    charts_2 = f"""
    <div class="charts-row">
      <div class="chart-card">
        <div class="chart-title">What Helped Most</div>
        {success_chart}
      </div>
      <div class="chart-card">
        <div class="chart-title">Outcomes</div>
        {outcomes_chart}
      </div>
    </div>"""

    # Friction
    friction_html = ""
    fa = s.get("friction_analysis", {})
    if fa.get("categories"):
        cards = []
        for c in fa["categories"]:
            examples = "".join(f"<li>{_esc(e)}</li>" for e in (c.get("examples") or []))
            cards.append(f"""
        <div class="friction-category">
          <div class="friction-title">{_esc(c.get('category', ''))}</div>
          <div class="friction-desc">{_esc(c.get('description', ''))}</div>
          <ul class="friction-examples">{examples}</ul>
        </div>""")
        intro = _esc(fa.get("intro", ""))
        friction_html = f"""
    <h2 id="section-friction">Where Things Go Wrong</h2>
    <p class="section-intro">{intro}</p>
    <div class="friction-categories">{''.join(cards)}
    </div>"""

    # Friction + satisfaction charts
    friction_chart = _bar_rows(stats.get("friction", {}), "#dc2626")
    satisfaction_chart = _bar_rows(stats.get("satisfaction", {}), "#eab308")
    charts_3 = f"""
    <div class="charts-row">
      <div class="chart-card">
        <div class="chart-title">Primary Friction Types</div>
        {friction_chart}
      </div>
      <div class="chart-card">
        <div class="chart-title">Inferred Satisfaction</div>
        {satisfaction_chart}
      </div>
    </div>"""

    # Suggestions: agent instruction additions
    suggestions_html = ""
    sg = s.get("suggestions", {})
    instruction_items = sg.get("claude_md_additions") or sg.get(f"{spec.name}_instruction_additions")
    if instruction_items:
        items = []
        for i, cmd in enumerate(instruction_items):
            addition = _esc(cmd.get("addition", ""))
            why = _esc(cmd.get("why", ""))
            items.append(f"""
        <div class="claude-md-item">
          <input type="checkbox" id="cmd-{i}" class="cmd-checkbox" checked data-text="{_esc(cmd.get('prompt_scaffold', ''))}&#10;&#10;{addition}">
          <label for="cmd-{i}">
            <code class="cmd-code">{addition}</code>
            <button class="copy-btn" onclick="copyCmdItem({i})">Copy</button>
          </label>
          <div class="cmd-why">{why}</div>
        </div>""")
        suggestions_html = f"""
    <h2 id="section-features">Existing {_esc(spec.display_name)} Features to Try</h2>
    <div class="claude-md-section">
      <h3>Suggested {_esc(spec.instruction_file)} Additions</h3>
      <p style="font-size: 12px; color: #64748b; margin-bottom: 12px;">Copy into {_esc(spec.display_name)} to add to your {_esc(spec.instruction_file)}.</p>
      <div class="claude-md-actions">
        <button class="copy-all-btn" onclick="copyAllCheckedClaudeMd()">Copy All Checked</button>
      </div>
      {''.join(items)}
    </div>"""

    # Features to try
    features_html = ""
    if sg.get("features_to_try"):
        cards = []
        for f in sg["features_to_try"]:
            code = _esc(f.get("example_code", ""))
            cards.append(f"""
        <div class="feature-card">
          <div class="feature-title">{_esc(f.get('feature', ''))}</div>
          <div class="feature-oneliner">{_esc(f.get('one_liner', ''))}</div>
          <div class="feature-why"><strong>Why for you:</strong> {_esc(f.get('why_for_you', ''))}</div>
          <div class="feature-examples"><div class="feature-example"><div class="example-code-row">
            <code class="example-code">{code}</code>
            <button class="copy-btn" onclick="copyText(this)">Copy</button>
          </div></div></div>
        </div>""")
        features_html = f"""
    <p style="font-size: 13px; color: #64748b; margin-bottom: 12px;">Copy into {_esc(spec.display_name)} and it'll set it up for you.</p>
    <div class="features-section">{''.join(cards)}
    </div>"""

    # Usage patterns
    patterns_html = ""
    if sg.get("usage_patterns"):
        cards = []
        for p in sg["usage_patterns"]:
            prompt = _esc(p.get("copyable_prompt", ""))
            cards.append(f"""
        <div class="pattern-card">
          <div class="pattern-title">{_esc(p.get('title', ''))}</div>
          <div class="pattern-summary">{_esc(p.get('suggestion', ''))}</div>
          <div class="pattern-detail">{_esc(p.get('detail', ''))}</div>
          <div class="copyable-prompt-section">
            <div class="prompt-label">Paste into {_esc(spec.display_name)}:</div>
            <div class="copyable-prompt-row">
              <code class="copyable-prompt">{prompt}</code>
              <button class="copy-btn" onclick="copyText(this)">Copy</button>
            </div>
          </div>
        </div>""")
        patterns_html = f"""
    <h2 id="section-patterns">New Ways to Use {_esc(spec.display_name)}</h2>
    <p style="font-size: 13px; color: #64748b; margin-bottom: 12px;">Copy into {_esc(spec.display_name)} and it'll walk you through it.</p>
    <div class="patterns-section">{''.join(cards)}
    </div>"""

    # On the horizon
    horizon_html = ""
    oh = s.get("on_the_horizon", {})
    if oh.get("opportunities"):
        cards = []
        for o in oh["opportunities"]:
            prompt = _esc(o.get("copyable_prompt", ""))
            cards.append(f"""
        <div class="horizon-card">
          <div class="horizon-title">{_esc(o.get('title', ''))}</div>
          <div class="horizon-possible">{_esc(o.get('whats_possible', ''))}</div>
          <div class="horizon-tip"><strong>Getting started:</strong> {_esc(o.get('how_to_try', ''))}</div>
          <div class="pattern-prompt"><div class="prompt-label">Paste into {_esc(spec.display_name)}:</div><code>{prompt}</code><button class="copy-btn" onclick="copyText(this)">Copy</button></div>
        </div>""")
        intro = _esc(oh.get("intro", ""))
        horizon_html = f"""
    <h2 id="section-horizon">On the Horizon</h2>
    <p class="section-intro">{intro}</p>
    <div class="horizon-section">{''.join(cards)}
    </div>"""

    # Fun ending
    fun_html = ""
    fe = s.get("fun_ending", {})
    if fe:
        fun_html = f"""
    <div class="fun-ending">
      <div class="fun-headline">{_esc(fe.get('headline', ''))}</div>
      <div class="fun-detail">{_esc(fe.get('detail', ''))}</div>
    </div>"""

    # Assemble
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{_esc(spec.display_name)} Insights</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; background: #f8fafc; color: #334155; line-height: 1.65; padding: 48px 24px; }}
    .container {{ max-width: 800px; margin: 0 auto; }}
    h1 {{ font-size: 32px; font-weight: 700; color: #0f172a; margin-bottom: 8px; }}
    h2 {{ font-size: 20px; font-weight: 600; color: #0f172a; margin-top: 48px; margin-bottom: 16px; }}
    .subtitle {{ color: #64748b; font-size: 15px; margin-bottom: 32px; }}
    .nav-toc {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 24px 0 32px 0; padding: 16px; background: white; border-radius: 8px; border: 1px solid #e2e8f0; }}
    .nav-toc a {{ font-size: 12px; color: #64748b; text-decoration: none; padding: 6px 12px; border-radius: 6px; background: #f1f5f9; transition: all 0.15s; }}
    .nav-toc a:hover {{ background: #e2e8f0; color: #334155; }}
    .stats-row {{ display: flex; gap: 24px; margin-bottom: 40px; padding: 20px 0; border-top: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0; flex-wrap: wrap; }}
    .stat {{ text-align: center; }}
    .stat-value {{ font-size: 24px; font-weight: 700; color: #0f172a; }}
    .stat-label {{ font-size: 11px; color: #64748b; text-transform: uppercase; }}
    .at-a-glance {{ background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 1px solid #f59e0b; border-radius: 12px; padding: 20px 24px; margin-bottom: 32px; }}
    .glance-title {{ font-size: 16px; font-weight: 700; color: #92400e; margin-bottom: 16px; }}
    .glance-sections {{ display: flex; flex-direction: column; gap: 12px; }}
    .glance-section {{ font-size: 14px; color: #78350f; line-height: 1.6; }}
    .glance-section strong {{ color: #92400e; }}
    .see-more {{ color: #b45309; text-decoration: none; font-size: 13px; white-space: nowrap; }}
    .see-more:hover {{ text-decoration: underline; }}
    .project-areas {{ display: flex; flex-direction: column; gap: 12px; margin-bottom: 32px; }}
    .project-area {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; }}
    .area-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
    .area-name {{ font-weight: 600; font-size: 15px; color: #0f172a; }}
    .area-count {{ font-size: 12px; color: #64748b; background: #f1f5f9; padding: 2px 8px; border-radius: 4px; }}
    .area-desc {{ font-size: 14px; color: #475569; line-height: 1.5; }}
    .narrative {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; margin-bottom: 24px; }}
    .narrative p {{ margin-bottom: 12px; font-size: 14px; color: #475569; line-height: 1.7; }}
    .key-insight {{ background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 12px 16px; margin-top: 12px; font-size: 14px; color: #166534; }}
    .section-intro {{ font-size: 14px; color: #64748b; margin-bottom: 16px; }}
    .big-wins {{ display: flex; flex-direction: column; gap: 12px; margin-bottom: 24px; }}
    .big-win {{ background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 16px; }}
    .big-win-title {{ font-weight: 600; font-size: 15px; color: #166534; margin-bottom: 8px; }}
    .big-win-desc {{ font-size: 14px; color: #15803d; line-height: 1.5; }}
    .friction-categories {{ display: flex; flex-direction: column; gap: 16px; margin-bottom: 24px; }}
    .friction-category {{ background: #fef2f2; border: 1px solid #fca5a5; border-radius: 8px; padding: 16px; }}
    .friction-title {{ font-weight: 600; font-size: 15px; color: #991b1b; margin-bottom: 6px; }}
    .friction-desc {{ font-size: 13px; color: #7f1d1d; margin-bottom: 10px; }}
    .friction-examples {{ margin: 0 0 0 20px; font-size: 13px; color: #334155; }}
    .friction-examples li {{ margin-bottom: 4px; }}
    .claude-md-section {{ background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 16px; margin-bottom: 20px; }}
    .claude-md-section h3 {{ font-size: 14px; font-weight: 600; color: #1e40af; margin: 0 0 12px 0; }}
    .claude-md-actions {{ margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid #dbeafe; }}
    .copy-all-btn {{ background: #2563eb; color: white; border: none; border-radius: 4px; padding: 6px 12px; font-size: 12px; cursor: pointer; font-weight: 500; }}
    .copy-all-btn:hover {{ background: #1d4ed8; }}
    .copy-all-btn.copied {{ background: #16a34a; }}
    .claude-md-item {{ display: flex; flex-wrap: wrap; align-items: flex-start; gap: 8px; padding: 10px 0; border-bottom: 1px solid #dbeafe; }}
    .claude-md-item:last-child {{ border-bottom: none; }}
    .cmd-code {{ background: white; padding: 8px 12px; border-radius: 4px; font-size: 12px; color: #1e40af; border: 1px solid #bfdbfe; font-family: monospace; display: block; white-space: pre-wrap; word-break: break-word; flex: 1; }}
    .cmd-why {{ font-size: 12px; color: #64748b; width: 100%; padding-left: 24px; margin-top: 4px; }}
    .features-section, .patterns-section {{ display: flex; flex-direction: column; gap: 12px; margin: 16px 0; }}
    .feature-card {{ background: #f0fdf4; border: 1px solid #86efac; border-radius: 8px; padding: 16px; }}
    .pattern-card {{ background: #f0f9ff; border: 1px solid #7dd3fc; border-radius: 8px; padding: 16px; }}
    .feature-title, .pattern-title {{ font-weight: 600; font-size: 15px; color: #0f172a; margin-bottom: 6px; }}
    .feature-oneliner {{ font-size: 14px; color: #475569; margin-bottom: 8px; }}
    .pattern-summary {{ font-size: 14px; color: #475569; margin-bottom: 8px; }}
    .feature-why, .pattern-detail {{ font-size: 13px; color: #334155; line-height: 1.5; }}
    .example-code-row {{ display: flex; align-items: flex-start; gap: 8px; }}
    .example-code {{ flex: 1; background: #f1f5f9; padding: 8px 12px; border-radius: 4px; font-family: monospace; font-size: 12px; color: #334155; overflow-x: auto; white-space: pre-wrap; }}
    .copyable-prompt-section {{ margin-top: 12px; padding-top: 12px; border-top: 1px solid #e2e8f0; }}
    .copyable-prompt-row {{ display: flex; align-items: flex-start; gap: 8px; }}
    .copyable-prompt {{ flex: 1; background: #f8fafc; padding: 10px 12px; border-radius: 4px; font-family: monospace; font-size: 12px; color: #334155; border: 1px solid #e2e8f0; white-space: pre-wrap; line-height: 1.5; }}
    .copy-btn {{ background: #e2e8f0; border: none; border-radius: 4px; padding: 4px 8px; font-size: 11px; cursor: pointer; color: #475569; flex-shrink: 0; }}
    .copy-btn:hover {{ background: #cbd5e1; }}
    .charts-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin: 24px 0; }}
    .chart-card {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; }}
    .chart-title {{ font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase; margin-bottom: 12px; }}
    .bar-row {{ display: flex; align-items: center; margin-bottom: 6px; }}
    .bar-label {{ width: 100px; font-size: 11px; color: #475569; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .bar-track {{ flex: 1; height: 6px; background: #f1f5f9; border-radius: 3px; margin: 0 8px; }}
    .bar-fill {{ height: 100%; border-radius: 3px; }}
    .bar-value {{ width: 28px; font-size: 11px; font-weight: 500; color: #64748b; text-align: right; }}
    .empty {{ color: #94a3b8; font-size: 13px; }}
    .horizon-section {{ display: flex; flex-direction: column; gap: 16px; }}
    .horizon-card {{ background: linear-gradient(135deg, #faf5ff 0%, #f5f3ff 100%); border: 1px solid #c4b5fd; border-radius: 8px; padding: 16px; }}
    .horizon-title {{ font-weight: 600; font-size: 15px; color: #5b21b6; margin-bottom: 8px; }}
    .horizon-possible {{ font-size: 14px; color: #334155; margin-bottom: 10px; line-height: 1.5; }}
    .horizon-tip {{ font-size: 13px; color: #6b21a8; background: rgba(255,255,255,0.6); padding: 8px 12px; border-radius: 4px; }}
    .prompt-label {{ font-size: 11px; font-weight: 600; text-transform: uppercase; color: #64748b; margin-bottom: 6px; }}
    .pattern-prompt {{ background: #f8fafc; padding: 12px; border-radius: 6px; margin-top: 12px; border: 1px solid #e2e8f0; }}
    .pattern-prompt code {{ font-family: monospace; font-size: 12px; color: #334155; display: block; white-space: pre-wrap; margin-bottom: 8px; }}
    .fun-ending {{ background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 1px solid #fbbf24; border-radius: 12px; padding: 24px; margin-top: 40px; text-align: center; }}
    .fun-headline {{ font-size: 18px; font-weight: 600; color: #78350f; margin-bottom: 8px; }}
    .fun-detail {{ font-size: 14px; color: #92400e; }}
    @media (max-width: 640px) {{ .charts-row {{ grid-template-columns: 1fr; }} .stats-row {{ justify-content: center; }} }}
  </style>
</head>
<body>
  <div class="container">
    <h1>{_esc(spec.display_name)} Insights</h1>
    <p class="subtitle">{_esc(subtitle)}</p>
    {at_a_glance_html}

    <nav class="nav-toc">
      <a href="#section-work">What You Work On</a>
      <a href="#section-usage">How You Use {_esc(spec.display_name)}</a>
      <a href="#section-wins">Impressive Things</a>
      <a href="#section-friction">Where Things Go Wrong</a>
      <a href="#section-features">Features to Try</a>
      <a href="#section-patterns">New Usage Patterns</a>
      <a href="#section-horizon">On the Horizon</a>
    </nav>

    {stats_row}
    {areas_html}
    {charts_1}
    {style_html}
    {wins_html}
    {charts_2}
    {friction_html}
    {charts_3}
    {suggestions_html}
    {features_html}
    {patterns_html}
    {horizon_html}
    {fun_html}
  </div>
  <script>
    function copyText(btn) {{
      const code = btn.previousElementSibling;
      navigator.clipboard.writeText(code.textContent).then(() => {{
        btn.textContent = 'Copied!';
        setTimeout(() => {{ btn.textContent = 'Copy'; }}, 2000);
      }});
    }}
    function copyCmdItem(idx) {{
      const checkbox = document.getElementById('cmd-' + idx);
      if (checkbox) {{
        const text = checkbox.dataset.text;
        navigator.clipboard.writeText(text).then(() => {{
          const btn = checkbox.nextElementSibling.querySelector('.copy-btn');
          if (btn) {{ btn.textContent = 'Copied!'; setTimeout(() => {{ btn.textContent = 'Copy'; }}, 2000); }}
        }});
      }}
    }}
    function copyAllCheckedClaudeMd() {{
      const checkboxes = document.querySelectorAll('.cmd-checkbox:checked');
      const texts = [];
      checkboxes.forEach(cb => {{ if (cb.dataset.text) {{ texts.push(cb.dataset.text); }} }});
      const combined = texts.join('\\n');
      const btn = document.querySelector('.copy-all-btn');
      if (btn) {{
        navigator.clipboard.writeText(combined).then(() => {{
          btn.textContent = 'Copied ' + texts.length + ' items!';
          btn.classList.add('copied');
          setTimeout(() => {{ btn.textContent = 'Copy All Checked'; btn.classList.remove('copied'); }}, 2000);
        }});
      }}
    }}
  </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def cmd_facet(args):
    """Generate a facet for a single session JSONL file."""
    console = stderr_console()
    spec = get_agent_spec("claude")
    path = Path(args.session_path)
    console.title("Facet", subtitle=str(path))
    if not path.exists():
        console.error(f"file not found: {path}")
        sys.exit(1)

    console.phase("Parse session")
    session = parse_agent_session(path, spec)
    if not session:
        console.error("failed to parse session")
        sys.exit(1)

    meta = extract_session_meta(session)
    console.detail(
        console.join([
            session.session_id[:8],
            f"{meta['user_message_count']} user msgs",
            f"{meta['duration_minutes']} min",
        ])
    )

    if is_insights_session(session):
        console.skip("skipping: this is an /insights session")
        sys.exit(0)

    if not meets_minimum_criteria(meta):
        console.skip(console.join(["skipping: below minimum criteria", "need >=2 user msgs and >=1 min"]))
        sys.exit(0)

    console.phase("Generate facet")
    facet = generate_facet(session, spec, dry_run=args.dry_run)
    if facet:
        print(json.dumps(facet, indent=2))
        if args.save:
            save_facet(facet, spec)
            facet_path = spec.out_facets_dir / f"{facet['session_id']}.json"
            console.success(f"saved to {facet_path}")
    else:
        console.skip("no facet generated")


def cmd_facets(args):
    """Generate facets for all sessions missing them."""
    console = stderr_console()
    spec = get_agent_spec("claude")
    console.title("Facets", subtitle=spec.display_name)
    console.phase("Discover sessions")
    all_sessions = discover_sessions(getattr(args, "project", None), spec)
    console.detail(f"{len(all_sessions)} session JSONL files found")

    console.phase("Load cached facets")
    need_facets = []

    for info in all_sessions:
        sid = info["session_id"]

        # Check if facet already exists
        existing = load_cached_facet(sid, spec)
        if existing:
            continue

        if len(need_facets) >= MAX_NEW_FACETS:
            break
        need_facets.append(info)

    console.detail(console.join([f"{len(need_facets)} sessions need facets", f"max {MAX_NEW_FACETS}"]))

    console.phase("Generate facets")
    generated = 0
    skipped_parse = 0
    skipped_insights = 0
    skipped_criteria = 0
    skipped_failed = 0

    for info in need_facets:
        session = parse_agent_session(info["path"], spec)
        if not session:
            skipped_parse += 1
            continue
        if is_insights_session(session):
            skipped_insights += 1
            continue

        meta = extract_session_meta(session)
        if not meets_minimum_criteria(meta):
            skipped_criteria += 1
            continue

        facet = generate_facet(session, spec, dry_run=args.dry_run)
        if facet:
            if not args.dry_run:
                save_facet(facet, spec)
                facet_path = spec.out_facets_dir / f"{facet['session_id']}.json"
                console.artifact("wrote", facet_path)
            generated += 1
            console.detail(
                console.join([
                    f"{generated}/{len(need_facets)}",
                    session.session_id[:8],
                    facet.get("brief_summary", "")[:80],
                ])
            )
        else:
            skipped_failed += 1

    console.success(f"generated {generated} new facets")
    if skipped_parse or skipped_insights or skipped_criteria or skipped_failed:
        console.skip(
            console.join([
                f"skipped {skipped_parse} unparseable",
                f"{skipped_insights} insights-about-insights",
                f"{skipped_criteria} below minimum criteria",
                f"{skipped_failed} LLM call failed",
            ])
        )


def run_report_for_agent(args, spec: AgentSpec) -> dict:
    """Full pipeline: discover → session-meta → facets → aggregate → report."""
    console = stderr_console()
    total_start = time.time()

    # Phase 1: Discover
    project = getattr(args, "project", None)
    scope = f"project {project}" if project else "all projects"
    console.title(spec.display_name, subtitle=console.join([f"scope {scope}", f"output {spec.output_dir}/"]))
    console.phase("Discover sessions")
    all_sessions = discover_sessions(project, spec)
    console.detail(f"{len(all_sessions)} {spec.display_name} sessions found")

    # Phase 2-3: Load cached data + parse new sessions
    console.phase("Load cache")
    session_metas: list[dict] = []
    facets: dict[str, dict] = {}
    need_facets: list[dict] = []
    parsed_sessions: dict[str, ParsedSession] = {}

    for info in all_sessions:
        sid = info["session_id"]

        # Try cached meta
        cached_meta = load_cached_session_meta(sid, spec)
        if cached_meta:
            session_metas.append(cached_meta)
        else:
            session = parse_agent_session(info["path"], spec)
            if session and not is_insights_session(session):
                meta = extract_session_meta(session)
                session_metas.append(meta)
                parsed_sessions[sid] = session

        # Try cached facet
        cached_facet = load_cached_facet(sid, spec)
        if cached_facet:
            facets[sid] = cached_facet
        elif len(need_facets) < MAX_NEW_FACETS:
            need_facets.append(info)

    console.detail(f"{len(session_metas)} session-meta records loaded")
    console.detail(f"{len(facets)} cached facets")
    console.detail(f"{len(need_facets)} sessions need new facets")

    # Phase 4: Generate new facets
    generated = 0
    if getattr(args, "skip_facets", False):
        console.phase("Generate facets", detail="skipped by --skip-facets", skipped=True)
    else:
        console.phase("Generate facets", detail=f"{len(need_facets)} sessions")
        console.detail(f"model {FACET_MODEL}")
        console.detail("max tokens 4096 per facet")

        skipped_parse = 0
        skipped_insights = 0
        skipped_criteria = 0
        skipped_failed = 0

        for i, info in enumerate(need_facets):
            sid = info["session_id"]

            if sid in parsed_sessions:
                session = parsed_sessions[sid]
            else:
                session = parse_agent_session(info["path"], spec)

            if not session:
                skipped_parse += 1
                continue
            if is_insights_session(session):
                skipped_insights += 1
                continue

            meta = extract_session_meta(session)
            if not meets_minimum_criteria(meta):
                skipped_criteria += 1
                continue

            facet = generate_facet(session, spec, dry_run=args.dry_run)
            if facet:
                facets[sid] = facet
                if not args.dry_run:
                    save_facet(facet, spec)
                    console.artifact("wrote", spec.out_facets_dir / f"{sid}.json")
                generated += 1
            else:
                skipped_failed += 1

        console.success(f"generated {generated} new facets")
        if skipped_parse or skipped_insights or skipped_criteria or skipped_failed:
            console.skip(
                console.join([
                    f"skipped {skipped_parse} unparseable",
                    f"{skipped_insights} insights-about-insights",
                    f"{skipped_criteria} below minimum criteria",
                    f"{skipped_failed} LLM call failed",
                ])
            )

    # Filter warmup sessions
    filtered_metas = [m for m in session_metas if meets_minimum_criteria(m) and not is_warmup_only(facets.get(m["session_id"], {}))]
    filtered_facets = {k: v for k, v in facets.items() if not is_warmup_only(v)}

    # Phase 5: Aggregate stats
    console.phase("Aggregate stats")
    stats = aggregate_stats(filtered_metas, filtered_facets)
    stats["total_sessions_scanned"] = len(all_sessions)
    stats["agent"] = spec.name
    stats["agent_display"] = spec.display_name
    console.detail(console.join([
        f"{stats['total_sessions']} sessions",
        f"{stats['sessions_with_facets']} with facets",
    ]))
    console.detail(f"date range {stats['date_range']['start']} to {stats['date_range']['end']}")

    # Phase 6: Generate report sections (7 parallel + 1 sequential)
    console.phase("Generate report sections", detail="7 LLM calls")
    console.detail(f"model {REPORT_MODEL}")
    console.detail("max tokens 8192 per section")
    context = build_report_context(stats, filtered_facets, spec)
    console.detail(f"context length {len(context)} chars")

    sections = {}
    for prompt_def in report_prompts(spec):
        name, result = generate_report_section(prompt_def, context, dry_run=args.dry_run)
        if result:
            sections[name] = result

    # at_a_glance runs after the 7 sections, using their results
    console.phase("Generate at a glance", detail="uses report section results")
    at_a_glance = generate_at_a_glance(context, sections, spec, dry_run=args.dry_run)
    if at_a_glance:
        sections["at_a_glance"] = at_a_glance

    # Phase 7: Output
    elapsed = time.time() - total_start
    report = {
        "agent": spec.name,
        "agent_display": spec.display_name,
        "stats": stats,
        "sections": sections,
    }

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = spec.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    console.phase("Write report")
    console.artifact("report json", report_path)

    # Phase 7b: Build HTML report
    html = build_report_html(stats, sections, spec)
    html_path = spec.output_dir / "report.html"
    html_path.write_text(html)
    console.artifact("report html", html_path, extra=f"{len(html):,} chars")

    facet_count = len(list(spec.out_facets_dir.glob("*.json"))) if spec.out_facets_dir.exists() else 0
    console.summary([
        ("elapsed", format_elapsed(elapsed)),
        ("output dir", f"{spec.output_dir}/"),
        ("report json", report_path),
        ("report html", f"file://{html_path}"),
        ("facets", f"{spec.out_facets_dir}/ ({facet_count} files)"),
    ])
    return report


def normalize_agents(agent_args: list[str] | None) -> list[str]:
    agents = agent_args or ["claude"]
    normalized = []
    for agent in agents:
        if agent not in AGENT_CHOICES:
            raise ValueError(f"Unsupported agent: {agent}")
        if agent not in normalized:
            normalized.append(agent)
    return normalized


def report_subprocess_command(agent: str) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "report", "--agent", agent]
    return [sys.executable, "-m", "agent_insights.cli", "report", "--agent", agent]


@dataclass
class AgentReportProcess:
    agent: str
    process: subprocess.Popen
    started_at: float
    stdout: list[str]
    threads: list[threading.Thread]


def drain_agent_report_stdout(stream, buffer: list[str]) -> None:
    for line in iter(stream.readline, ""):
        buffer.append(line)
    stream.close()


def stream_agent_report_stderr(agent: str, stream, console: ConsoleRenderer, label_width: int) -> None:
    for line in iter(stream.readline, ""):
        console.agent_line(agent, line.rstrip(), label_width=label_width)
    stream.close()


def run_agent_report_subprocesses(args, agents: list[str]) -> None:
    console = stderr_console()
    console.title(f"{len(agents)} agents in parallel")
    label_width = max(len(agent) for agent in agents)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    package_root = str(Path(__file__).resolve().parents[1])
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{package_root}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else package_root
    )

    processes = []
    for agent in agents:
        cmd = report_subprocess_command(agent)
        if getattr(args, "dry_run", False):
            cmd.append("--dry-run")
        if getattr(args, "project", None):
            cmd.extend(["--project", args.project])
        if getattr(args, "skip_facets", False):
            cmd.append("--skip-facets")
        console.agent_line(agent, f"{console.glyphs.phase} starting {' '.join(cmd)}", label_width=label_width)
        process = subprocess.Popen(
            cmd,
            cwd=Path.cwd(),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        started_at = time.time()
        stdout_buffer: list[str] = []
        stdout_thread = threading.Thread(
            target=drain_agent_report_stdout,
            args=(process.stdout, stdout_buffer),
        )
        stderr_thread = threading.Thread(
            target=stream_agent_report_stderr,
            args=(agent, process.stderr, console, label_width),
        )
        stdout_thread.start()
        stderr_thread.start()
        processes.append(AgentReportProcess(
            agent=agent,
            process=process,
            started_at=started_at,
            stdout=stdout_buffer,
            threads=[stdout_thread, stderr_thread],
        ))

    results = []
    failed = False
    for running in processes:
        returncode = running.process.wait()
        for thread in running.threads:
            thread.join()
        stdout = "".join(running.stdout)
        spec = get_agent_spec(running.agent, isolated_output=True)
        elapsed = time.time() - running.started_at
        if returncode == 0:
            console.agent_line(
                running.agent,
                f"{console.glyphs.success} {console.join(['finished', f'exit {returncode}', format_elapsed(elapsed)])}",
                label_width=label_width,
            )
        else:
            console.agent_line(
                running.agent,
                f"{console.glyphs.error} {console.join(['finished', f'exit {returncode}', format_elapsed(elapsed)])}",
                label_width=label_width,
            )
        if returncode != 0:
            failed = True
            if stdout:
                console.phase(f"{running.agent} stdout excerpt")
                for line in stdout.rstrip()[-4000:].splitlines():
                    console.detail(line)
        results.append({
            "agent": running.agent,
            "returncode": returncode,
            "output_dir": str(spec.output_dir),
            "report_json": str(spec.output_dir / "report.json"),
            "report_html": str(spec.output_dir / "report.html"),
        })

    print(json.dumps({"agents": results}, indent=2))
    if failed:
        sys.exit(1)


def cmd_report(args):
    agents = normalize_agents(getattr(args, "agent", None))
    explicit_agent = bool(getattr(args, "agent", None))
    if len(agents) > 1:
        run_agent_report_subprocesses(args, agents)
        return

    spec = get_agent_spec(agents[0], isolated_output=explicit_agent)
    report = run_report_for_agent(args, spec)
    print(json.dumps(report, indent=2))


def cmd_corrections(args):
    """Extract corrections from project sessions and synthesize CLAUDE.md rules."""
    console = stderr_console()
    console.title("Corrections")
    project = getattr(args, "project", None)
    if not project:
        # Auto-detect from CWD
        project = str(Path.cwd())
        console.detail(console.join(["no --project specified", f"using CWD {project}"]))

    max_sessions = args.max_sessions

    # Find CLAUDE.md
    claude_md_path = args.claude_md
    if not claude_md_path:
        candidate = Path(project) / "CLAUDE.md"
        if candidate.exists():
            claude_md_path = str(candidate)
    claude_md_content = ""
    if claude_md_path and Path(claude_md_path).exists():
        claude_md_content = Path(claude_md_path).read_text()
        console.detail(console.join([f"CLAUDE.md {claude_md_path}", f"{len(claude_md_content)} chars"]))
    else:
        console.warning(console.join(["no CLAUDE.md found", "will still extract corrections"]))

    # Phase 1: Discover project sessions
    console.phase("Discover sessions", detail=f"project {project}")
    all_sessions = discover_sessions(project)
    console.detail(f"{len(all_sessions)} sessions found")

    # Phase 2: Parse and filter
    console.phase("Parse sessions", detail=f"max {max_sessions}")
    sessions = []
    for info in all_sessions:
        if len(sessions) >= max_sessions:
            break
        session = parse_jsonl(info["path"])
        if not session:
            continue
        if is_insights_session(session):
            continue
        meta = extract_session_meta(session)
        if not meets_minimum_criteria(meta):
            continue
        sessions.append(session)

    console.detail(f"{len(sessions)} sessions to analyze")

    # Phase 3: Extract corrections from each session
    console.phase("Extract corrections")
    all_corrections = []
    for i, session in enumerate(sessions):
        result = extract_corrections(session, dry_run=args.dry_run)
        if result:
            n_corr = len(result.get("corrections", []))
            n_inst = len(result.get("repeated_instructions", []))
            if n_corr > 0 or n_inst > 0:
                all_corrections.append(result)
                console.detail(
                    console.join([
                        f"{i+1}/{len(sessions)}",
                        session.session_id[:8],
                        f"{n_corr} corrections",
                        f"{n_inst} instructions",
                    ])
                )
            else:
                console.detail(console.join([
                    f"{i+1}/{len(sessions)}",
                    session.session_id[:8],
                    "clean session",
                ]))

    if args.dry_run:
        console.skip(console.join(["dry run", f"would analyze {len(sessions)} sessions"]))
        return

    # Save raw corrections
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    corrections_path = OUTPUT_DIR / "corrections.json"
    corrections_path.write_text(json.dumps(all_corrections, indent=2))
    console.artifact("corrections", corrections_path)

    total_corr = sum(len(c.get("corrections", [])) for c in all_corrections)
    total_inst = sum(len(c.get("repeated_instructions", [])) for c in all_corrections)
    console.detail(
        console.join([
            f"total {total_corr} corrections",
            f"{total_inst} repeated instructions",
            f"{len(all_corrections)} sessions",
        ])
    )

    if not all_corrections:
        console.skip(console.join(["no corrections found", "nothing to synthesize"]))
        return

    # Phase 4: Synthesize into CLAUDE.md rules
    console.phase("Synthesize CLAUDE.md rules")
    rules = synthesize_rules(all_corrections, claude_md_content, dry_run=args.dry_run)

    if rules:
        rules_path = OUTPUT_DIR / "rules.json"
        rules_path.write_text(json.dumps(rules, indent=2))
        console.artifact("rules", rules_path)

        # Print human-readable summary
        console.phase("Suggested CLAUDE.md rules")
        for r in rules.get("rules", []):
            freq = r.get("frequency", 1)
            sev = r.get("severity", "?")
            section = r.get("section", "?")
            console.detail(console.join([sev, f"{freq}x", section]))
            console.detail(r["rule"])
            if r.get("evidence"):
                console.detail(f"evidence: {r['evidence']}")

        violated = rules.get("already_covered", [])
        if violated:
            console.phase("Existing rules being violated")
            for v in violated:
                console.detail(f"rule: {v['rule']}")
                console.detail(f"violated in: {v.get('but_violated_in', [])}")
                if v.get("suggestion"):
                    console.detail(f"suggestion: {v['suggestion']}")

    # Also dump full output to stdout as JSON
    output = {
        "project": project,
        "sessions_analyzed": len(sessions),
        "corrections": all_corrections,
        "rules": rules,
    }
    print(json.dumps(output, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="agent-insights",
        description="Generate AI agent session insights reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-v", "--version", action="version", version=f"agent-insights {__version__}")
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--dry-run", action="store_true", help="Show what would happen without making API calls")
    parent.add_argument("--project", help="Scope to a project path (e.g. /path/to/project)")
    sub = parser.add_subparsers(dest="command")

    # facet: single session
    p_facet = sub.add_parser("facet", help="Generate facet for a single session", parents=[parent])
    p_facet.add_argument("session_path", help="Path to session .jsonl file")
    p_facet.add_argument("--save", action="store_true", help="Save facet to ./insights-output/facets/")

    # facets: all missing
    p_facets = sub.add_parser("facets", help="Generate facets for all sessions missing them", parents=[parent])

    # report: full pipeline
    p_report = sub.add_parser("report", help="Run the full pipeline (facets + report)", parents=[parent])
    p_report.add_argument(
        "--agent",
        action="append",
        choices=AGENT_CHOICES,
        help="Agent session logs to analyze. Repeat to analyze multiple agents in parallel. Default: claude",
    )
    p_report.add_argument("--skip-facets", action="store_true", help="Skip facet generation, use only cached facets")

    # corrections: project-level correction extraction
    p_corr = sub.add_parser("corrections", help="Extract user corrections and generate CLAUDE.md rules", parents=[parent])
    p_corr.add_argument("--claude-md", help="Path to CLAUDE.md (default: auto-detect from --project)")
    p_corr.add_argument("--max-sessions", type=int, default=30, help="Max sessions to analyze (default: 30)")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "facet":
        cmd_facet(args)
    elif args.command == "facets":
        cmd_facets(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "corrections":
        cmd_corrections(args)


if __name__ == "__main__":
    main()
