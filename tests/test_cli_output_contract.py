from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CliOutputContractTests(unittest.TestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = os.environ.copy()
            env.update({
                "CLAUDE_CONFIG_DIR": str(root / "claude-home"),
                "CODEX_HOME": str(root / "codex-home"),
                "CURSOR_CONFIG_DIR": str(root / "cursor-home"),
                "GEMINI_CONFIG_DIR": str(root / "gemini-home"),
                "NO_COLOR": "1",
                "TERM": "dumb",
            })
            return subprocess.run(
                [sys.executable, "-m", "agent_insights.cli", *args],
                cwd=root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
            )

    def test_single_agent_report_keeps_json_on_stdout(self) -> None:
        result = self.run_cli("report", "--dry-run", "--skip-facets", "--agent", "claude")

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["agent"], "claude")
        self.assertEqual(payload["stats"]["total_sessions_scanned"], 0)
        self.assertIn("agent-insights | Claude Code", result.stderr)
        self.assertIn("* Discover sessions", result.stderr)
        self.assertNotIn("\x1b[", result.stdout)
        self.assertNotIn("·", result.stderr)

    def test_multi_agent_report_keeps_summary_json_on_stdout(self) -> None:
        result = self.run_cli(
            "report",
            "--dry-run",
            "--skip-facets",
            "--agent",
            "claude",
            "--agent",
            "codex",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual([agent["agent"] for agent in payload["agents"]], ["claude", "codex"])
        self.assertIn("agent-insights | 2 agents in parallel", result.stderr)
        self.assertIn("claude  * Discover sessions", result.stderr)
        self.assertIn("codex   * Discover sessions", result.stderr)
        self.assertNotIn("\x1b[", result.stdout)
        self.assertNotIn("·", result.stderr)


if __name__ == "__main__":
    unittest.main()
