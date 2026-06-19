from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CliOutputContractTests(unittest.TestCase):
    def run_cli_in(self, root: Path, *args: str) -> subprocess.CompletedProcess[str]:
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

    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as tmp:
            return self.run_cli_in(Path(tmp), *args)

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

    def test_single_agent_report_accepts_custom_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.run_cli_in(
                root,
                "report",
                "--dry-run",
                "--skip-facets",
                "--output",
                "analysis-output",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["agent"], "claude")
            self.assertTrue((root / "analysis-output" / "report.json").exists())
            self.assertTrue((root / "analysis-output" / "report.html").exists())
            self.assertFalse((root / "insights-output").exists())

    def test_multi_agent_report_forwards_custom_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.run_cli_in(
                root,
                "report",
                "--dry-run",
                "--skip-facets",
                "--output",
                "parallel-output",
                "--agent",
                "claude",
                "--agent",
                "codex",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(
                [agent["output_dir"] for agent in payload["agents"]],
                [
                    str(root.resolve() / "parallel-output" / "claude"),
                    str(root.resolve() / "parallel-output" / "codex"),
                ],
            )
            self.assertTrue((root / "parallel-output" / "claude" / "report.json").exists())
            self.assertTrue((root / "parallel-output" / "codex" / "report.html").exists())
            self.assertFalse((root / "insights-output").exists())


if __name__ == "__main__":
    unittest.main()
