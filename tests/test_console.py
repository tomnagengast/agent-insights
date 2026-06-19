from __future__ import annotations

import io
import unittest

from agent_insights.console import ConsoleRenderer, format_elapsed, supports_unicode


class TtyStringIO(io.StringIO):
    encoding = "utf-8"

    def isatty(self) -> bool:
        return True


class ConsoleRendererTests(unittest.TestCase):
    def test_plain_summary_output_is_deterministic(self) -> None:
        stream = io.StringIO()
        console = ConsoleRenderer(stream, width=80, color=False, unicode=False)

        console.title("Claude Code")
        console.context(["scope all projects", "output insights-output/"])
        console.phase("Discover", detail="1,248 sessions found")
        console.phase("Generate facets", detail="skipped by --skip-facets", skipped=True)
        console.summary([
            ("elapsed", "4.2s"),
            ("report html", "file://insights-output/report.html"),
        ])

        self.assertEqual(
            stream.getvalue(),
            "\n".join([
                "agent-insights | Claude Code",
                "scope all projects | output insights-output/",
                "* Discover",
                "  - 1,248 sessions found",
                "",
                "- Generate facets",
                "  - skipped by --skip-facets",
                "",
                "Summary",
                "  elapsed      4.2s",
                "  report html  file://insights-output/report.html",
                "",
            ]),
        )

    def test_unicode_glyphs_when_enabled(self) -> None:
        stream = io.StringIO()
        console = ConsoleRenderer(stream, width=80, color=False, unicode=True)

        console.title("Claude Code")
        console.phase("Discover", detail="sessions found")
        console.success("report complete")

        self.assertIn("agent-insights · Claude Code", stream.getvalue())
        self.assertIn("● Discover", stream.getvalue())
        self.assertIn("  └ sessions found", stream.getvalue())
        self.assertIn("  ✓ report complete", stream.getvalue())

    def test_supports_unicode_requires_tty_and_utf_encoding(self) -> None:
        self.assertFalse(supports_unicode(io.StringIO(), {"TERM": "xterm-256color"}))
        self.assertTrue(supports_unicode(TtyStringIO(), {"TERM": "xterm-256color"}))
        self.assertFalse(supports_unicode(TtyStringIO(), {"TERM": "dumb"}))

    def test_detail_wraps_with_stable_indent(self) -> None:
        stream = io.StringIO()
        console = ConsoleRenderer(stream, width=28, color=False, unicode=False)

        console.detail("this is a long detail message that should wrap predictably")

        self.assertEqual(
            stream.getvalue(),
            "\n".join([
                "  - this is a long detail",
                "    message that should wrap",
                "    predictably",
                "",
            ]),
        )

    def test_agent_line_normalizes_child_output(self) -> None:
        stream = io.StringIO()
        console = ConsoleRenderer(stream, width=80, color=False, unicode=False)

        console.agent_line("claude", "* Discover", label_width=6)
        console.agent_line("codex", "  - 12 sessions found", label_width=6)
        console.agent_line("gemini", "raw child line", label_width=6)

        self.assertEqual(
            stream.getvalue(),
            "\n".join([
                "claude  * Discover",
                "codex     - 12 sessions found",
                "gemini    - raw child line",
                "",
            ]),
        )

    def test_format_elapsed(self) -> None:
        self.assertEqual(format_elapsed(0.2), "0s")
        self.assertEqual(format_elapsed(59.4), "59s")
        self.assertEqual(format_elapsed(61), "1m 01s")
        self.assertEqual(format_elapsed(3661), "1h 01m 01s")


if __name__ == "__main__":
    unittest.main()
