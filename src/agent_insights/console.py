"""Human-facing terminal rendering for agent-insights.

Machine-readable command results stay on stdout. This module owns stderr output
so the rest of the CLI can talk in semantic events instead of ad hoc prints.
"""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
from dataclasses import dataclass
from typing import Iterable, TextIO

from rich.console import Console
from rich.text import Text


@dataclass(frozen=True)
class GlyphSet:
    phase: str
    detail: str
    success: str
    skip: str
    warning: str
    error: str
    separator: str
    arrow: str


UNICODE_GLYPHS = GlyphSet(
    phase="●",
    detail="└",
    success="✓",
    skip="○",
    warning="!",
    error="✗",
    separator="·",
    arrow="→",
)

ASCII_GLYPHS = GlyphSet(
    phase="*",
    detail="-",
    success="ok",
    skip="-",
    warning="!",
    error="x",
    separator="|",
    arrow="->",
)


def _stream_is_tty(stream: TextIO) -> bool:
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


def supports_color(stream: TextIO | None = None, env: dict[str, str] | None = None) -> bool:
    stream = stream or sys.stderr
    env = env or os.environ
    if env.get("NO_COLOR") is not None:
        return False
    if env.get("TERM") == "dumb":
        return False
    return _stream_is_tty(stream)


def supports_unicode(stream: TextIO | None = None, env: dict[str, str] | None = None) -> bool:
    stream = stream or sys.stderr
    env = env or os.environ
    if env.get("TERM") == "dumb":
        return False
    if not _stream_is_tty(stream):
        return False
    encoding = (getattr(stream, "encoding", None) or "").lower()
    return "utf" in encoding


def format_elapsed(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    if total < 60:
        return f"{total}s"
    if total < 3600:
        minutes, secs = divmod(total, 60)
        return f"{minutes}m {secs:02}s"
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}h {minutes:02}m {secs:02}s"


class ConsoleRenderer:
    """Small semantic wrapper around Rich for deterministic CLI output."""

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        width: int | None = None,
        color: bool | None = None,
        unicode: bool | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.stream = stream or sys.stderr
        self.env = env or os.environ
        self.width = width or shutil.get_terminal_size((100, 20)).columns
        self.color = supports_color(self.stream, self.env) if color is None else color
        self.unicode = supports_unicode(self.stream, self.env) if unicode is None else unicode
        self.glyphs = UNICODE_GLYPHS if self.unicode else ASCII_GLYPHS
        self._phase_count = 0
        self._console = Console(
            file=self.stream,
            width=self.width,
            force_terminal=self.color,
            no_color=not self.color,
            highlight=False,
            soft_wrap=True,
        )

    def blank(self) -> None:
        self._console.print()

    def raw(self, message: str = "") -> None:
        self._console.print(message)

    def title(self, title: str, *, subtitle: str | None = None) -> None:
        line = Text("agent-insights", style="bold cyan" if self.color else "")
        line.append(f" {self.glyphs.separator} ", style="dim" if self.color else "")
        line.append(title, style="bold" if self.color else "")
        self._console.print(line)
        if subtitle:
            self.context([subtitle])

    def context(self, parts: Iterable[str]) -> None:
        visible = [part for part in parts if part]
        if not visible:
            return
        self._console.print(Text(self.join(visible), style="dim" if self.color else ""))

    def join(self, parts: Iterable[object]) -> str:
        visible = []
        for part in parts:
            if part is None:
                continue
            text = str(part)
            if text:
                visible.append(text)
        return f" {self.glyphs.separator} ".join(visible)

    def phase(self, name: str, *, detail: str | None = None, skipped: bool = False) -> None:
        if self._phase_count:
            self.blank()
        self._phase_count += 1
        glyph = self.glyphs.skip if skipped else self.glyphs.phase
        style = "yellow" if skipped else "bold cyan"
        line = Text(f"{glyph} ", style=style if self.color else "")
        line.append(name, style="bold" if self.color else "")
        self._console.print(line)
        if detail:
            self.detail(detail)

    def detail(self, message: str, *, style: str = "dim") -> None:
        prefix = f"  {self.glyphs.detail} "
        subsequent = " " * len(prefix)
        for line in self._wrap(message, prefix, subsequent):
            self._console.print(Text(line, style=style if self.color else ""))

    def success(self, message: str) -> None:
        self.status(self.glyphs.success, message, "green")

    def skip(self, message: str) -> None:
        self.status(self.glyphs.skip, message, "yellow")

    def warning(self, message: str) -> None:
        self.status(self.glyphs.warning, message, "yellow")

    def error(self, message: str) -> None:
        self.status(self.glyphs.error, message, "red")

    def status(self, glyph: str, message: str, style: str) -> None:
        prefix = f"  {glyph} "
        subsequent = " " * len(prefix)
        for line in self._wrap(message, prefix, subsequent):
            self._console.print(Text(line, style=style if self.color else ""))

    def progress(self, label: str, message: str) -> None:
        label_text = f"[{label}] " if label else ""
        self.detail(f"{label_text}{message}")

    def artifact(self, label: str, path: object, *, extra: str | None = None) -> None:
        message = f"{label}: {path}"
        if extra:
            message = f"{message} ({extra})"
        self.detail(message, style="")

    def summary(self, rows: Iterable[tuple[str, object]]) -> None:
        row_list = [(label, str(value)) for label, value in rows]
        self.blank()
        self._console.print(Text("Summary", style="bold" if self.color else ""))
        self.metric_rows(row_list)

    def metric_rows(self, rows: Iterable[tuple[str, object]]) -> None:
        row_list = [(label, str(value)) for label, value in rows]
        if not row_list:
            return
        width = max(len(label) for label, _ in row_list)
        for label, value in row_list:
            self._console.print(f"  {label:<{width}}  {value}")

    def agent_line(self, agent: str, line: str, *, label_width: int | None = None) -> None:
        normalized = line.rstrip()
        if not normalized:
            return
        width = label_width or len(agent)
        prefix = f"{agent:<{width}}  "
        style = "magenta" if self.color else ""
        parsed = self._normalize_child_line(normalized)
        self._console.print(Text(prefix + parsed, style=style))

    def _normalize_child_line(self, line: str) -> str:
        is_indented = line.startswith(" ")
        stripped = line.strip()
        if not stripped:
            return ""

        phase_prefixes = (
            UNICODE_GLYPHS.phase,
            ASCII_GLYPHS.phase,
            UNICODE_GLYPHS.skip,
        )
        detail_prefixes = (
            f"{UNICODE_GLYPHS.detail} ",
            f"{ASCII_GLYPHS.detail} ",
            "- ",
        )
        status_prefixes = (
            UNICODE_GLYPHS.success,
            UNICODE_GLYPHS.error,
            UNICODE_GLYPHS.warning,
            ASCII_GLYPHS.success,
            ASCII_GLYPHS.error,
            ASCII_GLYPHS.warning,
        )

        if stripped.startswith(detail_prefixes):
            return f"  {stripped}"
        if stripped.startswith(phase_prefixes) and not is_indented:
            return stripped
        if stripped.startswith(status_prefixes):
            return stripped
        return f"  {self.glyphs.detail} {stripped}"

    def _wrap(self, message: str, initial_indent: str, subsequent_indent: str) -> list[str]:
        available = max(20, self.width)
        wrapped = textwrap.wrap(
            str(message),
            width=available,
            initial_indent=initial_indent,
            subsequent_indent=subsequent_indent,
            break_long_words=True,
            break_on_hyphens=False,
        )
        return wrapped or [initial_indent.rstrip()]


def stderr_console() -> ConsoleRenderer:
    return ConsoleRenderer(sys.stderr)
