"""Shared output handling for tools that can emit large text (shell, file read, …).

Design assumption: an agent can't reliably predict how big a command's or file's output
will be. So any bounded-output tool should:

  1. return a **bounded preview** (never dump unbounded text into the context),
  2. make the **full** output reachable (saved to a file the agent can page through), and
  3. report **precisely** what was shown vs. the total — total chars/lines, the exact
     shown range, and the next offset to continue from —

so the agent can decide to read the rest, read a specific slice, or move on. Most of the
time, once it sees "this is 8000 lines", it will read only the part it needs.
"""

from __future__ import annotations

from typing import Any


def line_count(s: str) -> int:
    """Number of lines in `s` (a trailing newline does not add an empty line)."""
    if not s:
        return 0
    return s.count("\n") + (0 if s.endswith("\n") else 1)


def clip_text(text: str, max_chars: int) -> tuple[str, dict[str, Any]]:
    """Clip `text` to ~max_chars, preferring a line boundary.

    Returns (preview, meta). meta always has total_chars/total_lines/shown_chars/
    shown_lines; when truncated it also has truncated=True and next_line_offset (the
    1-based line to resume from).
    """
    total_chars = len(text)
    total_lines = line_count(text)
    base = {"total_chars": total_chars, "total_lines": total_lines}

    if max_chars <= 0 or total_chars <= max_chars:
        return text, {**base, "truncated": False,
                      "shown_chars": total_chars, "shown_lines": total_lines}

    cut = max_chars
    nl = text.rfind("\n", 0, cut)
    if nl >= 0 and nl >= max_chars // 2:   # snap to a line boundary unless it wastes >half
        cut = nl + 1
    preview = text[:cut]
    shown_lines = preview.count("\n")       # fully-shown lines
    return preview, {**base, "truncated": True,
                     "shown_chars": cut, "shown_lines": shown_lines,
                     "next_line_offset": shown_lines + 1}


def truncation_note(meta: dict[str, Any], *, file_ref: str | None = None,
                    read_tool: str = "read_file") -> str:
    """A precise, human-readable note describing exactly what was shown."""
    if not meta.get("truncated"):
        return ""
    sc, tc = meta["shown_chars"], meta["total_chars"]
    sl, tl = meta["shown_lines"], meta["total_lines"]
    nxt = meta["next_line_offset"]
    note = f"Truncated — showed chars 0–{sc} of {tc} (lines 1–{sl} of {tl})."
    if file_ref:
        note += (f" Full output saved to `{file_ref}` — read more with "
                 f"{read_tool}(path='{file_ref}', offset={nxt}).")
    return note
