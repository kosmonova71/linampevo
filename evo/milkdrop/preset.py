"""Parse a Milkdrop ``.milk`` preset file into a structured object.

Milkdrop presets are INI-like files with a single ``[preset00]`` section
followed by ``key=value`` lines.  The ``per_frame_*`` and ``per_pixel_*``
keys are equations (``var = expr``) that Milkdrop evaluates once per
frame or once per pixel respectively.

This module is deliberately tolerant: unknown keys are preserved verbatim
so a preset that uses a parameter the renderer does not understand still
loads without error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Keys that hold Milkdrop equations rather than plain float parameters.
EQUATION_KEY_RE = re.compile(
    r"^(per_frame_\d+|per_pixel_\d+|wavecode_\d+_per_frame\d+|wave_\d+_per_point\d+|wave_\d+_per_frame\d+)$"
)


@dataclass
class MilkdropPreset:
    """Structured representation of a parsed Milkdrop preset."""

    name: str = ""
    path: str = ""
    per_frame_eqs: list[str] = field(default_factory=list)
    per_pixel_eqs: list[str] = field(default_factory=list)
    params: dict[str, str] = field(default_factory=dict)
    raw: dict[str, dict[str, str]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """True when the preset carries no equations and no parameters."""
        return not self.per_frame_eqs and not self.per_pixel_eqs and not self.params

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "per_frame_eqs": list(self.per_frame_eqs),
            "per_pixel_eqs": list(self.per_pixel_eqs),
            "params": dict(self.params),
        }


def _strip_comment(line: str) -> str:
    """Remove a trailing ``//`` comment from a preset line.

    Milkdrop allows ``//`` comments at the end of a line.  We only strip
    when the comment is not inside a string literal (Milkdrop has no string
    literals, so this is safe).
    """
    idx = line.find("//")
    if idx >= 0:
        return line[:idx]
    return line


def _is_equation_key(key: str) -> bool:
    """Return True for keys that hold Milkdrop equations."""
    return bool(EQUATION_KEY_RE.match(key))


def parse_milkdrop_text(text: str, name: str = "", path: str = "") -> MilkdropPreset:
    """Parse the textual body of a ``.milk`` preset.

    Args:
        text: File contents.
        name: Friendly preset name (defaults to the file stem).
        path: Source file path (for diagnostics).

    Returns:
        A populated :class:`MilkdropPreset`.  Never raises on malformed
        input — bad lines are skipped.
    """
    preset = MilkdropPreset(name=name, path=path)
    current_section: str | None = None
    section_params: dict[str, str] = {}

    for raw_line in text.splitlines():
        line = _strip_comment(raw_line).strip()
        if not line:
            continue

        # Section header: [preset00], [preset01], ...
        if line.startswith("[") and line.endswith("]"):
            if current_section is not None and section_params:
                preset.raw[current_section] = section_params
            current_section = line[1:-1].strip()
            section_params = {}
            continue

        if "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        section_params[key] = value

        # Milkdrop per_frame_* / per_pixel_* keys carry equations.
        # A key with no value after the ``=`` is a comment-only line
        # (e.g. ``per_frame_1000=// simple wave``) — skip it.
        if _is_equation_key(key) and not value:
            continue

        if _is_equation_key(key):
            if key.startswith("per_frame") or (key.startswith("wave_") and "per_frame" in key):
                preset.per_frame_eqs.append(f"{key}={value}")
            elif key.startswith("per_pixel") or (key.startswith("wave_") and "per_point" in key):
                preset.per_pixel_eqs.append(f"{key}={value}")
            else:
                preset.params[key] = value
        else:
            preset.params[key] = value

    if current_section is not None and section_params:
        preset.raw[current_section] = section_params

    if not preset.name and path:
        preset.name = Path(path).stem
    return preset


def parse_milkdrop_file(path: str | Path) -> MilkdropPreset:
    """Read and parse a ``.milk`` preset file.

    Args:
        path: Path to the preset file.

    Returns:
        A populated :class:`MilkdropPreset`.

    Raises:
        OSError: if the file cannot be read.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    return parse_milkdrop_text(text, name=p.stem, path=str(p))
