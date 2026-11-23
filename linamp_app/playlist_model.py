"""The playlist entry model and the m3u playlist format."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any

from .constants import AUDIO_EXTENSIONS

__all__ = ["PlaylistItem", "read_m3u_paths", "write_m3u_lines"]


@dataclass
class PlaylistItem:
    path: str
    title: str = ""
    duration: int = 0
    cover_art_path: str = ""
    id: str = ""

    def __post_init__(self) -> None:
        self.path = os.path.abspath(os.path.expanduser(str(self.path)))
        if not self.title:
            self.title = os.path.basename(self.path)
        if not self.id:
            self.id = hashlib.md5(f"{self.path}:{self.title}".encode()).hexdigest()[:32]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "title": self.title,
            "duration": self.duration,
            "cover_art_path": self.cover_art_path,
            "id": self.id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlaylistItem:
        if not isinstance(data, dict):
            raise TypeError("Input data must be a dictionary")
        item_data = data.copy()
        if "filename" in item_data and "path" not in item_data:
            item_data["path"] = item_data.pop("filename")
        if "path" not in item_data:
            raise ValueError("Playlist item must contain a 'path' or 'filename' field")
        # Only pass known fields to the constructor to support backward compatibility
        # with older playlist formats that may contain extra fields like 'artist'
        known_fields = cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in item_data.items() if k in known_fields}
        return cls(**filtered_data)

    def exists(self) -> bool:
        return os.path.isfile(self.path)

    def get_display_name(self) -> str:
        return self.title if self.title else os.path.basename(self.path)

    def has_cover_art(self) -> bool:
        return bool(self.cover_art_path and os.path.exists(self.cover_art_path))


def read_m3u_paths(filepath: str) -> list[str]:
    """Return the absolute audio paths referenced by an m3u file."""
    with open(filepath, encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    found = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not os.path.isabs(line):
            line = os.path.join(os.path.dirname(filepath), line)
        line = os.path.abspath(os.path.expanduser(line))
        if (
            os.path.exists(line)
            and os.path.isfile(line)
            and os.path.splitext(line)[1].lower() in AUDIO_EXTENSIONS
        ):
            found.append(line)
    return found


def write_m3u_lines(filepath: str, items: list[Any]) -> None:
    """Write playlist items to an m3u file, skipping missing paths."""
    with open(filepath, "w", encoding="utf-8") as handle:
        handle.write("#EXTM3U\n")
        for item in items:
            if not (hasattr(item, "path") and os.path.exists(item.path)):
                continue
            if hasattr(item, "duration") and item.duration > 0 and hasattr(item, "title"):
                handle.write(f"#EXTINF:{item.duration},{item.title}\n")
            handle.write(f"{item.path}\n")
