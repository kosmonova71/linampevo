"""Persisted player settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PlayerSettings:
    auto_play_next: bool = True
    shuffle_mode: bool = False
    repeat_mode: str = "none"
    beat_aware_enabled: bool = True
    beat_threshold: float = 0.1
    auto_download_covers: bool = False
    volume: float = 1.0
    position: float = 0.0
    window_size: tuple[int, int] = (600, 450)
    window_position: tuple[int, int] = (100, 100)
    equalizer_settings: list[float] = field(default_factory=lambda: [0.0] * 10)
    last_played_track: str = ""
    last_played_position: float = 0.0
    show_cover_art: bool = False
    cover_art_size: int = 650
    cover_art_shape: str = "square"
    milkdrop_auto_switch: bool = True

    def __post_init__(self) -> None:
        if self.equalizer_settings is None:
            self.equalizer_settings = [0.0] * 10

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (compatible with ``json.dump``)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlayerSettings:
        return cls(
            auto_play_next=data.get("auto_play_next", True),
            shuffle_mode=data.get("shuffle_mode", False),
            repeat_mode=data.get("repeat_mode", "none"),
            beat_aware_enabled=data.get("beat_aware_enabled", True),
            beat_threshold=data.get("beat_threshold", 0.1),
            auto_download_covers=data.get("auto_download_covers", False),
            volume=data.get("volume", 1.0),
            position=data.get("position", 0.0),
            window_size=tuple(data.get("window_size", (400, 300))),
            window_position=tuple(data.get("window_position", (100, 100))),
            equalizer_settings=data.get("equalizer_settings", None),
            last_played_track=data.get("last_played_track", ""),
            last_played_position=data.get("last_played_position", 0.0),
            show_cover_art=data.get("show_cover_art", False),
            cover_art_size=data.get("cover_art_size", 650),
            cover_art_shape=data.get("cover_art_shape", "square"),
            milkdrop_auto_switch=data.get("milkdrop_auto_switch", True),
        )
