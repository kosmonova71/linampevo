"""Data models for podcast feeds and episodes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

__all__ = ["PodcastEpisode", "PodcastFeed"]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_str(value: Any) -> str:
    return str(value) if value is not None else ""


@dataclass
class PodcastEpisode:
    guid: str = ""
    title: str = ""
    description: str = ""
    pub_date: str = ""
    duration: int = 0
    audio_url: str = ""
    artwork_url: str = ""
    played: bool = False
    position: float = 0.0
    downloaded_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "guid": self.guid,
            "title": self.title,
            "description": self.description,
            "pub_date": self.pub_date,
            "duration": self.duration,
            "audio_url": self.audio_url,
            "artwork_url": self.artwork_url,
            "played": bool(self.played),
            "position": self.position,
            "downloaded_path": self.downloaded_path,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PodcastEpisode:
        if not isinstance(data, dict):
            raise TypeError("Podcast episode data must be a dictionary")
        known_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in known_fields}
        episode = cls(
            guid=_safe_str(filtered.get("guid", "")),
            title=_safe_str(filtered.get("title", "")),
            description=_safe_str(filtered.get("description", "")),
            pub_date=_safe_str(filtered.get("pub_date", "")),
            duration=_safe_int(filtered.get("duration", 0)),
            audio_url=_safe_str(filtered.get("audio_url", "")),
            artwork_url=_safe_str(filtered.get("artwork_url", "")),
            played=bool(filtered.get("played", False)),
            position=_safe_float(filtered.get("position", 0.0)),
            downloaded_path=_safe_str(filtered.get("downloaded_path", "")),
        )
        return episode


@dataclass
class PodcastFeed:
    feed_id: str = ""
    title: str = ""
    author: str = ""
    description: str = ""
    artwork_url: str = ""
    homepage: str = ""
    feed_url: str = ""
    last_updated: str = ""
    episodes: list[PodcastEpisode] = field(default_factory=list)

    @staticmethod
    def generate_feed_id(feed_url: str) -> str:
        normalized = (feed_url or "").strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "feed_id": self.feed_id,
            "title": self.title,
            "author": self.author,
            "description": self.description,
            "artwork_url": self.artwork_url,
            "homepage": self.homepage,
            "feed_url": self.feed_url,
            "last_updated": self.last_updated,
            "episodes": [ep.to_dict() for ep in self.episodes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PodcastFeed:
        if not isinstance(data, dict):
            raise TypeError("Podcast feed data must be a dictionary")
        episodes = []
        raw_episodes = data.get("episodes")
        if isinstance(raw_episodes, list):
            for ep_data in raw_episodes:
                if not isinstance(ep_data, dict):
                    continue
                try:
                    episodes.append(PodcastEpisode.from_dict(ep_data))
                except (TypeError, ValueError, AttributeError, OverflowError):
                    continue
        feed = cls(
            feed_id=_safe_str(data.get("feed_id", "")),
            title=_safe_str(data.get("title", "")),
            author=_safe_str(data.get("author", "")),
            description=_safe_str(data.get("description", "")),
            artwork_url=_safe_str(data.get("artwork_url", "")),
            homepage=_safe_str(data.get("homepage", "")),
            feed_url=_safe_str(data.get("feed_url", "")),
            last_updated=_safe_str(data.get("last_updated", "")),
            episodes=episodes,
        )
        if not feed.feed_id and feed.feed_url:
            feed.feed_id = cls.generate_feed_id(feed.feed_url)
        return feed
