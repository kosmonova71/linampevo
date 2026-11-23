#!/usr/bin/env python3

"""Utility functions and helpers for EvoRenderer.

This module provides additional functionality for the evolutionary renderer:
  - Preset management and discovery
  - Synthetic audio generation for testing
  - Debugging helpers
  - Performance profiling (windowed frame-time stats)
  - Genome validation and sanitization
  - Configuration I/O helpers

Note: Actual audio file/stream input is not implemented; see the renderer's
live audio path for real FFT band data.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Shared parameter bounds from evo_advanced.PARAM_BOUNDS
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "warp": (0.0, 3.0),
    "glow": (0.0, 2.0),
    "rotation": (0.0, 6.2832),
    "noise_scale": (0.01, 5.0),
    "domain_warp": (0.0, 3.0),
    "turbulence": (0.0, 2.0),
    "vortex": (0.0, 3.0),
    "curl_strength": (0.0, 3.0),
    "feedback": (0.0, 1.0),
    "symmetry": (1, 12),
    "particle_density": (0.0, 1.0),
    "contrast": (0.0, 2.0),
    "flow_speed": (0.0, 2.0),
    "scale": (0.1, 3.0),
}


@dataclass
class PresetMetadata:
    """Metadata for a preset file."""

    path: Path
    timestamp: float
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)


class PresetManager:
    """Manager for discovering, loading, and organizing presets."""

    def __init__(self, preset_dir: Path | None = None):
        """Initialize preset manager.

        Args:
            preset_dir: Directory containing preset files
        """
        self.preset_dir = (
            Path(preset_dir) if preset_dir else Path.home() / ".evo_renderer" / "presets"
        )
        self.preset_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, PresetMetadata] = {}
        self._cache_mtime: float = 0.0
        self._cache_ttl: float = 2.0  # seconds

    def discover(self, force: bool = False) -> list[PresetMetadata]:
        """Discover all presets in the directory.

        Args:
            force: Force a fresh scan, ignoring cache

        Returns:
            List of discovered preset metadata
        """
        current_mtime = self.preset_dir.stat().st_mtime if self.preset_dir.exists() else 0.0
        if (
            not force
            and self._cache
            and (time.time() - self._cache_mtime) < self._cache_ttl
            and current_mtime <= self._cache_mtime
        ):
            return list(self._cache.values())

        presets = []
        try:
            for preset_file in self.preset_dir.glob("preset_*.json"):
                try:
                    with open(preset_file) as f:
                        data = json.load(f)

                    meta = PresetMetadata(
                        path=preset_file,
                        timestamp=data.get("timestamp", 0),
                        name=data.get("name", preset_file.stem),
                        description=data.get("description", ""),
                        tags=data.get("tags", []),
                    )
                    presets.append(meta)
                except Exception as _e:
                    logger.warning("Failed to load preset %s: %s", preset_file, _e)

            # Sort by timestamp (newest first)
            presets.sort(key=lambda p: p.timestamp, reverse=True)

            # Update cache
            self._cache = {p.name: p for p in presets}
            self._cache_mtime = current_mtime

            logger.info("Discovered %d presets", len(presets))
            return presets

        except Exception as _e:
            logger.error("Preset discovery failed: %s", _e)
            return []

    def load_preset(self, name: str) -> dict | None:
        """Load a preset by name, returning the full genome/config dict.

        Args:
            name: Preset name (without .json extension)

        Returns:
            Full preset data dictionary, or None if not found
        """
        preset_path = self.preset_dir / f"{name}.json"
        if not preset_path.exists():
            # Try with preset_ prefix
            preset_path = self.preset_dir / f"preset_{name}.json"
        if not preset_path.exists():
            return None

        try:
            with open(preset_path) as f:
                return json.load(f)
        except Exception as _e:
            logger.warning("Failed to load preset %s: %s", preset_path, _e)
            return None

    def save_preset(self, name: str, data: dict, tags: list[str] | None = None) -> bool:
        """Save a preset to disk.

        Expected JSON schema:
        {
            "timestamp": float,        # Unix timestamp
            "name": str,               # Preset name
            "description": str,        # Optional description
            "tags": list[str],         # Optional tags
            "genome": dict,            # Genome parameters
            "config": dict,            # Optional renderer config
        }

        Args:
            name: Preset name
            data: Preset data (genome, config, etc.)
            tags: Optional tags list

        Returns:
            True if successful
        """
        preset_data = {
            "timestamp": time.time(),
            "name": name,
            "description": data.get("description", ""),
            "tags": tags or [],
            "genome": data.get("genome", {}),
            "config": data.get("config", {}),
        }

        preset_path = self.preset_dir / f"preset_{name}.json"
        try:
            preset_path.parent.mkdir(parents=True, exist_ok=True)
            with open(preset_path, "w") as f:
                json.dump(preset_data, f, indent=2)
            # Invalidate cache
            self._cache.clear()
            logger.info("Preset saved to %s", preset_path)
            return True
        except Exception as _e:
            logger.error("Failed to save preset: %s", _e)
            return False

    def get_by_tag(self, tag: str) -> list[PresetMetadata]:
        """Get presets with a specific tag.

        Args:
            tag: Tag to search for

        Returns:
            List of matching presets
        """
        return [p for p in self.discover() if tag in p.tags]

    def get_recent(self, count: int = 10) -> list[PresetMetadata]:
        """Get most recent presets.

        Args:
            count: Number of presets to return

        Returns:
            List of recent presets
        """
        return self.discover()[:count]


class AudioBuffer:
    """Simple audio buffer for testing with synthetic audio.

    Note: This generates synthetic frequency bands for smoke testing only.
    It does NOT replace real FFT bands from the live audio input path.
    The bands sit around 0.5 +/- amplitude and use simple sine/square waves.
    """

    def __init__(self, sample_rate: int = 44100):
        """Initialize audio buffer.

        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.time: float = 0.0  # Current time in seconds

    def generate_frequencies(self, duration: float) -> dict[str, float]:
        """Generate synthetic audio frequency bands.

        Args:
            duration: Duration of audio in seconds

        Returns:
            Dictionary with 'bass', 'mid', 'treb', 'beat' keys (0.0-1.0)
        """
        t = self.time
        self.time += duration

        # Simple sine wave generators at different frequencies
        bass = 0.5 + 0.4 * self._sine_wave(t, 60.0)  # 60 Hz bass
        mid = 0.5 + 0.3 * self._sine_wave(t, 440.0)  # 440 Hz mid
        treb = 0.5 + 0.2 * self._sine_wave(t, 2000.0)  # 2000 Hz treble
        beat = 0.5 + 0.5 * self._square_wave(t, 2.0)  # 2 Hz beat

        return {
            "bass": min(1.0, max(0.0, bass)),
            "mid": min(1.0, max(0.0, mid)),
            "treb": min(1.0, max(0.0, treb)),
            "beat": min(1.0, max(0.0, beat)),
        }

    @staticmethod
    def _sine_wave(t: float, freq: float) -> float:
        """Generate sine wave value."""
        return math.sin(2.0 * math.pi * freq * t)

    @staticmethod
    def _square_wave(t: float, freq: float) -> float:
        """Generate square wave value."""
        phase = (2.0 * math.pi * freq * t) % (2.0 * math.pi)
        return 1.0 if phase < math.pi else -1.0


class FrameTimeProfiler:
    """Simple windowed frame-time profiler for monitoring render performance.

    For detailed GPU stage timing, use evo_perf.GpuProfiler instead.
    """

    def __init__(self, window_size: int = 60):
        """Initialize profiler.

        Args:
            window_size: Number of frames to average
        """
        self.window_size = window_size
        self._frame_times: list[float] = []
        self._start_time: float | None = None

    def start_frame(self) -> None:
        """Mark the start of a frame."""
        self._start_time = time.time()

    def end_frame(self) -> None:
        """Mark the end of a frame and record timing."""
        if self._start_time is not None:
            frame_time = time.time() - self._start_time
            self._frame_times.append(frame_time)

            if len(self._frame_times) > self.window_size:
                self._frame_times.pop(0)

    def get_stats(self) -> dict[str, float]:
        """Get current performance statistics.

        Returns:
            Dictionary with 'avg_ms', 'min_ms', 'max_ms', 'fps', 'frame_count' keys
        """
        if not self._frame_times:
            return {"avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "fps": 0.0, "frame_count": 0}

        times_ms = [t * 1000.0 for t in self._frame_times]
        avg_time = sum(times_ms) / len(times_ms)

        return {
            "avg_ms": avg_time,
            "min_ms": min(times_ms),
            "max_ms": max(times_ms),
            "fps": 1000.0 / avg_time if avg_time > 0 else 0.0,
            "frame_count": len(self._frame_times),
        }

    def log_stats(self) -> None:
        """Log current statistics."""
        stats = self.get_stats()
        logger.info(
            "Performance: %.2fms avg, %.2fms min, %.2fms max, %.1f FPS",
            stats["avg_ms"],
            stats["min_ms"],
            stats["max_ms"],
            stats["fps"],
        )


class GenomeValidator:
    """Validation helper for genome parameters.

    Uses shared PARAM_BOUNDS from evo_advanced for consistency.
    Only a subset of parameters are validated here; for full validation
    use the bounds table directly.
    """

    @staticmethod
    def validate_bounds(params: dict[str, float]) -> tuple[bool, list[str]]:
        """Validate that genome parameters are within expected bounds.

        Args:
            params: Genome parameters dictionary

        Returns:
            Tuple of (valid, error_list)
        """
        errors = []

        for key, (lo, hi) in PARAM_BOUNDS.items():
            if key not in params:
                errors.append(f"{key}: missing")
                continue

            val = params[key]
            if not isinstance(val, (int, float)):
                errors.append(f"{key}: must be numeric (got {type(val).__name__})")
                continue

            if not (lo <= val <= hi):
                errors.append(f"{key}: out of bounds [{lo}, {hi}], got {val}")

        return (len(errors) == 0, errors)

    @staticmethod
    def sanitize_params(params: dict[str, float]) -> dict[str, float]:
        """Sanitize genome parameters to valid ranges using PARAM_BOUNDS.

        Args:
            params: Genome parameters

        Returns:
            Sanitized parameters
        """
        result = copy.deepcopy(params)

        for key, (lo, hi) in PARAM_BOUNDS.items():
            if key in result:
                val = float(result[key])
                result[key] = max(lo, min(hi, val))

        # Special handling for integer params
        if "symmetry" in result:
            result["symmetry"] = int(max(1, min(12, int(result["symmetry"]))))

        return result


class ConfigurationBuilder:
    """Helper for building and loading EvoConfig from files."""

    @staticmethod
    def load(path: Path) -> dict | None:
        """Load configuration from JSON file.

        Args:
            path: Configuration file path

        Returns:
            Configuration dictionary or None if failed
        """
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as _e:
            logger.error("Failed to load config from %s: %s", path, _e)
            return None

    @staticmethod
    def save(config: dict, path: Path) -> bool:
        """Save configuration to JSON file.

        Args:
            config: Configuration dictionary
            path: Output file path

        Returns:
            True if successful
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(config, f, indent=2)
            logger.info("Config saved to %s", path)
            return True
        except Exception as _e:
            logger.error("Failed to save config: %s", _e)
            return False

    @staticmethod
    def get_default_config() -> dict:
        """Get default configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            "feedback_alpha": 0.85,
            "feedback_res": 512,
            "mutation_strength": 0.1,
            "mutation_strength_min": 0.01,
            "mutation_strength_max": 0.5,
            "debug": False,
            "enable_logging": True,
            "brightness_base": 0.85,
            "brightness_audio_scale": 0.35,
        }

    @staticmethod
    def merge_with_defaults(config: dict) -> dict:
        """Merge user config with defaults, ensuring all keys exist.

        Args:
            config: User configuration (may be partial)

        Returns:
            Complete configuration with defaults filled in
        """
        defaults = ConfigurationBuilder.get_default_config()
        result = defaults.copy()
        result.update(config)
        return result
