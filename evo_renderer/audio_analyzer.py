#!/usr/bin/env python3

"""evo_renderer.audio_analyzer

Music DNA analysis engine.

Computes a rich MusicSignature from raw audio levels and FFT data:
  - bpm
  - energy
  - danceability
  - spectral_centroid
  - beat_strength
  - harmonic_complexity
  - dynamic_range
  - key
  - mood

Designed to work with the existing AudioState (bass/mid/treble/beat/energy)
and optional numpy FFT frames.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


# Musical keys (circle of fifths ordering for smoother transitions)
KEYS = ["C", "G", "D", "A", "E", "B", "Gb", "Db", "Ab", "Eb", "Bb", "F"]

MOODS = [
    "calm",
    "energetic",
    "aggressive",
    "mysterious",
    "euphoric",
    "melancholic",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MusicDNA:
    """Full music fingerprint (point 31).

    Every field is 0..1 unless noted otherwise, so it can be fed directly
    into evolutionary pressure and shader parameter mapping.
    """

    bpm: float = 0.0
    energy: float = 0.0
    danceability: float = 0.0
    spectral_centroid: float = 0.0
    beat_strength: float = 0.0
    harmonic_complexity: float = 0.0
    dynamic_range: float = 0.0
    key: str = "C"
    mood: str = "calm"

    # Smoothed audio bands (legacy compatibility)
    bass: float = 0.0
    mid: float = 0.0
    treble: float = 0.0
    beat: float = 0.0
    section: str = "ambient"

    # Derived
    tempo: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def feature_vector(self) -> list[float]:
        return [
            self.bpm / 200.0,
            self.energy,
            self.danceability,
            self.spectral_centroid,
            self.beat_strength,
            self.harmonic_complexity,
            self.dynamic_range,
            self.bass,
            self.mid,
            self.treble,
            self.beat,
        ]

    def similarity(self, other: MusicDNA) -> float:
        a = self.feature_vector()
        b = other.feature_vector()
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a)) or 1.0
        mag_b = math.sqrt(sum(x * x for x in b)) or 1.0
        return dot / (mag_a * mag_b)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bpm": self.bpm,
            "energy": self.energy,
            "danceability": self.danceability,
            "spectral_centroid": self.spectral_centroid,
            "beat_strength": self.beat_strength,
            "harmonic_complexity": self.harmonic_complexity,
            "dynamic_range": self.dynamic_range,
            "key": self.key,
            "mood": self.mood,
            "bass": self.bass,
            "mid": self.mid,
            "treble": self.treble,
            "beat": self.beat,
            "section": self.section,
            "tempo": self.tempo,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MusicDNA:
        return cls(
            bpm=float(data.get("bpm", 0.0)),
            energy=float(data.get("energy", 0.0)),
            danceability=float(data.get("danceability", 0.0)),
            spectral_centroid=float(data.get("spectral_centroid", 0.0)),
            beat_strength=float(data.get("beat_strength", 0.0)),
            harmonic_complexity=float(data.get("harmonic_complexity", 0.0)),
            dynamic_range=float(data.get("dynamic_range", 0.0)),
            key=str(data.get("key", "C")),
            mood=str(data.get("mood", "calm")),
            bass=float(data.get("bass", 0.0)),
            mid=float(data.get("mid", 0.0)),
            treble=float(data.get("treble", 0.0)),
            beat=float(data.get("beat", 0.0)),
            section=str(data.get("section", "ambient")),
            tempo=float(data.get("tempo", 0.0)),
            timestamp=float(data.get("timestamp", time.time())),
        )


# ---------------------------------------------------------------------------
# Audio analyzer
# ---------------------------------------------------------------------------


class AudioAnalyzer:
    """Compute a MusicDNA fingerprint from audio state and optional FFT data.

    Works with:
      - duck-typed audio objects (bass/mid/treble/beat/energy)
      - raw numpy FFT frames (``fft_frame: np.ndarray``)
      - optional previous DNA for smoothing
    """

    def __init__(self, smoothing: float = 0.85):
        self.smoothing = float(smoothing)
        self._prev_bpm: float = 0.0
        self._prev_energy: float = 0.0
        self._beat_history: list[float] = []
        self._energy_history: list[float] = []
        self._fft_history: list[Any] = []
        self._max_history: int = 128
        self._last_beat_time: float = 0.0
        self._beat_times: list[float] = []
        self._max_beat_times: int = 32

    # -- public API ----------------------------------------------------------

    def analyze(self, audio: Any, fft_frame: Any | None = None) -> MusicDNA:
        """Return a fresh MusicDNA for the current audio frame."""
        bass = _safe_float(getattr(audio, "bass", getattr(audio, "audio", {}).get("bass", 0.0)))
        mid = _safe_float(getattr(audio, "mid", getattr(audio, "audio", {}).get("mid", 0.0)))
        treble = _safe_float(
            getattr(audio, "treble", getattr(audio, "audio", {}).get("treb", 0.0))
        )
        beat = _safe_float(getattr(audio, "beat", 0.0))
        energy = _safe_float(getattr(audio, "energy", bass * 0.5 + mid * 0.3 + treble * 0.2))

        energy = _clamp01(energy)
        bass = _clamp01(bass)
        mid = _clamp01(mid)
        treble = _clamp01(treble)
        beat = _clamp01(beat)

        now = time.time()

        # BPM from beat intervals
        bpm = self._estimate_bpm(beat, now)

        # Spectral features from FFT
        spectral_centroid = (
            self._spectral_centroid_from_fft(fft_frame) if fft_frame is not None else mid * 0.8
        )
        harmonic_complexity = (
            self._harmonic_complexity_from_fft(fft_frame)
            if fft_frame is not None
            else _clamp01(bass * 0.5 + treble * 0.5)
        )

        # Dynamic range from energy history
        dynamic_range = self._dynamic_range()

        # Beat strength
        beat_strength = self._beat_strength(beat, now)

        # Danceability
        danceability = self._danceability(bpm, beat, energy, bass)

        # Key detection
        key = self._detect_key(fft_frame) if fft_frame is not None else "C"

        # Mood
        mood = self._detect_mood(energy, bass, mid, treble, beat, bpm)

        # Section
        if energy > 1.2:
            section = "drop"
        elif energy > 0.7:
            section = "buildup"
        elif energy > 0.3:
            section = "verse"
        else:
            section = "ambient"

        # Smooth
        self._prev_bpm = _clamp(
            self._prev_bpm * self.smoothing + bpm * (1.0 - self.smoothing), 0.0, 200.0
        )
        self._prev_energy = _clamp01(
            self._prev_energy * self.smoothing + energy * (1.0 - self.smoothing)
        )

        return MusicDNA(
            bpm=self._prev_bpm,
            energy=self._prev_energy,
            danceability=_clamp01(danceability),
            spectral_centroid=_clamp01(spectral_centroid),
            beat_strength=_clamp01(beat_strength),
            harmonic_complexity=_clamp01(harmonic_complexity),
            dynamic_range=_clamp01(dynamic_range),
            key=key,
            mood=mood,
            bass=bass,
            mid=mid,
            treble=treble,
            beat=beat,
            section=section,
            tempo=self._prev_bpm,
        )

    # -- private feature extractors ------------------------------------------

    def _estimate_bpm(self, beat: float, now: float) -> float:
        if beat > 0.6 and (now - self._last_beat_time) > 0.2:
            interval = now - self._last_beat_time
            if 0.2 < interval < 2.0:
                self._beat_times.append(now)
                if len(self._beat_times) > self._max_beat_times:
                    self._beat_times.pop(0)
            self._last_beat_time = now

        if len(self._beat_times) < 3:
            return self._prev_bpm

        intervals = [
            self._beat_times[i + 1] - self._beat_times[i] for i in range(len(self._beat_times) - 1)
        ]
        if not intervals:
            return self._prev_bpm

        median_interval = sorted(intervals)[len(intervals) // 2]
        if median_interval <= 0:
            return self._prev_bpm

        bpm = 60.0 / median_interval
        return _clamp(bpm, 40.0, 200.0)

    def _spectral_centroid_from_fft(self, fft_frame: Any) -> float:
        try:
            import numpy as np

            arr = np.asarray(fft_frame, dtype=np.float32)
            if arr.ndim == 0 or arr.size < 2:
                return 0.5
            mag = np.abs(arr[: arr.size // 2])
            if mag.sum() < 1e-10:
                return 0.5
            freqs = np.arange(mag.size)
            centroid = (mag * freqs).sum() / (mag.sum() * freqs.size)
            return _clamp01(centroid * 2.0)
        except (TypeError, ValueError):
            return 0.5

    def _harmonic_complexity_from_fft(self, fft_frame: Any) -> float:
        try:
            import numpy as np

            arr = np.asarray(fft_frame, dtype=np.float32)
            if arr.ndim == 0 or arr.size < 8:
                return 0.3
            mag = np.abs(arr[: arr.size // 2])
            if mag.sum() < 1e-10:
                return 0.3
            hist, _ = np.histogram(mag, bins=8)
            hist = hist / max(hist.sum(), 1e-10)
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            return _clamp01(entropy / 3.0)
        except (TypeError, ValueError):
            return 0.3

    def _dynamic_range(self) -> float:
        if len(self._energy_history) < 8:
            return 0.5
        recent = self._energy_history[-32:]
        if not recent:
            return 0.5
        return _clamp01((max(recent) - min(recent)) / max(max(recent), 0.01))

    def _beat_strength(self, beat: float, now: float) -> float:
        self._beat_history.append(beat)
        if len(self._beat_history) > self._max_history:
            self._beat_history.pop(0)
        if not self._beat_history:
            return 0.0
        avg = sum(self._beat_history) / len(self._beat_history)
        return _clamp01(avg * 1.5)

    def _danceability(self, bpm: float, beat: float, energy: float, bass: float) -> float:
        if bpm < 1.0:
            return _clamp01(beat * 0.5 + bass * 0.3)
        tempo_fit = 1.0 - min(abs(bpm - 128.0) / 80.0, 1.0)
        return _clamp01(0.4 * tempo_fit + 0.3 * beat + 0.2 * energy + 0.1 * bass)

    def _detect_key(self, fft_frame: Any) -> str:
        try:
            import numpy as np

            arr = np.asarray(fft_frame, dtype=np.float32)
            if arr.ndim == 0 or arr.size < 12:
                return "C"
            mag = np.abs(arr[: max(12, arr.size // 4)])
            idx = int(np.argmax(mag[:12])) % 12
            return KEYS[idx]
        except (TypeError, ValueError):
            return "C"

    def _detect_mood(
        self,
        energy: float,
        bass: float,
        mid: float,
        treble: float,
        beat: float,
        bpm: float,
    ) -> str:
        if energy < 0.25 and bass < 0.3:
            return "calm"
        if energy > 0.8 and bass > 0.7 and beat > 0.6:
            return "aggressive"
        if energy > 0.6 and bpm > 120 and beat > 0.5:
            return "euphoric"
        if treble > 0.6 and energy < 0.4:
            return "mysterious"
        if energy < 0.35 and bass > 0.4:
            return "melancholic"
        if energy > 0.5 and beat > 0.4:
            return "energetic"
        return "calm"

    def update(self, audio: Any, fft_frame: Any | None = None) -> MusicDNA:
        """Process one frame of audio and return the updated MusicDNA.

        Also updates internal history buffers for tempo/energy tracking.
        """
        energy = _safe_float(getattr(audio, "energy", 0.0))
        self._energy_history.append(energy)
        if len(self._energy_history) > self._max_history:
            self._energy_history.pop(0)

        return self.analyze(audio, fft_frame=fft_frame)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def analyze_audio(audio: Any, fft_frame: Any | None = None, smoothing: float = 0.85) -> MusicDNA:
    """One-shot audio analysis."""
    analyzer = AudioAnalyzer(smoothing=smoothing)
    return analyzer.analyze(audio, fft_frame=fft_frame)
