"""evo_bio.music_pressure

Natural selection driven by the music being played (point 13).

Instead of music only nudging uniforms, it now shapes *who survives*:

    Bass-heavy  -> rewards vortex / spirals / curl / mandelbrot / sdf
    Ambient     -> rewards slow feedback / diffusion / soft palettes
    Metal       -> rewards sharp edges / fractals / rapid motion
    Energetic   -> rewards bloom / particles / high speed
    Melancholic -> rewards deep colour / slow movement

The population gradually adapts to whatever is on the turntable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class MusicSignature:
    """Fingerprint of a music section for pattern matching."""

    bass: float = 0.0
    mid: float = 0.0
    treble: float = 0.0
    energy: float = 0.0
    beat: float = 0.0
    tempo: float = 0.0
    section: str = "ambient"
    bpm: float = 0.0
    danceability: float = 0.0
    spectral_centroid: float = 0.0
    beat_strength: float = 0.0
    harmonic_complexity: float = 0.0
    dynamic_range: float = 0.0
    key: str = "C"
    mood: str = "calm"

    @classmethod
    def from_audio(cls, audio: dict[str, float]) -> MusicSignature:
        bass = _clamp01(float(audio.get("bass", 0.0)))
        mid = _clamp01(float(audio.get("mid", 0.0)))
        treble = _clamp01(float(audio.get("treb", audio.get("treble", 0.0))))
        energy = _clamp01(float(audio.get("energy", bass * 0.5 + mid * 0.3 + treble * 0.2)))
        beat = _clamp01(float(audio.get("beat", 0.0)))
        tempo = float(audio.get("tempo", 0.0))
        if energy > 1.2:
            section = "drop"
        elif energy > 0.7:
            section = "buildup"
        elif energy > 0.3:
            section = "verse"
        else:
            section = "ambient"
        return cls(
            bass=bass,
            mid=mid,
            treble=treble,
            energy=energy,
            beat=beat,
            tempo=tempo,
            section=section,
            mood=cls._detect_mood(energy, bass, mid, treble, beat),
            bpm=cls._estimate_bpm(beat, tempo, audio),
            spectral_centroid=_clamp01(
                float(audio.get("spectral_centroid", treble * 0.6 + mid * 0.4))
            ),
            beat_strength=_clamp01(float(audio.get("beat_strength", beat))),
            harmonic_complexity=_clamp01(float(audio.get("harmonic_complexity", 0.3))),
            dynamic_range=_clamp01(float(audio.get("dynamic_range", 0.5))),
        )

    # ------------------------------------------------------------------
    # Mood inference (lightweight, from frequency bands — no FFT needed)
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_mood(energy: float, bass: float, mid: float, treble: float, beat: float) -> str:
        """Infer a qualitative mood from smoothed audio bands."""
        if energy < 0.25 and bass < 0.3:
            return "calm"
        if energy > 0.8 and bass > 0.7 and beat > 0.6:
            return "aggressive"
        if energy > 0.6 and beat > 0.5:
            return "euphoric"
        if treble > 0.6 and energy < 0.4:
            return "mysterious"
        if energy < 0.35 and bass > 0.4:
            return "melancholic"
        if energy > 0.5 and beat > 0.4:
            return "energetic"
        return "calm"

    @staticmethod
    def _estimate_bpm(beat: float, tempo: float, audio: dict[str, float]) -> float:
        """Estimate BPM from explicit tempo, audio field, or beat interval hint."""
        t = float(audio.get("bpm", tempo))
        if t < 1.0:
            # Fall back to beat-based estimate if beat is active.
            t = 60.0 if beat > 0.5 else 0.0
        return _clamp(t, 0.0, 300.0)

    # ------------------------------------------------------------------
    # Compatibility helpers (also used by EvolutionMemory pattern matching)
    # ------------------------------------------------------------------
    @property
    def bass_profile(self) -> float:
        return self.bass

    @property
    def mid_profile(self) -> float:
        return self.mid

    @property
    def treble_profile(self) -> float:
        return self.treble

    def feature_vector(self) -> list[float]:
        """Ordered numeric vector for cross-signature similarity comparison."""
        return [
            self.bass,
            self.mid,
            self.treble,
            self.energy,
            self.beat,
            self.tempo,
            self.bpm / 300.0,
            self.danceability,
            self.spectral_centroid,
            self.beat_strength,
            self.harmonic_complexity,
            self.dynamic_range,
        ]

    def similarity(self, other: Any) -> float:
        """Cosine similarity of feature vectors with another signature."""
        try:
            a = self.feature_vector()
            b = (
                other.feature_vector()
                if hasattr(other, "feature_vector")
                else MusicSignature.from_dict(getattr(other, "to_dict", dict)()).feature_vector()
            )
        except Exception:
            return 0.0
        n = min(len(a), len(b))
        if n == 0:
            return 0.0
        a, b = a[:n], b[:n]
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a)) or 1.0
        mag_b = math.sqrt(sum(y * y for y in b)) or 1.0
        return max(0.0, dot / (mag_a * mag_b))

    def to_dict(self) -> dict[str, Any]:
        return {
            "bass": self.bass,
            "mid": self.mid,
            "treble": self.treble,
            "energy": self.energy,
            "beat": self.beat,
            "tempo": self.tempo,
            "section": self.section,
            "bpm": self.bpm,
            "danceability": self.danceability,
            "spectral_centroid": self.spectral_centroid,
            "beat_strength": self.beat_strength,
            "harmonic_complexity": self.harmonic_complexity,
            "dynamic_range": self.dynamic_range,
            "key": self.key,
            "mood": self.mood,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MusicSignature:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------
def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


@dataclass
class MusicPressure:
    """Selection pressure weights derived from a music fingerprint."""

    bass: float = 0.0
    mid: float = 0.0
    treble: float = 0.0
    beat: float = 0.0
    energy: float = 0.0
    section: str = "ambient"
    mood: str = "calm"
    bpm: float = 0.0
    danceability: float = 0.0
    harmonic_complexity: float = 0.0
    key: str = "C"
    spectral_centroid: float = 0.0
    dynamic_range: float = 0.0
    pressure_weight: float = 0.3

    @classmethod
    def from_signature(cls, sig: MusicSignature) -> MusicPressure:
        return cls(
            bass=sig.bass,
            mid=sig.mid,
            treble=sig.treble,
            beat=sig.beat,
            energy=sig.energy,
            section=sig.section,
            mood=sig.mood,
            bpm=sig.bpm,
            danceability=sig.danceability,
            harmonic_complexity=sig.harmonic_complexity,
            key=sig.key,
            spectral_centroid=getattr(sig, "spectral_centroid", 0.0),
            dynamic_range=getattr(sig, "dynamic_range", 0.0),
        )

    @classmethod
    def from_audio(cls, audio: dict[str, float], section: str = "ambient") -> MusicPressure:
        sig = MusicSignature.from_audio(audio)
        sig.section = section
        return cls.from_signature(sig)

    def reward_traits(self) -> dict[str, float]:
        """Return a weight dict keyed by trait the current music favours."""
        r: dict[str, float] = {}
        # Bass -> structure / rotation.
        if self.bass > 0.6:
            r.update(
                {
                    "vortex": 0.3,
                    "spiral": 0.25,
                    "curl": 0.2,
                    "mandelbrot": 0.2,
                    "sdf": 0.2,
                    "twist": 0.15,
                }
            )
        # Ambient -> calm diffusion.
        if self.energy < 0.4:
            r.update(
                {
                    "reaction_diffusion": 0.25,
                    "feedback": 0.2,
                    "domain_warp": 0.15,
                    "cosine": 0.15,
                    "plasma": 0.1,
                }
            )
        # Metal -> sharp / fast.
        if self.section in ("drop", "buildup") or self.energy > 1.0:
            r.update(
                {
                    "neon_edge": 0.25,
                    "mandelbrot": 0.2,
                    "fractal_layers": 0.2,
                    "flow_speed": 0.2,
                    "chromatic": 0.15,
                    "bloom": 0.15,
                }
            )
        # Treble -> glow / particles.
        if self.treble > 0.6:
            r.update({"glow": 0.2, "particle_density": 0.2})
        # Beat -> motion.
        if self.beat > 0.5:
            r.update({"flow_speed": 0.15, "feedback": 0.1})
        # Dynamic range -> fractal depth.
        if self.dynamic_range > 0.5:
            r.update({"fractal_layers": 0.15, "spiral": 0.1})
        # Spectral centroid (brightness) -> high-frequency emphasis.
        if self.spectral_centroid > 0.5:
            r.update({"glow": 0.15, "chromatic": 0.1, "bloom": 0.1})
        # Mood-specific pressure.
        if self.mood in ("energetic", "euphoric"):
            r.update({"bloom": 0.2, "flow_speed": 0.2, "particle_density": 0.15})
        if self.mood in ("melancholic", "calm"):
            r.update({"cosine": 0.15, "reaction_diffusion": 0.15, "domain_warp": 0.1})
        if self.mood == "dark":
            r.update({"neon_edge": 0.2, "chromatic": 0.15, "twist": 0.1})
        if self.mood == "euphoric":
            r.update({"chromatic": 0.2, "bloom": 0.2, "glow": 0.15})
        return r

    def pressure_for(self, traits: dict[str, float]) -> float:
        """Score an organism's trait dict against the current pressure."""
        rewards = self.reward_traits()
        score = 0.0
        for trait, val in traits.items():
            if trait in rewards:
                score += rewards[trait] * float(val)
        return min(score, 1.0)

    def preferred_specialisations(self) -> list[str]:
        if self.bass > 0.6:
            return ["geometric", "fractal", "crystalline"]
        if self.energy < 0.4:
            return ["organic", "plasma"]
        if self.section in ("drop", "buildup"):
            return ["geometric", "fractal"]
        return ["organic", "plasma", "crystalline"]

    def summary(self) -> dict[str, float]:
        return {
            "bass": self.bass,
            "mid": self.mid,
            "treble": self.treble,
            "beat": self.beat,
            "energy": self.energy,
            "bpm": self.bpm,
            "danceability": self.danceability,
            "harmonic_complexity": self.harmonic_complexity,
            "spectral_centroid": self.spectral_centroid,
            "dynamic_range": self.dynamic_range,
        }
