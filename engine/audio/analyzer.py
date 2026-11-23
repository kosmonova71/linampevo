#!/usr/bin/env python3
"""
ASCEND V9 - Audio Engine (Upgraded)

Enhanced audio analysis with smooth values, beat detection, energy envelope.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class AudioFeatures:
    """Enhanced audio features."""

    # Raw values
    bass: float
    mid: float
    treb: float

    # Smoothed values
    bass_smooth: float
    mid_smooth: float
    treb_smooth: float

    # Peak values
    bass_peak: float
    mid_peak: float
    treb_peak: float

    # Beat detection
    beat: bool
    beat_strength: float

    # Energy envelope
    energy: float
    energy_smooth: float

    # Spectral features
    spectral_centroid: float
    spectral_bandwidth: float


class AudioAnalyzer:
    """Enhanced audio analyzer for V9.

    Provides:
    - Smoothed frequency bands
    - Beat detection
    - Energy envelope tracking
    - Spectral features
    """

    def __init__(self, sample_rate: int = 44100, fft_size: int = 1024):
        """Initialize audio analyzer.

        Args:
            sample_rate: Audio sample rate
            fft_size: FFT size for analysis
        """
        self.sample_rate = sample_rate
        self.fft_size = fft_size

        # Smoothing factors
        self.smooth_factor = 0.1
        self.peak_decay = 0.95

        # State
        self.bass_smooth = 0.0
        self.mid_smooth = 0.0
        self.treb_smooth = 0.0

        self.bass_peak = 0.0
        self.mid_peak = 0.0
        self.treb_peak = 0.0

        self.energy_smooth = 0.0
        self.prev_energy = 0.0

        # Beat detection
        self.beat_threshold = 0.5
        self.beat_history: list[bool] = []
        self.beat_cooldown = 0.1  # seconds
        self.last_beat_time = 0.0

    def analyze(self, samples: np.ndarray) -> AudioFeatures:
        """Analyze audio samples.

        Args:
            samples: Audio samples (mono or stereo)

        Returns:
            AudioFeatures with enhanced features
        """
        # Convert to mono if stereo
        if len(samples.shape) > 1:
            samples = samples.mean(axis=1)

        # Compute FFT
        fft = np.fft.rfft(samples)
        magnitude = np.abs(fft)

        # Frequency bands
        freqs = np.fft.rfftfreq(len(samples), 1 / self.sample_rate)

        # Split into bands
        bass_mask = freqs < 250
        mid_mask = (freqs >= 250) & (freqs < 2000)
        treb_mask = freqs >= 2000

        bass = magnitude[bass_mask].mean() if bass_mask.any() else 0.0
        mid = magnitude[mid_mask].mean() if mid_mask.any() else 0.0
        treb = magnitude[treb_mask].mean() if treb_mask.any() else 0.0

        # Normalize
        bass = min(bass / 1000.0, 1.0)
        mid = min(mid / 500.0, 1.0)
        treb = min(treb / 200.0, 1.0)

        # Smooth values
        self.bass_smooth = self._smooth(self.bass_smooth, bass)
        self.mid_smooth = self._smooth(self.mid_smooth, mid)
        self.treb_smooth = self._smooth(self.treb_smooth, treb)

        # Peak tracking
        self.bass_peak = max(bass, self.bass_peak * self.peak_decay)
        self.mid_peak = max(mid, self.mid_peak * self.peak_decay)
        self.treb_peak = max(treb, self.treb_peak * self.peak_decay)

        # Energy envelope
        energy = (bass + mid + treb) / 3.0
        self.energy_smooth = self._smooth(self.energy_smooth, energy)

        # Beat detection
        beat = self._detect_beat(bass, time.time())
        beat_strength = self._calculate_beat_strength(bass, self.bass_smooth)

        # Spectral features
        spectral_centroid = self._calculate_spectral_centroid(magnitude, freqs)
        spectral_bandwidth = self._calculate_spectral_bandwidth(
            magnitude, freqs, spectral_centroid
        )

        return AudioFeatures(
            bass=bass,
            mid=mid,
            treb=treb,
            bass_smooth=self.bass_smooth,
            mid_smooth=self.mid_smooth,
            treb_smooth=self.treb_smooth,
            bass_peak=self.bass_peak,
            mid_peak=self.mid_peak,
            treb_peak=self.treb_peak,
            beat=beat,
            beat_strength=beat_strength,
            energy=energy,
            energy_smooth=self.energy_smooth,
            spectral_centroid=spectral_centroid,
            spectral_bandwidth=spectral_bandwidth,
        )

    def _smooth(self, prev: float, current: float) -> float:
        """Apply exponential smoothing.

        Args:
            prev: Previous value
            current: Current value

        Returns:
            Smoothed value
        """
        return prev + self.smooth_factor * (current - prev)

    def _detect_beat(self, bass: float, current_time: float) -> bool:
        """Detect beat from bass signal.

        Args:
            bass: Current bass value
            current_time: Current time in seconds

        Returns:
            True if beat detected
        """
        # Check cooldown
        if current_time - self.last_beat_time < self.beat_cooldown:
            return False

        # Detect beat if bass exceeds threshold and is rising
        if bass > self.beat_threshold and bass > self.bass_smooth * 1.5:
            self.last_beat_time = current_time
            self.beat_history.append(True)
            if len(self.beat_history) > 100:
                self.beat_history.pop(0)
            return True

        return False

    def _calculate_beat_strength(self, bass: float, bass_smooth: float) -> float:
        """Calculate beat strength.

        Args:
            bass: Current bass value
            bass_smooth: Smoothed bass value

        Returns:
            Beat strength (0-1)
        """
        if bass_smooth > 0:
            return min((bass - bass_smooth) / bass_smooth, 1.0)
        return 0.0

    def _calculate_spectral_centroid(self, magnitude: np.ndarray, freqs: np.ndarray) -> float:
        """Calculate spectral centroid (brightness).

        Args:
            magnitude: FFT magnitude
            freqs: Frequencies

        Returns:
            Spectral centroid in Hz
        """
        total = magnitude.sum()
        if total == 0:
            return 0.0
        return (freqs * magnitude).sum() / total

    def _calculate_spectral_bandwidth(
        self, magnitude: np.ndarray, freqs: np.ndarray, centroid: float
    ) -> float:
        """Calculate spectral bandwidth.

        Args:
            magnitude: FFT magnitude
            freqs: Frequencies
            centroid: Spectral centroid

        Returns:
            Spectral bandwidth in Hz
        """
        total = magnitude.sum()
        if total == 0:
            return 0.0
        return np.sqrt(((freqs - centroid) ** 2 * magnitude).sum() / total)

    def get_bpm(self) -> float | None:
        """Estimate BPM from beat history.

        Returns:
            Estimated BPM, or None if insufficient data
        """
        if len(self.beat_history) < 10:
            return None

        # Count beats in recent history
        recent_beats = self.beat_history[-60:]  # Last 60 beats
        beat_count = sum(recent_beats)

        if beat_count < 2:
            return None

        # Estimate BPM (assuming 60 FPS)
        bpm = (beat_count / len(recent_beats)) * 60
        return bpm
