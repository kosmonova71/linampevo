"""Rendering support classes: feedback buffer, audio state, fitness evaluator, tracking."""

from __future__ import annotations

import logging
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from OpenGL import GL

from evo_bio.pareto import Objectives
from evo_gpu_pool import FboPool
from evo_visualizer.constants import (
    AUDIO_CLAMP_MAX,
    AUDIO_CLAMP_MIN,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:
    import cairo
except ImportError:
    cairo = None

logger = logging.getLogger(__name__)


# Canonical audio-key aliases accepted by :class:`AudioState`.  Kept at module
# level (rather than as a class attribute) so the dict is shared read-only and
# ruff's RUF012 mutable-class-default warning does not fire.
_AUDIO_KEY_ALIASES: dict[str, str] = {
    "treb": "treb",
    "treble": "treb",
    "bass": "bass",
    "mid": "mid",
    "beat": "beat",
    "beat_phase": "beat_phase",
    "phase": "beat_phase",
}


class FeedbackBuffer:
    """Ping-pong framebuffer for feedback effects with optional temporal history.

    Maintains two textures and framebuffers for iterative feedback rendering.
    Resolution matches the viewport (not a fixed square) so the feedback
    effect stays sharp and the fitness readback is consistent with what is
    displayed.

    If ``history > 0``, a ring buffer of older frames is maintained so the
    shader can sample echoes / delayed reactions (spec #6).
    """

    def __init__(
        self, width: int, height: int | None = None, history: int = 0, pool: FboPool | None = None
    ):
        if height is None:
            height = width
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.res = self.width
        self.history = max(0, int(history))
        # Optional shared FBO pool (texture + framebuffer pooling). When set, the
        # ping-pong pair and history ring draw from one bounded pool of GL
        # objects instead of allocating a fresh pair on every resize.
        self._pool = pool
        self.fbo: list[int | None] = [None, None]
        self.tex: list[int | None] = [None, None]
        self.ping = 0
        self._hist_fbo: list[int | None] = []
        self._hist_tex: list[int | None] = []
        self._hist_idx = 0

    def _alloc_fbo_tex(self) -> tuple[int, int]:
        """Allocate one (fbo, tex) pair, preferring the shared pool."""
        if self._pool is not None:
            return self._pool.acquire_fbo_with_texture(self.width, self.height)
        fbo = GL.glGenFramebuffers(1)
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA16F,
            self.width,
            self.height,
            0,
            GL.GL_RGBA,
            GL.GL_HALF_FLOAT,
            None,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D,
            tex,
            0,
        )
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Feedback FBO incomplete: {hex(status)}")
        return int(fbo), int(tex)

    def _ensure_history(self) -> None:
        n = self.history
        if n == len(self._hist_fbo):
            return
        # Grow the ring (the only allocation path on startup / history growth).
        while len(self._hist_fbo) < n:
            self._hist_fbo.append(None)
            self._hist_tex.append(None)
        for i in range(n):
            if self._hist_fbo[i] is None:
                fbo, tex = self._alloc_fbo_tex()
                self._hist_fbo[i] = fbo
                self._hist_tex[i] = tex

    def _create_textures(self) -> None:
        for i in (0, 1):
            if self.tex[i] is None:
                self.fbo[i], self.tex[i] = self._alloc_fbo_tex()
        self._ensure_history()
        for i in (0, 1):
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo[i])
            GL.glViewport(0, 0, self.width, self.height)
            GL.glClearColor(0.0, 0.0, 0.0, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        for i in range(self.history):
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._hist_fbo[i])
            GL.glViewport(0, 0, self.width, self.height)
            GL.glClearColor(0.0, 0.0, 0.0, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        self.ping = 0
        self._hist_idx = 0

    def ensure_history(self, history: int) -> None:
        """Resize the history ring buffer to ``history`` frames.

        The adaptive memory watchdog calls this to grow / shrink the temporal
        echo history as GPU/process memory pressure changes. Growing allocates
        new FBO+texture pairs (pooled); shrinking releases the surplus pairs
        back to the pool (or deletes them) so the GL objects are reclaimed
        immediately rather than left resident.
        """
        history = max(0, int(history))
        if history == self.history:
            return
        if history < self.history:
            # Release surplus history slots back to the pool / GL.
            for i in range(history, len(self._hist_fbo)):
                if self._hist_fbo[i] is not None:
                    if self._pool is not None:
                        self._pool.release(self._hist_fbo[i], self._hist_tex[i])
                    else:
                        try:
                            GL.glDeleteFramebuffers(1, [self._hist_fbo[i]])
                        except Exception:
                            logger.debug("Suppressed exception in cleanup")
                        if self._hist_tex[i] is not None:
                            try:
                                GL.glDeleteTextures(1, [self._hist_tex[i]])
                            except Exception:
                                logger.debug("Suppressed exception in cleanup")
            self._hist_fbo = self._hist_fbo[:history]
            self._hist_tex = self._hist_tex[:history]
            if self._hist_idx >= history > 0:
                self._hist_idx %= history
        else:
            self.history = history
            self._ensure_history()
        self.history = history

    def init_gl(self) -> None:
        """Initialize OpenGL resources."""
        try:
            self._create_textures()
            logger.info(f"FeedbackBuffer initialized: {self.width}x{self.height}")
        except Exception as e:
            logger.error(f"FeedbackBuffer initialization failed: {e}")
            raise

    def resize(self, width: int, height: int | None = None) -> None:
        if height is None:
            height = width
        width = max(1, int(width))
        height = max(1, int(height))
        if width == self.width and height == self.height and self.tex[0] is not None:
            return
        self.cleanup()
        self.width = width
        self.height = height
        self.res = self.width
        self._create_textures()

    def src_tex(self) -> int:
        return int(self.tex[1 - self.ping] or 0)

    def dst_fbo(self) -> int:
        return int(self.fbo[self.ping] or 0)

    def history_tex(self, offset: int = 0) -> int:
        if offset < 0 or offset >= self.history or not self._hist_tex:
            return 0
        idx = (self._hist_idx - 1 - offset) % self.history
        return int(self._hist_tex[idx] or 0)

    def swap(self, default_fbo: int = 0) -> None:
        # Guard against partially-initialised slots. If either ping-pong
        # FBO is missing (init_gl failed, resize hit the pool trim cap, or
        # the context was lost) the blit below would bind FBO 0 as the read
        # buffer and raise GL_INVALID_FRAME_BUFFER_OPERATION (0x0506).
        if self.fbo[0] is None or self.fbo[1] is None:
            self.ping = 1 - self.ping
            return

        if self.history > 0 and self._hist_fbo:
            src = self.src_tex()
            dst = self._hist_fbo[self._hist_idx]
            if src and dst:
                GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo[1 - self.ping])
                GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, dst)
                GL.glBlitFramebuffer(
                    0,
                    0,
                    self.width,
                    self.height,
                    0,
                    0,
                    self.width,
                    self.height,
                    GL.GL_COLOR_BUFFER_BIT,
                    GL.GL_NEAREST,
                )
            GL.glBindFramebuffer(
                GL.GL_FRAMEBUFFER,
                default_fbo if default_fbo else GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING),
            )
            self._hist_idx = (self._hist_idx + 1) % self.history
        self.ping = 1 - self.ping

    def cleanup(self) -> None:
        """Clean up OpenGL resources with individual error handling.

        Ensures each resource is cleaned up even if others fail, preventing
        partial cleanup that could lead to GPU memory leaks. When a shared
        :class:`FboPool` is in use the GL objects are returned to it (so they
        can be recycled by other passes) instead of being deleted outright.
        """
        cleanup_errors = []

        # Clean up main textures and framebuffers
        for i in (0, 1):
            if self.tex[i] is not None:
                if self._pool is not None:
                    self._pool.release(self.fbo[i], self.tex[i])
                else:
                    if self.fbo[i] is not None:
                        try:
                            GL.glDeleteFramebuffers(1, [self.fbo[i]])
                        except Exception as e:
                            cleanup_errors.append(f"Failed to delete framebuffer {i}: {e}")
                    try:
                        GL.glDeleteTextures(1, [self.tex[i]])
                    except Exception as e:
                        cleanup_errors.append(f"Failed to delete texture {i}: {e}")
                self.tex[i] = None
                self.fbo[i] = None

        # Clean up history textures and framebuffers
        for i in range(len(self._hist_tex)):
            if self._hist_tex[i] is not None:
                if self._pool is not None:
                    self._pool.release(self._hist_fbo[i], self._hist_tex[i])
                else:
                    try:
                        GL.glDeleteTextures(1, [self._hist_tex[i]])
                    except Exception as e:
                        cleanup_errors.append(f"Failed to delete history texture {i}: {e}")
                    try:
                        GL.glDeleteFramebuffers(1, [self._hist_fbo[i]])
                    except Exception as e:
                        cleanup_errors.append(f"Failed to delete history framebuffer {i}: {e}")
                self._hist_tex[i] = None
                self._hist_fbo[i] = None

        # Reset state
        self._hist_idx = 0
        self._hist_fbo = []
        self._hist_tex = []

        if cleanup_errors:
            logger.error(f"FeedbackBuffer cleanup encountered errors: {'; '.join(cleanup_errors)}")
        else:
            logger.debug("FeedbackBuffer cleanup complete")


class AudioState:
    """Tracks audio levels for reactive rendering.

    Maintains bass, mid, treble, and beat frequencies in normalized range.
    The internal storage key is ``"treb"`` (short form) but callers may pass
    either ``"treb"`` or ``"treble"``; both are normalised on input so the
    mixed naming across modules does not leak into downstream consumers.
    """

    __slots__ = ("audio", "raw", "sample_rate")

    def __init__(self):
        """Initialize audio state with zero levels."""
        self.audio: dict[str, float] = {
            "bass": 0.0,
            "mid": 0.0,
            "treb": 0.0,
            "beat": 0.0,
            "beat_phase": 0.0,
        }
        self.raw: Any | None = None
        self.sample_rate: float = 44100.0

    def update(self, audio_levels: dict[str, float]) -> None:
        """Update audio levels with clamping.

        Args:
            audio_levels: Dictionary with 'bass', 'mid', 'treb'/'treble',
                'beat' keys (either spelling of the treble band is accepted).
        """
        for raw_key, val in audio_levels.items():
            key = _AUDIO_KEY_ALIASES.get(str(raw_key).lower())
            if key is None:
                continue
            self.audio[key] = max(AUDIO_CLAMP_MIN, min(AUDIO_CLAMP_MAX, float(val)))

    def update_with_audio(
        self,
        audio_levels: dict[str, float],
        fft_samples: Any | None = None,
        sample_rate: float = 44100.0,
    ) -> None:
        """Update audio levels and optionally carry raw FFT samples.

        Args:
            audio_levels: Dict with 'bass', 'mid', 'treb'/'treble', 'beat'.
            fft_samples: Optional numpy array of raw FFT / waveform data for
                deeper audio DNA analysis.
            sample_rate: Sampling rate of the raw audio (Hz).
        """
        self.update(audio_levels)
        self.raw = fft_samples
        self.sample_rate = float(sample_rate)

    @property
    def bass(self) -> float:
        return self.audio["bass"]

    @property
    def mid(self) -> float:
        return self.audio["mid"]

    @property
    def treb(self) -> float:
        return self.audio["treb"]

    @property
    def treble(self) -> float:
        return self.audio["treb"]

    @property
    def beat(self) -> float:
        return self.audio["beat"]

    def uvec(self) -> tuple[float, float, float, float]:
        """Get audio levels as tuple for shader uniforms.

        Returns:
            (bass, mid, treble, beat)
        """
        return (
            self.audio["bass"],
            self.audio["mid"],
            self.audio["treb"],
            self.audio["beat"],
        )

    @property
    def energy(self) -> float:
        """Return the overall audio energy as a weighted sum of bands."""
        return self.audio["bass"] * 0.5 + self.audio["mid"] * 0.3 + self.audio["treb"] * 0.2


class FitnessEvaluator:
    """Evaluates genome fitness using visual image analysis."""

    def __init__(
        self, res: int = 64, novelty_window_size: int = 200, novelty_weight: float = 0.15
    ):
        self.res = res
        self._previous_image: bytes | None = None
        self._novelty_window_size = max(1, int(novelty_window_size))
        self._novelty_weight = max(0.0, min(1.0, float(novelty_weight)))
        self._novelty_history: deque = deque(maxlen=self._novelty_window_size)
        self._last_sig_hash: int | None = None
        self._frame_energy_history: deque = deque(maxlen=16)
        self._beat_sync_history: deque = deque(maxlen=32)
        self._prev_beat: float = 0.0
        self._prev_frame_sig: bytes | None = None
        self._temporal_stable_score: float = 1.0
        # Phenotype signatures of recently scored genomes, used to give
        # graph genomes a deterministic novelty axis (no RNG in fitness).
        self._genome_sig_history: deque = deque(maxlen=self._novelty_window_size)

    def _to_gray(self, image_data: bytes) -> NDArray | None:
        if image_data is None or np is None or len(image_data) < self.res * self.res * 4:
            return None
        arr = np.frombuffer(image_data, dtype=np.uint8)
        arr = arr.reshape(self.res, self.res, 4)
        rgb = arr[:, :, :3].astype(np.float32) / 255.0
        return rgb.mean(axis=2)

    def _to_rgb(self, image_data: bytes) -> NDArray | None:
        if image_data is None or np is None or len(image_data) < self.res * self.res * 4:
            return None
        arr = np.frombuffer(image_data, dtype=np.uint8)
        arr = arr.reshape(self.res, self.res, 4)
        return arr[:, :, :3].astype(np.float32) / 255.0

    def _compute_signature(self, image_data: bytes) -> bytes | None:
        if image_data is None or len(image_data) < self.res * self.res * 4:
            return None
        if np is None:
            return None
        try:
            arr = np.frombuffer(image_data, dtype=np.uint8)
            arr = arr.reshape(self.res, self.res, 4)
            rgb = arr[:, :, :3].astype(np.float32) / 255.0
            h_blocks = 8
            w_blocks = 8
            h_per = self.res // h_blocks
            w_per = self.res // w_blocks
            if h_per == 0 or w_per == 0:
                return None
            crop = rgb[: h_blocks * h_per, : w_blocks * w_per, :]
            small = crop.reshape(h_blocks, h_per, w_blocks, w_per, 3).mean(axis=(1, 3))
            return small.astype(np.float32).tobytes()
        except Exception:
            return None

    def _novelty_bonus(self, sig: bytes) -> float:
        if not self._novelty_history or np is None:
            return 0.0
        curr = np.frombuffer(sig, dtype=np.float32).reshape(8, 8, 3)
        min_dist = float("inf")
        for past_sig in self._novelty_history:
            past = np.frombuffer(past_sig, dtype=np.float32).reshape(8, 8, 3)
            dist = float(np.abs(curr - past).mean())
            min_dist = min(min_dist, dist)
        return min(1.0, max(0.0, min_dist / 0.3))

    def _eval_entropy(self, gray: NDArray) -> float:
        hist, _ = np.histogram(gray, bins=16, range=(0.0, 1.0))
        hist = hist / max(hist.sum(), 1)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return min(entropy / 4.0, 1.0)

    def _eval_edges(self, gray: NDArray) -> float:
        dx = np.abs(gray[:, 2:] - gray[:, :-2])
        dy = np.abs(gray[2:, :] - gray[:-2, :])
        edges = (dx.mean() + dy.mean()) / 2.0
        return min(edges * 10.0, 1.0)

    def _eval_colour(self, rgb: NDArray) -> float:
        colour_var = rgb.var(axis=(0, 1)).mean()
        return min(colour_var * 20.0, 1.0)

    def _eval_motion(self, gray: NDArray, prev_gray: NDArray | None) -> float:
        if prev_gray is None:
            return 0.0
        motion = np.abs(gray - prev_gray).mean()
        return min(motion * 15.0, 1.0)

    def _eval_symmetry(self, gray: NDArray, genome: Any) -> float:
        params = getattr(genome, "params", None)
        sym = int(params.get("symmetry", 0)) if isinstance(params, dict) else 0
        if sym > 1:
            half = gray[:, : self.res // 2]
            mirror_half = np.fliplr(half)
            symmetry_score = 1.0 - np.abs(half - mirror_half).mean()
            return symmetry_score
        return 0.0

    def _eval_audio_sync(
        self, gray: NDArray, prev_gray: NDArray | None, audio_data: dict[str, float] | None
    ) -> float:
        if audio_data is None or prev_gray is None:
            return 0.0
        beat = float(audio_data.get("beat", 0.0))
        bass = float(audio_data.get("bass", 0.0))
        mid = float(audio_data.get("mid", 0.0))
        treb = float(audio_data.get("treb", 0.0))
        audio_energy = (bass + mid + treb) / 3.0
        visual_change = np.abs(gray - prev_gray).mean()
        beat_delta = abs(beat - self._prev_beat)
        self._prev_beat = beat
        if beat_delta < 0.01 and audio_energy < 0.05:
            return 0.0
        sync_score = visual_change * (beat + audio_energy * 0.5)
        self._beat_sync_history.append(sync_score)
        consistency = 0.0
        if len(self._beat_sync_history) > 4:
            arr = np.array(list(self._beat_sync_history), dtype=np.float32)
            consistency = float(1.0 - np.std(arr) / max(np.mean(arr), 1e-6))
        return min(1.0, sync_score * 3.0 + consistency * 0.3)

    def _eval_species_diversity(self, ecosystem: Any | None) -> float:
        if ecosystem is None:
            return 0.0
        try:
            species = getattr(ecosystem, "species", None)
            if not species:
                return 0.0
            counts = []
            for sp in species:
                cnt = getattr(sp, "population", 0) or getattr(sp, "count", 0)
                if isinstance(cnt, list):
                    cnt = len(cnt)
                counts.append(max(1, int(cnt)))
            if not counts:
                return 0.0
            total = sum(counts)
            probs = np.array(counts, dtype=np.float32) / max(total, 1)
            shannon = float(-np.sum(probs * np.log2(probs + 1e-10)))
            max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
            return min(1.0, shannon / max(max_entropy, 1e-6))
        except Exception:
            return 0.0

    def _eval_temporal_stability(
        self, gray: NDArray, previous_frames: list[bytes] | None
    ) -> float:
        if not previous_frames or len(previous_frames) < 2:
            sig = self._compute_signature(gray)
            if sig is None:
                return 1.0
            if self._prev_frame_sig is None:
                self._prev_frame_sig = sig
                return 1.0
            curr = np.frombuffer(sig, dtype=np.float32).reshape(8, 8, 3)
            prev = np.frombuffer(self._prev_frame_sig, dtype=np.float32).reshape(8, 8, 3)
            dist = float(np.abs(curr - prev).mean())
            self._prev_frame_sig = sig
            self._temporal_stable_score = max(0.0, 1.0 - dist / 0.5)
            return self._temporal_stable_score
        sigs = []
        for frame in previous_frames[-4:]:
            # ``frame`` is raw RGBA bytes from the readback path; rebuild the
            # 8x8x3 signature the same way _compute_signature does for the
            # live image so the two are directly comparable.
            s = self._compute_signature(frame)
            if s:
                sigs.append(s)
        if len(sigs) < 2:
            return 1.0
        diffs = []
        for i in range(len(sigs) - 1):
            a = np.frombuffer(sigs[i], dtype=np.float32).reshape(8, 8, 3)
            b = np.frombuffer(sigs[i + 1], dtype=np.float32).reshape(8, 8, 3)
            diffs.append(float(np.abs(a - b).mean()))
        if not diffs:
            return 1.0
        mean_diff = np.mean(diffs)
        self._temporal_stable_score = max(0.0, min(1.0, 1.0 - mean_diff / 0.4))
        return self._temporal_stable_score

    def _eval_camera_movement(self, genome: Any, gray: NDArray) -> float:
        params = getattr(genome, "params", None)
        if not isinstance(params, dict):
            return 0.0
        cam_speed = float(params.get("camera_speed", 0.0))
        cam_orbit = float(params.get("camera_orbit", 0.0))
        cam_roll = float(params.get("camera_roll", 0.0))
        cam_fov = float(params.get("camera_fov", 0.0))
        cam_shake = float(params.get("camera_shake", 0.0))
        cam_zoom = float(params.get("camera_depth", 3.0))
        movement_intensity = (
            abs(cam_speed)
            + abs(cam_orbit)
            + abs(cam_roll)
            + abs(cam_fov - 1.5)
            + cam_shake
            + abs(cam_zoom - 3.0)
        ) / 6.0
        edges = self._eval_edges(gray)
        return min(1.0, movement_intensity * 0.5 + edges * 0.3)

    def _eval_flow_coherence(self, gray: NDArray) -> float:
        dx = np.abs(gray[:, 2:] - gray[:, :-2])
        dy = np.abs(gray[2:, :] - gray[:-2, :])
        grad_mag = np.sqrt(dx[1:-1, :] ** 2 + dy[:, 1:-1] ** 2 + 1e-10)
        coherence = 1.0 - np.std(grad_mag) / max(np.mean(grad_mag), 1e-6)
        return float(np.clip(coherence, 0.0, 1.0))

    def _eval_depth_complexity(self, rgb: NDArray) -> float:
        small = rgb[::2, ::2, :]
        h, w = small.shape[:2]
        if h < 4 or w < 4:
            return 0.0
        blocks = []
        bh, bw = h // 4, w // 4
        for i in range(4):
            for j in range(4):
                block = small[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw, :]
                blocks.append(block.mean(axis=(0, 1)))
        block_std = np.std(np.array(blocks, dtype=np.float32), axis=0)
        depth_score = float(block_std.mean()) * 8.0
        return min(1.0, depth_score)

    def _eval_motion_vectors(self, gray: NDArray, prev_gray: NDArray | None) -> float:
        if prev_gray is None or np is None:
            return 0.0
        block_size = 8
        h, w = gray.shape
        bh, bw = h // block_size, w // block_size
        if bh < 2 or bw < 2:
            return 0.0
        vectors = []
        magnitudes = []
        for i in range(bh):
            for j in range(bw):
                y0, y1 = i * block_size, (i + 1) * block_size
                x0, x1 = j * block_size, (j + 1) * block_size
                curr_block = gray[y0:y1, x0:x1].mean()
                best_dx, best_dy, best_diff = 0, 0, float("inf")
                for dy in range(-4, 5, 2):
                    for dx in range(-4, 5, 2):
                        ny0, ny1 = y0 + dy, y1 + dy
                        nx0, nx1 = x0 + dx, x1 + dx
                        if 0 <= ny0 and ny1 < h and 0 <= nx0 and nx1 < w:
                            prev_block = prev_gray[ny0:ny1, nx0:nx1].mean()
                            diff = abs(curr_block - prev_block)
                            if diff < best_diff:
                                best_diff = diff
                                best_dx, best_dy = dx, dy
                vectors.append((best_dx, best_dy))
                magnitudes.append(math.hypot(best_dx, best_dy))
        if not magnitudes:
            return 0.0
        mag_arr = np.array(magnitudes, dtype=np.float32)
        avg_mag = float(mag_arr.mean())
        diversity = float(np.std(mag_arr)) / max(avg_mag, 1e-6)
        score = min(1.0, avg_mag * 0.4 + diversity * 0.3)
        return score

    def _eval_visual_rhythm(self, gray: NDArray) -> float:
        _h, _w = gray.shape
        row_means = gray.mean(axis=1)
        col_means = gray.mean(axis=0)

        def periodicity(signal: NDArray) -> float:
            n = len(signal)
            if n < 8:
                return 0.0
            ac = np.correlate(signal - signal.mean(), signal - signal.mean(), mode="full")[n - 1 :]
            ac = ac / max(ac[0], 1e-10)
            peaks = 0
            for i in range(2, len(ac) - 1):
                if ac[i] > ac[i - 1] and ac[i] > ac[i + 1] and ac[i] > 0.1:
                    peaks += 1
            return min(1.0, peaks * 0.2)

        row_score = periodicity(row_means)
        col_score = periodicity(col_means)
        return (row_score + col_score) * 0.5

    def _eval_silhouette_quality(self, gray: NDArray) -> float:
        dx = np.abs(gray[:, 2:] - gray[:, :-2])
        dy = np.abs(gray[2:, :] - gray[:-2, :])
        edge_mag = np.sqrt(dx[1:-1, :] ** 2 + dy[:, 1:-1] ** 2 + 1e-10)
        threshold = 0.05
        strong = edge_mag > threshold
        if not np.any(strong):
            return 0.0
        strong_ratio = float(strong.mean())
        continuity = 0.0
        if strong_ratio > 0.001:
            row_strong = strong.mean(axis=1)
            col_strong = strong.mean(axis=0)
            continuity = (float(np.std(row_strong)) + float(np.std(col_strong))) * 2.0
        return float(np.clip(strong_ratio * 8.0 + continuity, 0.0, 1.0))

    def _eval_frequency_spectrum(self, audio_data: dict[str, float] | None) -> float:
        if audio_data is None:
            return 0.0
        bass = float(audio_data.get("bass", 0.0))
        mid = float(audio_data.get("mid", 0.0))
        treb = float(audio_data.get("treb", 0.0))
        components = np.array([bass, mid, treb], dtype=np.float32)
        spread = float(np.std(components))
        richness = min(1.0, spread * 3.0)
        energy = min(1.0, float(components.mean()) * 1.5)
        return (richness + energy) * 0.5

    def _eval_optical_flow(self, gray: NDArray, prev_gray: NDArray | None) -> float:
        if prev_gray is None or np is None:
            return 0.0
        h, w = gray.shape
        step = 4
        flow_vectors = []
        for y in range(step, h - step, step):
            for x in range(step, w - step, step):
                window = prev_gray[y - step : y + step, x - step : x + step]
                if window.size == 0:
                    continue
                center_val = gray[y, x]
                best_dx, best_dy, best_err = 0, 0, float("inf")
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            err = abs(center_val - prev_gray[ny, nx])
                            if err < best_err:
                                best_err = err
                                best_dx, best_dy = dx, dy
                flow_vectors.append((best_dx, best_dy))
        if len(flow_vectors) < 10:
            return 0.0
        mags = [math.hypot(dx, dy) for dx, dy in flow_vectors]
        mag_arr = np.array(mags, dtype=np.float32)
        avg_mag = float(mag_arr.mean())
        coherence = 1.0 - float(np.std(mag_arr)) / max(avg_mag, 1e-6)
        return float(np.clip(avg_mag * 0.5 + coherence * 0.4, 0.0, 1.0))

    def evaluate(
        self,
        genome: Any,
        image_data: bytes,
        previous_data: bytes | None = None,
        audio_data: dict[str, float] | None = None,
        previous_frames: list[bytes] | None = None,
        ecosystem: Any | None = None,
    ) -> Objectives:
        """Evaluate multi-objective fitness from image analysis.

        Returns
        -------
        Objectives
            Eight-axis objective vector (beauty, novelty, audio_sync,
            performance, species_health, motion, complexity, memory).
        """

        if image_data is None or np is None or len(image_data) < self.res * self.res * 4:
            return Objectives()

        try:
            rgb = self._to_rgb(image_data)
            gray = self._to_gray(image_data)
            if rgb is None or gray is None:
                return Objectives()

            prev_gray = self._to_gray(previous_data) if previous_data else None

            entropy_score = self._eval_entropy(gray)
            edge_score = self._eval_edges(gray)
            colour_score = self._eval_colour(rgb)
            motion_score = self._eval_motion(gray, prev_gray)
            sym_score = self._eval_symmetry(gray, genome)
            brightness = float(rgb.mean())

            audio_sync = self._eval_audio_sync(gray, prev_gray, audio_data)

            sig = self._compute_signature(image_data)
            novelty = 0.0
            if sig is not None:
                novelty = self._novelty_bonus(sig)
                sig_hash = hash(sig)
                if sig_hash != self._last_sig_hash:
                    self._novelty_history.append(sig)
                    self._last_sig_hash = sig_hash

            species_div = self._eval_species_diversity(ecosystem)
            temporal_stab = self._eval_temporal_stability(gray, previous_frames)
            cam_movement = self._eval_camera_movement(genome, gray)
            flow_coh = self._eval_flow_coherence(gray)
            depth_comp = self._eval_depth_complexity(rgb)
            motion_vec = self._eval_motion_vectors(gray, prev_gray)
            vis_rhythm = self._eval_visual_rhythm(gray)
            sil_qual = self._eval_silhouette_quality(gray)
            self._eval_frequency_spectrum(audio_data)
            opt_flow = self._eval_optical_flow(gray, prev_gray)

            beauty = min(
                1.0,
                entropy_score * 0.20
                + edge_score * 0.15
                + colour_score * 0.12
                + sym_score * 0.08
                + flow_coh * 0.06
                + depth_comp * 0.05
                + sil_qual * 0.05
                + vis_rhythm * 0.04
                - max(brightness - 0.8, 0.0) * 0.15,
            )

            species_health = min(1.0, species_div * 0.6 + temporal_stab * 0.4)

            motion_total = min(
                1.0,
                motion_score * 0.35 + motion_vec * 0.25 + opt_flow * 0.20 + cam_movement * 0.20,
            )

            complexity_total = min(1.0, (entropy_score + depth_comp + vis_rhythm) / 3.0)

            performance = 1.0
            memory_score = temporal_stab

            return Objectives(
                beauty=beauty,
                novelty=novelty,
                audio_sync=audio_sync,
                performance=performance,
                species_health=species_health,
                motion=motion_total,
                complexity=complexity_total,
                memory=memory_score,
            )

        except Exception as e:
            logger.debug(f"Fitness evaluation failed: {e}")
            return Objectives()

    def _graph_objectives(self, genome: Any, graph: Any) -> Objectives | None:
        """Structural objectives for shader-graph genomes.

        ``score_from_genome``'s legacy branches inspect flat numeric params
        (``warp``, ``feedback``, ...) which graph genomes do not have, so they
        used to return near-constant vectors decided by a random novelty term.
        This scorer derives every axis from the graph itself: node-category
        composition, shader cost metrics, audio-reactive wiring and a
        deterministic novelty measure against recently seen phenotypes.
        """
        nodes = getattr(graph, "nodes", None)
        if not isinstance(nodes, dict) or not nodes:
            return None
        try:
            from evo_shader_graph import NODE_COST_METRICS, NODE_LIBRARY
        except ImportError:
            return None

        ntypes: dict[str, int] = {}
        cats: dict[str, int] = {}
        audio_inputs = 0
        audio_uniform_nodes = 0
        for node in nodes.values():
            ntype = str(getattr(node, "ntype", "?"))
            ntypes[ntype] = ntypes.get(ntype, 0) + 1
            tpl = NODE_LIBRARY.get(ntype, {})
            cat = str(tpl.get("category", "other"))
            cats[cat] = cats.get(cat, 0) + 1
            if cat == "input" and ntype.startswith("input_audio"):
                audio_inputs += 1
            for uniform in tpl.get("uniforms", {}):
                if uniform in ("bass", "mid", "treble", "beat"):
                    audio_uniform_nodes += 1
                    break

        total = max(1, sum(cats.values()))

        def frac(*names: str) -> float:
            return sum(cats.get(c, 0) for c in names) / total

        try:
            metrics = genome.compute_shader_metrics()
        except (AttributeError, TypeError, ValueError):
            metrics = {"instructions": 0, "alu": 0, "texture_reads": 0, "branches": 0}
            for ntype, count in ntypes.items():
                cost = NODE_COST_METRICS.get(ntype, {})
                for key in metrics:
                    metrics[key] += int(cost.get(key, 0)) * count
        instructions = float(metrics.get("instructions", 0))
        branches = float(metrics.get("branches", 0))
        texture_reads = float(metrics.get("texture_reads", 0))

        try:
            fps_cost = float(genome.fps_contribution())
        except (AttributeError, TypeError, ValueError):
            fps_cost = instructions / 8.0

        # Aesthetic richness: colour, post-processing, fields, lighting and
        # feedback loops are what make the output look evolved rather than
        # plain noise.
        beauty = min(
            1.0,
            0.24 * min(1.0, cats.get("palette", 0) / 2.0)
            + 0.20 * min(1.0, cats.get("post", 0) / 3.0)
            + 0.18 * min(1.0, cats.get("field", 0) / 2.0)
            + 0.12 * min(1.0, cats.get("lighting", 0) / 2.0)
            + 0.10 * min(1.0, cats.get("warp", 0) / 2.0)
            + 0.08 * min(1.0, cats.get("composite", 0) / 2.0)
            + 0.08 * min(1.0, cats.get("feedback", 0) / 2.0),
        )

        motion = min(
            1.0,
            0.30 * min(1.0, cats.get("warp", 0) / 2.0)
            + 0.25 * min(1.0, cats.get("camera", 0))
            + 0.20 * min(1.0, cats.get("particle", 0))
            + 0.15 * min(1.0, cats.get("feedback", 0) / 2.0)
            + 0.10 * min(1.0, cats.get("sample", 0)),
        )

        complexity = min(
            1.0,
            instructions / 300.0 + branches / 40.0 + len(ntypes) / 60.0 + texture_reads / 30.0,
        )

        # Lower GPU cost -> higher performance score.
        performance = 1.0 / (1.0 + fps_cost / 20.0)

        audio_sync = min(1.0, 0.30 * audio_inputs + 0.12 * audio_uniform_nodes)

        # Deterministic novelty: histogram distance to the most similar
        # recently-seen phenotype signature.
        sig = tuple(sorted(ntypes.items()))
        novelty = 0.0
        best_sim = 0.0
        for past in self._genome_sig_history:
            keys = set(dict(sig)) | set(dict(past))
            inter = sum(min(dict(sig).get(k, 0), dict(past).get(k, 0)) for k in keys)
            union = sum(max(dict(sig).get(k, 0), dict(past).get(k, 0)) for k in keys)
            sim = inter / max(union, 1)
            best_sim = max(best_sim, sim)
        if self._genome_sig_history:
            novelty = min(1.0, max(0.0, 1.0 - best_sim))
        self._genome_sig_history.append(sig)

        return Objectives(
            beauty=beauty,
            novelty=novelty,
            audio_sync=audio_sync,
            performance=performance,
            species_health=0.5,
            motion=motion,
            complexity=complexity,
            memory=0.5,
        )

    def score_from_genome(
        self, genome: Any, weights: dict[str, float] | None = None
    ) -> Objectives:
        """Fallback objectives when image fitness is unavailable."""

        graph = getattr(genome, "graph", None)
        if graph is not None:
            objectives = self._graph_objectives(genome, graph)
            if objectives is not None:
                return objectives

        if hasattr(genome, "params") and isinstance(genome.params, dict):
            p = genome.params
            beauty = min(
                1.0,
                p.get("warp", 0.0) * 0.35
                + p.get("noise_scale", 0.0) * 0.15
                + p.get("feedback", 0.0) * 0.15
                + p.get("glow", 0.0) * 0.10
                + p.get("symmetry", 0) * 0.10
                + p.get("contrast", 1.0) * 0.10
                + p.get("turbulence", 0.0) * 0.05,
            )
            novelty = min(
                1.0,
                p.get("turbulence", 0.0) * 0.5
                + p.get("noise_scale", 0.0) * 0.3
                + abs(p.get("vortex", 0.0)) * 0.2,
            )
            motion = min(
                1.0,
                abs(p.get("vortex", 0.0)) * 0.3
                + p.get("flow_speed", 1.0) * 0.2
                + abs(p.get("camera_speed", 0.0)) * 0.3
                + abs(p.get("camera_shake", 0.0)) * 0.15
                + abs(p.get("camera_tracking", 0.5) - 0.5) * 0.2,
            )
            complexity = min(
                1.0,
                (p.get("fbm_octaves", 4) / 8.0) * 0.5
                + len(p.get("post", [])) * 0.1
                + p.get("turbulence", 0.0) * 0.2,
            )
            audio_sync = min(1.0, p.get("audio_reactive", 0.5))
            performance = 1.0
            species_health = 0.5
            memory = 0.5
            return Objectives(
                beauty=beauty,
                novelty=novelty,
                audio_sync=audio_sync,
                performance=performance,
                species_health=species_health,
                motion=motion,
                complexity=complexity,
                memory=memory,
            )

        score = 0.0
        if hasattr(genome, "field"):
            field = str(genome.field)
            score += {
                "fbm": 0.35,
                "voronoi": 0.25,
                "reaction_diffusion": 0.45,
                "mandelbrot": 0.5,
                "sdf": 0.3,
            }.get(field, 0.25)
        if hasattr(genome, "warp"):
            warp = str(genome.warp)
            score += {"curl": 0.2, "vortex": 0.15, "twist": 0.1, "none": 0.05}.get(warp, 0.1)
        if hasattr(genome, "palette"):
            score += 0.15
        if hasattr(genome, "post") and isinstance(genome.post, list):
            score += 0.05 * len(set(genome.post))
            score += 0.08 if "glow" in genome.post else 0.0
            score += 0.08 if "bloom" in genome.post else 0.0
            score += 0.05 if "nonlinear_trails" in genome.post else 0.0
            score += 0.05 if "audio_topology" in genome.post else 0.0
            score += 0.05 if "depth_cloud" in genome.post else 0.0
            score += 0.05 if "neon_edge" in genome.post else 0.0
        fb_mode = str(getattr(genome, "feedback_mode", "spiral"))
        score += 0.05 if fb_mode in ("kaleido", "mirror", "zoom") else 0.0

        beauty = min(1.0, max(0.0, score))
        return Objectives(
            beauty=beauty,
            novelty=min(1.0, beauty * 0.5),
            audio_sync=0.5,
            performance=1.0,
            species_health=0.5,
            motion=beauty * 0.5,
            complexity=beauty * 0.7,
            memory=0.5,
        )


class GeneFitnessTracker:
    """Tracks per-parameter fitness impact to bias mutations toward weak genes."""

    def __init__(self, decay: float = 0.92):
        self._delta: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._decay = max(0.8, min(1.0, decay))

    def record(self, param: str, fitness_delta: float) -> None:
        self._delta[param] = self._delta.get(param, 0.0) * self._decay + fitness_delta
        self._counts[param] = self._counts.get(param, 0) + 1

    def weights(self) -> dict[str, float]:
        if not self._delta:
            return {}
        worst = min(self._delta.values())
        span = max(self._delta.values()) - worst
        if span < 1e-6:
            return {k: 1.0 for k in self._delta}
        return {k: max(0.1, 1.0 + (v - worst) / span) for k, v in self._delta.items()}

    def pick_weak_gene(self, params: dict[str, Any]) -> str:
        weights = self.weights()
        keys = list(params.keys())
        if not weights or random.random() < 0.3:
            return random.choice(keys)
        w = [weights.get(k, 0.1) for k in keys]
        return random.choices(keys, weights=w, k=1)[0]


@dataclass
class _HeatRow:
    """Per-parameter mutation statistics for the heatmap."""

    param: str
    fitness_gain: float = 0.0  # EMA of accepted fitness deltas
    times_mutated: int = 0
    accepts: int = 0
    rejects: int = 0


class MutationHeatmap:
    """Records per-parameter mutation activity so evolution becomes explainable.

    For every gene the heatmap keeps four counters that the on-screen overlay
    renders as horizontal bars:

      * ``fitness_gain`` -- exponentially weighted average of the fitness delta
        produced by accepted mutations that touched this parameter. A large
        positive bar means "this gene is where the wins came from".
      * ``times_mutated`` -- how often evolution reached for this parameter.
      * ``acceptance`` -- accept / (accept + reject), i.e. how often touching
        this gene paid off.

    Records are keyed on the genome parameter name. Mutations are logged with a
    short ``param`` token (the same prefix the genome appends to its
    ``mutations`` history), so both the steady-state and the single-organism
    evolution paths can feed the same heatmap.
    """

    def __init__(self, gain_decay: float = 0.9, max_rows: int = 64):
        self._rows: dict[str, _HeatRow] = {}
        self._gain_decay = max(0.8, min(0.99, float(gain_decay)))
        self._max_rows = max(8, int(max_rows))
        self._total_mutated = 0
        self._total_accepted = 0

    def record(self, param: str, delta: float, accepted: bool) -> None:
        """Record one mutation event for ``param``.

        Args:
            param: Genome parameter name (or "unknown" if undeterminable).
            delta: Fitness delta produced by the mutation (0.0 if not measured).
            accepted: Whether the mutation was accepted into the genome.
        """
        if not param:
            param = "unknown"
        row = self._rows.get(param)
        if row is None:
            if len(self._rows) >= self._max_rows:
                return
            row = _HeatRow(param=param)
            self._rows[param] = row
        row.times_mutated += 1
        self._total_mutated += 1
        if accepted:
            row.accepts += 1
            self._total_accepted += 1
            row.fitness_gain = row.fitness_gain * self._gain_decay + float(delta) * (
                1.0 - self._gain_decay
            )
        else:
            row.rejects += 1

    def acceptance(self, param: str) -> float:
        """Return 0..1 acceptance rate for ``param`` (0.0 if never touched)."""
        row = self._rows.get(param)
        if row is None or row.times_mutated == 0:
            return 0.0
        return row.accepts / row.times_mutated

    def snapshot(self) -> list[_HeatRow]:
        """Return rows sorted by times-mutated, descending (most-active first)."""
        return sorted(
            self._rows.values(), key=lambda r: (r.times_mutated, r.accepts), reverse=True
        )

    def totals(self) -> tuple[int, int]:
        """Return (total mutations, total acceptances)."""
        return self._total_mutated, self._total_accepted

    def reset(self) -> None:
        """Clear all recorded statistics."""
        self._rows.clear()
        self._total_mutated = 0
        self._total_accepted = 0

    def merge(self, other: MutationHeatmap) -> None:
        """Fold another heatmap's statistics into this one (used by replay)."""
        if other is None:
            return
        for param, row in other._rows.items():
            mine = self._rows.get(param)
            if mine is None:
                if len(self._rows) >= self._max_rows:
                    continue
                mine = _HeatRow(param=param)
                self._rows[param] = mine
            mine.times_mutated += row.times_mutated
            mine.accepts += row.accepts
            mine.rejects += row.rejects
            mine.fitness_gain += row.fitness_gain
        self._total_mutated += other._total_mutated
        self._total_accepted += other._total_accepted
