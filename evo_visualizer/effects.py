"""Evolving post-processing effects driven by the ``effects/`` shader folder.

Every ``*.frag`` in the configured directory becomes a candidate full-screen
effect.  One effect is active at a time:

* Its numeric uniforms are *genes* — values in ``[0, 1]`` that drift every
  frame (continuous mutation) and are re-seeded/inherited when the active
  effect changes (selection evolution).
* On a timer the manager picks a new effect from the pool (novelty +
  audio-affinity scoring, recent repeats penalised) and crossfades to it.
  The crossfade itself is rendered by a *transition* shader from the same
  pool (any spec exposing a secondary input texture), falling back to a
  built-in mix shader.

The manager owns three ping-pong RGBA16F targets plus one scene target that
the renderer draws the raw frame into; the final composite is blitted to the
GTK default framebuffer, so effects never feed back into the genome's own
feedback buffer.
"""

from __future__ import annotations

import logging
import math
import random
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from OpenGL import GL

from evo_visualizer.constants import QUAD_VERTEX_COUNT

try:  # Folder texture loading is optional: Pillow + numpy.
    import numpy as np
    from PIL import Image, ImageOps

    _IMAGE_DEPS = True
except Exception:  # pragma: no cover - exercised only without Pillow/numpy
    np = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]
    _IMAGE_DEPS = False

if TYPE_CHECKING:
    from evo_visualizer.config import EvoConfig
    from evo_visualizer.renderer import EvoRenderer

logger = logging.getLogger(__name__)

_TWO_PI = 2.0 * math.pi

# The effects/*.frag files come from two dialects: post shaders declare
# ``in vec2 v_texcoord`` (glitch, motion_blur, kaleidoscope) while the
# shadertoy-style transition shaders declare ``in vec2 v_uv``.  The vertex
# shader below writes *both* so a single program layout satisfies either.
_VERTEX_SRC = """#version 330 core
layout(location = 0) in vec2 position;
out vec2 v_texcoord;
out vec2 v_uv;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    v_texcoord = position * 0.5 + 0.5;
    v_uv = v_texcoord;
}
"""

# Public alias so the renderer can compile effect programs without reaching
# into private names.
VERTEX_SHADER_SRC = _VERTEX_SRC

_MIX_FRAG = """#version 330 core
in vec2 v_texcoord;
out vec4 FragColor;
uniform sampler2D u_texture_from;
uniform sampler2D u_texture_to;
uniform float u_progress;
void main() {
    FragColor = mix(
        texture(u_texture_from, v_texcoord),
        texture(u_texture_to, v_texcoord),
        u_progress
    );
}
"""

_UNIFORM_RE = re.compile(
    r"^\s*uniform\s+(?:lowp\s+|mediump\s+|highp\s+)?"
    r"(sampler2D|float|int|ivec2|ivec3|ivec4|vec2|vec3|vec4|bool)"
    r"\s+([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*;",
    re.MULTILINE,
)
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_IN_VEC2_RE = re.compile(r"\bin\s+vec2\s+(\w+)")
_MAIN_RE = re.compile(r"(void\s+main\s*\([^)]*\)\s*\{)")
_FRAGCOORD_DECL_RE = re.compile(r"^\s*uniform\s+vec2\s+fragCoord\s*;[^\n]*\n?", re.MULTILINE)
_RESOLUTION_DECL_RE = re.compile(r"uniform\s+vec[23]\s+(iResolution|u_resolution)\b")

# Substring -> (lo, hi) mapping for float gene uniforms.  First match wins.
_FLOAT_RANGES: tuple[tuple[str, float, float], ...] = (
    ("angle", 0.0, _TWO_PI),
    ("rotat", 0.0, _TWO_PI),
    ("speed", 0.0, 3.0),
    ("scale", 0.2, 3.0),
    ("zoom", 0.5, 2.5),
    ("size", 0.1, 2.0),
    ("freq", 0.0, 6.0),
    ("threshold", 0.0, 1.0),
    ("intens", 0.0, 1.0),
    ("strength", 0.0, 1.0),
    ("amount", 0.0, 1.0),
    ("distort", 0.0, 1.0),
    ("warp", 0.0, 1.0),
    ("glitch", 0.0, 1.0),
    ("blur", 0.0, 1.0),
    ("radius", 0.0, 1.0),
    ("mix", 0.0, 1.0),
    ("opacity", 0.0, 1.0),
    ("alpha", 0.0, 1.0),
    ("cx", 0.0, 1.0),
    ("cy", 0.0, 1.0),
)

_INT_RANGES: tuple[tuple[str, int, int], ...] = (
    ("segment", 2, 24),
    ("sample", 1, 16),
    ("tap", 1, 16),
    ("octave", 1, 8),
    ("count", 1, 16),
    ("level", 1, 8),
    ("pass", 1, 8),
    ("mip", 0, 8),
)

# Uniform names that react to the music level in addition to their genes.
_AUDIO_PARAM_RE = re.compile(
    r"intens|strength|speed|amount|energy|power|glitch|distort|warp|blur|ampl|pulse"
)

# Name fragments that steer selection toward audio-appropriate effects.
_BASS_AFFINITY_RE = re.compile(r"glitch|distort|warp|noise|static|kaleido|datamosh")
_TREB_AFFINITY_RE = re.compile(r"shimmer|glow|sparkle|star|plasma|kaleido|shine")
_MID_AFFINITY_RE = re.compile(r"wave|fluid|smoke|cloud|ripple|flow")

_VEC_TYPES = {"vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4"}
_INT_TYPES = {"int", "bool"}


def _strip_comments(src: str) -> str:
    """Remove ``//`` and ``/* */`` comments so uniform parsing ignores them."""
    src = _BLOCK_COMMENT_RE.sub("", src)
    return re.sub(r"//[^\n]*", "", src)


def _patch_fragcoord(src: str) -> str:
    """Turn the shadertoy-style ``uniform vec2 fragCoord`` into a per-pixel
    varying computed from the fragment's uv coordinate.

    These transition shaders were written for Shadertoy where ``fragCoord``
    is a built-in per-fragment value; as a plain uniform it would be constant
    across the whole draw.  When a resolution uniform exists we inject
    ``fragCoord = uv * resolution`` inside ``main()`` and drop the uniform
    declaration.  If no resolution uniform is present the uniform is kept and
    the manager supplies a best-effort constant instead.
    """
    if not _FRAGCOORD_DECL_RE.search(src):
        return src
    res = _RESOLUTION_DECL_RE.search(src)
    uv = _IN_VEC2_RE.search(src)
    if res is None or uv is None:
        return src
    res_name = res.group(1)
    uv_name = uv.group(1)
    patched = _FRAGCOORD_DECL_RE.sub("", src, count=1)
    patched = _MAIN_RE.sub(
        rf"\g<1>\n    vec2 fragCoord = {uv_name} * {res_name}.xy;",
        patched,
        count=1,
    )
    return patched


def _sampler_role(name: str) -> str:
    """Classify a sampler uniform into primary / secondary / previous-frame.

    Heuristics cover the naming dialects in the folder: ``u_source`` /
    ``u_texture_from`` / ``u_texture1`` / ``iChannel0`` are primary inputs,
    ``u_texture_to`` / ``u_texture2`` / ``iChannel1+`` are secondary inputs,
    and ``u_previous_frame`` / ``*history*`` sample the previous output.
    """
    n = name.lower()
    if "prev" in n or "history" in n or "last" in n:
        return "prev"
    # Folder-image samplers are checked before "secondary" because names
    # like ``u_photo`` contain the substring "to".
    if (
        "image_next" in n
        or "next_image" in n
        or "image_b" in n
        or "second_image" in n
    ):
        return "image_next"
    if (
        "image" in n
        or "photo" in n
        or n in ("u_texture", "u_tex", "texture", "tex")
        or n.endswith("_tex")
        or n.endswith("_image")
    ):
        return "image"
    if (
        "to" in n
        or "target" in n
        or n.endswith("2")
        or "channel1" in n
        or "channel2" in n
        or "channel3" in n
    ):
        return "secondary"
    return "primary"


@dataclass
class EffectSpec:
    """A parsed ``.frag`` effect shader."""

    name: str
    path: Path
    source: str
    uniforms: dict[str, str] = field(default_factory=dict)  # name -> gl type
    sampler_roles: dict[str, str] = field(default_factory=dict)  # name -> role
    broken: bool = False
    last_used: float = -1e9

    @property
    def has_secondary_input(self) -> bool:
        return "secondary" in self.sampler_roles.values()

    @property
    def numeric_uniforms(self) -> list[str]:
        return [n for n, t in self.uniforms.items() if t != "sampler2D"]


class EffectLibrary:
    """Scans a directory of ``.frag`` shaders and classifies them."""

    def __init__(self, directory: Path | str | None):
        self.directory = Path(directory) if directory else None
        self.specs: list[EffectSpec] = []
        self.load()

    def load(self) -> None:
        self.specs = []
        if self.directory is None or not self.directory.is_dir():
            return
        for path in sorted(self.directory.glob("*.frag")):
            try:
                raw = path.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                logger.warning("Effects: cannot read %s: %s", path, exc)
                continue
            spec = self._parse(path, raw)
            if spec is not None:
                self.specs.append(spec)
        logger.info(
            "Effects library: %d shaders from %s",
            len(self.specs),
            self.directory,
        )

    @staticmethod
    def _parse(path: Path, raw: str) -> EffectSpec | None:
        src = _strip_comments(raw)
        if "void main" not in src:
            return None
        if not src.lstrip().startswith("#version"):
            src = "#version 330 core\n" + src
        src = _patch_fragcoord(src)
        uniforms: dict[str, str] = {}
        roles: dict[str, str] = {}
        for m in _UNIFORM_RE.finditer(src):
            utype, uname = m.group(1), m.group(2)
            uniforms[uname] = utype
            if utype == "sampler2D":
                roles[uname] = _sampler_role(uname)
        if not uniforms:
            return None
        # Require at least one way to sample the source frame.
        if not any(t == "sampler2D" for t in uniforms.values()):
            return None
        return EffectSpec(
            name=path.stem,
            path=path,
            source=src,
            uniforms=uniforms,
            sampler_roles=roles,
        )


class _PassTarget:
    """One RGBA16F color target used by the effect ping-pong chain."""

    __slots__ = ("fbo", "h", "tex", "w")

    def __init__(self) -> None:
        self.fbo = 0
        self.tex = 0
        self.w = 0
        self.h = 0

    def alloc(self, w: int, h: int) -> None:
        self.release()
        fbo = int(GL.glGenFramebuffers(1))
        tex = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA16F,
            w,
            h,
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
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, tex, 0
        )
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            GL.glDeleteFramebuffers(1, [fbo])
            GL.glDeleteTextures(1, [tex])
            raise RuntimeError(f"Effect target FBO incomplete: {hex(status)}")
        self.fbo, self.tex, self.w, self.h = fbo, tex, w, h

    def release(self) -> None:
        if self.fbo:
            try:
                GL.glDeleteFramebuffers(1, [int(self.fbo)])
            except Exception:
                pass
            self.fbo = 0
        if self.tex:
            try:
                GL.glDeleteTextures(1, [int(self.tex)])
            except Exception:
                pass
            self.tex = 0
        self.w = self.h = 0


_IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
    ".tif",
    ".tiff",
}


class _FolderTexturePool:
    """LRU cache of GL textures loaded from an image folder.

    The pool scans a directory of still images and exposes the *current*
    one to effect shaders.  A timer rotates ``index`` towards ``next_index``
    so the imagery evolves over time; ``blend()`` reports the rotation
    phase so shaders can mix ``current`` into ``next`` for a smooth
    transition.  Textures upload lazily on first use and are evicted beyond
    ``cache_size`` to bound VRAM.
    """

    __slots__ = (
        "_bad",
        "_cache",
        "_interval",
        "_timer",
        "cache_size",
        "directory",
        "evolve_secs",
        "index",
        "log",
        "max_size",
        "next_index",
        "paths",
        "rng",
    )

    def __init__(
        self,
        directory: Path | str,
        *,
        max_size: int = 512,
        cache_size: int = 8,
        evolve_secs: float = 20.0,
        rng: random.Random,
        log: logging.Logger,
    ) -> None:
        self.directory = Path(directory)
        self.max_size = max(64, int(max_size))
        self.cache_size = max(1, int(cache_size))
        self.evolve_secs = max(1.0, float(evolve_secs))
        self.rng = rng
        self.log = log
        self.paths = self._scan()
        self.index = 0
        self.next_index = self._pick(exclude=0) if len(self.paths) > 1 else 0
        self._timer = self.evolve_secs * 0.5
        self._interval = self.evolve_secs
        self._cache: OrderedDict[Path, tuple[int, int, int]] = OrderedDict()
        self._bad: set[Path] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        return len(self.paths)

    @property
    def available(self) -> bool:
        return _IMAGE_DEPS and bool(self.paths)

    def update(self, dt: float) -> None:
        """Advance the rotation timer; swaps current -> next on expiry."""
        if not self.paths:
            return
        self._timer -= max(0.0, float(dt))
        if self._timer <= 0.0:
            # Promote the queued texture, then queue a new one.  ``blend``
            # resets to 0 right after the swap, matching the texture that
            # was being blended towards — the transition stays continuous.
            self.index = self.next_index
            self.next_index = self._pick(exclude=self.index)
            self._interval = self.evolve_secs * self.rng.uniform(0.75, 1.35)
            self._timer = self._interval

    def blend(self) -> float:
        """Rotation phase in ``[0, 1]``: 0 = showing ``index`` fully,
        approaching 1 = ``next_index`` fully blended in."""
        if self._interval <= 0.0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - self._timer / self._interval))

    def current_tex(self) -> int:
        return self._entry(self.index)[0]

    def next_tex(self) -> int:
        return self._entry(self.next_index)[0]

    def aspect(self) -> float:
        """Width/height of the current image (1.0 until loaded)."""
        _tex, w, h = self._entry(self.index)
        return float(w) / float(max(1, h)) if h else 1.0

    def seed(self) -> int:
        """Stable per-image seed derived from the current filename."""
        if not self.paths:
            return 0
        h = 0
        for ch in self.paths[self.index].name:
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
        return h

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "count": self.count,
            "index": self.index,
            "name": self.paths[self.index].name if self.paths else None,
            "cached": len(self._cache),
        }

    def release(self) -> None:
        for _path, (tex, _w, _h) in self._cache.items():
            if tex:
                try:
                    GL.glDeleteTextures(1, [int(tex)])
                except Exception:
                    pass
        self._cache.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _scan(self) -> list[Path]:
        if not _IMAGE_DEPS or not self.directory.is_dir():
            return []
        paths = [
            p
            for p in sorted(self.directory.iterdir())
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        ]
        self.log.info(
            "Effects textures: %d images from %s", len(paths), self.directory
        )
        return paths

    def _pick(self, *, exclude: int) -> int:
        n = len(self.paths)
        if n <= 1:
            return 0
        for _ in range(8):
            idx = self.rng.randrange(n)
            if idx != exclude:
                return idx
        return (exclude + 1) % n

    def _entry(self, idx: int) -> tuple[int, int, int]:
        """Return ``(tex, w, h)`` for ``paths[idx]``, loading on demand."""
        if not self.paths:
            return (0, 0, 0)
        path = self.paths[idx % len(self.paths)]
        if path in self._bad:
            return (0, 0, 0)
        ent = self._cache.get(path)
        if ent is not None:
            self._cache.move_to_end(path)
            return ent
        ent = self._load(path)
        if ent is None:
            return (0, 0, 0)
        self._cache[path] = ent
        while len(self._cache) > self.cache_size:
            _old_path, old_ent = self._cache.popitem(last=False)
            if old_ent[0]:
                try:
                    GL.glDeleteTextures(1, [int(old_ent[0])])
                except Exception:
                    pass
        return ent

    def _load(self, path: Path) -> tuple[int, int, int] | None:
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGBA")
            img.thumbnail(
                (self.max_size, self.max_size), Image.Resampling.LANCZOS
            )
            arr = np.asarray(img, dtype=np.uint8)
        except Exception as exc:
            self.log.debug("Effects texture: cannot decode %s: %s", path, exc)
            self._bad.add(path)
            return None
        # GL's texture origin is bottom-left; PIL rows start at the top.
        arr = np.ascontiguousarray(arr[::-1])
        h, w = int(arr.shape[0]), int(arr.shape[1])
        tex = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA8,
            w,
            h,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            arr.tobytes(),
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return (tex, w, h)


class EffectManager:
    """Owns the effect shader pool, the evolving selection state, and the
    GL targets needed to composite effects over the rendered scene."""

    def __init__(self, config: EvoConfig, logger_: logging.Logger | None = None):
        self.config = config
        self.log = logger_ or logger
        seed = getattr(config, "effects_seed", None)
        self.rng = random.Random(seed if seed is not None else None)
        self.library = EffectLibrary(getattr(config, "effects_dir", None))
        self.enabled = bool(getattr(config, "enable_effects", True)) and bool(
            self.library.specs
        )

        # GL resources (allocated lazily once a GL context is current).
        self._scene = _PassTarget()
        self._bufs = [_PassTarget(), _PassTarget(), _PassTarget()]
        self._programs: dict[str, int] = {}  # spec.name -> program id (0 = failed)
        self._locs: dict[int, dict[str, int]] = {}
        self._mix_program = 0
        self._mix_locs: dict[str, int] = {}

        # Evolution state.
        self._active_spec: EffectSpec | None = None
        self._active_genes: dict[str, float] = {}
        self._out_spec: EffectSpec | None = None
        self._out_genes: dict[str, float] = {}
        self._in_spec: EffectSpec | None = None
        self._in_genes: dict[str, float] = {}
        self._xfade_shader: EffectSpec | None = None
        self._xfading = False
        self._xfade_t = 0.0
        self._evolve_timer = float(getattr(config, "effects_evolve_secs", 14.0)) * 0.6
        self._frame = 0
        self._prev_final_tex = 0
        self._ping = 0
        self._rand_static: dict[str, tuple[int, int, int, int]] = {}
        # Audio levels cached by update() for selection-time affinity scoring.
        self._last_bass = 0.0
        self._last_mid = 0.0
        self._last_treb = 0.0
        # Folder-image pool (``textures/``) sampled via u_image samplers.
        self._textures = self._init_texture_pool()

        if self.enabled:
            # Pick the opening effect without a crossfade.
            spec = self._compile_safe(self._pick_spec())
            if spec is not None:
                self._active_spec = spec
                self._active_genes = self._spawn_genes(spec, {})
                self.log.info("Effects: initial effect '%s'", spec.name)
            else:
                self.enabled = False

    def _init_texture_pool(self) -> _FolderTexturePool | None:
        """Build the folder-image pool, or None when disabled/unavailable."""
        if not getattr(self.config, "enable_effect_textures", True):
            return None
        directory = getattr(self.config, "effects_texture_dir", None)
        if directory is None:
            directory = Path(__file__).resolve().parents[1] / "textures"
        directory = Path(directory)
        if not directory.is_dir():
            self.log.info("Effects textures: directory %s not found", directory)
            return None
        return _FolderTexturePool(
            directory,
            max_size=int(getattr(self.config, "effects_texture_max_size", 512)),
            cache_size=int(getattr(self.config, "effects_texture_cache_size", 8)),
            evolve_secs=float(
                getattr(self.config, "effects_texture_evolve_secs", 20.0)
            ),
            rng=self.rng,
            log=self.log,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scene_fbo(self) -> int:
        """Framebuffer the renderer should draw the raw frame into."""
        return int(self._scene.fbo)

    def scene_tex(self) -> int:
        return int(self._scene.tex)

    def ensure_size(self, w: int, h: int) -> None:
        w, h = max(1, int(w)), max(1, int(h))
        if self._scene.w == w and self._scene.h == h:
            return
        try:
            self._scene.alloc(w, h)
            for buf in self._bufs:
                buf.alloc(w, h)
        except Exception as exc:
            self.log.warning("Effects: target alloc failed: %s", exc)
            self.enabled = False
        self._prev_final_tex = 0

    def update(self, dt: float, t: float, bass: float, mid: float, treb: float, beat: float) -> None:
        """Advance gene drift, evolution timer and crossfade state."""
        if not self.enabled:
            return
        dt = max(0.0, min(0.25, float(dt)))
        self._last_bass = float(bass)
        self._last_mid = float(mid)
        self._last_treb = float(treb)

        if self._textures is not None:
            try:
                self._textures.update(dt)
            except Exception as exc:
                self.log.debug("Effects texture rotation failed: %s", exc)

        # Continuous mutation: every gene performs a small random walk.
        if self._active_genes:
            sigma = float(getattr(self.config, "effects_param_mutation", 0.15))
            if sigma > 0.0:
                scale = sigma * math.sqrt(dt)
                for key in self._active_genes:
                    self._active_genes[key] = self._clamp01(
                        self._active_genes[key] + self.rng.gauss(0.0, scale)
                    )

        if self._xfading:
            self._xfade_t += dt
            dur = self._xfade_duration()
            if dur <= 0.0 or self._xfade_t >= dur:
                # Crossfade complete: promote the incoming effect.
                if self._in_spec is not None:
                    self._active_spec = self._in_spec
                    self._active_genes = self._in_genes
                self._xfading = False
                self._out_spec = self._in_spec = None
                self._out_genes = self._in_genes = {}
                self._xfade_shader = None

        self._evolve_timer -= dt
        if self._evolve_timer <= 0.0:
            interval = max(1.0, float(getattr(self.config, "effects_evolve_secs", 14.0)))
            self._evolve_timer = interval * self.rng.uniform(0.75, 1.35)
            self.evolve_now()

    def evolve_now(self) -> bool:
        """Force the next evolution step (also used by the timer)."""
        if not self.enabled or self._xfading:
            return False
        spec = self._compile_safe(self._pick_spec())
        if spec is None or spec is self._active_spec:
            return False
        self._out_spec = self._active_spec
        self._out_genes = dict(self._active_genes)
        self._in_spec = spec
        self._in_genes = self._spawn_genes(spec, self._active_genes)
        self._xfade_shader = self._compile_safe(self._pick_xfade_shader())
        self._xfading = True
        self._xfade_t = 0.0
        spec.last_used = time.time()
        self.log.info(
            "Effects: evolving '%s' -> '%s'",
            self._out_spec.name if self._out_spec else "-",
            spec.name,
        )
        return True

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "active": self._active_spec.name if self._active_spec else None,
            "incoming": self._in_spec.name if self._in_spec else None,
            "xfading": self._xfading,
            "pool_size": len(self.library.specs),
            "broken": sum(1 for s in self.library.specs if s.broken),
            "textures": self._textures.status() if self._textures else None,
        }

    def disable(self) -> None:
        self.enabled = False

    def cleanup(self) -> None:
        for prog in list(self._programs.values()):
            if prog:
                try:
                    GL.glDeleteProgram(int(prog))
                except Exception:
                    pass
        self._programs.clear()
        self._locs.clear()
        if self._mix_program:
            try:
                GL.glDeleteProgram(int(self._mix_program))
            except Exception:
                pass
            self._mix_program = 0
        for buf in [self._scene, *self._bufs]:
            buf.release()
        if self._textures is not None:
            try:
                self._textures.release()
            except Exception:
                pass
        self._prev_final_tex = 0

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def present(
        self,
        renderer: EvoRenderer,
        dst_fbo: int,
        w: int,
        h: int,
        *,
        t: float,
        dt: float,
        bass: float,
        mid: float,
        treb: float,
        beat: float,
    ) -> bool:
        """Composite the current effect chain from ``scene_tex`` to ``dst_fbo``.

        Returns False when effects are unavailable so the caller can fall
        back to drawing straight to the default framebuffer.
        """
        if not self.enabled or not self._scene.tex or not self._bufs[0].tex:
            return False
        try:
            if self._xfading and self._in_spec is not None:
                # Render both effect outputs, then blend with a transition
                # shader (or the built-in mix fallback).
                self._render_spec(
                    renderer,
                    self._out_spec,
                    self._out_genes,
                    dst=self._bufs[0],
                    w=w,
                    h=h,
                    t=t,
                    dt=dt,
                    bass=bass,
                    mid=mid,
                    treb=treb,
                    beat=beat,
                    progress=self._idle_progress(t),
                )
                self._render_spec(
                    renderer,
                    self._in_spec,
                    self._in_genes,
                    dst=self._bufs[1],
                    w=w,
                    h=h,
                    t=t,
                    dt=dt,
                    bass=bass,
                    mid=mid,
                    treb=treb,
                    beat=beat,
                    progress=self._xfade_progress(),
                )
                self._render_xfade(renderer, w, h, t, dt, bass, mid, treb, beat)
                self._blit_to(self._bufs[2], dst_fbo, w, h)
                self._prev_final_tex = self._bufs[2].tex
            else:
                dst = self._bufs[self._ping]
                ok = self._render_spec(
                    renderer,
                    self._active_spec,
                    self._active_genes,
                    dst=dst,
                    w=w,
                    h=h,
                    t=t,
                    dt=dt,
                    bass=bass,
                    mid=mid,
                    treb=treb,
                    beat=beat,
                    progress=self._idle_progress(t),
                )
                if ok:
                    self._blit_to(dst, dst_fbo, w, h)
                    self._prev_final_tex = dst.tex
                else:
                    # Effect unavailable: present the raw scene untouched.
                    self._blit_to(self._scene, dst_fbo, w, h)
                    self._prev_final_tex = self._scene.tex
                self._ping ^= 1
        except Exception as exc:
            self.log.warning("Effects: present failed, disabling: %s", exc)
            self.disable()
            return False
        finally:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, int(dst_fbo))
        self._frame += 1
        return True

    # ------------------------------------------------------------------
    # Internals — GL
    # ------------------------------------------------------------------

    def _program_for(self, spec: EffectSpec) -> int:
        """Compile (once) and return the GL program for ``spec``."""
        if spec is None:
            return 0
        cached = self._programs.get(spec.name)
        if cached is not None:
            return cached
        prog = 0
        try:
            prog = int(
                self._compile_fn(VERTEX_SHADER_SRC, spec.source, f"fx:{spec.name}")
            )
        except Exception as exc:
            self.log.warning("Effects: shader '%s' failed to compile: %s", spec.name, exc)
        self._programs[spec.name] = prog
        if prog:
            self._locs[prog] = self._uniform_locations(prog, spec)
        else:
            spec.broken = True
        return prog

    def _compile_fn(self, vs_src: str, fs_src: str, cache_key: str) -> int:
        """Indirection so tests can stub compilation; the renderer injects
        its own ``_compile_program`` via :meth:`bind_compiler`."""
        fn = getattr(self, "_compiler", None)
        if fn is None:
            raise RuntimeError("EffectManager has no GL compiler bound")
        return int(fn(vs_src, fs_src, cache_key))

    def bind_compiler(self, fn: Any) -> None:
        """Provide the renderer's program compiler.

        ``fn(vs_src, fs_src, cache_key)`` must compile+link a program and
        return its id — the same contract as ``EvoRenderer._compile_program``.
        """
        self._compiler = fn

    @staticmethod
    def _uniform_locations(program: int, spec: EffectSpec) -> dict[str, int]:
        locs: dict[str, int] = {}
        for uname in spec.uniforms:
            try:
                locs[uname] = int(GL.glGetUniformLocation(program, uname))
            except Exception:
                locs[uname] = -1
        return locs

    def _mix_program_obj(self) -> tuple[int, dict[str, int]]:
        if not self._mix_program:
            prog = int(
                self._compile_fn(VERTEX_SHADER_SRC, _MIX_FRAG, "fx:mix")
            )
            self._mix_program = prog
            locs: dict[str, int] = {}
            for uname in ("u_texture_from", "u_texture_to", "u_progress"):
                try:
                    locs[uname] = int(GL.glGetUniformLocation(prog, uname))
                except Exception:
                    locs[uname] = -1
            self._mix_locs = locs
        return self._mix_program, self._mix_locs

    def _render_spec(
        self,
        renderer: EvoRenderer,
        spec: EffectSpec | None,
        genes: dict[str, float],
        *,
        dst: _PassTarget,
        w: int,
        h: int,
        t: float,
        dt: float,
        bass: float,
        mid: float,
        treb: float,
        beat: float,
        progress: float,
    ) -> bool:
        if spec is None or spec.broken:
            return False
        prog = self._program_for(spec)
        if not prog:
            return False
        locs = self._locs.get(prog, {})
        scene_tex = self._scene.tex
        prev_tex = self._prev_final_tex or scene_tex
        if prev_tex == dst.tex:
            # Never sample the texture we are rendering into.
            prev_tex = scene_tex

        GL.glUseProgram(prog)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, dst.fbo)
        GL.glViewport(0, 0, w, h)
        GL.glDisable(GL.GL_BLEND)
        GL.glBindVertexArray(int(renderer._vao))

        role_tex = {
            "primary": scene_tex,
            "secondary": scene_tex,
            "prev": prev_tex,
        }
        role_unit = {"primary": 0, "secondary": 1, "prev": 2, "image": 3, "image_next": 4}
        pool = self._textures
        cur = nxt = 0
        if pool is not None and pool.available:
            cur = pool.current_tex()
            nxt = pool.next_tex()
        role_tex["image"] = cur or scene_tex
        role_tex["image_next"] = nxt or cur or scene_tex
        for uname, role in spec.sampler_roles.items():
            loc = locs.get(uname, -1)
            if loc is None or loc < 0:
                continue
            unit = role_unit[role]
            tex = role_tex.get(role) or scene_tex
            if not tex:
                tex = scene_tex
            GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
            GL.glBindTexture(GL.GL_TEXTURE_2D, int(tex))
            GL.glUniform1i(loc, unit)

        ctx = _UniformContext(
            t=t,
            dt=dt,
            frame=self._frame,
            w=w,
            h=h,
            progress=progress,
            xfade_dur=self._xfade_duration(),
            bass=bass,
            mid=mid,
            treb=treb,
            beat=beat,
            spec_name=spec.name,
            rand_static=self._rand_static_for(spec.name),
        )
        for uname, utype in spec.uniforms.items():
            if utype == "sampler2D":
                continue
            loc = locs.get(uname, -1)
            if loc is None or loc < 0:
                continue
            value = self._uniform_value(uname, utype, genes, ctx)
            self._set_uniform(loc, utype, value)

        GL.glDrawArrays(GL.GL_TRIANGLES, 0, QUAD_VERTEX_COUNT)
        return True

    def _render_xfade(
        self,
        renderer: EvoRenderer,
        w: int,
        h: int,
        t: float,
        dt: float,
        bass: float,
        mid: float,
        treb: float,
        beat: float,
    ) -> None:
        """Blend the outgoing/incoming effect outputs into ``bufs[2]``.

        The blend target is a real texture (not the screen) so the result
        becomes next frame's ``prev`` sample for temporal effects.
        """
        blend = self._bufs[2]
        prog_spec = self._xfade_shader
        if prog_spec is not None and not prog_spec.broken:
            prog = self._program_for(prog_spec)
            if prog:
                locs = self._locs.get(prog, {})
                GL.glUseProgram(prog)
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, blend.fbo)
                GL.glViewport(0, 0, w, h)
                GL.glDisable(GL.GL_BLEND)
                GL.glBindVertexArray(int(renderer._vao))
                # from = outgoing output, to = incoming output.
                for uname, role in prog_spec.sampler_roles.items():
                    loc = locs.get(uname, -1)
                    if loc is None or loc < 0:
                        continue
                    if role == "secondary":
                        unit, tex = 1, self._bufs[1].tex
                    elif role == "prev":
                        ptex = self._prev_final_tex or self._scene.tex
                        if ptex == blend.tex:
                            ptex = self._scene.tex
                        unit, tex = 2, ptex
                    elif role in ("image", "image_next") and self._textures is not None:
                        unit = 3 if role == "image" else 4
                        tex = (
                            self._textures.current_tex()
                            if role == "image"
                            else self._textures.next_tex()
                        ) or self._scene.tex
                    else:
                        unit, tex = 0, self._bufs[0].tex
                    GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
                    GL.glBindTexture(GL.GL_TEXTURE_2D, int(tex))
                    GL.glUniform1i(loc, unit)
                ctx = _UniformContext(
                    t=t,
                    dt=dt,
                    frame=self._frame,
                    w=w,
                    h=h,
                    progress=self._xfade_progress(),
                    xfade_dur=self._xfade_duration(),
                    bass=bass,
                    mid=mid,
                    treb=treb,
                    beat=beat,
                    spec_name=prog_spec.name,
                    rand_static=self._rand_static_for(prog_spec.name),
                )
                for uname, utype in prog_spec.uniforms.items():
                    if utype == "sampler2D":
                        continue
                    loc = locs.get(uname, -1)
                    if loc is None or loc < 0:
                        continue
                    value = self._uniform_value(
                        uname, utype, self._in_genes or {}, ctx
                    )
                    self._set_uniform(loc, utype, value)
                GL.glDrawArrays(GL.GL_TRIANGLES, 0, QUAD_VERTEX_COUNT)
                return
        # Fallback: built-in crossfade mix.
        prog, locs = self._mix_program_obj()
        GL.glUseProgram(prog)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, blend.fbo)
        GL.glViewport(0, 0, w, h)
        GL.glDisable(GL.GL_BLEND)
        GL.glBindVertexArray(int(renderer._vao))
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, int(self._bufs[0].tex))
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, int(self._bufs[1].tex))
        for uname, unit in (("u_texture_from", 0), ("u_texture_to", 1)):
            loc = locs.get(uname, -1)
            if loc is not None and loc >= 0:
                GL.glUniform1i(loc, unit)
        loc = locs.get("u_progress", -1)
        if loc is not None and loc >= 0:
            GL.glUniform1f(loc, self._xfade_progress_eased())
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, QUAD_VERTEX_COUNT)

    def _blit_to(self, src: _PassTarget, dst_fbo: int, w: int, h: int) -> None:
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, src.fbo)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, int(dst_fbo))
        GL.glBlitFramebuffer(
            0, 0, src.w, src.h, 0, 0, w, h,
            GL.GL_COLOR_BUFFER_BIT, GL.GL_LINEAR,
        )
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, int(dst_fbo))

    # ------------------------------------------------------------------
    # Internals — uniforms
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp01(v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    def _xfade_duration(self) -> float:
        return max(
            0.05,
            float(getattr(self.config, "effects_transition_secs", 1.4)),
        )

    def _xfade_progress(self) -> float:
        return self._clamp01(self._xfade_t / self._xfade_duration())

    @staticmethod
    def _smoothstep(p: float) -> float:
        p = max(0.0, min(1.0, p))
        return p * p * (3.0 - 2.0 * p)

    @staticmethod
    def _smootherstep(p: float) -> float:
        p = max(0.0, min(1.0, p))
        return p * p * p * (p * (p * 6.0 - 15.0) + 10.0)

    def _xfade_progress_eased(self) -> float:
        return self._smoothstep(self._xfade_progress())

    def _idle_progress(self, t: float) -> float:
        """Slow LFO driving progress-style uniforms on steady-state effects."""
        speed = 0.35
        if self._active_genes:
            speed += 0.8 * self._active_genes.get("u_speed", 0.5)
        return 0.5 + 0.5 * math.sin(t * speed)

    def _rand_static_for(self, spec_name: str) -> tuple[int, int, int, int]:
        val = self._rand_static.get(spec_name)
        if val is None:
            val = tuple(self.rng.randint(0, 65535) for _ in range(4))  # type: ignore[assignment]
            self._rand_static[spec_name] = val
        return val  # type: ignore[return-value]

    def _uniform_value(
        self,
        uname: str,
        utype: str,
        genes: dict[str, float],
        ctx: _UniformContext,
    ) -> Any:
        n = uname.lower()

        # --- system-driven uniforms -------------------------------------
        if "randstatic" in n:
            return list(ctx.rand_static)
        if "randframe" in n:
            return [self.rng.randint(0, 65535) for _ in range(4)]
        if n in ("u_time", "itime", "utime", "u_time_sec", "time"):
            return ctx.t
        if "timedelta" in n or "time_delta" in n:
            return ctx.dt
        if "framerate" in n or n == "fps":
            return ctx.fps
        if n in ("i_frame", "u_frame", "iframe"):
            return ctx.frame
        if "duration" in n:
            return ctx.xfade_dur
        if "progresscosine" in n:
            return 0.5 - 0.5 * math.cos(math.pi * ctx.progress)
        if "progressbicubic" in n:
            return self._smootherstep(ctx.progress)
        if "progress" in n:
            return ctx.progress
        if "resolution" in n:
            if utype == "vec3":
                return (float(ctx.w), float(ctx.h), 1.0)
            if utype == "vec2":
                return (float(ctx.w), float(ctx.h))
            return float(max(ctx.w, ctx.h))
        if "fragcoord" in n:
            if utype in ("vec2", "vec3", "vec4"):
                comps = [float(ctx.w), float(ctx.h), 1.0, 1.0][: _vec_len(utype)]
                return tuple(comps)
            return float(ctx.w)
        if "texel" in n:
            tw, th = 1.0 / max(1, ctx.w), 1.0 / max(1, ctx.h)
            if utype == "vec2":
                return (tw, th)
            return tw
        if n.endswith("aspect") or n == "u_aspect":
            return ctx.w / max(1, ctx.h)

        # --- folder texture pool ---------------------------------------
        pool = self._textures
        if pool is not None and pool.count:
            if "image_blend" in n or "texture_blend" in n:
                return pool.blend()
            if "image_index" in n or "texture_index" in n:
                return float(pool.index)
            if "image_phase" in n or "texture_phase" in n:
                return pool.blend()
            if "image_count" in n or "texture_count" in n:
                return pool.count
            if "image_aspect" in n or "texture_aspect" in n:
                return pool.aspect()
            if "image_seed" in n or "texture_seed" in n:
                return float(pool.seed() % 1000) / 1000.0

        # --- audio-driven uniforms --------------------------------------
        audio_val = self._audio_uniform(n, ctx)
        if audio_val is not None:
            return audio_val

        # --- evolved genes ------------------------------------------------
        g = self._clamp01(genes.get(uname, self._default_gene(uname)))
        if (
            getattr(self.config, "effects_audio_reactive", True)
            and utype in ("float", "int")
            and _AUDIO_PARAM_RE.search(n)
        ):
            level = self._clamp01(
                0.45 * ctx.bass + 0.35 * ctx.mid + 0.20 * ctx.treb + 0.30 * ctx.beat
            )
            g = self._clamp01(g * (0.45 + 0.9 * level))
        return self._gene_to_value(uname, utype, g)

    @staticmethod
    def _audio_uniform(n: str, ctx: _UniformContext) -> float | None:
        """Map iBass/iMid/iTreb(-style) uniforms to the live audio bands."""
        att = "att" in n
        if "bass" in n:
            return ctx.bass * (0.5 if att else 1.0)
        if "treb" in n:
            return ctx.treb * (0.5 if att else 1.0)
        if "mid" in n:
            return ctx.mid * (0.5 if att else 1.0)
        return None

    @staticmethod
    def _default_gene(uname: str) -> float:
        # Deterministic pseudo-random default so specs render sensibly even
        # before their gene dict is populated.
        h = 0
        for ch in uname:
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
        return ((h % 1000) + 0.5) / 1000.0

    def _gene_to_value(self, uname: str, utype: str, g: float) -> Any:
        n = uname.lower()
        if utype == "float":
            lo, hi = 0.0, 1.0
            for key, k_lo, k_hi in _FLOAT_RANGES:
                if key in n:
                    lo, hi = k_lo, k_hi
                    break
            return lo + g * (hi - lo)
        if utype == "bool":
            return 1 if g > 0.5 else 0
        # Vector types (incl. ivec*): replicate the scalar gene per component.
        if utype in _VEC_TYPES:
            length = _vec_len(utype)
            lo, hi = 0.0, 1.0
            for key, k_lo, k_hi in _FLOAT_RANGES:
                if key in n:
                    lo, hi = k_lo, k_hi
                    break
            val = lo + g * (hi - lo)
            if utype.startswith("i"):
                return tuple(round(val) for _ in range(length))
            return tuple(float(val) for _ in range(length))
        if utype in _INT_TYPES:
            if any(k in n for k in ("enable", "flag", "use_", "_use")):
                return 1 if g > 0.5 else 0
            lo, hi = 1, 8
            for key, k_lo, k_hi in _INT_RANGES:
                if key in n:
                    lo, hi = k_lo, k_hi
                    break
            return round(lo + g * (hi - lo))
        return g

    @staticmethod
    def _set_uniform(loc: int, utype: str, value: Any) -> None:
        try:
            if utype == "float":
                GL.glUniform1f(loc, float(value))
            elif utype == "int":
                GL.glUniform1i(loc, int(value))
            elif utype == "bool":
                GL.glUniform1i(loc, int(value))
            elif utype == "vec2":
                GL.glUniform2f(loc, float(value[0]), float(value[1]))
            elif utype == "vec3":
                GL.glUniform3f(loc, float(value[0]), float(value[1]), float(value[2]))
            elif utype == "vec4":
                GL.glUniform4f(
                    loc,
                    float(value[0]),
                    float(value[1]),
                    float(value[2]),
                    float(value[3]),
                )
            elif utype == "ivec2":
                GL.glUniform2iv(loc, [int(v) for v in value])
            elif utype == "ivec3":
                GL.glUniform3iv(loc, [int(v) for v in value])
            elif utype == "ivec4":
                GL.glUniform4iv(loc, [int(v) for v in value])
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internals — evolution
    # ------------------------------------------------------------------

    def _compile_safe(self, spec: EffectSpec | None) -> EffectSpec | None:
        """Return ``spec`` after verifying its shader compiles, else None."""
        if spec is None:
            return None
        if spec.broken:
            return None
        if self._programs.get(spec.name) is None and not self._compiler_bound():
            return spec  # defer compile-check to first render
        prog = self._program_for(spec)
        return spec if prog else None

    def _compiler_bound(self) -> bool:
        return getattr(self, "_compiler", None) is not None

    def _pick_spec(self) -> EffectSpec | None:
        """Choose the next effect: novelty vs. recent use + audio affinity."""
        candidates = [s for s in self.library.specs if not s.broken]
        if not candidates:
            return None
        now = time.time()
        interval = max(1.0, float(getattr(self.config, "effects_evolve_secs", 14.0)))
        best: EffectSpec | None = None
        best_score = -1e9
        for spec in candidates:
            if spec is self._active_spec:
                continue
            age = now - spec.last_used
            novelty = min(age / (interval * 2.0), 3.0) / 3.0
            # Penalise very recent repeats so the pool actually rotates.
            if age < interval * 0.5:
                novelty *= 0.2
            score = self.rng.random() * 0.6 + novelty * 0.4
            name = spec.name.lower()
            if self._last_bass > 0.55 and _BASS_AFFINITY_RE.search(name):
                score += 0.15
            if self._last_treb > 0.55 and _TREB_AFFINITY_RE.search(name):
                score += 0.15
            if self._last_mid > 0.55 and _MID_AFFINITY_RE.search(name):
                score += 0.10
            if score > best_score:
                best_score = score
                best = spec
        return best if best is not None else self.rng.choice(candidates)

    def _pick_xfade_shader(self) -> EffectSpec | None:
        """Pick a transition-capable shader for the crossfade blend."""
        candidates = [
            s
            for s in self.library.specs
            if not s.broken and s.has_secondary_input and s is not self._active_spec
        ]
        if not candidates:
            return None
        now = time.time()
        best: EffectSpec | None = None
        best_score = -1e9
        for spec in candidates:
            age = now - spec.last_used
            score = self.rng.random() * 0.5 + min(age / 120.0, 1.0) * 0.5
            if score > best_score:
                best_score = score
                best = spec
        return best

    def _spawn_genes(
        self, spec: EffectSpec, inherited: dict[str, float]
    ) -> dict[str, float]:
        """Create the gene dict for ``spec``: inherit shared uniforms from the
        outgoing effect (with mutation) and randomise the rest."""
        genes: dict[str, float] = {}
        burst = 0.18
        for uname in spec.numeric_uniforms:
            if uname in inherited:
                genes[uname] = self._clamp01(
                    inherited[uname] + self.rng.gauss(0.0, burst)
                )
            else:
                genes[uname] = self.rng.random()
        return genes


def _vec_len(utype: str) -> int:
    if utype.endswith("2"):
        return 2
    if utype.endswith("3"):
        return 3
    return 4


@dataclass
class _UniformContext:
    t: float
    dt: float
    frame: int
    w: int
    h: int
    progress: float
    xfade_dur: float
    bass: float
    mid: float
    treb: float
    beat: float
    spec_name: str
    rand_static: tuple[int, int, int, int]

    @property
    def fps(self) -> float:
        return 1.0 / max(self.dt, 1e-3)
