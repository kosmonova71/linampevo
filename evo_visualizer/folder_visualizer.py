#!/usr/bin/env python3
"""Folder-driven visualiser built only from the repo asset folders.

Three pools evolve over time, independently of the genetic genome:

* ``shaders/*.glsl`` — full-screen base shaders (GLSL 1.30 feedback style:
  ``u_time``, ``u_resolution``, ``u_prev_frame``, ``u_audio``, ``u_feedback``).
  One shader is active at a time; its numeric uniforms act as genes that
  drift every frame, and on a timer the active shader crossfades into a
  newly picked one.
* ``effects/*.frag`` — post-processing layer, delegated to
  :class:`evo_visualizer.effects.EffectManager` (which already evolves its
  own selection and crossfades).
* ``textures/*`` — still-image pool sampled through ``u_texture`` /
  ``u_image``-style samplers; the active image rotates on a timer via
  :class:`evo_visualizer.effects._FolderTexturePool`.

Render order per frame::

    base shader (shaders/)  ->  ping-pong frame targets
         |
         v  (blit into the effects scene target when enabled)
    EffectManager.present() -> evolving effects/ post chain -> screen
         |
         v  (plain blit when the effects layer is off)
    GTK default framebuffer

Audio reactivity mirrors the genome renderer: ``u_audio`` is a vec4 of
``(bass, mid, treb, beat)`` after the perceptual response curve, band-named
uniforms receive their band, and uniforms whose names look intensity-like
are boosted by the current mix level.
"""

from __future__ import annotations

import logging
import math
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from OpenGL import GL

from evo_visualizer.constants import QUAD_VERTEX_COUNT
from evo_visualizer.effects import (
    _AUDIO_PARAM_RE,
    _BASS_AFFINITY_RE,
    _FLOAT_RANGES,
    _INT_RANGES,
    _INT_TYPES,
    _MID_AFFINITY_RE,
    _MIX_FRAG,
    _TREB_AFFINITY_RE,
    _UNIFORM_RE,
    _VEC_TYPES,
    VERTEX_SHADER_SRC,
    EffectManager,
    _FolderTexturePool,
    _PassTarget,
    _sampler_role,
    _strip_comments,
)

if TYPE_CHECKING:
    from evo_visualizer.config import EvoConfig
    from evo_visualizer.renderer import EvoRenderer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shader-source patching
# ---------------------------------------------------------------------------

_PY_STRING_RE = re.compile(r'=\s*"""(.*)"""', re.DOTALL)
_VERSION_RE = re.compile(r"^\s*#version\s+(\d+)", re.MULTILINE)
_FRAGCOLOR_DECL_RE = re.compile(
    r"^\s*(?:in|out|varying)\s+vec4\s+gl_FragColor\s*;[^\n]*\n?",
    re.MULTILINE,
)
# ``in``/``varying`` vecN declarations that need a matching vertex output.
_VARYING_RE = re.compile(
    r"^\s*(?:in|varying)\s+(vec[234])\s+(\w+)\s*;", re.MULTILINE
)
_OUT_FRAGCOLOR_RE = re.compile(r"\bout\s+vec4\s+fragColor\s*;")


def _extract_glsl(raw: str) -> str:
    """Unwrap ``NAME = \"\"\" ... \"\"\"`` Python-string shader files."""
    m = _PY_STRING_RE.search(raw)
    if m is not None:
        return m.group(1)
    return raw


def _normalize_source(raw: str) -> str | None:
    """Return GLSL ready to compile as ``#version 330 core``.

    The ``shaders/`` folder mixes dialects: most files are GLSL 1.30
    (``gl_FragColor``, ``texture2D``, ``varying``) which core-profile
    contexts reject.  Everything is mechanically upgraded to 330 core —
    ``#version 330 core`` shaders compile on both core and compatibility
    contexts, so this is strictly more portable than keeping 1.30.
    """
    src = _extract_glsl(raw)
    if "main" not in src:
        return None
    src = _FRAGCOLOR_DECL_RE.sub("", src)
    m = _VERSION_RE.search(src)
    version = int(m.group(1)) if m else 0
    if version >= 330:
        return src
    # Upgrade legacy GLSL to 330 core.
    body = _VERSION_RE.sub("", src, count=1) if m else src
    body = re.sub(r"\btexture2D\s*\(", "texture(", body)
    body = re.sub(r"\bgl_FragColor\b", "fragColor", body)
    # Fragment-shader inputs use ``in`` in 330 core (``varying`` is gone).
    body = re.sub(r"\bvarying\b", "in", body)
    if "fragColor" in body and not _OUT_FRAGCOLOR_RE.search(body):
        # Precision statements must precede global declarations, so the
        # output variable goes right after the precision line when present.
        pm = re.search(r"precision\s+\w+\s+\w+\s*;", body)
        decl = "\nout vec4 fragColor;\n"
        if pm is not None:
            body = body[: pm.end()] + decl + body[pm.end() :]
        else:
            body = decl + body
    return "#version 330 core\n" + body


def _make_vertex_330(frag_src: str) -> str:
    """Build a ``#version 330 core`` vertex shader that feeds every vecN
    input the fragment shader declares, plus the common ``v_texcoord`` /
    ``v_uv`` aliases."""
    outs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for vtype, vname in _VARYING_RE.findall(frag_src):
        if vname in seen or vname.startswith("gl_"):
            continue
        seen.add(vname)
        outs.append((vtype, vname))
    for alias in ("v_texcoord", "v_uv"):
        if alias not in seen:
            seen.add(alias)
            outs.append(("vec2", alias))
    decls = "".join(f"out {vt} {vn};\n" for vt, vn in outs)
    assigns = "".join(
        f"    {vn} = vec{vt[3]}(uv"
        + (", 0.0" if vt == "vec3" else ", 0.0, 1.0" if vt == "vec4" else "")
        + ");\n"
        for vt, vn in outs
    )
    return (
        "#version 330 core\n"
        "layout(location = 0) in vec2 position;\n"
        f"{decls}"
        "void main() {\n"
        "    gl_Position = vec4(position, 0.0, 1.0);\n"
        "    vec2 uv = position * 0.5 + 0.5;\n"
        f"{assigns}"
        "}\n"
    )


# ---------------------------------------------------------------------------
# Shader pool
# ---------------------------------------------------------------------------


@dataclass
class FolderShaderSpec:
    """A parsed ``shaders/*.glsl`` full-screen shader."""

    name: str
    path: Path
    source: str
    vertex_source: str
    uniforms: dict[str, str] = field(default_factory=dict)
    sampler_roles: dict[str, str] = field(default_factory=dict)
    broken: bool = False
    last_used: float = -1e9

    @property
    def numeric_uniforms(self) -> list[str]:
        return [n for n, t in self.uniforms.items() if t != "sampler2D"]


class FolderShaderLibrary:
    """Scans a directory of ``.glsl``/``.frag`` shaders and classifies them."""

    def __init__(self, directory: Path | str | None):
        self.directory = Path(directory) if directory else None
        self.specs: list[FolderShaderSpec] = []
        self.load()

    def load(self) -> None:
        self.specs = []
        if self.directory is None or not self.directory.is_dir():
            return
        paths = sorted(
            p
            for p in self.directory.iterdir()
            if p.is_file() and p.suffix.lower() in (".glsl", ".frag")
        )
        for path in paths:
            try:
                raw = path.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                logger.warning("Folder shaders: cannot read %s: %s", path, exc)
                continue
            spec = self._parse(path, raw)
            if spec is not None:
                self.specs.append(spec)
        logger.info(
            "Folder shader library: %d shaders from %s",
            len(self.specs),
            self.directory,
        )

    @staticmethod
    def _parse(path: Path, raw: str) -> FolderShaderSpec | None:
        src = _normalize_source(raw)
        if src is None:
            return None
        stripped = _strip_comments(src)
        uniforms: dict[str, str] = {}
        roles: dict[str, str] = {}
        for m in _UNIFORM_RE.finditer(stripped):
            utype, uname = m.group(1), m.group(2)
            uniforms[uname] = utype
            if utype == "sampler2D":
                roles[uname] = _sampler_role(uname)
        if not any(t == "sampler2D" for t in uniforms.values()):
            # Generative shaders without any input texture cannot feed back
            # and would render a static frame; skip them.
            return None
        return FolderShaderSpec(
            name=path.stem,
            path=path,
            source=src,
            vertex_source=_make_vertex_330(stripped),
            uniforms=uniforms,
            sampler_roles=roles,
        )


# ---------------------------------------------------------------------------
# Uniform value resolution (mirrors EffectManager._uniform_value)
# ---------------------------------------------------------------------------

# Uniform names driven by frame state rather than genes.
_CONTEXT_FLOATS = {
    "u_time": "t",
    "itime": "t",
    "utime": "t",
    "u_time_sec": "t",
    "time": "t",
    "u_delta_time": "dt",
    "u_dt": "dt",
    "u_frame": "frame",
    "u_beat": "beat",
    "beat": "beat",
}


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _audio_response(v: float) -> float:
    """Same perceptual lift the genome renderer applies to band levels."""
    v = max(0.0, min(1.5, float(v)))
    return v**0.85 if v > 0.0 else 0.0


def _vec_len(utype: str) -> int:
    if utype.endswith("2"):
        return 2
    if utype.endswith("3"):
        return 3
    return 4


def _gene_to_value(uname: str, utype: str, g: float) -> Any:
    n = uname.lower()
    g = _clamp01(g)
    if utype == "float":
        lo, hi = 0.0, 1.0
        for key, k_lo, k_hi in _FLOAT_RANGES:
            if key in n:
                lo, hi = k_lo, k_hi
                break
        return lo + g * (hi - lo)
    if utype == "bool":
        return 1 if g > 0.5 else 0
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
    pool: _FolderTexturePool | None = None

    @property
    def fps(self) -> float:
        return 1.0 / max(self.dt, 1e-3)


# ---------------------------------------------------------------------------
# FolderVisualizer
# ---------------------------------------------------------------------------


class FolderVisualizer:
    """Evolving visualiser driven by the ``shaders/`` + ``effects/`` +
    ``textures/`` folders.

    The base layer picks a full-screen shader from ``shaders/`` and evolves
    the selection on a timer; numeric uniforms drift continuously as genes.
    When an :class:`EffectManager` is attached, its evolving ``effects/``
    post chain (with the rotating ``textures/`` image pool) composites on
    top.  Without it the base frame is blitted straight to the screen.
    """

    def __init__(
        self,
        config: EvoConfig,
        effects: EffectManager | None = None,
        logger_: logging.Logger | None = None,
    ):
        self.config = config
        self.log = logger_ or logger
        # Effects/ post layer: reuse the renderer's EffectManager when one
        # was passed in, otherwise own a private instance so the folder
        # visualiser works even with ``enable_effects=False``.
        self._owns_fx = False
        self._fx: EffectManager | None = None
        if getattr(config, "folder_use_effects", True):
            if effects is not None:
                self._fx = effects
            else:
                mgr = EffectManager(config, logger_)
                if mgr.enabled:
                    self._owns_fx = True
                    self._fx = mgr
        seed = getattr(config, "folder_seed", None)
        self.rng = random.Random(seed if seed is not None else None)
        self.library = FolderShaderLibrary(getattr(config, "folder_shaders_dir", None))
        self.enabled = bool(getattr(config, "enable_folder_visualizer", False)) and bool(
            self.library.specs
        )

        # GL resources, allocated lazily once a context is current.
        self._rt = [_PassTarget(), _PassTarget(), _PassTarget()]
        self._prev_i = 0
        self._programs: dict[str, int] = {}
        self._locs: dict[int, dict[str, int]] = {}
        self._mix_program = 0
        self._mix_locs: dict[str, int] = {}
        self._rand_static: dict[str, tuple[int, int, int, int]] = {}

        # Base-shader texture pool (textures/ folder).
        self._textures = self._init_texture_pool()

        # Evolution state.
        self._active_spec: FolderShaderSpec | None = None
        self._active_genes: dict[str, float] = {}
        self._in_spec: FolderShaderSpec | None = None
        self._in_genes: dict[str, float] = {}
        self._out_spec: FolderShaderSpec | None = None
        self._out_genes: dict[str, float] = {}
        self._xfading = False
        self._xfade_t = 0.0
        self._evolve_timer = float(getattr(config, "folder_evolve_secs", 18.0)) * 0.6
        self._frame = 0
        self._last_bass = self._last_mid = self._last_treb = 0.0

        if self.enabled:
            spec = self._compile_safe(self._pick_spec())
            if spec is not None:
                self._active_spec = spec
                self._active_genes = self._spawn_genes(spec, {})
                self.log.info("Folder visualizer: initial shader '%s'", spec.name)
            else:
                self.enabled = False

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _init_texture_pool(self) -> _FolderTexturePool | None:
        if not getattr(self.config, "enable_effect_textures", True):
            return None
        directory = getattr(self.config, "effects_texture_dir", None)
        if directory is None:
            directory = Path(__file__).resolve().parent.parent / "textures"
        directory = Path(directory)
        if not directory.is_dir():
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

    def bind_compiler(self, fn: Any) -> None:
        """Provide the renderer's program compiler.

        ``fn(vs_src, fs_src, cache_key)`` must compile+link a program and
        return its id — the same contract as ``EffectManager.bind_compiler``.
        """
        self._compiler = fn
        if self._fx is not None:
            self._fx.bind_compiler(fn)

    def ensure_size(self, w: int, h: int) -> None:
        w, h = max(1, int(w)), max(1, int(h))
        if all(t.w == w and t.h == h for t in self._rt):
            return
        try:
            for t in self._rt:
                t.alloc(w, h)
                # Start from black so feedback shaders don't sample
                # uninitialised texture memory on the first frames.
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, t.fbo)
                GL.glClearColor(0.0, 0.0, 0.0, 1.0)
                GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        except Exception as exc:
            self.log.warning("Folder visualizer: target alloc failed: %s", exc)
            self.enabled = False
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def update(
        self,
        dt: float,
        t: float,
        bass: float,
        mid: float,
        treb: float,
        beat: float,
    ) -> None:
        """Advance gene drift, texture rotation and shader-selection timers."""
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
                self.log.debug("Folder texture rotation failed: %s", exc)

        # A privately owned EffectManager is not ticked by the renderer, so
        # advance it here.  A shared one is updated by the renderer already.
        if self._owns_fx and self._fx is not None:
            try:
                self._fx.update(dt, t, bass, mid, treb, beat)
            except Exception as exc:
                self.log.debug("Folder effects update failed: %s", exc)

        if self._active_genes:
            sigma = float(getattr(self.config, "folder_param_mutation", 0.12))
            if sigma > 0.0:
                scale = sigma * math.sqrt(dt)
                for key in self._active_genes:
                    self._active_genes[key] = _clamp01(
                        self._active_genes[key] + self.rng.gauss(0.0, scale)
                    )

        if self._xfading:
            self._xfade_t += dt
            if self._xfade_t >= self._xfade_duration():
                if self._in_spec is not None:
                    self._active_spec = self._in_spec
                    self._active_genes = self._in_genes
                self._xfading = False
                self._out_spec = self._in_spec = None
                self._out_genes = self._in_genes = {}

        self._evolve_timer -= dt
        if self._evolve_timer <= 0.0:
            interval = max(1.0, float(getattr(self.config, "folder_evolve_secs", 18.0)))
            self._evolve_timer = interval * self.rng.uniform(0.75, 1.35)
            self.evolve_now()

    def evolve_now(self) -> bool:
        """Force the next shader-selection step (also used by the timer)."""
        if not self.enabled or self._xfading:
            return False
        spec = self._compile_safe(self._pick_spec())
        if spec is None or spec is self._active_spec:
            return False
        self._out_spec = self._active_spec
        self._out_genes = dict(self._active_genes)
        self._in_spec = spec
        self._in_genes = self._spawn_genes(spec, self._active_genes)
        self._xfading = True
        self._xfade_t = 0.0
        spec.last_used = time.time()
        self.log.info(
            "Folder visualizer: evolving '%s' -> '%s'",
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
            "effects": self._fx.status() if self._fx is not None else None,
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
        for t in self._rt:
            t.release()
        if self._textures is not None:
            try:
                self._textures.release()
            except Exception:
                pass
        if self._owns_fx and self._fx is not None:
            try:
                self._fx.cleanup()
            except Exception:
                pass
            self._fx = None
        self._prev_i = 0

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(
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
        """Render the evolving folder-shader frame and composite it.

        When an EffectManager is attached the base frame is blitted into its
        scene target and ``present()`` runs the evolving effects/ chain; the
        result lands on ``dst_fbo``.  Otherwise the base frame is blitted
        straight to ``dst_fbo``.  Returns False when disabled.
        """
        if not self.enabled or not self._rt[0].tex:
            return False
        try:
            base = self._render_base(renderer, w, h, t=t, dt=dt, bass=bass,
                                     mid=mid, treb=treb, beat=beat)
            if base is None:
                return False
            fx = self._fx
            if fx is not None and fx.enabled:
                try:
                    fx.ensure_size(w, h)
                    if fx.scene_tex():
                        self._blit(base, fx.scene_fbo(), w, h)
                        if fx.present(
                            renderer, dst_fbo, w, h,
                            t=t, dt=dt, bass=bass, mid=mid, treb=treb, beat=beat,
                        ):
                            self._frame += 1
                            return True
                except Exception as exc:
                    self.log.debug("Folder effects composite failed: %s", exc)
            self._blit(base, int(dst_fbo), w, h)
        except Exception as exc:
            self.log.warning("Folder visualizer render failed, disabling: %s", exc)
            self.disable()
            return False
        finally:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, int(dst_fbo))
        self._frame += 1
        return True

    def _render_base(
        self,
        renderer: EvoRenderer,
        w: int,
        h: int,
        *,
        t: float,
        dt: float,
        bass: float,
        mid: float,
        treb: float,
        beat: float,
    ) -> _PassTarget | None:
        """Render the active (or crossfading) base shader; return the target
        holding this frame's image."""
        prev = self._rt[self._prev_i]
        if self._xfading and self._in_spec is not None:
            dst_i = (self._prev_i + 1) % 3
            tmp_i = (self._prev_i + 2) % 3
            dst = self._rt[dst_i]
            tmp = self._rt[tmp_i]
            ok_out = self._render_spec(
                renderer, self._out_spec, self._out_genes,
                dst=dst, prev_tex=prev.tex, w=w, h=h,
                t=t, dt=dt, bass=bass, mid=mid, treb=treb, beat=beat,
                progress=self._idle_progress(t),
            )
            ok_in = self._render_spec(
                renderer, self._in_spec, self._in_genes,
                dst=tmp, prev_tex=prev.tex, w=w, h=h,
                t=t, dt=dt, bass=bass, mid=mid, treb=treb, beat=beat,
                progress=self._xfade_progress_eased(),
            )
            if ok_out and ok_in:
                # Blend into the *previous* target: both inputs are already
                # drawn, so overwriting it is safe and keeps the ring small.
                # The mixed frame lands where the old previous frame was, so
                # ``_prev_i`` stays valid and now points at the new frame.
                self._render_mix(renderer, dst, tmp, prev, w, h, t, dt,
                                 bass, mid, treb, beat)
                return prev
            if ok_in:
                self._prev_i = tmp_i
                return tmp
            if ok_out:
                self._prev_i = dst_i
                return dst
            return self._render_spec_fallback(renderer, dst, w, h)

        dst_i = (self._prev_i + 1) % 3
        dst = self._rt[dst_i]
        ok = self._render_spec(
            renderer, self._active_spec, self._active_genes,
            dst=dst, prev_tex=prev.tex, w=w, h=h,
            t=t, dt=dt, bass=bass, mid=mid, treb=treb, beat=beat,
            progress=self._idle_progress(t),
        )
        if not ok:
            res = self._render_spec_fallback(renderer, dst, w, h)
            if res is dst:
                self._prev_i = dst_i
            return res
        self._prev_i = dst_i
        return dst

    def _render_spec_fallback(
        self, renderer: EvoRenderer, dst: _PassTarget, w: int, h: int
    ) -> _PassTarget | None:
        """No usable shader: copy the previous frame forward so the display
        never goes black."""
        src = self._rt[self._prev_i]
        if src is dst or not src.tex:
            return dst if dst.tex else None
        try:
            self._blit(src, dst.fbo, w, h)
        except Exception:
            pass
        return dst

    # ------------------------------------------------------------------
    # GL internals
    # ------------------------------------------------------------------

    def _compile_fn(self, vs_src: str, fs_src: str, cache_key: str) -> int:
        fn = getattr(self, "_compiler", None)
        if fn is None:
            raise RuntimeError("FolderVisualizer has no GL compiler bound")
        return int(fn(vs_src, fs_src, cache_key))

    def _program_for(self, spec: FolderShaderSpec) -> int:
        if spec is None:
            return 0
        cached = self._programs.get(spec.name)
        if cached is not None:
            return cached
        prog = 0
        try:
            prog = int(
                self._compile_fn(
                    spec.vertex_source, spec.source, f"folder:{spec.name}"
                )
            )
        except Exception as exc:
            self.log.warning(
                "Folder shader '%s' failed to compile: %s", spec.name, exc
            )
        self._programs[spec.name] = prog
        if prog:
            locs: dict[str, int] = {}
            for uname in spec.uniforms:
                try:
                    locs[uname] = int(GL.glGetUniformLocation(prog, uname))
                except Exception:
                    locs[uname] = -1
            self._locs[prog] = locs
        else:
            spec.broken = True
        return prog

    def _mix_program_obj(self) -> tuple[int, dict[str, int]]:
        if not self._mix_program:
            prog = int(self._compile_fn(VERTEX_SHADER_SRC, _MIX_FRAG, "folder:mix"))
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
        spec: FolderShaderSpec | None,
        genes: dict[str, float],
        *,
        dst: _PassTarget,
        prev_tex: int,
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
        pool = self._textures
        cur = nxt = 0
        if pool is not None and pool.available:
            cur = pool.current_tex()
            nxt = pool.next_tex()

        GL.glUseProgram(prog)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, dst.fbo)
        GL.glViewport(0, 0, w, h)
        GL.glDisable(GL.GL_BLEND)
        GL.glBindVertexArray(int(renderer._vao))

        # Sampler roles -> texture units.  Everything that is not a folder
        # image samples the previous composited base frame.
        role_tex = {
            "prev": prev_tex,
            "primary": prev_tex,
            "secondary": prev_tex,
            "image": cur or prev_tex,
            "image_next": nxt or cur or prev_tex,
        }
        role_unit = {"prev": 0, "primary": 1, "secondary": 1, "image": 2, "image_next": 3}
        for uname, role in spec.sampler_roles.items():
            loc = locs.get(uname, -1)
            if loc is None or loc < 0:
                continue
            unit = role_unit[role]
            tex = role_tex.get(role) or prev_tex
            GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
            GL.glBindTexture(GL.GL_TEXTURE_2D, int(tex))
            GL.glUniform1i(loc, unit)

        ctx = _UniformContext(
            t=t, dt=dt, frame=self._frame, w=w, h=h,
            progress=progress,
            xfade_dur=self._xfade_duration(),
            bass=_audio_response(bass), mid=_audio_response(mid),
            treb=_audio_response(treb), beat=_audio_response(beat),
            spec_name=spec.name,
            rand_static=self._rand_static_for(spec.name),
            pool=pool,
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

    def _render_mix(
        self,
        renderer: EvoRenderer,
        src_a: _PassTarget,
        src_b: _PassTarget,
        dst: _PassTarget,
        w: int,
        h: int,
        t: float,
        dt: float,
        bass: float,
        mid: float,
        treb: float,
        beat: float,
    ) -> None:
        """Crossfade ``src_a`` -> ``src_b`` into ``dst`` with the built-in mix."""
        prog, locs = self._mix_program_obj()
        GL.glUseProgram(prog)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, dst.fbo)
        GL.glViewport(0, 0, w, h)
        GL.glDisable(GL.GL_BLEND)
        GL.glBindVertexArray(int(renderer._vao))
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, int(src_a.tex))
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, int(src_b.tex))
        for uname, unit in (("u_texture_from", 0), ("u_texture_to", 1)):
            loc = locs.get(uname, -1)
            if loc is not None and loc >= 0:
                GL.glUniform1i(loc, unit)
        loc = locs.get("u_progress", -1)
        if loc is not None and loc >= 0:
            GL.glUniform1f(loc, self._xfade_progress_eased())
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, QUAD_VERTEX_COUNT)

    @staticmethod
    def _blit(src: _PassTarget, dst_fbo: int, w: int, h: int) -> None:
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, src.fbo)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, int(dst_fbo))
        GL.glBlitFramebuffer(
            0, 0, src.w, src.h, 0, 0, w, h,
            GL.GL_COLOR_BUFFER_BIT, GL.GL_LINEAR,
        )
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, int(dst_fbo))

    # ------------------------------------------------------------------
    # Uniform values
    # ------------------------------------------------------------------

    def _uniform_value(
        self,
        uname: str,
        utype: str,
        genes: dict[str, float],
        ctx: _UniformContext,
    ) -> Any:
        n = uname.lower()

        if "randstatic" in n:
            return list(ctx.rand_static)
        if "randframe" in n:
            return [self.rng.randint(0, 65535) for _ in range(4)]
        if n == "u_audio" or n == "audio":
            if utype == "vec4":
                return (ctx.bass, ctx.mid, ctx.treb, ctx.beat)
            if utype == "vec3":
                return (ctx.bass, ctx.mid, ctx.treb)
            if utype == "vec2":
                return (ctx.bass, ctx.treb)
            return _clamp01(0.5 * ctx.bass + 0.3 * ctx.mid + 0.2 * ctx.treb)
        key = _CONTEXT_FLOATS.get(n)
        if key is not None:
            return float(getattr(ctx, key))
        if "timedelta" in n or "time_delta" in n:
            return ctx.dt
        if "framerate" in n or n == "fps":
            return ctx.fps
        if "duration" in n:
            return ctx.xfade_dur
        if "progresscosine" in n:
            return 0.5 - 0.5 * math.cos(math.pi * ctx.progress)
        if n == "progressbicubic" or "smoother" in n:
            p = _clamp01(ctx.progress)
            return p * p * p * (p * (p * 6.0 - 15.0) + 10.0)
        if "progress" in n:
            return ctx.progress
        if "resolution" in n or n in ("u_res", "res"):
            if utype == "vec3":
                return (float(ctx.w), float(ctx.h), 1.0)
            if utype == "vec2":
                return (float(ctx.w), float(ctx.h))
            return float(max(ctx.w, ctx.h))
        if "fragcoord" in n:
            if utype in ("vec2", "vec3", "vec4"):
                comps = [float(ctx.w), float(ctx.h), 1.0, 1.0][:_vec_len(utype)]
                return tuple(comps)
            return float(ctx.w)
        if "texel" in n:
            tw, th = 1.0 / max(1, ctx.w), 1.0 / max(1, ctx.h)
            if utype == "vec2":
                return (tw, th)
            return tw
        if n.endswith("aspect") or n == "u_aspect":
            return ctx.w / max(1, ctx.h)
        if n in ("u_width", "width"):
            return float(ctx.w)
        if n in ("u_height", "height"):
            return float(ctx.h)

        # Folder texture pool state uniforms.
        pool = ctx.pool
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

        # Audio band uniforms.
        att = "att" in n
        if "bass" in n:
            return ctx.bass * (0.5 if att else 1.0)
        if "treble" in n or "treb" in n:
            return ctx.treb * (0.5 if att else 1.0)
        if "mid" in n:
            return ctx.mid * (0.5 if att else 1.0)

        # Evolved genes.
        g = _clamp01(genes.get(uname, self._default_gene(uname)))
        if utype in ("float", "int") and _AUDIO_PARAM_RE.search(n):
            level = _clamp01(
                0.45 * ctx.bass + 0.35 * ctx.mid + 0.20 * ctx.treb + 0.30 * ctx.beat
            )
            g = _clamp01(g * (0.45 + 0.9 * level))
        return _gene_to_value(uname, utype, g)

    @staticmethod
    def _default_gene(uname: str) -> float:
        h = 0
        for ch in uname:
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
        return ((h % 1000) + 0.5) / 1000.0

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
                    float(value[0]), float(value[1]),
                    float(value[2]), float(value[3]),
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
    # Evolution
    # ------------------------------------------------------------------

    def _compile_safe(self, spec: FolderShaderSpec | None) -> FolderShaderSpec | None:
        if spec is None or spec.broken:
            return None
        if self._programs.get(spec.name) is None and not self._compiler_bound():
            return spec
        return spec if self._program_for(spec) else None

    def _compiler_bound(self) -> bool:
        return getattr(self, "_compiler", None) is not None

    def _pick_spec(self) -> FolderShaderSpec | None:
        """Choose the next base shader: novelty vs. recent use + audio affinity."""
        candidates = [s for s in self.library.specs if not s.broken]
        if not candidates:
            return None
        now = time.time()
        interval = max(1.0, float(getattr(self.config, "folder_evolve_secs", 18.0)))
        best: FolderShaderSpec | None = None
        best_score = -1e9
        for spec in candidates:
            if spec is self._active_spec:
                continue
            age = now - spec.last_used
            novelty = min(age / (interval * 2.0), 3.0) / 3.0
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

    def _spawn_genes(
        self, spec: FolderShaderSpec, inherited: dict[str, float]
    ) -> dict[str, float]:
        """Gene dict for ``spec``: inherit shared uniforms (mutated) and
        randomise the rest."""
        genes: dict[str, float] = {}
        burst = 0.18
        for uname in spec.numeric_uniforms:
            if uname in inherited:
                genes[uname] = _clamp01(inherited[uname] + self.rng.gauss(0.0, burst))
            else:
                genes[uname] = self.rng.random()
        return genes

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _xfade_duration(self) -> float:
        return max(
            0.05,
            float(getattr(self.config, "folder_transition_secs", 1.6)),
        )

    def _xfade_progress(self) -> float:
        return _clamp01(self._xfade_t / self._xfade_duration())

    @staticmethod
    def _smoothstep(p: float) -> float:
        p = max(0.0, min(1.0, p))
        return p * p * (3.0 - 2.0 * p)

    def _xfade_progress_eased(self) -> float:
        return self._smoothstep(self._xfade_progress())

    def _idle_progress(self, t: float) -> float:
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
