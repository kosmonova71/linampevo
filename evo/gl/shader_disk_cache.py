#!/usr/bin/env python3

"""shader_disk_cache.py

Disk-backed shader cache for the evolutionary renderer.

Caches compiled shader binaries, specializations, and pipeline objects to
disk so identical shaders need not be recompiled every session.  On OpenGL
the primary portable artefact is the generated source; the driver-specific
binary is cached additionally when ARB_get_program_binary is available.

Design
------
* In-memory in/out caches layered on top of a thread-safe disk store, so
  the renderer can check a fast O(1) dict before touching the filesystem.
* LRU eviction: a tiny ``index.json`` tracks access order; the stalest
  entries are removed when the max size is exceeded.
* Key composition: ``genome_shader_signature_hash`` + ``driver_id``,
  so a shader whose source is unchanged but is linked on a new driver
  (or GL version) safely misses and recompiles rather than silently
  accepting an incompatible binary.
* Specializations: extra uniform/concept parameter deltas are folded into
  the hash so "shader A with brightness=0.5" and "shader A with
  brightness=0.7" are distinct entries.
* Pipeline objects (multi-pass composition, FBO attachments, and
  rendertarget formats) are stored as a JSON description that the
  MultiPassPipeline can reconstruct on reload.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from OpenGL import GL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: stable JSON serialisation
# ---------------------------------------------------------------------------


def _dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(obj)


def _hash(obj: Any) -> str:
    return hashlib.sha256(_dumps(obj).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Driver / GL version fingerprinting
# ---------------------------------------------------------------------------


class DriverFingerprint:
    """Lightweight driver identity so the cache knows when binaries are
    stale.

    On modern Mesa / NVIDIA / AMD the driver version and GL renderer
    string are stable across runs on the same install, and they change
    when a driver update invalidates cached binaries.
    """

    def __init__(self):
        self.renderer = self._safe_query(GL.GL_RENDERER)
        self.version = self._safe_query(GL.GL_VERSION)
        self.vendor = self._safe_query(GL.GL_VENDOR)
        self.sl_version = self._safe_query(GL.GL_SHADING_LANGUAGE_VERSION)
        # ARB_get_program_binary availability — determinant for whether we
        # can persist compiled binaries at all.
        ext_str = self._get_extensions()
        self.has_program_binary = "GL_ARB_get_program_binary" in ext_str
        if not self.has_program_binary:
            self.has_program_binary = "GL_OES_get_program_binary" in ext_str
        self._id = _hash(
            {
                "renderer": self.renderer,
                "version": self.version,
                "vendor": self.vendor,
                "sl": self.sl_version,
            }
        )

    @staticmethod
    def _safe_str(val) -> str:
        if val is None:
            return ""
        if isinstance(val, bytes):
            return val.decode("utf-8", errors="replace")
        return str(val)

    @staticmethod
    def _safe_query(enum: int) -> str:
        """Query a string parameter, returning "" if the call fails."""
        try:
            return DriverFingerprint._safe_str(GL.glGetString(enum))
        except Exception:
            DriverFingerprint._clear_errors()
            return ""

    @staticmethod
    def _clear_errors() -> None:
        """Drain the GL error queue so a failed probe leaves no stale flag."""
        try:
            while GL.glGetError() != GL.GL_NO_ERROR:
                pass
        except Exception:
            pass

    @staticmethod
    def _get_extensions() -> str:
        """Return the space-joined extension string.

        ``glGetString(GL_EXTENSIONS)`` was removed in OpenGL 3.1 core
        profiles (it generates GL_INVALID_ENUM there), so prefer
        ``glGetStringi``/``GL_NUM_EXTENSIONS`` and only fall back to the
        legacy call when the modern path is unavailable (GLES 2 / GL 2.x).
        """
        try:
            count = GL.glGetIntegerv(GL.GL_NUM_EXTENSIONS)
            if isinstance(count, (list, tuple)):
                count = int(count[0]) if count else 0
            else:
                count = int(count)
            names = []
            for i in range(count):
                names.append(DriverFingerprint._safe_str(GL.glGetStringi(GL.GL_EXTENSIONS, i)))
            return " ".join(names)
        except Exception:
            DriverFingerprint._clear_errors()
        try:
            return DriverFingerprint._safe_str(GL.glGetString(GL.GL_EXTENSIONS))
        except Exception:
            DriverFingerprint._clear_errors()
            return ""

    def __str__(self) -> str:
        status = "binary-capable" if self.has_program_binary else "source-only"
        return f"DriverFingerprint(id={self._id[:16]}, {status})"


# ---------------------------------------------------------------------------
# Cache entry dataclasses (serialised to JSON on disk)
# ---------------------------------------------------------------------------


@dataclass
class SourceEntry:
    """A cached GLSL source fragment."""

    source: str
    shader_type: int = 0  # GL.GL_VERTEX_SHADER / GL.GL_FRAGMENT_SHADER
    genome_hash: str = ""
    specialization_key: str = ""
    driver_id: str = ""
    created_ts: float = field(default_factory=time.time)
    accessed_ts: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0

    def __post_init__(self):
        if isinstance(self.size_bytes, str):
            self.size_bytes = (
                int(self.size_bytes) if self.size_bytes.isdigit() else len(self.source.encode())
            )
        if not self.size_bytes and self.source:
            self.size_bytes = len(self.source.encode())


@dataclass
class BinaryEntry:
    """A cached compiled program binary (if extension available)."""

    binary_data: bytes = b""
    binary_format: int = 0
    source_hash: str = ""  # hash of source used to build this binary
    driver_id: str = ""
    created_ts: float = field(default_factory=time.time)
    accessed_ts: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0

    def __post_init__(self):
        if isinstance(self.binary_format, str):
            try:
                self.binary_format = int(self.binary_format)
            except ValueError:
                self.binary_format = 0
        if isinstance(self.size_bytes, str):
            self.size_bytes = (
                int(self.size_bytes) if self.size_bytes.isdigit() else len(self.binary_data)
            )
        if not self.size_bytes and self.binary_data:
            self.size_bytes = len(self.binary_data)


@dataclass
class PipelineEntry:
    """Serialised pipeline / multi-pass configuration.

    Stores the pass graph, FBO render-target formats, and any shader
    keys so the pipeline can be reconstructed without re-generating
    topology.
    """

    passes: list = field(default_factory=list)
    rt_formats: list = field(default_factory=list)
    shader_keys: list = field(default_factory=list)
    global_params: dict = field(default_factory=dict)
    driver_id: str = ""
    created_ts: float = field(default_factory=time.time)
    accessed_ts: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0

    def __post_init__(self):
        if isinstance(self.size_bytes, str):
            self.size_bytes = (
                int(self.size_bytes) if self.size_bytes.isdigit() else len(_dumps(self).encode())
            )
        if not self.size_bytes:
            self.size_bytes = len(_dumps(self).encode())


# ---------------------------------------------------------------------------
# ShaderDiskCache
# ---------------------------------------------------------------------------


class ShaderDiskCache:
    """Disk-backed shader cache.

    Directory layout
    ----------------
    <cache_dir>/
        index.json          -- LRU index (all entries)
        sources/<key>.frag  -- GLSL fragment source
        sources/<key>.vert  -- GLSL vertex source
        binaries/<key>.bin  -- driver-native shader binary
        pipelines/<key>.json -- serialised pipeline config

    Parameters
    ----------
    cache_dir : str | Path
        Root directory for the cache.  Created if missing.
    max_size : int
        Maximum number of entries across all tiers.
    """

    def __init__(self, cache_dir: Any, max_size: int = 256):
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.max_size = max(1, int(max_size))
        self._lock = threading.RLock()
        self._fp: DriverFingerprint | None = None
        self._sources_dir = self.cache_dir / "sources"
        self._binaries_dir = self.cache_dir / "binaries"
        self._pipelines_dir = self.cache_dir / "pipelines"
        self._index_path = self.cache_dir / "index.json"
        # In-memory lookups are layered over disk I/O.
        self._src_mem: dict[str, SourceEntry] = {}
        self._bin_mem: dict[str, BinaryEntry] = {}
        self._pipe_mem: dict[str, PipelineEntry] = {}
        # LRU ordering: key -> timestamp (most-recent first).
        self._lru: OrderedDict[str, float] = OrderedDict()
        self._ensure_dirs()
        self._load_index()

    # -- lifecycle ----------------------------------------------------------

    def init_gl(self) -> DriverFingerprint | None:
        """Initialise the driver fingerprint.

        Never raises: if fingerprinting fails the cache degrades to
        driver-unaware mode (empty driver ids, source-only caching).
        """
        with self._lock:
            if self._fp is None:
                try:
                    self._fp = DriverFingerprint()
                    logger.info("ShaderDiskCache initialised: %s", self._fp)
                except Exception as exc:
                    logger.warning("ShaderDiskCache driver fingerprint failed: %s", exc)
            return self._fp

    @property
    def driver(self) -> DriverFingerprint | None:
        return self._fp

    def clear(self) -> None:
        with self._lock:
            for d in (self._sources_dir, self._binaries_dir, self._pipelines_dir):
                try:
                    for f in d.glob("*"):
                        f.unlink()
                except OSError:
                    logger.debug("ShaderDiskCache clear failed to unlink entries in %s", d)
            self._src_mem.clear()
            self._bin_mem.clear()
            self._pipe_mem.clear()
            self._lru.clear()
            self._save_index()

    # -- directory helpers --------------------------------------------------

    def _ensure_dirs(self) -> None:
        for d in (self.cache_dir, self._sources_dir, self._binaries_dir, self._pipelines_dir):
            d.mkdir(parents=True, exist_ok=True)

    # -- index (LRU) -------------------------------------------------------

    def _load_index(self) -> None:
        if not self._index_path.exists():
            return
        try:
            with open(self._index_path, encoding="utf-8") as f:
                data = json.load(f)
            for tier in ("sources", "binaries", "pipelines"):
                for key, ts in data.get(tier, {}).items():
                    self._lru[(tier, key)] = float(ts)
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("ShaderDiskCache index load failed: %s", exc)

    def _save_index(self) -> None:
        index: dict[str, dict[str, float]] = {"sources": {}, "binaries": {}, "pipelines": {}}
        for (tier, key), ts in self._lru.items():
            index[tier][key] = ts
        try:
            tmp = self._index_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=1)
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(self._index_path)
        except OSError as exc:
            logger.debug("ShaderDiskCache index save failed: %s", exc)

    def _touch(self, tier: str, key: str) -> None:
        now = time.time()
        self._lru[(tier, key)] = now
        # Move any existing entry to the end.
        if (tier, key) in self._lru:
            # OrderedDict already tracks order via dict keys, but we
            # re-insert to promote recency.
            del self._lru[(tier, key)]
        self._lru[(tier, key)] = now
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        # Count active entries across all tiers.
        total = 0
        tier_counts: dict[str, int] = {}
        for tier, _key in self._lru:
            total += 1
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        if total <= self.max_size:
            return
        # Evict the stalest entries (beginning of OrderedDict).
        excess = total - self.max_size
        evicted = 0
        while evicted < excess and self._lru:
            (tier, key), _ = next(iter(self._lru.items()))
            self._evict_key(tier, key)
            evicted += 1
        self._save_index()

    def _evict_key(self, tier: str, key: str) -> None:
        try:
            del self._lru[(tier, key)]
        except KeyError:
            pass
        try:
            if tier == "sources":
                f = self._sources_dir / key
                if f.exists():
                    f.unlink()
                self._src_mem.pop(key, None)
            elif tier == "binaries":
                f = self._binaries_dir / key
                if f.exists():
                    f.unlink()
                self._bin_mem.pop(key, None)
            elif tier == "pipelines":
                f = self._pipelines_dir / key
                if f.exists():
                    f.unlink()
                self._pipe_mem.pop(key, None)
        except OSError as exc:
            logger.debug("ShaderDiskCache evict %s/%s failed: %s", tier, key, exc)

    # -- key composition ---------------------------------------------------

    def _make_key(
        self,
        genome_hash: str,
        *,
        driver_id: str = "",
        specialization: str = "",
        suffix: str = "frag",
    ) -> str:
        parts = [
            genome_hash,
            driver_id or (self._fp._id if self._fp else ""),
            specialization,
            suffix,
        ]
        combined = "|".join(parts)
        return hashlib.sha256(combined.encode()).hexdigest()

    def _source_key(
        self, genome_hash: str, shader_type: int, specialization: str = "", driver_id: str = ""
    ) -> str:
        suffix = "frag" if shader_type == GL.GL_FRAGMENT_SHADER else "vert"
        return self._make_key(
            genome_hash, driver_id=driver_id, specialization=specialization, suffix=suffix
        )

    def _binary_key(self, source_hash: str, driver_id: str = "") -> str:
        combined = "|".join([source_hash, driver_id or (self._fp._id if self._fp else ""), "bin"])
        return hashlib.sha256(combined.encode()).hexdigest()

    def _pipeline_key(self, genome_hash: str, driver_id: str = "") -> str:
        return self._make_key(genome_hash, driver_id=driver_id, suffix="pipeline")

    # -- source cache ------------------------------------------------------

    def put_source(
        self,
        key: str,
        source: str,
        *,
        shader_type: int = GL.GL_FRAGMENT_SHADER,
        genome_hash: str = "",
        specialization: str = "",
    ) -> None:
        with self._lock:
            entry = SourceEntry(
                source=source,
                shader_type=int(shader_type),
                genome_hash=genome_hash,
                specialization_key=specialization,
                driver_id=self._fp._id if self._fp else "",
                size_bytes=len(source.encode("utf-8")),
            )
            self._src_mem[key] = entry
            self._touch("sources", key)
            # Persist asynchronously is fine since this is called from the
            # render thread. We use a blocking write because the source is
            # already in memory and the write is fast.
            path = self._sources_dir / key
            try:
                tmp = path.with_suffix(".tmp")
                with open(tmp, "w", encoding="utf-8", newline="") as f:
                    f.write(source)
                    f.flush()
                    os.fsync(f.fileno())
                tmp.replace(path)
            except OSError as exc:
                logger.debug("ShaderDiskCache source put failed: %s", exc)

    def get_source(self, key: str) -> str | None:
        with self._lock:
            entry = self._src_mem.get(key)
            if entry is not None:
                entry.accessed_ts = time.time()
                entry.access_count += 1
                self._touch("sources", key)
                self._save_index()
                return entry.source
            # Disk fallback.
            path = self._sources_dir / key
            if not path.exists():
                return None
            try:
                with open(path, encoding="utf-8") as f:
                    source = f.read()
                entry = SourceEntry(
                    source=source,
                    genome_hash="",
                    specialization_key="",
                    driver_id="",
                    size_bytes=len(source.encode("utf-8")),
                )
                self._src_mem[key] = entry
                self._touch("sources", key)
                self._save_index()
                return source
            except OSError as exc:
                logger.debug("ShaderDiskCache source get failed: %s", exc)
                return None

    # -- binary cache (requires ARB_get_program_binary) --------------------

    def put_binary(
        self,
        key: str,
        binary_data: bytes,
        binary_format: int,
        source_hash: str = "",
        driver_id: str = "",
    ) -> None:
        with self._lock:
            if not (self._fp and self._fp.has_program_binary):
                return
            entry = BinaryEntry(
                binary_data=binary_data,
                binary_format=int(binary_format),
                source_hash=source_hash,
                driver_id=driver_id or (self._fp._id if self._fp else ""),
                size_bytes=len(binary_data),
            )
            self._bin_mem[key] = entry
            self._touch("binaries", key)
            bin_path = self._binaries_dir / key
            meta_path = self._binaries_dir / f"{key}.json"
            try:
                tmp_bin = bin_path.with_suffix(".tmp")
                with open(tmp_bin, "wb") as f:
                    f.write(binary_data)
                    f.flush()
                    os.fsync(f.fileno())
                tmp_bin.replace(bin_path)
                meta = {
                    "binary_format": int(binary_format),
                    "size_bytes": len(binary_data),
                }
                tmp_meta = meta_path.with_suffix(".tmp")
                with open(tmp_meta, "w", encoding="utf-8") as f:
                    json.dump(meta, f)
                    f.flush()
                    os.fsync(f.fileno())
                tmp_meta.replace(meta_path)
            except OSError as exc:
                logger.debug("ShaderDiskCache binary put failed: %s", exc)

    def get_binary(self, key: str) -> tuple[int, bytes] | None:
        """Return ``(binary_format, binary_data)`` if available, else None."""
        with self._lock:
            entry = self._bin_mem.get(key)
            if entry is not None and entry.binary_data:
                entry.accessed_ts = time.time()
                entry.access_count += 1
                self._touch("binaries", key)
                self._save_index()
                return (entry.binary_format, entry.binary_data)
            path = self._binaries_dir / key
            meta_path = self._binaries_dir / f"{key}.json"
            if not path.exists():
                return None
            try:
                data = path.read_bytes()
                binary_format = 0
                if meta_path.exists():
                    try:
                        with open(meta_path, encoding="utf-8") as f:
                            meta = json.load(f)
                        binary_format = int(meta.get("binary_format", 0))
                    except (OSError, json.JSONDecodeError) as exc:
                        logger.debug("ShaderDiskCache binary meta load failed: %s", exc)
                entry = BinaryEntry(
                    binary_data=data,
                    binary_format=binary_format,
                    source_hash="",
                    driver_id="",
                    size_bytes=len(data),
                )
                self._bin_mem[key] = entry
                self._touch("binaries", key)
                self._save_index()
                return (binary_format, data)
            except OSError as exc:
                logger.debug("ShaderDiskCache binary get failed: %s", exc)
                return None

    def extract_binary(self, program: int) -> tuple[int, bytes] | None:
        """Call ``glGetProgramBinary`` on ``program`` if available.

        Returns the binary blob and its format token on success, or None
        if the extension is missing / the call fails.
        """
        if not (self._fp and self._fp.has_program_binary):
            return None
        try:
            length = GL.glGetProgramiv(program, GL.GL_PROGRAM_BINARY_LENGTH)
            if length <= 0:
                return None
            buf = GL.glGetProgramBinary(program, length)
            # glGetProgramBinary signature varies; attempt both orderings.
            if isinstance(buf, tuple):
                if len(buf) == 2 and isinstance(buf[1], int):
                    binary_format, data = int(buf[1]), bytes(buf[0])
                elif len(buf) == 2 and isinstance(buf[0], int):
                    binary_format, data = int(buf[0]), bytes(buf[1])
                else:
                    return None
            else:
                return None
            return (binary_format, data)
        except (RuntimeError, GL.error.GLError) as exc:
            logger.debug("ShaderDiskCache extract_binary failed: %s", exc)
            return None

    def try_load_binary(self, target: int, key: str) -> bool:
        """Attempt to load a cached binary into target using glProgramBinary.

        Returns True on success.
        """
        if not (self._fp and self._fp.has_program_binary):
            return False
        entry = self._bin_mem.get(key)
        if entry is None:
            return False
        try:
            GL.glProgramBinary(target, entry.binary_format, entry.binary_data)
            # Verify link succeeded.
            linked = GL.glGetProgramiv(target, GL.GL_LINK_STATUS)
            return bool(linked)
        except (RuntimeError, GL.error.GLError) as exc:
            logger.debug("ShaderDiskCache try_load_binary failed: %s", exc)
            # Remove a bad binary so we don't keep retrying.
            self._bin_mem.pop(key, None)
            return False

    # -- pipeline cache ----------------------------------------------------

    def put_pipeline(self, key: str, config: dict[str, Any], driver_id: str = "") -> None:
        with self._lock:
            entry = PipelineEntry(
                passes=config.get("passes", []),
                rt_formats=config.get("rt_formats", []),
                shader_keys=config.get("shader_keys", []),
                global_params=config.get("global_params", {}),
                driver_id=driver_id or (self._fp._id if self._fp else ""),
                size_bytes=len(_dumps(config).encode("utf-8")),
            )
            self._pipe_mem[key] = entry
            self._touch("pipelines", key)
            path = self._pipelines_dir / key
            try:
                payload = json.dumps(asdict(entry), indent=1, default=str)
                tmp = path.with_suffix(".tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(payload)
                    f.flush()
                    os.fsync(f.fileno())
                tmp.replace(path)
            except OSError as exc:
                logger.debug("ShaderDiskCache pipeline put failed: %s", exc)

    def get_pipeline(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._pipe_mem.get(key)
            if entry is not None:
                entry.accessed_ts = time.time()
                entry.access_count += 1
                self._touch("pipelines", key)
                return asdict(entry)
            path = self._pipelines_dir / key
            if not path.exists():
                return None
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                entry = PipelineEntry(**data)
                self._pipe_mem[key] = entry
                self._touch("pipelines", key)
                self._save_index()
                return asdict(entry)
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug("ShaderDiskCache pipeline get failed: %s", exc)
                return None

    # -- stats --------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "sources_in_mem": len(self._src_mem),
                "binaries_in_mem": len(self._bin_mem),
                "pipelines_in_mem": len(self._pipe_mem),
                "lru_total_entries": len(self._lru),
                "max_size": self.max_size,
                "has_program_binary": (
                    bool(self._fp and self._fp.has_program_binary) if self._fp else False
                ),
                "driver_id": self._fp._id[:16] + "…" if self._fp else "uninitialised",
            }
