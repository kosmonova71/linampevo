#!/usr/bin/env python3
"""Per-module GLSL source cache with dirty tracking.

Each ShaderModule receives a stable cache key.  Identical selections hit the
cache and bypass Python-side code generation.  The combined module
fingerprint enables frame-to-frame short-circuit: when the fingerprint
matches the currently-bound program, the renderer skips the entire
compilation pipeline and draws with the existing program.

In OpenGL 3.3, a fragment shader must compile as a single unit, so the
cache targets the *source generation* step.  The resulting full source is
fed through the existing binary program cache (``ShaderDiskCache`` + in-memory
``ShaderCache``) which skips ``glCompileShader``/``glLinkProgram`` entirely
when the combined fingerprint is unchanged.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from .base import ModuleSections

# ============================================================================
# Per-module identity and caching
# ============================================================================


@dataclass(frozen=True, order=True)
class ModuleKey:
    """Stable cache identity for a ShaderModule."""

    category: str
    name: str
    param_hash: str


@dataclass
class ModuleCacheEntry:
    """One cached module's GLSL sections and derived hash."""

    sections: ModuleSections
    hash: str
    hits: int = 0


class ModuleSourceCache:
    """LRU cache for generated module GLSL sections.

    Usage::

        cache = ModuleSourceCache()
        key = cache.key_for("field", "fbm", {"scale": 2.0, "octaves": 4})
        sections = cache.get(key)
        if sections is None:
            module = registry.create("field", "fbm", params)
            sections = module.sections()
            cache.put(key, sections)
    """

    def __init__(self, max_size: int = 512):
        self._cache: OrderedDict[ModuleKey, ModuleCacheEntry] = OrderedDict()
        self.max_size = max(1, int(max_size))
        self.hits = 0
        self.misses = 0

    # ---- Identity --------------------------------------------------------

    @staticmethod
    def _hash_params(params: dict[str, Any] | None) -> str:
        if not params:
            return "empty"
        try:
            raw = json.dumps(params, sort_keys=True, default=str).encode()
            return hashlib.sha256(raw).hexdigest()[:16]
        except (TypeError, ValueError):
            return "fallback"

    def key_for(self, category: str, name: str, params: dict[str, Any] | None = None) -> ModuleKey:
        """Build a stable cache key for a module."""
        return ModuleKey(category, name, self._hash_params(params))

    # ---- CRUD ------------------------------------------------------------

    def get(self, key: ModuleKey) -> ModuleSections | None:
        entry = self._cache.get(key)
        if entry is not None:
            self._cache.move_to_end(key)
            entry.hits += 1
            self.hits += 1
            return entry.sections
        self.misses += 1
        return None

    def put(self, key: ModuleKey, sections: ModuleSections) -> None:
        sec_hash = _section_hash(sections)
        entry = ModuleCacheEntry(sections=sections, hash=sec_hash)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = entry
            return
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        self._cache[key] = entry

    def module_hash(self, key: ModuleKey) -> str:
        entry = self._cache.get(key)
        return entry.hash if entry else ""

    def invalidate(self, key: ModuleKey | None = None) -> None:
        if key is None:
            self._cache.clear()
        else:
            self._cache.pop(key, None)

    def stats(self) -> dict[str, Any]:
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
        }


# ============================================================================
# Full-assembly fingerprint
# ============================================================================


def _section_hash(sections: ModuleSections) -> str:
    combined = (sections.header or "") + (sections.footer or "") + (sections.call or "")
    try:
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    except (TypeError, ValueError):
        return "unknown"


class ModuleFingerprint:
    """Combined fingerprint of a full shader assembly's modules.

    Tracks each module category's identity.  A matching fingerprint means
    the full shader source is byte-identical, enabling the renderer to
    short-circuit to the existing program.
    """

    def __init__(self):
        self._items: list[tuple[str, str]] = []

    def update(self, category: str, hash_value: str) -> None:
        self._items.append((category, hash_value))

    def combined_hash(self) -> str:
        try:
            material = "|".join(f"{k}:{v}" for k, v in sorted(self._items))
            return hashlib.sha256(material.encode()).hexdigest()[:16]
        except (TypeError, ValueError):
            return ""

    def matches(self, other: ModuleFingerprint) -> bool:
        return self._items == other._items

    def __repr__(self):
        items = ", ".join(f"{k}={v}" for k, v in self._items)
        return f"ModuleFingerprint([{items}])"
