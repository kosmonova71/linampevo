#!/usr/bin/env python3
"""
ASCEND V9 - Shader Cache (Hash-based Lookup)

Cache compiled shaders using SHA256 hash of IR bytes.
Deterministic: same IR always produces same cache key.
"""

from __future__ import annotations

from engine.ir.ssa import IRBlock, hash_ir


class ShaderCache:
    """Cache for compiled GLSL shaders.

    Uses SHA256 hash of IR bytes as cache key.
    Enables cross-preset shader reuse.
    """

    def __init__(self, max_size: int = 1000):
        """Initialize shader cache.

        Args:
            max_size: Maximum number of cached shaders
        """
        self.cache: dict[str, str] = {}  # hash → GLSL source
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.access_order: list[str] = []  # Track access order for LRU eviction

    def get(self, block: IRBlock) -> str | None:
        """Get cached shader for IR block.

        Args:
            block: SSA IR block

        Returns:
            Cached GLSL source if found, None otherwise
        """
        key = hash_ir(block)

        if key in self.cache:
            self.hits += 1
            # Move to end of access_order (most recently used)
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]

        self.misses += 1
        return None

    def put(self, block: IRBlock, glsl: str) -> None:
        """Cache shader for IR block.

        Args:
            block: SSA IR block
            glsl: GLSL source code
        """
        key = hash_ir(block)

        # Evict least recently used if cache is full and key not already in cache
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_lru()

        self.cache[key] = glsl
        # Add to access_order if not already present
        if key not in self.access_order:
            self.access_order.append(key)

    def _evict_lru(self, count: int = 1) -> None:
        """Evict least recently used item(s) from cache.

        Args:
            count: Number of items to evict
        """
        for _ in range(count):
            if not self.access_order:
                return
            lru_key = self.access_order.pop(0)
            if lru_key in self.cache:
                del self.cache[lru_key]

    def invalidate(self, block: IRBlock) -> None:
        """Invalidate cache entry for IR block.

        Args:
            block: SSA IR block
        """
        key = hash_ir(block)
        if key in self.cache:
            del self.cache[key]
        if key in self.access_order:
            self.access_order.remove(key)

    def clear(self) -> None:
        """Clear entire cache."""
        self.cache.clear()
        self.access_order.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }

    def __len__(self):
        return len(self.cache)

    def __contains__(self, block: IRBlock):
        return hash_ir(block) in self.cache


# Global shader cache instance
_global_shader_cache: ShaderCache | None = None


def get_shader_cache(max_size: int = 1000) -> ShaderCache:
    """Get global shader cache instance.

    Args:
        max_size: Maximum cache size

    Returns:
        ShaderCache instance
    """
    global _global_shader_cache
    if _global_shader_cache is None:
        _global_shader_cache = ShaderCache(max_size)
    return _global_shader_cache
