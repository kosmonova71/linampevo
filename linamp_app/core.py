"""Shared helpers for podcast/feed fetching and on-disk state.

This module is imported by :mod:`linamp_app.podcasts`; it must not import
anything from ``podcasts`` or the UI layer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import socket
import sys
import threading
from pathlib import Path
from urllib.parse import urlparse

from .constants import CACHE_DIR, CONFIG_DIR
from .deps import (
    PIL_AVAILABLE,
    REQUESTS_AVAILABLE,
    BytesIO,
    Image,
    requests,
)

_PODCAST_STATE_LOCK = threading.Lock()

MAX_EPISODES_PER_FEED = 500
MAX_FEED_CONTENT_SIZE = 10 * 1024 * 1024  # 10 MiB

try:
    import defusedxml.ElementTree as DefusedET
except ImportError:  # pragma: no cover
    from xml.etree import ElementTree as DefusedET  # noqa: F401

PODCASTS_CONFIG_DIR = CONFIG_DIR / "podcasts"
PODCASTS_FILE = CONFIG_DIR / "podcasts.json"
PODCAST_EPISODES_DIR = CONFIG_DIR / "podcast_episodes"
PODCAST_STATE_JOURNAL = PODCASTS_CONFIG_DIR / "state.journal"
PODCAST_ART_CACHE_DIR = CACHE_DIR / "podcast_art"

_SAFE_FEED_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")


def _linamp_attr(name: str, default):
    """Resolve *name* on the podcasts module (or package) for test overrides."""
    for mod_name in ("linamp_app.podcasts", "linamp_app"):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, name):
            return getattr(mod, name)
    return default


def _is_safe_feed_id(feed_id: str) -> bool:
    """Feed IDs become filenames; reject anything path-like."""
    if not feed_id or not isinstance(feed_id, str):
        return False
    if ".." in feed_id or "/" in feed_id or "\\" in feed_id:
        return False
    return bool(_SAFE_FEED_ID_RE.match(feed_id))


def _fsync_dir(dir_path) -> None:
    try:
        fd = os.open(str(dir_path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write_json(path, data) -> bool:
    """Write *data* as JSON via a temp file + rename, fsyncing the directory."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        _fsync_dir(path.parent)
        return True
    except (OSError, TypeError, ValueError) as e:
        logging.getLogger(__name__).debug("Atomic JSON write failed for %s: %s", path, e)
        try:
            tmp_path.unlink()
        except (OSError, UnboundLocalError):
            pass
        return False


def _image_cache_path(url: str, cache_dir) -> Path:
    """Deterministic cache path for a remote image URL."""
    key = hashlib.md5(url.encode("utf-8")).hexdigest()[:24]
    ext = "jpg"
    try:
        suffix = Path(urlparse(url).path).suffix.lower().lstrip(".")
        if suffix in ("png", "jpg", "jpeg", "webp", "gif", "bmp"):
            ext = "png" if suffix in ("jpg", "jpeg") else suffix
    except Exception:
        pass
    return Path(cache_dir) / f"{key}.{ext}"


def _validate_url_dns(url: str) -> bool:
    """Accept only public-looking http(s) URLs whose host resolves."""
    try:
        parsed = urlparse(url or "")
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False
    hostname = parsed.hostname
    if not hostname:
        return False
    try:
        socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    except OSError:
        return True
    return True


def _safe_http_get(url: str, timeout=15.0, stream: bool = False, **kwargs):
    """GET *url* with a browser UA; returns the response or None on failure."""
    if not _validate_url_dns(url):
        logging.getLogger(__name__).warning("HTTP GET blocked by URL validation: %s", url)
        return None
    if not REQUESTS_AVAILABLE:
        return None
    try:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "LinAmp/1.0"
                )
            }
        )
        return session.get(url, timeout=timeout, stream=stream, verify=True, **kwargs)
    except requests.RequestException as e:
        logging.getLogger(__name__).debug("HTTP GET failed for %s: %s", url, e)
        return None


def fetch_image_to_cache(
    url: str,
    cache_dir,
    size: tuple[int, int] = (300, 300),
    timeout: float = 8.0,
) -> Path | None:
    """Download *url* to *cache_dir* (resized via PIL) and return the path."""
    if not url or not isinstance(url, str):
        return None
    if not url.startswith(("http://", "https://")):
        p = Path(url).expanduser()
        return p if p.is_file() else None
    if not REQUESTS_AVAILABLE or not PIL_AVAILABLE:
        return None
    cache_path = _image_cache_path(url, cache_dir)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        resp = _safe_http_get(url, timeout=timeout)
        if resp is None or resp.status_code != 200 or not resp.content:
            return None
        try:
            img = Image.open(BytesIO(resp.content))
            img.load()
        except Exception:
            return None
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
        else:
            img = img.convert("RGB")
        img.thumbnail(size, Image.Resampling.LANCZOS)
        img.save(cache_path, format="PNG", optimize=True)
        return cache_path
    except (OSError, ValueError) as e:
        logging.getLogger(__name__).debug("Image cache fetch failed for %s: %s", url, e)
        return None
