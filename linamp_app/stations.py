"""Radio station model, on-disk config, and the station logo cache."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .constants import RADIO_LOGO_CACHE_DIR, RADIO_STATIONS_FILE
from .deps import PIL_AVAILABLE, REQUESTS_AVAILABLE, BytesIO, Image, requests

DEFAULT_RADIO_STATIONS = [
    {
        "name": "AfterHours FM (AH.FM)",
        "genre": "Trance",
        "url": "http://198.244.201.21:8000/live",
        "logo": "https://ah.fm/wp-content/uploads/2023/03/transparent-white.png",
        "homepage": "https://ah.fm/",
    },
    {
        "name": "Party107",
        "genre": "Trance",
        "url": "https://party107.com/listen/party107-192.mp3",
        "logo": "https://www.party107.com/images/logo.jpg",
        "homepage": "https://www.party107.com/",
    },
    {
        "name": "Ganorium Radio",
        "genre": "Uplifting Trance",
        "url": "https://stream.ganorium.com",
        "logo": "https://ik.imagekit.io/8lskxw561nyg/ganoriumradio/wp-content/uploads/2024/06/ganorium-radio-2024-icon.png",
        "homepage": "https://www.ganorium.com/",
    },
    {
        "name": "SKellyWays Radio",
        "genre": "Trance",
        "url": "https://s31.ssl-stream.com:8200/radio.mp3",
        "logo": "https://skellywaysradio.com/SkellyWaysBadge.png",
        "homepage": "https://skellywaysradio.com/",
    },
    {
        "name": "SKellyWays Classic Trance",
        "genre": "Classic Trance",
        "url": "https://s31.ssl-stream.com:8210/radio.mp3",
        "logo": "https://skellywaysradio.com/SkellyWaysBadge.png",
        "homepage": "https://skellywaysradio.com/",
    },
    {
        "name": "SomaFM - The Trip",
        "genre": "Progressive Trance",
        "url": "https://ice5.somafm.com/thetrip-128-mp3",
        "logo": "https://somafm.com/logos/400/thetrip400.jpg",
        "homepage": "https://somafm.com/thetrip/",
    },
    {
        "name": "Intense Radio",
        "genre": "Dance / Trance",
        "url": "https://stream.intenseradio.net:8000/stream",
        "logo": "https://www.intenseradio.net/wp-content/uploads/2023/09/intense2017-logo3.png",
        "homepage": "https://www.intenseradio.net/",
    },
]


def load_radio_stations() -> list[dict[str, str]]:
    defaults_by_name = {s.get("name", ""): s for s in DEFAULT_RADIO_STATIONS}

    def _merge_defaults(stations: list) -> list[dict[str, str]]:
        """Fill missing logo/homepage fields from built-in defaults."""
        merged = []
        for s in stations:
            if not isinstance(s, dict):
                continue
            item = dict(s)
            base = defaults_by_name.get(item.get("name", ""), {})
            if not item.get("logo") and base.get("logo"):
                item["logo"] = base["logo"]
            if not item.get("homepage") and base.get("homepage"):
                item["homepage"] = base["homepage"]
            merged.append(item)
        return merged

    if not RADIO_STATIONS_FILE.exists():
        return DEFAULT_RADIO_STATIONS.copy()
    try:
        with open(RADIO_STATIONS_FILE, encoding="utf-8") as f:
            stations = json.load(f)
        if isinstance(stations, list) and stations:
            return _merge_defaults(stations)
    except (json.JSONDecodeError, OSError):
        logging.getLogger(__name__).debug("Failed to load radio stations config, using defaults")
    return DEFAULT_RADIO_STATIONS.copy()


def save_radio_stations(stations: list[dict[str, str]]) -> bool:
    config_dir = RADIO_STATIONS_FILE.parent
    config_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(RADIO_STATIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(stations, f, indent=2, ensure_ascii=False)
        return True
    except OSError:
        return False


def _radio_logo_cache_path(logo_url: str) -> Path:
    key = hashlib.md5(logo_url.encode("utf-8")).hexdigest()[:16]
    # Prefer png extension for cached files
    return RADIO_LOGO_CACHE_DIR / f"{key}.png"


_FAILED_LOGO_URLS: set[str] = set()


def fetch_radio_logo(logo_url: str, timeout: float = 8.0) -> Path | None:
    """Download a station logo to the local cache and return its path.

    Returns None if the logo cannot be fetched or requests/PIL are unavailable.
    Failed URLs are remembered for the session so dead hosts are not retried
    on every list rebind; transient errors get one retry.
    """
    if not logo_url or not isinstance(logo_url, str):
        return None
    if not logo_url.startswith(("http://", "https://")):
        # Local path support
        p = Path(logo_url).expanduser()
        return p if p.is_file() else None
    if logo_url in _FAILED_LOGO_URLS:
        return None
    cache_path = _radio_logo_cache_path(logo_url)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path
    if not REQUESTS_AVAILABLE or not PIL_AVAILABLE:
        return None
    try:
        RADIO_LOGO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        session = requests.Session()
        session.headers.update(
            {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 LinAmp/1.0"}
        )
        resp = None
        for attempt in range(2):
            try:
                resp = session.get(logo_url, timeout=timeout, verify=True)
                break
            except requests.RequestException:
                if attempt == 1:
                    raise
                time.sleep(0.75)
        if resp.status_code != 200 or not resp.content:
            return None
        img = Image.open(BytesIO(resp.content))
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
        else:
            img = img.convert("RGB")
        img.thumbnail((128, 128), Image.Resampling.LANCZOS)
        img.save(cache_path, format="PNG", optimize=True)
        _FAILED_LOGO_URLS.discard(logo_url)
        return cache_path
    except (requests.RequestException, OSError, Image.UnidentifiedImageError) as e:
        _FAILED_LOGO_URLS.add(logo_url)
        logging.getLogger(__name__).debug("Failed to fetch radio logo from %s: %s", logo_url, e)
        return None


def preload_radio_logos(stations: list[dict]) -> None:
    """Best-effort background preload of station logos into the cache."""
    if not REQUESTS_AVAILABLE or not PIL_AVAILABLE:
        return

    def _worker() -> None:
        for s in stations:
            url = (s or {}).get("logo") or ""
            if url:
                try:
                    fetch_radio_logo(url)
                except Exception as e:
                    logging.getLogger(__name__).debug("Radio logo preload failed: %s", e)

    threading.Thread(target=_worker, daemon=True, name="RadioLogoPreload").start()


@dataclass
class RadioStation:
    name: str = ""
    genre: str = ""
    url: str = ""
    logo: str = ""
    homepage: str = ""

    def to_dict(self) -> dict[str, str]:
        d = {"name": self.name, "genre": self.genre, "url": self.url}
        if self.logo:
            d["logo"] = self.logo
        if self.homepage:
            d["homepage"] = self.homepage
        return d

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> RadioStation:
        return cls(
            name=data.get("name", ""),
            genre=data.get("genre", ""),
            url=data.get("url", ""),
            logo=data.get("logo", ""),
            homepage=data.get("homepage", ""),
        )

    @staticmethod
    def validate_url(url: str) -> bool:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https")
