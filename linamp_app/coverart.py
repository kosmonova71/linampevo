"""Cover art discovery: embedded tags, folder scans, cache, and download."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote, urlparse

from .constants import COVER_CACHE_DIR
from .deps import (
    COVER_ART_DEPENDENCIES,
    COVER_DOWNLOAD_AVAILABLE,
    FLAC,
    ID3,
    MP3,
    MP4,
    BytesIO,
    File,
    Image,
    ImageEnhance,
    requests,
)

COVER_ART_PATTERNS = [
    "cover.jpg",
    "cover.jpeg",
    "cover.png",
    "cover.webp",
    "cover.gif",
    "cover.bmp",
    "folder.jpg",
    "folder.jpeg",
    "folder.png",
    "folder.webp",
    "folder.gif",
    "folder.bmp",
    "album.jpg",
    "album.jpeg",
    "album.png",
    "album.webp",
    "album.gif",
    "album.bmp",
    "art.jpg",
    "art.jpeg",
    "art.png",
    "art.webp",
    "art.gif",
    "art.bmp",
    "front.jpg",
    "front.jpeg",
    "front.png",
    "front.webp",
    "front.gif",
    "front.bmp",
    ".cover.jpg",
    ".cover.jpeg",
    ".cover.png",
    ".cover.webp",
    ".cover.gif",
    ".cover.bmp",
]


# Cache for computed cache keys to avoid repeated stat+md5
@lru_cache(maxsize=256)
def _get_cached_cache_key(audio_path: str) -> str:
    """Compute and cache the cache key for an audio file."""
    try:
        stat = os.stat(audio_path)
        file_data = f"{audio_path}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.md5(file_data.encode()).hexdigest()[:16]
    except OSError:
        return ""


@dataclass
class CoverArt:
    path: str = ""
    source: str = "embedded"
    width: int = 0
    height: int = 0
    format: str = "unknown"
    bytes_data: bytes | None = None

    @staticmethod
    def get_cache_key(audio_path: str) -> str:
        return _get_cached_cache_key(audio_path)

    @staticmethod
    def normalize_format(format_str: str) -> str:
        fmt = format_str.lower().strip(" .")
        if fmt in ("jpg", "jpeg"):
            return "jpg"
        if fmt in ("png", "gif", "webp", "bmp"):
            return fmt
        return "jpg"

    @classmethod
    def get_cache_path(cls, audio_path: str) -> Path | None:
        key = cls.get_cache_key(audio_path)
        if not key:
            return None
        # Use direct path construction instead of glob for speed
        for ext in ("jpg", "jpeg", "png", "webp", "gif", "bmp"):
            cache_path = COVER_CACHE_DIR / f"{key}.{ext}"
            if cache_path.exists():
                return cache_path
        return None

    def save_to_cache(self, audio_path: str) -> str | None:
        if not self.bytes_data:
            return None
        key = self.get_cache_key(audio_path)
        if not key:
            return None
        ext = self.normalize_format(self.format)
        cache_path = COVER_CACHE_DIR / f"{key}.{ext}"
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                f.write(self.bytes_data)
            return str(cache_path)
        except OSError:
            return None

    @classmethod
    def get_cache_path_for_audio(cls, audio_path: str) -> Path | None:
        return cls.get_cache_path(audio_path)

    @classmethod
    def get_cached_cover_path(cls, audio_path: str) -> Path | None:
        return cls.get_cache_path_for_audio(audio_path)

    @classmethod
    def load_from_cache(cls, audio_path: str) -> CoverArt | None:
        cache_path = cls.get_cache_path_for_audio(audio_path)
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    data = f.read()
                ext = cache_path.suffix.lower().lstrip(".")
                return cls(
                    path=str(cache_path),
                    source="cache",
                    bytes_data=data,
                    format=ext,
                )
            except OSError:
                return None
        return None


def _tag_key_matches(key: str | tuple, prefix: str) -> bool:
    if isinstance(key, str):
        return key.startswith(prefix)
    if isinstance(key, tuple):
        return any(isinstance(item, str) and item.startswith(prefix) for item in key)
    return str(key).startswith(prefix)


def _extract_cover_art_from_file(file_path: str) -> CoverArt | None:
    if not COVER_ART_DEPENDENCIES.get("mutagen", {}).get("available", False):
        return None
    audio = File(file_path)
    if audio is None:
        return None

    def _make_cover(data: bytes, fmt: str = "jpg") -> CoverArt:
        return CoverArt(
            path="",
            source="embedded",
            bytes_data=data,
            format=CoverArt.normalize_format(fmt),
        )

    def _try_prefix_tags() -> CoverArt | None:
        if not hasattr(audio, "tags") or not audio.tags:
            return None
        for tag_name in ["APIC:", "APIC", "cover", "Cover", "albumart", "coverart"]:
            if tag_name.endswith(":"):
                for key in audio.tags:
                    if _tag_key_matches(key, tag_name):
                        tag = audio.tags[key]
                        if hasattr(tag, "data"):
                            return _make_cover(tag.data, getattr(tag, "mime", "jpg"))
                        elif hasattr(tag, "value"):
                            return _make_cover(tag.value)
            else:
                if tag_name in audio.tags:
                    tag = audio.tags[tag_name]
                    if hasattr(tag, "data"):
                        return _make_cover(tag.data, getattr(tag, "mime", "jpg"))
                    elif hasattr(tag, "value"):
                        return _make_cover(tag.value)
        return None

    def _try_covr() -> CoverArt | None:
        if not (hasattr(audio, "tags") and audio.tags and "covr" in audio.tags):
            return None
        covr_data = audio.tags["covr"]
        if isinstance(covr_data, list) and covr_data:
            cover_image = covr_data[0]
            if hasattr(cover_image, "data"):
                return _make_cover(cover_image.data, getattr(cover_image, "mime", "jpg"))
            elif isinstance(cover_image, bytes):
                return _make_cover(cover_image)
            return None

    def _try_pictures() -> CoverArt | None:
        if hasattr(audio, "pictures") and audio.pictures:
            picture = audio.pictures[0]
            if hasattr(picture, "data") and picture.data:
                return _make_cover(picture.data, getattr(picture, "mime", "jpg"))
        return None

    def _try_alternatives() -> CoverArt | None:
        methods = [
            ("artwork", "artwork"),
            ("cover", "cover"),
            ("coverart", "coverart"),
            ("albumart", "albumart"),
            ("covers", "covers"),
            ("pictures", "pictures"),
            ("covr", "covr"),
        ]
        for attr, _key in methods:
            try:
                val = getattr(audio, attr, None)
                if val:
                    return _make_cover(val)
            except (AttributeError, TypeError, ValueError) as e:
                logging.getLogger(__name__).debug("Cover extraction failed: %s", e)
        return None

    for func in [_try_prefix_tags, _try_covr, _try_pictures, _try_alternatives]:
        result = func()
        if result:
            return result
    return None


def _find_cover_in_folder(audio_path: str) -> CoverArt | None:
    audio_dir = os.path.dirname(os.path.abspath(audio_path))
    if not os.path.isdir(audio_dir):
        return None
    for pattern in COVER_ART_PATTERNS:
        cover_path = os.path.join(audio_dir, pattern)
        if os.path.isfile(cover_path):
            try:
                with open(cover_path, "rb") as f:
                    data = f.read()
                ext = os.path.splitext(pattern)[1].lstrip(".")
                return CoverArt(path=cover_path, source="folder", bytes_data=data, format=ext)
            except OSError:
                logging.getLogger(__name__).debug(
                    "Failed to read cover art from %s: %s", cover_path, sys.exc_info()[1]
                )
    return None


def _validate_file_path(file_path: str, allow_symlinks: bool = False) -> bool:
    if not file_path or not isinstance(file_path, str):
        return False
    if "\0" in file_path:
        return False
    if not allow_symlinks and os.path.islink(file_path):
        return False
    return os.path.isfile(file_path)


def _validate_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    return parsed.scheme != "file"


def extract_cover_art(
    audio_path: str, allow_download: bool = False
) -> tuple[CoverArt | None, str | None]:
    if not _validate_file_path(audio_path):
        return None, None
    cached = CoverArt.load_from_cache(audio_path)
    if cached:
        return cached, cached.path if cached.path else None
    cover = _extract_cover_art_from_file(audio_path)
    if cover:
        cache_path = cover.save_to_cache(audio_path)
        if cache_path:
            cover.path = cache_path
        return cover, cache_path
    cover = _find_cover_in_folder(audio_path)
    if cover:
        cache_path = cover.save_to_cache(audio_path)
        if cache_path:
            cover.path = cache_path
        return cover, cache_path
    return cover, None


class CoverArtDownloader:
    def __init__(self) -> None:
        if not COVER_DOWNLOAD_AVAILABLE:
            raise ImportError("requests and PIL are required for cover downloading")
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        self.timeout = (3, 5)

    def get_metadata(self, file_path: str) -> tuple[str, str]:
        if not _validate_file_path(file_path):
            return "", ""
        if not COVER_ART_DEPENDENCIES.get("mutagen", {}).get("available"):
            return "", ""
        audio_path = Path(file_path)
        if audio_path.suffix.lower() == ".mp3":
            audio = MP3(file_path, ID3=ID3)
            artist = audio.tags.get("TPE1", [""])[0] if audio.tags else ""
            album = audio.tags.get("TALB", [""])[0] if audio.tags else ""
        elif audio_path.suffix.lower() == ".flac":
            audio = FLAC(file_path)
            artist = audio.get("artist", [""])[0]
            album = audio.get("album", [""])[0]
        elif audio_path.suffix.lower() in [".m4a", ".mp4"]:
            audio = MP4(file_path)
            artist = audio.get("\xa9ART", [""])[0] if audio.get("\xa9ART") else ""
            album = audio.get("\xa9alb", [""])[0] if audio.get("\xa9alb") else ""
        else:
            return "", ""
        return artist.strip(), album.strip()

    def search_cover(self, artist: str, album: str) -> str | None:
        if not artist or not album:
            return None
        query = f"{artist} {album} album cover"
        sources = [
            self._search_deezer,
            self._search_lastfm,
        ]
        for source_func in sources:
            url = source_func(query, artist, album)
            if url:
                return url
            time.sleep(0.5)
        return None

    def _search_deezer(self, query: str, artist: str, album: str) -> str | None:
        search_url = f"https://api.deezer.com/search/album?q={quote(query)}"
        if not _validate_url(search_url):
            return None
        response = self.session.get(search_url, timeout=self.timeout, verify=True)
        if response.status_code == 200:
            data = response.json()
            if data.get("data") and len(data["data"]) > 0:
                album_data = data["data"][0]
                cover_url = (
                    album_data.get("cover_xl")
                    or album_data.get("cover_big")
                    or album_data.get("cover_medium")
                )
                if cover_url and _validate_url(cover_url):
                    return cover_url

    def _search_google_images(self, query: str, artist: str, album: str) -> str | None:
        search_url = f"https://www.google.com/search?tbm=isch&q={quote(query)}"
        response = self.session.get(search_url, timeout=self.timeout)
        if response.status_code == 200:
            pattern = r'\["(https://[^"]+?\.(?:jpg|jpeg|png|webp))",\d+,\d+\]'
            matches = re.findall(pattern, response.text)
            if matches:
                for url in matches[:3]:
                    if self._is_likely_album_cover(url):
                        return url

    def _search_lastfm(self, query: str, artist: str, album: str) -> str | None:
        search_url = f"https://www.last.fm/music/{quote(artist)}/{quote(album)}"
        if not _validate_url(search_url):
            return None
        response = self.session.get(search_url, timeout=self.timeout, verify=True)
        if response.status_code == 200:
            pattern = r'https://lastfm\.freetls\.fastly\.net/i/u/[^"]+\.(?:jpg|jpeg|png)'
            matches = re.findall(pattern, response.text)
            if matches:
                valid_matches = [url for url in matches if _validate_url(url)]
                if valid_matches:
                    return max(valid_matches, key=len)

    def _is_likely_album_cover(self, url: str) -> bool:
        exclude_patterns = ["icon", "logo", "avatar", "thumb", "small", "tiny"]
        url_lower = url.lower()
        for pattern in exclude_patterns:
            if pattern in url_lower:
                return False
        return True

    def download_cover(self, url: str) -> bytes | None:
        if not _validate_url(url):
            return None
        response = self.session.get(url, timeout=self.timeout, verify=True)
        if response.status_code != 200:
            return None
        img = Image.open(BytesIO(response.content))
        if img.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = background
        img.thumbnail((500, 500), Image.Resampling.LANCZOS)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.1)
        output = BytesIO()
        img.save(output, format="JPEG", quality=85, optimize=True)
        return output.getvalue()


_downloader = None


def get_cover_downloader() -> CoverArtDownloader | None:
    global _downloader
    if _downloader is None and COVER_DOWNLOAD_AVAILABLE:
        _downloader = CoverArtDownloader()
    return _downloader


def download_cover_for_audio(audio_path: str) -> CoverArt | None:
    downloader = get_cover_downloader()
    if not downloader:
        return None
    artist, album = downloader.get_metadata(audio_path)
    if not artist or not album:
        return None
    cover_url = downloader.search_cover(artist, album)
    if not cover_url:
        return None
    cover_data = downloader.download_cover(cover_url)
    if not cover_data:
        return None
    cover = CoverArt(path="", source="downloaded", bytes_data=cover_data, format="jpg")
    cache_path = cover.save_to_cache(audio_path)
    if cache_path:
        cover.path = cache_path
    return cover
