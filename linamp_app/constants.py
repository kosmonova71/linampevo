"""Filesystem locations and tuning constants shared across the app."""

from __future__ import annotations

from pathlib import Path

# Anchor resource paths to the application directory rather than the
# current working directory, so launching from elsewhere still works.
APP_DIR = Path(__file__).resolve().parent.parent

DEFAULT_VOLUME = 0.7
BEAT_DETECTION_INTERVAL = 100
BEAT_HISTORY_SIZE = 10
UPDATE_DISPLAY_INTERVAL = 250
AUTO_SAVE_DELAY = 2000
PERIODIC_SAVE_INTERVAL = 30000
AUDIO_EXTENSIONS = {
    ".mp3",
    ".mp4",
    ".flac",
    ".ogg",
    ".wav",
    ".m4a",
    ".wma",
    ".aac",
    ".opus",
}

CONFIG_DIR = Path.home() / ".config" / "linamp"
CACHE_DIR = Path.home() / ".cache" / "linamp"

RADIO_STATIONS_FILE = CONFIG_DIR / "radio_stations.json"
PLAYLIST_FILE = CONFIG_DIR / "playlist.json"
RADIO_LOGO_CACHE_DIR = CACHE_DIR / "radio_logos"
COVER_CACHE_DIR = CACHE_DIR / "covers"

SETTINGS_FILE = CONFIG_DIR / "linamp_settings.json"
THEME_CSS_FILE = APP_DIR / "theme" / "gtk.css"
EQ_PRESETS_PATH = APP_DIR / "configs" / "eq_presets.json"
PRESETS_DIR = APP_DIR / "presets"
