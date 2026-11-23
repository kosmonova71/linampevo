"""Playlist state, persistence, and track navigation.

Mixed into the window; see :mod:`linamp_app.audio` for the mixin rationale.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .constants import (
    AUDIO_EXTENSIONS,
    CONFIG_DIR,
    COVER_CACHE_DIR,
)
from .coverart import (
    CoverArt,
    download_cover_for_audio,
    extract_cover_art,
)
from .deps import COVER_DOWNLOAD_AVAILABLE, GLib, Gst
from .playlist_model import PlaylistItem


class PlaylistController:
    """Owns ``self.playlist`` and everything that reads or writes it."""

    def _invalidate_playlist_index(self) -> None:
        self._playlist_index = {}

    def _get_playlist_index(self) -> dict[str, int]:
        if (
            not hasattr(self, "_playlist_index")
            or not self._playlist_index
            or len(self._playlist_index) != len(self.playlist)
        ):
            self._playlist_index = {item.id: i for i, item in enumerate(self.playlist)}
        return self._playlist_index

    def add_to_playlist(self, file_paths):
        added_items = []
        existing_paths = {
            os.path.realpath(item.path) for item in self.playlist if hasattr(item, "path")
        }
        for path in file_paths:
            try:
                normalized_path = os.path.abspath(os.path.expanduser(str(path)))
                if "\0" in normalized_path:
                    continue
                real_path = os.path.realpath(normalized_path)
                if not os.path.exists(real_path):
                    continue
                if not os.path.isfile(real_path):
                    continue
                if real_path.startswith(("/proc/", "/sys/")):
                    continue
                if real_path in existing_paths:
                    continue
                existing_paths.add(real_path)
            except (OSError, ValueError):
                continue
            item = PlaylistItem(path=normalized_path, title=os.path.basename(normalized_path))
            self.playlist.append(item)
            added_items.append(item)
        if hasattr(self, "playlist_tab") and added_items:
            for item in added_items:
                self.playlist_tab.playlist_store.append(f"{item.title}|{item.id}")
            self.playlist_tab.update_statistics()
            self._invalidate_playlist_index()
        if added_items:
            self._schedule_playlist_save()

    def add_folder_to_playlist(self, folder_path):
        try:
            normalized_path = os.path.abspath(os.path.expanduser(str(folder_path)))
            if "\0" in normalized_path:
                return
            if not os.path.exists(normalized_path) or not os.path.isdir(normalized_path):
                return
        except (OSError, ValueError):
            return
        GLib.idle_add(
            self.set_status_message, f"Scanning folder: {os.path.basename(normalized_path)}..."
        )

        def scan_folder():
            added_items = []
            errors = []
            try:
                for root, dirs, files in os.walk(normalized_path):
                    dirs[:] = [d for d in dirs if not d.startswith(".")]
                    for file in files:
                        try:
                            file_path = os.path.join(root, file)
                            file_ext = os.path.splitext(file)[1].lower()
                            if file_ext in AUDIO_EXTENSIONS:
                                rel_path = os.path.relpath(file_path, normalized_path)
                                title = rel_path if rel_path != file_path else file
                                item = PlaylistItem(path=file_path, title=title)
                                added_items.append(item)
                        except (OSError, ValueError) as e:
                            errors.append(f"{file}: {e}")
            except (OSError, PermissionError) as e:
                errors.append(f"Scan failed: {e}")
            GLib.idle_add(self._on_folder_scan_complete, added_items, errors)

        thread = threading.Thread(target=scan_folder, daemon=True)
        thread.start()

    def _on_folder_scan_complete(self, added_items, errors=None):
        if errors:
            for error in errors:
                pass
        if added_items:
            existing_paths = {
                os.path.realpath(item.path) for item in self.playlist if hasattr(item, "path")
            }
            unique_items = []
            for item in added_items:
                real_path = os.path.realpath(item.path)
                if real_path not in existing_paths:
                    existing_paths.add(real_path)
                    unique_items.append(item)
            added_items = unique_items
        if added_items:
            self.playlist.extend(added_items)
            if hasattr(self, "playlist_tab"):
                for item in added_items:
                    self.playlist_tab.playlist_store.append(f"{item.title}|{item.id}")
                self.playlist_tab.update_statistics()
                self._invalidate_playlist_index()
            self._schedule_playlist_save()
            GLib.idle_add(self.set_status_message, f"Added {len(added_items)} tracks to playlist")
        else:
            GLib.idle_add(self.set_status_message, "No audio files found in folder")

    def _schedule_playlist_save(self):
        """Debounced playlist save - only writes after 2 seconds of no changes."""
        if hasattr(self, "_playlist_save_timer") and self._playlist_save_timer:
            self._safe_source_remove(self._playlist_save_timer)
        self._playlist_save_timer = GLib.timeout_add(2000, self._do_playlist_save)
        if not hasattr(self, "_active_timers"):
            self._active_timers = set()
        self._active_timers.add(self._playlist_save_timer)

    def _do_playlist_save(self):
        self.save_playlist()
        self._playlist_save_timer = 0
        return False

    def save_playlist(self, filepath: str | None = None) -> bool:
        if not hasattr(self, "playlist"):
            return False
        if not filepath:
            playlist_dir = str(CONFIG_DIR)
            filepath = os.path.join(playlist_dir, "playlist.json")
        else:
            filepath = os.path.abspath(os.path.expanduser(filepath))
            playlist_dir = os.path.dirname(filepath)
        os.makedirs(playlist_dir, exist_ok=True)
        playlist_data = []
        for item in self.playlist:
            if hasattr(item, "to_dict") and callable(item.to_dict):
                if hasattr(item, "path") and item.path and os.path.exists(item.path):
                    playlist_data.append(item.to_dict())
                else:
                    pass
        temp_file = None
        fd, temp_file = tempfile.mkstemp(
            prefix=".playlist_", suffix=".tmp", dir=playlist_dir, text=True
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(playlist_data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        if os.path.exists(filepath):
            os.replace(temp_file, filepath)
        else:
            shutil.move(temp_file, filepath)
        return True

    def load_playlist(self, filepath: str | None = None) -> bool:
        if not filepath:
            playlist_dir = str(CONFIG_DIR)
            filepath = os.path.join(playlist_dir, "playlist.json")
        else:
            filepath = os.path.abspath(os.path.expanduser(filepath))
        if not os.path.exists(filepath):
            return False
        if not os.access(filepath, os.R_OK):
            return False
        try:
            with open(filepath, encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    return False
            if not isinstance(data, list):
                return False
            if len(data) > 100000:
                return False
        except OSError:
            return False
        playlist = []
        invalid_items = 0
        for i, item_data in enumerate(data, 1):
            if not isinstance(item_data, dict):
                invalid_items += 1
                continue
            try:
                item = PlaylistItem.from_dict(item_data)
                if item.exists():
                    playlist.append(item)
            except ValueError:
                invalid_items += 1
        if invalid_items > 0:
            pass
        self.playlist = playlist
        self._invalidate_playlist_index()
        if hasattr(self, "playlist_tab") and self.playlist_tab:
            self._clear_playlist_store()
            self._update_playlist_display()
        return len(playlist) > 0

    def cleanup_invalid_tracks(self):
        if not self.playlist:
            return
        original_count = len(self.playlist)
        valid_tracks = []
        for item in self.playlist:
            if item.exists():
                valid_tracks.append(item)
            else:
                pass
        if len(valid_tracks) != original_count:
            self.playlist = valid_tracks
            self._invalidate_playlist_index()
            self._update_playlist_display()
            self._schedule_playlist_save()
            self.set_status_message(f"Removed {original_count - len(valid_tracks)} invalid tracks")

    def _clear_playlist_store(self) -> bool:
        if not hasattr(self, "playlist_tab") or not hasattr(self.playlist_tab, "playlist_store"):
            return False
        store = self.playlist_tab.playlist_store
        store.freeze_notify()
        if hasattr(store, "splice") and callable(store.splice):
            store.splice(0, store.get_n_items(), [])
        else:
            while store.get_n_items() > 0:
                store.remove(store.get_n_items() - 1)
        store.thaw_notify()
        return True

    def _update_playlist_display(self) -> bool:
        if not hasattr(self, "playlist_tab") or not hasattr(self.playlist_tab, "playlist_store"):
            return False
        store = self.playlist_tab.playlist_store
        if not self.playlist:
            self._clear_playlist_store()
            return True
        store.freeze_notify()
        current_count = store.get_n_items()
        target_count = len(self.playlist)
        if current_count > target_count:
            if hasattr(store, "splice") and callable(store.splice):
                store.splice(target_count, current_count - target_count, [])
            else:
                while store.get_n_items() > target_count:
                    store.remove(store.get_n_items() - 1)
        display_names = []
        for i, item in enumerate(self.playlist):
            if isinstance(item, PlaylistItem):
                display_names.append(f"{item.get_display_name()}|{item.id}")
            elif isinstance(item, dict):
                title = item.get("title") or os.path.basename(item.get("path", ""))
                path = item.get("path", "")
                track_id = (
                    item.get("id") or hashlib.md5(f"{path}:{title}".encode()).hexdigest()[:16]
                )
                display_names.append(f"{title}|{track_id}")
            else:
                s = str(item)
                track_id = hashlib.md5(s.encode()).hexdigest()[:16]
                display_names.append(f"{s}|{track_id}")
        if hasattr(store, "splice") and callable(store.splice):
            if current_count < target_count:
                store.splice(current_count, 0, display_names[current_count:])
            else:
                store.splice(0, min(current_count, target_count), display_names[:target_count])
        else:
            while store.get_n_items() > 0:
                store.remove(store.get_n_items() - 1)
            for name in display_names:
                store.append(name)
        store.thaw_notify()
        return True

    def play_track(self, index):
        if 0 <= index < len(self.playlist):
            item = self.playlist[index]
            if self.repeat_mode == "one" and index == self.current_track:
                GLib.idle_add(self.player.set_state, Gst.State.NULL)
            if self.play_file(item.path, item.title):
                self.current_track = index
                self.save_settings_on_track_change()
                if self.shuffle_mode and index in self.shuffled_indices:
                    self.shuffle_position = self.shuffled_indices.index(index)
                if hasattr(self, "playlist_tab") and hasattr(self.playlist_tab, "selection_model"):
                    self.playlist_tab.selection_model.set_selected(index)
                self._load_cover_art_for_track(item)
                return True
        return False

    def _load_cover_art_for_track(self, item):
        if not hasattr(self, "player_tab") or not hasattr(self.player_tab, "cover_art_widget"):
            return
        cover_widget = self.player_tab.cover_art_widget
        current_audio_path = cover_widget.current_cover_path
        if current_audio_path:
            if current_audio_path.startswith(str(COVER_CACHE_DIR)):
                try:
                    cache_filename = Path(current_audio_path).name
                    cache_key = cache_filename.split(".")[0]
                    item_cache_key = CoverArt.get_cache_key(item.path)
                    if cache_key == item_cache_key:
                        return
                except (OSError, ValueError):
                    logging.getLogger(__name__).debug(
                        "Cover art cache check failed for %s: %s", item.path, sys.exc_info()[1]
                    )
            elif current_audio_path == item.path:
                return
        allow_download = getattr(self, "auto_download_covers", False)
        cover, _cache_path = extract_cover_art(item.path, allow_download=False)
        if cover:
            cover_widget.load_cover(cover, item.path)
        else:
            if allow_download:
                if not hasattr(self, "_download_executor"):
                    self._download_executor = ThreadPoolExecutor(max_workers=2)

                def download_and_load():
                    cover = download_cover_for_audio(item.path)
                    if cover:
                        GLib.idle_add(cover_widget.load_cover, cover, item.path)

                self._download_executor.submit(download_and_load)
            else:
                cover_widget.load_cover(None, item.path)

    def on_prev(self, button):
        self.play_previous_track()

    def play_previous_track(self):
        if not self.playlist:
            return
        prev_index = self.get_previous_track_index()
        if prev_index is not None:
            self.play_track(prev_index)

    def get_previous_track_index(self):
        if not self.playlist:
            return None
        if self.shuffle_mode:
            return self.get_previous_shuffled_index()
        else:
            if self.current_track > 0:
                return self.current_track - 1
            elif self.repeat_mode == "all":
                return len(self.playlist) - 1
            else:
                return None

    def get_previous_shuffled_index(self):
        if not self.shuffled_indices:
            self.regenerate_shuffle_list()
        if self.shuffle_position > 0:
            self.shuffle_position -= 1
            return self.shuffled_indices[self.shuffle_position]
        elif self.repeat_mode == "all":
            self.shuffle_position = len(self.shuffled_indices) - 1
            return self.shuffled_indices[self.shuffle_position]
        else:
            return None

    def on_next(self, button):
        self.play_next_track()

    def play_next_track(self):
        if not self.playlist:
            return
        next_index = self.get_next_track_index()
        if next_index is not None:
            if hasattr(self, "playlist_tab") and hasattr(self.playlist_tab, "selection_model"):
                self.playlist_tab.selection_model.set_selected(next_index)
            self.play_track(next_index)

    def get_next_track_index(self):
        if not self.playlist:
            return None
        if self.shuffle_mode:
            return self.get_next_shuffled_index()
        else:
            if self.current_track < len(self.playlist) - 1:
                return self.current_track + 1
            elif self.repeat_mode == "all":
                return 0
            else:
                return None

    def get_next_shuffled_index(self):
        if not self.shuffled_indices:
            self.regenerate_shuffle_list()
        if self.shuffle_position < len(self.shuffled_indices) - 1:
            self.shuffle_position += 1
            return self.shuffled_indices[self.shuffle_position]
        elif self.repeat_mode == "all":
            self.regenerate_shuffle_list()
            self.shuffle_position = 0
            return self.shuffled_indices[0]
        else:
            return None

    def regenerate_shuffle_list(self):
        self.original_indices = list(range(len(self.playlist)))
        self.shuffled_indices = self.original_indices.copy()
        random.shuffle(self.shuffled_indices)
        self.shuffle_position = -1

    def toggle_repeat_mode(self):
        modes = ["none", "one", "all"]
        current_index = modes.index(self.repeat_mode)
        self.repeat_mode = modes[(current_index + 1) % len(modes)]
        self.update_status_display()
        self.auto_save_settings()

    def toggle_shuffle_mode(self):
        self.shuffle_mode = not self.shuffle_mode
        if self.shuffle_mode:
            self.regenerate_shuffle_list()
            if self.current_track >= 0:
                try:
                    self.shuffle_position = self.shuffled_indices.index(self.current_track)
                except ValueError:
                    self.shuffle_position = -1
        self.update_status_display()
        self.auto_save_settings()

    def toggle_auto_play_next(self):
        self.auto_play_next = not self.auto_play_next
        self.update_status_display()
        self.auto_save_settings()

    def toggle_beat_aware(self):
        self.beat_aware_enabled = not self.beat_aware_enabled
        self.update_status_display()
        self.auto_save_settings()

    def toggle_auto_download_covers(self):
        if not COVER_DOWNLOAD_AVAILABLE:
            self.set_status_message("Cover download requires requests and PIL libraries")
            return
        self.auto_download_covers = not self.auto_download_covers
        status = "enabled" if self.auto_download_covers else "disabled"
        self.set_status_message(f"Auto download covers: {status}")
        self.update_status_display()
        self.auto_save_settings()
        if (
            self.auto_download_covers
            and self.current_track >= 0
            and self.current_track < len(self.playlist)
        ):
            current_item = self.playlist[self.current_track]
            self._load_cover_art_for_track(current_item)

    def update_status_display(self):
        status_parts = []
        if self.repeat_mode != "none":
            status_parts.append(f"Repeat: {self.repeat_mode.upper()}")
        if self.shuffle_mode:
            status_parts.append("SHUFFLE")
        if not self.auto_play_next:
            status_parts.append("NO AUTO-NEXT")
        if self.beat_aware_enabled:
            status_parts.append("BEAT-AWARE")
        if self.auto_download_covers:
            status_parts.append("AUTO-DL")
        status_text = " | ".join(status_parts) if status_parts else "Ready"
        GLib.idle_add(self.set_status_message, status_text)
        self.update_button_states()

    def update_button_states(self):
        if not hasattr(self, "player_tab"):
            return
        if hasattr(self.player_tab, "repeat_btn"):
            if self.repeat_mode == "none":
                self.player_tab.repeat_btn.remove_css_class("active")
                self.player_tab.repeat_btn.set_tooltip_text("Repeat: Off")
            elif self.repeat_mode == "one":
                self.player_tab.repeat_btn.add_css_class("active")
                self.player_tab.repeat_btn.set_tooltip_text("Repeat: One Track")
            else:
                self.player_tab.repeat_btn.add_css_class("active")
                self.player_tab.repeat_btn.set_tooltip_text("Repeat: All")
        if hasattr(self.player_tab, "shuffle_btn"):
            if self.shuffle_mode:
                self.player_tab.shuffle_btn.add_css_class("active")
                self.player_tab.shuffle_btn.set_tooltip_text("Shuffle: On")
            else:
                self.player_tab.shuffle_btn.remove_css_class("active")
                self.player_tab.shuffle_btn.set_tooltip_text("Shuffle: Off")
        if hasattr(self.player_tab, "beat_btn"):
            if self.beat_aware_enabled:
                self.player_tab.beat_btn.add_css_class("active")
                self.player_tab.beat_btn.set_tooltip_text("Beat Detection: On")
            else:
                self.player_tab.beat_btn.remove_css_class("active")
                self.player_tab.beat_btn.set_tooltip_text("Beat Detection: Off")
        if hasattr(self.player_tab, "download_btn"):
            if self.auto_download_covers:
                self.player_tab.download_btn.add_css_class("active")
                self.player_tab.download_btn.set_tooltip_text("Auto Download Covers: On")
            else:
                self.player_tab.download_btn.remove_css_class("active")
                self.player_tab.download_btn.set_tooltip_text("Auto Download Covers: Off")
        if hasattr(self.player_tab, "autonext_btn"):
            if self.auto_play_next:
                self.player_tab.autonext_btn.add_css_class("active")
                self.player_tab.autonext_btn.set_tooltip_text("Auto Next: On")
            else:
                self.player_tab.autonext_btn.remove_css_class("active")
                self.player_tab.autonext_btn.set_tooltip_text("Auto Next: Off")
