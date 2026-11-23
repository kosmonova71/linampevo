"""The main application window: widget composition and window state."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict

from .audio import AudioEngine
from .constants import (
    AUTO_SAVE_DELAY,
    PERIODIC_SAVE_INTERVAL,
    PRESETS_DIR,
    SETTINGS_FILE,
    THEME_CSS_FILE,
    UPDATE_DISPLAY_INTERVAL,
)
from .coverart import CoverArt
from .deps import (
    EVO_AVAILABLE,
    Gdk,
    Gio,
    GLib,
    Gst,
    Gtk,
    Pango,
    get_default_screen_size,
)
from .playlist import PlaylistController
from .playlist_model import PlaylistItem
from .settings_model import PlayerSettings
from .ui.equalizer_tab import EqualizerTab
from .ui.player_tab import PlayerTab
from .ui.playlist_tab import PlaylistTab
from .ui.podcasts_tab import PodcastsTab
from .ui.radio_tab import RadioTab
from .ui.visualizer_tab import EvoVisualizerTab


class WinampWindow(AudioEngine, PlaylistController, Gtk.ApplicationWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_title("LinAmp")
        self.add_css_class("linamp-window")
        self.header_bar = Gtk.HeaderBar()
        self.header_bar.add_css_class("app-header")
        self.set_titlebar(self.header_bar)
        title_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        title_box.add_css_class("title-box")
        title_label = Gtk.Label(label="LinAmp")
        title_label.add_css_class("app-title")
        title_label.set_halign(Gtk.Align.START)
        self.header_nowplaying = Gtk.Label(label="Ready")
        self.header_nowplaying.add_css_class("app-subtitle")
        self.header_nowplaying.set_halign(Gtk.Align.START)
        self.header_nowplaying.set_ellipsize(Pango.EllipsizeMode.END)
        title_box.append(title_label)
        title_box.append(self.header_nowplaying)
        self.header_bar.set_title_widget(title_box)
        search_btn = Gtk.Button()
        search_btn.set_tooltip_text("Search playlist")
        search_btn.add_css_class("icon-button")
        search_btn.set_child(Gtk.Image.new_from_icon_name("system-search-symbolic"))
        search_btn.connect("clicked", lambda *_: self.focus_playlist_search())
        self.header_bar.pack_end(search_btn)
        self.set_resizable(True)
        self._active_timers = set()
        self.playlist = []
        self._playlist_index = {}
        self.current_track = -1
        self.shuffled_indices = []
        self.shuffle_position = 0
        self._is_muted = False
        self._previous_volume = 1.0
        self._playback_in_progress = False
        self._is_radio_stream = False
        self._radio_stream_uri = ""
        self._radio_stream_title = ""
        self._radio_start_time = 0
        self.beat_detection_timer = 0
        self._last_avg_level = 0.0
        self._last_beat_timestamp = 0.0
        self._bus_watch_id = None
        self._main_bus_signal_watch_attached = False
        self._internal_buses_signal_watches_attached = []
        self.is_compact_mode = False
        self.current_window_width = 650
        self._cleanup_complete = False
        self._podcast_art_generation = 0
        self._cover_executor = None
        self._podcast_persist_gen = {}
        self._current_podcast_episode = None
        self._current_podcast_feed = None
        self.last_beat_time = 0
        self.beat_threshold = 0.1
        self.beat_interval_history = []
        self.settings = PlayerSettings()
        self.load_settings()
        self.auto_play_next = self.settings.auto_play_next
        if self.settings.window_size[0] > 0 and self.settings.window_size[1] > 0:
            self.set_default_size(*self.settings.window_size)
        self.shuffle_mode = self.settings.shuffle_mode
        self.repeat_mode = self.settings.repeat_mode
        self.beat_aware_enabled = self.settings.beat_aware_enabled
        self.beat_threshold = self.settings.beat_threshold
        self.auto_download_covers = self.settings.auto_download_covers
        self.milkdrop_auto_switch = self.settings.milkdrop_auto_switch
        self.connect("close-request", self.on_close_request)
        self.connect("destroy", self.on_window_destroy)
        self.connect("notify::default-width", self.on_window_size_changed)
        self.connect("notify::default-height", self.on_window_size_changed)
        self.connect("notify::width", self.on_window_size_changed)
        self.connect("notify::height", self.on_window_size_changed)

        def suppress_source_remove_warning(log_domain, log_level, message, user_data=None):
            if "Source ID" in str(message) and "was not found" in str(message):
                return
            return

        GLib.log_set_handler(
            "GLib",
            GLib.LogLevelFlags.LEVEL_WARNING | GLib.LogLevelFlags.LEVEL_CRITICAL,
            suppress_source_remove_warning,
            None,
        )
        self.setup_player()
        self.apply_xmms_css()
        self.main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.main_box.add_css_class("main-container")
        self.main_box.set_hexpand(True)
        self.main_box.set_vexpand(True)
        self.main_box.set_halign(Gtk.Align.FILL)
        self.main_box.set_valign(Gtk.Align.FILL)
        self.main_box.set_hexpand_set(True)
        self.main_box.set_vexpand_set(True)
        self.set_child(self.main_box)
        self.notebook = Gtk.Notebook()
        self.notebook.add_css_class("main-notebook")
        self.notebook.set_hexpand(True)
        self.notebook.set_vexpand(True)
        self.notebook.set_halign(Gtk.Align.FILL)
        self.notebook.set_valign(Gtk.Align.FILL)
        self.notebook.set_hexpand_set(True)
        self.notebook.set_vexpand_set(True)
        self.notebook.set_scrollable(False)

        def on_critical_log(log_domain, log_level, message, user_data=None):
            if "CRITICAL" in str(log_level):
                pass

        GLib.log_set_handler("Gtk", GLib.LogLevelFlags.LEVEL_CRITICAL, on_critical_log, None)
        self.main_box.append(self.notebook)
        self.player_tab = PlayerTab(self)
        self.equalizer_tab = EqualizerTab(self)
        self.playlist_tab = PlaylistTab(self)
        self.available_presets = []
        presets_dir = PRESETS_DIR
        if presets_dir.exists():
            self.available_presets = [p for p in presets_dir.rglob("*.milk") if p.exists()]
        if EVO_AVAILABLE:
            try:
                self.visualizer_tab = EvoVisualizerTab(self, self.available_presets)
            except Exception as e:
                logging.getLogger(__name__).error("EvoVisualizerTab creation failed: %s", e)
                self.visualizer_tab = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
                self.visualizer_tab.add_css_class("visualizer-widget")
                error_label = Gtk.Label(label="Visualizer failed to load")
                error_label.add_css_class("placeholder-label")
                self.visualizer_tab.append(error_label)
        else:
            self.visualizer_tab = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
            self.visualizer_tab.add_css_class("visualizer-widget")
            error_label = Gtk.Label(label="Visualizer unavailable: MilkdropRenderer missing")
            error_label.add_css_class("placeholder-label")
            self.visualizer_tab.append(error_label)
        for tab in (self.player_tab, self.equalizer_tab, self.playlist_tab):
            tab.set_hexpand(True)
            tab.set_vexpand(True)
        self.notebook.append_page(
            self.player_tab, self._create_tab_widget("Player", "media-playback-start-symbolic")
        )
        self.notebook.append_page(
            self.equalizer_tab, self._create_tab_widget("Equalizer", "audio-equalizer-symbolic")
        )
        self.notebook.append_page(
            self.playlist_tab, self._create_tab_widget("Playlist", "view-list-symbolic")
        )
        self.notebook.append_page(
            self.visualizer_tab,
            self._create_tab_widget("Visualizer", "preferences-desktop-wallpaper-symbolic"),
        )
        self.notebook.set_current_page(0)
        self.notebook.set_hexpand(True)
        self.notebook.set_vexpand(True)
        self.notebook.set_halign(Gtk.Align.FILL)
        self.notebook.set_valign(Gtk.Align.FILL)
        self.notebook.set_hexpand_set(True)
        self.notebook.set_vexpand_set(True)
        self.radio_tab = RadioTab(self)
        self.radio_tab.set_hexpand(True)
        self.radio_tab.set_vexpand(True)
        self.radio_tab.set_halign(Gtk.Align.FILL)
        self.radio_tab.set_valign(Gtk.Align.FILL)
        radio_tab_widget = self._create_tab_widget("Radio", "radio-symbolic")
        self.notebook.append_page(self.radio_tab, radio_tab_widget)
        self.podcasts_tab = PodcastsTab(self)
        self.podcasts_tab.set_hexpand(True)
        self.podcasts_tab.set_vexpand(True)
        self.podcasts_tab.set_halign(Gtk.Align.FILL)
        self.podcasts_tab.set_valign(Gtk.Align.FILL)
        podcasts_tab_widget = self._create_tab_widget("Podcasts", "podcast-symbolic")
        self.notebook.append_page(self.podcasts_tab, podcasts_tab_widget)

        def on_switch_page(notebook, page, page_num):
            if (
                page is self.player_tab
                and hasattr(self, "visualizer_tab")
                and hasattr(self.visualizer_tab, "cleanup")
            ):
                self.visualizer_tab.cleanup()
            if (
                page is not self.radio_tab
                and hasattr(self, "radio_tab")
                and hasattr(self.radio_tab, "_is_playing_radio")
                and self.radio_tab._is_playing_radio
            ):
                self.radio_tab._is_playing_radio = False
                if hasattr(self.radio_tab, "_live_timer_id") and self.radio_tab._live_timer_id:
                    GLib.source_remove(self.radio_tab._live_timer_id)
                    self.radio_tab._live_timer_id = 0
                self.radio_tab.live_indicator.set_text("")

        self.notebook.connect("switch-page", on_switch_page)
        GLib.idle_add(self.apply_responsive_layout)

        # --- Player controls ---
        self.player_tab.play_btn.connect("clicked", self.on_play)
        self.player_tab.pause_btn.connect("clicked", self.on_pause)
        self.player_tab.stop_btn.connect("clicked", self.on_stop)
        self.player_tab.prev_btn.connect("clicked", self.on_prev)
        self.player_tab.next_btn.connect("clicked", self.on_next)
        self.player_tab.volume_scale.connect("value-changed", self.on_volume_changed)
        self.player_tab.repeat_btn.connect("clicked", lambda b: self.toggle_repeat_mode())
        self.player_tab.shuffle_btn.connect("clicked", lambda b: self.toggle_shuffle_mode())
        self.player_tab.beat_btn.connect("clicked", lambda b: self.toggle_beat_aware())
        self.player_tab.download_btn.connect(
            "clicked", lambda b: self.toggle_auto_download_covers()
        )
        self.player_tab.autonext_btn.connect("clicked", lambda b: self.toggle_auto_play_next())

        # --- Status bar ---
        self.status_bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.status_bar.set_size_request(-1, 28)
        self.status_bar.add_css_class("statusbar")
        self.status_label = Gtk.Label()
        self.status_label.set_halign(Gtk.Align.START)
        self.status_label.set_hexpand(True)
        self.status_label.set_margin_start(12)
        self.status_label.set_margin_end(12)
        self.status_bar.append(self.status_label)
        self.main_box.append(self.status_bar)

        # Cache UI references for update_display to avoid hasattr lookups
        self._ui_time_label = self.player_tab.time_label if hasattr(self.player_tab, "time_label") else None
        self._ui_progress = self.player_tab.progress if hasattr(self.player_tab, "progress") else None
        self._ui_track_label = self.player_tab.track_label if hasattr(self.player_tab, "track_label") else None
        self._ui_header_nowplaying = self.header_nowplaying

        # Cache last displayed values to avoid redundant updates
        self._last_position_sec = -1
        self._last_duration_sec = -1
        self._last_status_text = ""
        self._last_header_text = ""
        self._progress_css_state = 0  # 0=none, 1=halfway, 2=near-end

        # --- Final setup ---
        self.setup_drag_drop()
        timer1 = GLib.timeout_add(500, self._apply_equalizer_settings_delayed)

        # Wayland-safe MilkDrop hotkeys: window-level CAPTURE controller forwards
        # keys to the visualizer when its tab is active (GLArea often has no focus).
        self._viz_key_controller = Gtk.EventControllerKey()
        try:
            self._viz_key_controller.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        except Exception as e:
            logging.getLogger(__name__).debug(
                "Failed to set key controller propagation phase: %s", e
            )
        self._viz_key_controller.connect("key-pressed", self._on_window_key_pressed)
        self.add_controller(self._viz_key_controller)

        def _on_viz_page_focus(notebook, page, page_num):
            try:
                if page is getattr(self, "visualizer_tab", None):
                    GLib.idle_add(self._focus_visualizer)
            except Exception as e:
                logging.getLogger(__name__).debug(
                    "Failed to focus visualizer on tab switch: %s", e
                )

        self.notebook.connect("switch-page", _on_viz_page_focus)

        self._active_timers.add(timer1)
        timer2 = GLib.timeout_add(1000, self._apply_ui_settings_delayed)
        self._active_timers.add(timer2)
        timer3 = GLib.timeout_add(UPDATE_DISPLAY_INTERVAL, self.update_display)
        self._active_timers.add(timer3)
        timer4 = GLib.timeout_add(PERIODIC_SAVE_INTERVAL, self.periodic_auto_save)
        self._active_timers.add(timer4)
        self.load_playlist()

    def on_window_size_changed(self, widget, pspec):
        width = widget.get_width()
        if width <= 0:
            width = self.current_window_width
        was_compact = self.is_compact_mode
        if width < 600:
            self.is_compact_mode = True
        else:
            self.is_compact_mode = False
        if was_compact != self.is_compact_mode or width != self.current_window_width:
            self.apply_responsive_layout()
        self.current_window_width = width

    def _apply_control_button_sizes(self, buttons, size, margin):
        for btn in buttons:
            btn.set_size_request(size[0], size[1])
            btn.set_margin_start(margin)
            btn.set_margin_end(margin)

    def _apply_search_entry_margins(self, margin):
        if hasattr(self.playlist_tab, "search_entry"):
            self.playlist_tab.search_entry.set_margin_start(margin)
            self.playlist_tab.search_entry.set_margin_end(margin)
            self.playlist_tab.search_entry.set_size_request(-1, margin + 28)

    def _apply_cover_art_size(self, cover_size):
        if hasattr(self.player_tab, "cover_art_widget"):
            self.player_tab.cover_art_widget.set_size(cover_size)

    def apply_responsive_layout(self):
        window_width = self.current_window_width if hasattr(self, "current_window_width") else 600
        if window_width <= 400:
            cover_size = 200
        elif window_width <= 500:
            cover_size = 100
        elif window_width <= 600:
            cover_size = 150
        else:
            cover_size = 250
        control_buttons = [
            self.player_tab.prev_btn,
            self.player_tab.play_btn,
            self.player_tab.pause_btn,
            self.player_tab.stop_btn,
            self.player_tab.next_btn,
        ]
        if self.is_compact_mode:
            if hasattr(self.player_tab, "volume_scale"):
                self.player_tab.volume_scale.set_size_request(120, -1)
            self._apply_control_button_sizes(control_buttons, (28, 24), 2)
            self._apply_search_entry_margins(4)
            if hasattr(self.player_tab, "cover_art_widget"):
                compact_size = max(60, cover_size // 2)
                self._apply_cover_art_size(compact_size)
                cover_size = compact_size
            if hasattr(self.player_tab, "progress"):
                self.player_tab.progress.set_size_request(-1, 20)
        elif self.current_window_width < 800:
            if hasattr(self.player_tab, "volume_scale"):
                self.player_tab.volume_scale.set_size_request(160, -1)
            self._apply_control_button_sizes(control_buttons, (36, 30), 4)
            self._apply_search_entry_margins(8)
            if hasattr(self.player_tab, "cover_art_widget"):
                medium_size = max(100, cover_size - 30)
                self._apply_cover_art_size(medium_size)
                cover_size = medium_size
            if hasattr(self.player_tab, "progress"):
                self.player_tab.progress.set_size_request(-1, 24)
        else:
            if hasattr(self.player_tab, "volume_scale"):
                self.player_tab.volume_scale.set_size_request(200, -1)
            self._apply_control_button_sizes(control_buttons, (44, 36), 6)
            self._apply_search_entry_margins(16)
            if self.current_window_width >= 1200:
                large_size = min(350, cover_size + 100)
            elif self.current_window_width >= 1000:
                large_size = min(300, cover_size + 50)
            else:
                large_size = cover_size
            if hasattr(self.player_tab, "cover_art_widget"):
                cover_widget = self.player_tab.cover_art_widget
                if (
                    0 <= self.current_track < len(self.playlist)
                    and isinstance(self.playlist[self.current_track], PlaylistItem)
                ):
                    item = self.playlist[self.current_track]
                    cached_cover_path = CoverArt.get_cached_cover_path(item.path)
                    if cached_cover_path and cached_cover_path.exists():
                        cover_widget.load_cover_from_path(cached_cover_path)
                self._apply_cover_art_size(large_size)
                cover_size = large_size
        if hasattr(self.player_tab, "progress"):
            self.player_tab.progress.set_size_request(-1, 28)
        if self.is_compact_mode:
            self.main_box.add_css_class("compact-layout")
            if hasattr(self, "notebook"):
                self.notebook.add_css_class("compact-notebook")
        else:
            self.main_box.remove_css_class("compact-layout")
            if hasattr(self, "notebook"):
                self.notebook.remove_css_class("compact-notebook")

    def set_status_message(self, message):
        self.status_label.set_label(message)

    def _create_tab_widget(self, text, icon_name):
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        box.add_css_class("tab-widget")
        box.set_visible(True)
        icon = Gtk.Image.new_from_icon_name(icon_name)
        icon.set_pixel_size(16)
        icon.set_visible(True)
        box.append(icon)
        label = Gtk.Label(label=text)
        label.add_css_class("tab-label")
        box.append(label)
        return box

    def _schedule_settings_save(self):
        """Debounced settings save - only writes after 2 seconds of no changes."""
        if hasattr(self, "_settings_save_timer") and self._settings_save_timer:
            self._safe_source_remove(self._settings_save_timer)
        self._settings_save_timer = GLib.timeout_add(AUTO_SAVE_DELAY, self._do_settings_save)
        self._active_timers.add(self._settings_save_timer)

    def _do_settings_save(self):
        self._update_settings_from_state()
        self.save_settings()
        self._settings_save_timer = 0
        return False

    def save_settings(self):
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Use atomic write with temp file
        import tempfile
        fd, temp_path = tempfile.mkstemp(
            prefix=".linamp_settings_", suffix=".tmp", dir=SETTINGS_FILE.parent, text=True
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(asdict(self.settings), f, indent=2)
            os.replace(temp_path, SETTINGS_FILE)
        except OSError:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def load_settings(self):
        if SETTINGS_FILE.exists():
            try:
                with open(SETTINGS_FILE, encoding="utf-8") as f:
                    content = f.read()
            except OSError:
                return
            if not content.strip():
                return
            try:
                settings_dict = json.loads(content)
                self.settings = PlayerSettings.from_dict(settings_dict)
            except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                logging.getLogger(__name__).warning(
                    "Ignoring invalid settings file: %s", SETTINGS_FILE
                )
                return
            self._apply_settings_to_state()
            if hasattr(self.settings, "last_played_track") and self.settings.last_played_track:
                self._resume_last_played()

    def _register_timer(self, interval_ms, callback):
        """Register a GLib timeout tracked for cleanup; returns the source id."""
        timer_id = GLib.timeout_add(interval_ms, callback)
        if not hasattr(self, "_active_timers"):
            self._active_timers = set()
        self._active_timers.add(timer_id)
        return timer_id

    def _invalidate_podcast_persists(self, feed_id: str) -> None:
        """Invalidate in-flight podcast persist work for *feed_id*."""
        gen = self._podcast_persist_gen
        gen[feed_id] = gen.get(feed_id, 0) + 1

    def _flush_persist_futures(self, timeout: float = 1.0) -> bool:
        """Podcast persists run synchronously on the caller thread, so there
        are no persist futures to flush. Kept for API compatibility with the
        podcasts tab, which calls this before feed removal."""
        return True

    def _save_podcast_snapshot(self, feed_id: str, episodes, feeds) -> bool:
        """Persist a podcast feed's episodes + the feed index atomically."""
        try:
            from .podcasts import save_podcast_state

            return bool(save_podcast_state(list(feeds or []), feed_id, list(episodes or [])))
        except Exception:
            logging.getLogger(__name__).debug(
                "Podcast snapshot save failed for %s", feed_id, exc_info=True
            )
            return False

    def _safe_source_remove(self, timer_id):
        if timer_id is None or timer_id == 0:
            return True
        if not hasattr(self, "_active_timers"):
            self._active_timers = set()
        if timer_id in self._active_timers:
            self._active_timers.discard(timer_id)
            try:
                GLib.source_remove(timer_id)
                return True
            except (TypeError, ValueError):
                return False
        else:
            return False

    def cleanup(self):
        if hasattr(self, "_cleanup_in_progress") and self._cleanup_in_progress:
            return
        self._cleanup_in_progress = True
        self._update_settings_from_state()
        self.save_settings()
        self.stop_beat_detection()
        if hasattr(self, "_active_timers"):
            for timer_id in list(self._active_timers):
                self._safe_source_remove(timer_id)
            self._active_timers.clear()
        if hasattr(self, "player") and self.player:
            bus = self.player.get_bus()
            if bus:
                if hasattr(self, "_bus_watch_id") and self._bus_watch_id is not None:
                    bus.disconnect(self._bus_watch_id)
                    self._bus_watch_id = None
                if self._main_bus_signal_watch_attached:
                    bus.remove_signal_watch()
                    self._main_bus_signal_watch_attached = False
                self.player.set_state(Gst.State.NULL)
        if hasattr(self, "equalizer") and self.equalizer:
            self.equalizer.set_state(Gst.State.NULL)
        if hasattr(self, "system_audio_pipeline") and self.system_audio_pipeline:
            self.system_audio_pipeline.set_state(Gst.State.NULL)
            self.system_audio_pipeline = None
        if hasattr(self, "_internal_buses") and self._internal_buses:
            for internal_bus in self._internal_buses:
                if internal_bus in self._internal_buses_signal_watches_attached:
                    internal_bus.remove_signal_watch()
                    self._internal_buses_signal_watches_attached.remove(internal_bus)
            self._internal_buses = []
        if hasattr(self, "player") and self.player:
            self.player.set_state(Gst.State.NULL)
            self.player = None
        if hasattr(self, "equalizer") and self.equalizer:
            self.equalizer.set_state(Gst.State.NULL)
            self.equalizer = None
        if (
            hasattr(self, "cover_art_widget")
            and self.cover_art_widget
            and hasattr(self.cover_art_widget, "executor")
            and self.cover_art_widget.executor
        ):
            self.cover_art_widget.executor.shutdown(wait=False)
        self._cleanup_in_progress = False

    def on_window_destroy(self, *args):
        if hasattr(self, "full_cleanup") and not getattr(self, "_cleanup_in_progress", False):
            self.full_cleanup()
        if self.get_application():
            self.get_application().quit()

    def _update_settings_from_state(self):
        if not hasattr(self, "settings"):
            return
        self.settings.auto_play_next = self.auto_play_next
        self.settings.shuffle_mode = self.shuffle_mode
        self.settings.repeat_mode = self.repeat_mode
        self.settings.beat_aware_enabled = self.beat_aware_enabled
        self.settings.beat_threshold = self.beat_threshold
        self.settings.auto_download_covers = self.auto_download_covers
        self.settings.milkdrop_auto_switch = self.milkdrop_auto_switch
        if self.current_track >= 0 and self.current_track < len(self.playlist):
            current_item = self.playlist[self.current_track]
            self.settings.last_played_track = current_item.path
            if self.playing and hasattr(self, "player") and self.player:
                success, position = self.player.query_position(Gst.Format.TIME)
                if success:
                    self.settings.last_played_position = position / Gst.SECOND
                else:
                    self.settings.last_played_position = 0.0
        if hasattr(self, "player") and self.player:
            volume = self.player.get_property("volume")
            if volume is not None:
                self.settings.volume = volume
        if hasattr(self, "get_width") and hasattr(self, "get_height"):
            width = self.get_width()
            height = self.get_height()
            if width > 0 and height > 0:
                self.settings.window_size = (width, height)
            elif hasattr(self, "get_default_size"):
                size = self.get_default_size()
                if size.width > 0 and size.height > 0:
                    self.settings.window_size = (size.width, size.height)
        elif hasattr(self, "get_default_size"):
            size = self.get_default_size()
            if size.width > 0 and size.height > 0:
                self.settings.window_size = (size.width, size.height)
        if hasattr(self, "get_position"):
            self.settings.window_position = self.get_position()
        if hasattr(self, "equalizer") and self.equalizer:
            self.settings.equalizer_settings = []
            for i in range(10):
                value = self.equalizer.get_property("band" + str(i))
                self.settings.equalizer_settings.append(value)

    def _apply_settings_to_state(self):
        if not hasattr(self, "settings"):
            return
        self.auto_play_next = self.settings.auto_play_next
        self.shuffle_mode = self.settings.shuffle_mode
        self.repeat_mode = self.settings.repeat_mode
        self.beat_aware_enabled = self.settings.beat_aware_enabled
        self.beat_threshold = self.settings.beat_threshold
        self.auto_download_covers = self.settings.auto_download_covers
        self.milkdrop_auto_switch = self.settings.milkdrop_auto_switch
        if hasattr(self, "player") and self.player:
            volume = max(0.0, min(1.0, self.settings.volume))
            self.player.set_property("volume", volume)
        if hasattr(self, "player_tab") and hasattr(self.player_tab, "volume_scale"):
            volume_percent = self.settings.volume * 100
            clamped_volume = max(0, min(100, volume_percent))
            self.player_tab.volume_scale.set_value(clamped_volume)
            if hasattr(self.player_tab, "volume_label"):
                self.player_tab.volume_label.set_label(f"{int(clamped_volume)}%")
        if hasattr(self, "move") and self.settings.window_position:
            x, y = self.settings.window_position
            screen_size = get_default_screen_size()
            if screen_size is None:
                raise RuntimeError("Display geometry unavailable")
            screen_width, screen_height = screen_size
            max_x = screen_width - 450
            max_y = screen_height - 350
            x = max(50, min(x, max_x))
            y = max(50, min(y, max_y))
            self.move(x, y)
        if hasattr(self, "equalizer") and self.equalizer and self.settings.equalizer_settings:
            for i, value in enumerate(self.settings.equalizer_settings[:10]):
                self.equalizer.set_property("band" + str(i), value)
        if hasattr(self, "equalizer_tab") and self.settings.equalizer_settings:
            for i, value in enumerate(self.settings.equalizer_settings[:10]):
                if i < len(self.equalizer_tab.band_scales):
                    scale, value_label = self.equalizer_tab.band_scales[i]
                    clamped_value = max(-12, min(12, value))
                    scale.set_value(clamped_value)
                    value_label.set_text(f"{clamped_value:+.1f} dB")
        if (
            hasattr(self, "visualizer_tab")
            and hasattr(self.visualizer_tab, "visualizer")
            and self.visualizer_tab.visualizer is not None
        ):
            self.visualizer_tab.visualizer._disable_auto_switching = (
                not self.settings.milkdrop_auto_switch
            )

    def _apply_ui_settings_delayed(self):
        if not hasattr(self, "settings"):
            return False
        if hasattr(self, "player_tab") and hasattr(self.player_tab, "volume_scale"):
            volume_percent = self.settings.volume * 100
            clamped_volume = max(0, min(100, volume_percent))
            self.player_tab.volume_scale.set_value(clamped_volume)
            if hasattr(self.player_tab, "volume_label"):
                self.player_tab.volume_label.set_label(f"{int(clamped_volume)}%")
        if hasattr(self, "equalizer_tab") and self.settings.equalizer_settings:
            for i, value in enumerate(self.settings.equalizer_settings[:10]):
                if i < len(self.equalizer_tab.band_scales):
                    scale, value_label = self.equalizer_tab.band_scales[i]
                    clamped_value = max(-12, min(12, value))
                    scale.set_value(clamped_value)
                    value_label.set_text(f"{value:+.1f} dB")
        return

    def _apply_equalizer_settings_delayed(self):
        if (
            hasattr(self, "equalizer")
            and self.equalizer
            and hasattr(self, "settings")
            and self.settings.equalizer_settings
        ):
            for i, value in enumerate(self.settings.equalizer_settings[:10]):
                self.equalizer.set_property("band" + str(i), value)
        return False

    def save_settings_on_track_change(self):
        self._update_settings_from_state()
        self._schedule_settings_save()

    def save_settings_on_stop(self):
        self._update_settings_from_state()
        self._schedule_settings_save()

    def periodic_auto_save(self):
        if self.playing:
            self.auto_save_settings()
        return True

    def apply_xmms_css(self):
        css_file = str(THEME_CSS_FILE)
        css_provider = Gtk.CssProvider()
        css_provider.load_from_path(css_file)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(), css_provider, Gtk.STYLE_PROVIDER_PRIORITY_USER
        )

    def _visualizer_tab_active(self) -> bool:
        """True when the MilkDrop visualizer notebook page is selected."""
        try:
            if not hasattr(self, "notebook") or not hasattr(self, "visualizer_tab"):
                return False
            page = self.notebook.get_nth_page(self.notebook.get_current_page())
            return page is self.visualizer_tab
        except Exception:
            return False

    def _focus_visualizer(self) -> None:
        viz = getattr(getattr(self, "visualizer_tab", None), "visualizer", None)
        if viz is None:
            return
        try:
            viz.grab_focus()
        except Exception as e:
            logging.getLogger(__name__).debug("Failed to grab visualizer focus: %s", e)

    def _on_window_key_pressed(self, controller, keyval, keycode, state):
        """Forward key events to MilkDrop when its tab is active (Wayland)."""
        if not self._visualizer_tab_active():
            return False
        viz = getattr(getattr(self, "visualizer_tab", None), "visualizer", None)
        if viz is None:
            return False
        # Let entry/search widgets keep their keys
        focus = self.get_focus()
        if focus is not None and not isinstance(focus, Gtk.GLArea):
            # Allow visualizer shortcuts only when focus is GLArea or nothing focusable
            widget_type = type(focus).__name__
            if widget_type in ("Entry", "TextView", "SearchEntry", "SpinButton"):
                return False
        handler = getattr(viz, "handle_keyval", None) or getattr(viz, "on_key_pressed", None)
        if handler is None:
            return False
        try:
            if handler.__name__ == "handle_keyval":
                return bool(handler(keyval, keycode, state))
            return bool(handler(None, keyval, keycode, state))
        except Exception:
            return False

    def full_cleanup(self):
        if getattr(self, "_cleanup_complete", False):
            return
        if hasattr(self, "_cleanup_in_progress") and self._cleanup_in_progress:
            return
        self._cleanup_in_progress = True
        self._cleanup_complete = True
        self.playing = False
        self.stop_beat_detection()
        if hasattr(self, "podcasts_tab") and self.podcasts_tab is not None:
            try:
                self.podcasts_tab.cleanup()
            except Exception:
                logging.getLogger(__name__).debug(
                    "Podcasts tab cleanup failed", exc_info=True
                )
        if hasattr(self, "_active_timers"):
            for timer_id in list(self._active_timers):
                self._safe_source_remove(timer_id)
                self._active_timers.clear()
        if hasattr(self, "player") and self.player:
            self.player.set_state(Gst.State.NULL)
            self.player.get_state(Gst.CLOCK_TIME_NONE)
        if hasattr(self, "player") and self.player:
            self.cleanup_pipeline(self.player)
            self.player = None
        if hasattr(self, "equalizer") and self.equalizer:
            self.equalizer.set_state(Gst.State.NULL)
            self.equalizer = None
        if hasattr(self, "system_audio_pipeline") and self.system_audio_pipeline:
            self.system_audio_pipeline.set_state(Gst.State.NULL)
            self.system_audio_pipeline.get_state(Gst.CLOCK_TIME_NONE)
            if hasattr(self, "system_audio_probe_id") and self.system_audio_probe_id is not None:
                capsfilter = self.system_audio_pipeline.get_by_name("system_audio_caps")
                if capsfilter:
                    sink_pad = capsfilter.get_static_pad("src")
                    if sink_pad:
                        sink_pad.remove_probe(self.system_audio_probe_id)
                self.system_audio_probe_id = None
            if hasattr(self, "_system_audio_consume_id") and self._system_audio_consume_id:
                GLib.source_remove(self._system_audio_consume_id)
            self._system_audio_consume_id = None
            self.system_audio_pipeline = None

        if hasattr(self, "player_tab") and hasattr(self.player_tab, "cover_art_widget"):
            self.player_tab.cover_art_widget.cleanup()
        cover_executor = getattr(self, "_cover_executor", None)
        if cover_executor is not None:
            try:
                cover_executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._cover_executor = None
        self._cleanup_in_progress = False

    def on_close_request(self, window):
        self.full_cleanup()
        if self.get_application():
            self.get_application().quit()
        return True

    def setup_drag_drop(self):
        self.dnd = Gtk.DropTarget.new(Gio.File, Gdk.DragAction.COPY)
        self.dnd.connect("drop", self.on_file_dropped)
        self.add_controller(self.dnd)

    def on_file_dropped(self, drop, file, x, y):
        path = file.get_path()
        if path:
            self.add_to_playlist([path])
            if not self.playing and self.playlist:
                self.play_track(0)
        return True

    def _check_visualizer_allocation(self, viz):
        if (
            hasattr(self.visualizer_tab, "visualizer")
            and self.visualizer_tab.visualizer is not None
        ):
            alloc_w = viz.get_allocated_width()
            alloc_h = viz.get_allocated_height()
            tab_alloc_w = self.visualizer_tab.get_allocated_width()
            tab_alloc_h = self.visualizer_tab.get_allocated_height()
            hexpand = viz.get_hexpand()
            vexpand = viz.get_vexpand()
            print(f"GLArea alloc (delayed)={alloc_w}x{alloc_h} expand=h={hexpand} v={vexpand}")
            print(f"VisualizerTab alloc (delayed)={tab_alloc_w}x{tab_alloc_h}")
        return False

    def update_display(self):
        is_radio = getattr(self, "_is_radio_stream", False)
        if hasattr(self, "player") and self.player and self.playing:
            if is_radio:
                elapsed = (
                    int(time.time() - self._radio_start_time)
                    if hasattr(self, "_radio_start_time")
                    else 0
                )
                mins = elapsed // 60
                secs = elapsed % 60
                live_str = f"LIVE {mins}:{secs:02d}"
                if self._ui_time_label:
                    self._ui_time_label.set_text(live_str)
                if self._ui_progress:
                    self._ui_progress.set_value(0)
                if self._ui_header_nowplaying:
                    title = (
                        self._radio_stream_title
                        if hasattr(self, "_radio_stream_title")
                        else "Radio"
                    )
                    header_text = f"{title}  •  {live_str}"
                    if header_text != self._last_header_text:
                        self._ui_header_nowplaying.set_text(header_text)
                        self._last_header_text = header_text
            else:
                success, position = self.player.query_position(Gst.Format.TIME)
                if not success:
                    position = 0
                success, duration = self.player.query_duration(Gst.Format.TIME)
                if not success:
                    duration = 0
                position_sec = position // Gst.SECOND
                duration_sec = duration // Gst.SECOND

                # Only update if position/duration changed significantly
                if position_sec != self._last_position_sec or duration_sec != self._last_duration_sec:
                    self._last_position_sec = position_sec
                    self._last_duration_sec = duration_sec

                    pos_str = f"{position_sec // 60}:{position_sec % 60:02d}"
                    dur_str = f"{duration_sec // 60}:{duration_sec % 60:02d}"

                    if self._ui_time_label:
                        self._ui_time_label.set_text(f"{pos_str} / {dur_str}")

                    if self._ui_progress and duration_sec > 0 and position_sec >= 0:
                        fraction = position_sec / duration_sec
                        fraction = max(0.0, min(1.0, fraction))
                        current_value = self._ui_progress.get_value()
                        if abs(fraction * 100 - current_value) > 0.1:
                            self._ui_progress.set_value(fraction * 100)
                            # Update CSS classes only when state changes
                            if fraction > 0.75 and self._progress_css_state != 2:
                                self._ui_progress.remove_css_class("halfway")
                                self._ui_progress.add_css_class("near-end")
                                self._progress_css_state = 2
                            elif fraction > 0.5 and self._progress_css_state != 1:
                                self._ui_progress.remove_css_class("near-end")
                                self._ui_progress.add_css_class("halfway")
                                self._progress_css_state = 1
                            elif fraction <= 0.5 and self._progress_css_state != 0:
                                self._ui_progress.remove_css_class("near-end")
                                self._ui_progress.remove_css_class("halfway")
                                self._progress_css_state = 0
                    elif self._ui_progress:
                        self._ui_progress.set_value(0)
                        if self._progress_css_state != 0:
                            self._ui_progress.remove_css_class("near-end")
                            self._ui_progress.remove_css_class("halfway")
                            self._progress_css_state = 0

                    if self._ui_header_nowplaying and self.playlist and 0 <= self.current_track < len(self.playlist):
                        current = self.playlist[self.current_track]
                        title = getattr(current, "title", "") or "Playing"
                        header_text = f"{title}  •  {pos_str} / {dur_str}"
                        if header_text != self._last_header_text:
                            self._ui_header_nowplaying.set_text(header_text)
                            self._last_header_text = header_text

        # Build status text
        status_parts = []
        if self.repeat_mode != "none":
            status_parts.append(f"Repeat: {self.repeat_mode.upper()}")
        if self.shuffle_mode:
            status_parts.append("SHUFFLE")
        if not self.auto_play_next:
            status_parts.append("NO AUTO-NEXT")
        if self.beat_aware_enabled:
            status_parts.append("BEAT-AWARE")
        status_text = " | ".join(status_parts) if status_parts else "Ready"
        if self.playing:
            status_text = f"🎵 {status_text}"
            if self.status_bar:
                self.status_bar.add_css_class("playing")
        else:
            if self.status_bar:
                self.status_bar.remove_css_class("playing")
            if self._ui_header_nowplaying and self.playlist and 0 <= self.current_track < len(self.playlist):
                current = self.playlist[self.current_track]
                title = getattr(current, "title", "") or "Ready"
                if title != self._last_header_text:
                    self._ui_header_nowplaying.set_text(title)
                    self._last_header_text = title
            elif self._ui_header_nowplaying and self._last_header_text != "Ready":
                self._ui_header_nowplaying.set_text("Ready")
                self._last_header_text = "Ready"

        # Only update status label if text changed
        if status_text != self._last_status_text:
            self._last_status_text = status_text
            self.status_label.set_label(status_text)

        return True

    def focus_playlist_search(self):
        if hasattr(self, "notebook"):
            self.notebook.set_current_page(2)
        if (
            hasattr(self, "playlist_tab")
            and hasattr(self.playlist_tab, "search_entry")
            and self.playlist_tab.search_entry
        ):
            self.playlist_tab.search_entry.grab_focus()

    def auto_save_settings(self):
        self._schedule_settings_save()

    def _auto_save_callback(self):
        # Handled by _do_settings_save
        self._settings_save_timer = 0
        return False
