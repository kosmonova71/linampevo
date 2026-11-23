"""Internet radio tab: station list, filtering, and playback."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import ClassVar

from ..deps import Gdk, GLib, Gtk, Pango
from ..stations import (
    DEFAULT_RADIO_STATIONS,
    RadioStation,
    _radio_logo_cache_path,
    fetch_radio_logo,
    load_radio_stations,
    preload_radio_logos,
    save_radio_stations,
)
from .widgets import create_icon_button


class RadioTab(Gtk.Box):
    def __init__(self, player):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.player = player
        self.add_css_class("radio-tab")
        self.set_spacing(8)
        self.set_margin_top(8)
        self.set_margin_bottom(8)
        self.set_margin_start(8)
        self.set_margin_end(8)
        self._is_playing_radio = False
        self._radio_buffer_timer = 0
        self._live_timer_id = 0
        self._search_debounce_id = None
        self._stations = load_radio_stations()
        self._visible_stations = []
        self._stream_tag_update_id = None
        # Warm logo cache in background so list rows paint quickly
        try:
            preload_radio_logos(self._stations)
        except Exception as e:
            logging.getLogger(__name__).debug("Radio logo preload failed: %s", e)

        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        header.set_halign(Gtk.Align.START)
        self.append(header)
        title_label = Gtk.Label(label="Internet Radio")
        title_label.add_css_class("section-header")
        title_label.set_halign(Gtk.Align.START)
        header.append(title_label)

        self.live_indicator = Gtk.Label(label="")
        self.live_indicator.add_css_class("live-indicator")
        self.live_indicator.set_halign(Gtk.Align.END)
        header.append(self.live_indicator)

        main_layout = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.append(main_layout)

        toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        main_layout.append(toolbar)

        add_btn = create_icon_button("Add Stream", "list-add-symbolic")
        add_btn.connect("clicked", self.on_add_stream)
        toolbar.append(add_btn)

        remove_btn = create_icon_button("Remove", "list-remove-symbolic")
        remove_btn.connect("clicked", self.on_remove_station)
        toolbar.append(remove_btn)

        defaults_btn = create_icon_button("Load Defaults", "document-open-symbolic")
        defaults_btn.connect("clicked", self.on_load_defaults)
        toolbar.append(defaults_btn)

        refresh_btn = create_icon_button("Refresh", "view-refresh-symbolic")
        refresh_btn.connect("clicked", self.on_refresh)
        toolbar.append(refresh_btn)

        search_container = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        main_layout.append(search_container)

        self.search_entry = Gtk.Entry()
        self.search_entry.set_placeholder_text("Filter stations...")
        self.search_entry.set_hexpand(True)
        self.search_entry.add_css_class("search-entry")
        self.search_entry.connect("changed", self.on_search_changed)
        search_container.append(self.search_entry)

        self.genre_filter = Gtk.DropDown.new_from_strings(self.GENRES)
        self.genre_filter.add_css_class("sort-dropdown")
        self.genre_filter.set_selected(0)
        self.genre_filter.connect("notify::selected", self.on_genre_filter_changed)
        search_container.append(self.genre_filter)

        station_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        station_container.set_margin_top(8)
        main_layout.append(station_container)

        station_header = Gtk.Label(label="Stations")
        station_header.add_css_class("section-header")
        station_header.set_halign(Gtk.Align.START)
        station_container.append(station_header)

        scrolled = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.AUTOMATIC, vscrollbar_policy=Gtk.PolicyType.AUTOMATIC
        )
        scrolled.add_css_class("radio-scrolled")
        scrolled.set_hexpand(True)
        scrolled.set_vexpand(True)
        station_container.append(scrolled)

        self.store = Gtk.StringList()
        self._populate_store()

        self.selection_model = Gtk.SingleSelection(model=self.store)
        self.column_view = Gtk.ColumnView(model=self.selection_model)
        self.column_view.add_css_class("radio-view")
        self.column_view.set_hexpand(True)
        self.column_view.set_vexpand(True)
        factory = Gtk.SignalListItemFactory()
        factory.connect("setup", self._on_station_factory_setup)
        factory.connect("bind", self._on_station_factory_bind)
        col = Gtk.ColumnViewColumn(title="Stations", factory=factory)
        self.column_view.append_column(col)
        self.column_view.connect("activate", self.on_station_activated)
        scrolled.set_child(self.column_view)
        self.selection_model.connect("selection-changed", self.on_selection_changed)

    GENRES: ClassVar[list[str]] = [
        "All Genres",
        "Electronic",
        "Jazz",
        "Classical",
        "Indie",
        "Alternative",
        "Rock",
        "Pop",
        "Lounge",
        "Mixed",
        "Varied",
    ]

    def _populate_store(self):
        self.store.splice(0, self.store.get_n_items())
        filter_text = (
            self.search_entry.get_text().lower() if hasattr(self, "search_entry") else ""
        )
        genre_idx = self.genre_filter.get_selected() if hasattr(self, "genre_filter") else 0
        genre = self.GENRES[genre_idx] if 0 <= genre_idx < len(self.GENRES) else "All Genres"
        self._visible_stations = []
        for station in self._stations:
            name = station.get("name", "")
            station_genre = station.get("genre", "")
            url = station.get("url", "")
            if filter_text and filter_text not in name.lower() and filter_text not in url.lower():
                continue
            if genre != "All Genres" and station_genre.lower() != genre.lower():
                continue
            display = f"{name}  ·  {station_genre}" if station_genre else name
            self.store.append(display)
            self._visible_stations.append(station)

    def _on_station_factory_setup(self, factory, list_item):
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        row.set_margin_top(4)
        row.set_margin_bottom(4)
        row.set_margin_start(8)
        row.set_margin_end(8)

        logo = Gtk.Image()
        logo.set_size_request(40, 40)
        logo.set_pixel_size(40)
        logo.add_css_class("radio-station-logo")
        logo.set_from_icon_name("audio-x-generic")
        row.append(logo)

        text_col = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        text_col.set_hexpand(True)
        text_col.set_valign(Gtk.Align.CENTER)

        title_label = Gtk.Label(label="")
        title_label.set_hexpand(True)
        title_label.set_halign(Gtk.Align.START)
        title_label.set_ellipsize(Pango.EllipsizeMode.END)
        title_label.add_css_class("radio-station-name")
        text_col.append(title_label)

        genre_label = Gtk.Label(label="")
        genre_label.add_css_class("radio-station-genre")
        genre_label.set_halign(Gtk.Align.START)
        genre_label.set_ellipsize(Pango.EllipsizeMode.END)
        text_col.append(genre_label)

        row.append(text_col)

        play_btn = Gtk.Button()
        play_btn.add_css_class("flat")
        play_btn.set_tooltip_text("Play station")
        play_btn.set_valign(Gtk.Align.CENTER)
        play_img = Gtk.Image.new_from_icon_name("media-playback-start-symbolic")
        play_btn.set_child(play_img)
        play_btn.connect("clicked", self._on_station_play_clicked, list_item)
        row.append(play_btn)

        list_item.set_child(row)

    def _on_station_factory_bind(self, factory, list_item):
        item = list_item.get_item()
        if not item:
            return
        station_data = None
        for s in self._stations:
            name = s.get("name", "")
            genre = s.get("genre", "")
            display = f"{name}  ·  {genre}" if genre else name
            if item.get_string() == display:
                station_data = s
                break
        if not station_data:
            return

        row = list_item.get_child()
        if not row or not isinstance(row, Gtk.Box):
            return

        logo = row.get_first_child()
        text_col = logo.get_next_sibling() if logo else None
        title_label = text_col.get_first_child() if text_col else None
        genre_label = title_label.get_next_sibling() if title_label else None

        if title_label and isinstance(title_label, Gtk.Label):
            title_label.set_text(station_data.get("name", ""))
        if genre_label and isinstance(genre_label, Gtk.Label):
            genre_label.set_text(station_data.get("genre", ""))

        if logo and isinstance(logo, Gtk.Image):
            logo_url = station_data.get("logo", "")
            self._apply_station_logo(logo, logo_url)

    def _apply_station_logo(self, image_widget: Gtk.Image, logo_url: str) -> None:
        """Load a station logo into a Gtk.Image (cached; async-safe via idle)."""
        if not logo_url:
            image_widget.set_from_icon_name("audio-x-generic")
            return

        def _set_from_path(path_str: str) -> bool:
            try:
                texture = Gdk.Texture.new_from_filename(path_str)
                image_widget.set_from_paintable(texture)
                image_widget.set_size_request(40, 40)
            except Exception:
                image_widget.set_from_icon_name("audio-x-generic")
            return False

        # Local bundled / already-cached path
        local = Path(logo_url).expanduser()
        if local.is_file():
            _set_from_path(str(local))
            return

        cache = _radio_logo_cache_path(logo_url)
        if cache.exists():
            _set_from_path(str(cache))
            return

        # Placeholder while downloading
        image_widget.set_from_icon_name("audio-x-generic")

        def _download():
            path = fetch_radio_logo(logo_url)
            if path and path.exists():
                GLib.idle_add(_set_from_path, str(path))

        threading.Thread(target=_download, daemon=True, name="RadioLogoFetch").start()

    def on_selection_changed(self, selection, position, n_items):
        pass  # Intentional: selection changes are handled by row activation

    def on_search_changed(self, entry):
        def apply_filter():
            self._search_debounce_id = None
            self._populate_store()
            return False

        if self._search_debounce_id:
            GLib.source_remove(self._search_debounce_id)
        self._search_debounce_id = GLib.timeout_add(200, apply_filter)

    def on_genre_filter_changed(self, dropdown, pspec):
        self._populate_store()

    def on_station_activated(self, column_view, position):
        if position < 0 or position >= len(self._visible_stations):
            return
        station = self._visible_stations[position]
        self._play_station(station)

    def _on_station_play_clicked(self, button, list_item):
        item = list_item.get_item()
        if not item:
            return
        display = item.get_string()
        for station in self._visible_stations:
            name = station.get("name", "")
            genre = station.get("genre", "")
            match = f"{name}  ·  {genre}" if genre else name
            if display == match:
                self._play_station(station)
                break

    def _play_station(self, station):
        if not station or not station.get("url"):
            return
        url = station["url"]
        name = station.get("name", url)
        if not RadioStation.validate_url(url):
            self.player.set_status_message("Invalid stream URL")
            return
        if hasattr(self.player, "play_uri") and self.player.play_uri(url, name):
            self._is_playing_radio = True
            self._radio_start_time = time.time()
            self._update_live_indicator()
            GLib.idle_add(self._start_buffer_timer)
            # Show station logo on the player cover-art widget when available
            logo_url = station.get("logo") or ""
            if logo_url:

                def _show_logo():
                    path = fetch_radio_logo(logo_url)
                    cover = getattr(
                        getattr(self.player, "player_tab", None), "cover_art_widget", None
                    )
                    if path and cover and hasattr(cover, "load_cover_from_path"):
                        GLib.idle_add(cover.load_cover_from_path, str(path))
                    return False

                threading.Thread(target=_show_logo, daemon=True, name="RadioLogoPlay").start()

    def _start_buffer_timer(self):
        self._radio_buffer_timer = GLib.timeout_add(2000, self._check_buffer)
        return False

    def _check_buffer(self):
        if self._is_playing_radio and hasattr(self.player, "playing") and self.player.playing:
            self.live_indicator.set_text("")
            return True
        return False

    def _update_live_indicator(self):
        if self._live_timer_id:
            GLib.source_remove(self._live_timer_id)
            self._live_timer_id = 0
        if not self._is_playing_radio:
            GLib.idle_add(self.live_indicator.set_text, "")
            return
        elapsed = (
            int(time.time() - self._radio_start_time) if hasattr(self, "_radio_start_time") else 0
        )
        mins = elapsed // 60
        secs = elapsed % 60
        GLib.idle_add(self.live_indicator.set_text, f"LIVE {mins}:{secs:02d}")
        self._live_timer_id = GLib.timeout_add(1000, self._update_live_indicator)

    def on_refresh(self, button):
        self._stations = load_radio_stations()
        self._populate_store()
        self.player.set_status_message("Radio stations refreshed")

    def on_load_defaults(self, button):
        self._stations = DEFAULT_RADIO_STATIONS.copy()
        save_radio_stations(self._stations)
        self._populate_store()
        self.player.set_status_message("Default stations loaded")

    def on_add_stream(self, button):
        dialog = Gtk.Dialog(title="Add Radio Stream", transient_for=self.get_root())
        dialog.set_modal(True)
        dialog.add_css_class("radio-dialog")

        content = dialog.get_content_area()
        content.set_spacing(8)
        content.set_margin_top(12)
        content.set_margin_bottom(12)
        content.set_margin_start(12)
        content.set_margin_end(12)

        name_entry = Gtk.Entry()
        name_entry.set_placeholder_text("Station name")
        content.append(name_entry)

        genre_entry = Gtk.Entry()
        genre_entry.set_placeholder_text("Genre (optional)")
        content.append(genre_entry)

        url_entry = Gtk.Entry()
        url_entry.set_placeholder_text("http:// or https:// stream URL")
        content.append(url_entry)

        dialog.add_button("Cancel", Gtk.ResponseType.CANCEL)
        dialog.add_button("Add", Gtk.ResponseType.OK)
        dialog.set_default_response(Gtk.ResponseType.OK)

        def on_response(dlg, response_id):
            if response_id == Gtk.ResponseType.OK:
                name = name_entry.get_text().strip()
                genre = genre_entry.get_text().strip()
                url = url_entry.get_text().strip()
                if not url or not RadioStation.validate_url(url):
                    self.player.set_status_message(
                        "Invalid URL — must start with http:// or https://"
                    )
                    dialog.destroy()
                    return
                if not name:
                    name = url
                self._stations.append({"name": name, "genre": genre, "url": url})
                save_radio_stations(self._stations)
                self._populate_store()
                self.player.set_status_message(f"Added station: {name}")
            dialog.destroy()

        dialog.connect("response", on_response)
        dialog.set_visible(True)

    def on_remove_station(self, button):
        selected = self.selection_model.get_selected()
        if selected is None or selected == Gtk.INVALID_LIST_POSITION:
            self.player.set_status_message("No station selected")
            return
        if selected < 0 or selected >= len(self._visible_stations):
            return
        station = self._visible_stations[selected]
        try:
            self._stations.remove(station)
        except ValueError:
            return
        save_radio_stations(self._stations)
        self._populate_store()
        name = station.get("name", "Unknown")
        self.player.set_status_message(f"Removed station: {name}")
