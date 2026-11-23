"""Playlist tab: search, sort, import/export, and the track list."""

from __future__ import annotations

import hashlib
import os
import random
import threading
from typing import Any

from ..deps import (
    METADATA_FIXER_AVAILABLE,
    Gio,
    GLib,
    Gtk,
    MetadataFixer,
)
from ..playlist_model import PlaylistItem, read_m3u_paths, write_m3u_lines
from .widgets import create_icon_button


class PlaylistTab(Gtk.Box):
    def __init__(self, player: Any) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.player = player
        self.add_css_class("playlist-tab")
        self.playlist_store = Gtk.StringList()
        self.filtered_store = None
        self.filter_model = None
        self._search_debounce_id = None
        self._playlist_by_id = {}
        self._filter_func_ref = None  # Cache filter function
        self._last_search_text = ""  # Cache last search to avoid redundant filters
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        main_box.set_margin_top(16)
        main_box.set_margin_bottom(16)
        main_box.set_margin_start(16)
        main_box.set_margin_end(16)
        self.append(main_box)
        search_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        main_box.append(search_section)
        search_header = Gtk.Label(label="Search Playlist")
        search_header.add_css_class("section-header")
        search_header.set_halign(Gtk.Align.START)
        search_section.append(search_header)
        search_container = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        search_section.append(search_container)
        self.search_entry = Gtk.Entry()
        self.search_entry.set_placeholder_text("Type to filter playlist...")
        self.search_entry.set_hexpand(True)
        self.search_entry.add_css_class("search-entry")
        self.search_entry.connect("changed", self.on_search_changed)
        search_container.append(self.search_entry)
        clear_search_btn = create_icon_button("Clear", "edit-clear-symbolic")
        clear_search_btn.connect("clicked", self.on_clear_search)
        search_container.append(clear_search_btn)
        toolbar_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        main_box.append(toolbar_section)
        toolbar_header = Gtk.Label(label="Playlist Actions")
        toolbar_header.add_css_class("section-header")
        toolbar_header.set_halign(Gtk.Align.START)
        toolbar_section.append(toolbar_header)
        main_toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        main_toolbar.set_homogeneous(True)
        toolbar_section.append(main_toolbar)
        add_btn = create_icon_button("Add Files", "list-add-symbolic")
        add_btn.connect("clicked", self.on_add_files)
        main_toolbar.append(add_btn)
        add_folder_btn = create_icon_button("Add Folder", "folder-open-symbolic")
        add_folder_btn.connect("clicked", self.on_add_folder)
        main_toolbar.append(add_folder_btn)
        remove_btn = create_icon_button("Remove", "list-remove-symbolic")
        remove_btn.connect("clicked", self.on_remove)
        main_toolbar.append(remove_btn)
        clear_btn = create_icon_button("Clear All", "edit-delete-symbolic")
        clear_btn.connect("clicked", self.on_clear)
        main_toolbar.append(clear_btn)
        shuffle_btn = create_icon_button("Shuffle", "media-playlist-shuffle-symbolic")
        shuffle_btn.connect("clicked", self.on_shuffle)
        main_toolbar.append(shuffle_btn)
        secondary_toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        secondary_toolbar.set_homogeneous(True)
        toolbar_section.append(secondary_toolbar)
        sort_dropdown = Gtk.DropDown.new_from_strings(
            ["Default Order", "By Title", "By Path", "By Duration"]
        )
        sort_dropdown.add_css_class("sort-dropdown")
        sort_dropdown.connect("notify::selected", self.on_sort_changed)
        secondary_toolbar.append(sort_dropdown)
        self.sort_dropdown = sort_dropdown
        import_btn = create_icon_button("Import", "document-open-symbolic")
        import_btn.connect("clicked", self.on_import_m3u)
        secondary_toolbar.append(import_btn)
        export_btn = create_icon_button("Export", "document-save-symbolic")
        export_btn.connect("clicked", self.on_export_m3u)
        secondary_toolbar.append(export_btn)
        remove_dups_btn = create_icon_button("Remove Dups", "view-refresh-symbolic")
        remove_dups_btn.connect("clicked", self.on_remove_duplicates)
        secondary_toolbar.append(remove_dups_btn)
        if METADATA_FIXER_AVAILABLE:
            fix_tags_btn = create_icon_button("Fix Tags", "document-edit-symbolic")
            fix_tags_btn.connect("clicked", self.on_fix_tags)
            secondary_toolbar.append(fix_tags_btn)
            self.fix_tags_btn = fix_tags_btn
        else:
            self.fix_tags_btn = None
        stats_container = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        stats_container.set_halign(Gtk.Align.END)
        stats_container.set_margin_top(8)
        toolbar_section.append(stats_container)
        self.stats_label = Gtk.Label(label="0 tracks")
        self.stats_label.add_css_class("stats-label")
        stats_container.append(self.stats_label)
        playlist_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        playlist_container.set_margin_top(12)
        main_box.append(playlist_container)
        playlist_header = Gtk.Label(label="Track List")
        playlist_header.add_css_class("section-header")
        playlist_header.set_halign(Gtk.Align.START)
        playlist_container.append(playlist_header)
        scrolled = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.AUTOMATIC, vscrollbar_policy=Gtk.PolicyType.AUTOMATIC
        )
        scrolled.add_css_class("playlist-scrolled")
        scrolled.set_hexpand(True)
        scrolled.set_vexpand(True)
        playlist_container.append(scrolled)
        self.sort_model = Gtk.SortListModel(model=self.playlist_store)
        self.filter_model = Gtk.FilterListModel(model=self.sort_model)
        self.selection_model = Gtk.SingleSelection(model=self.filter_model)
        self.column_view = Gtk.ColumnView(model=self.selection_model)
        self.column_view.add_css_class("playlist-view")
        self.column_view.set_hexpand(True)
        self.column_view.set_vexpand(True)
        self.column_view.set_size_request(-1, 180)
        factory = Gtk.SignalListItemFactory()
        factory.connect("setup", self._on_factory_setup)
        factory.connect("bind", self._on_factory_bind)
        column = Gtk.ColumnViewColumn(title="Tracks", factory=factory)
        self.column_view.append_column(column)
        scrolled.set_child(self.column_view)
        self.selection_model.connect("selection-changed", self.on_selection_changed)
        self.column_view.connect("activate", self.on_row_activated)

    def _rebuild_playlist_index(self):
        """Rebuild the playlist ID index for O(1) lookups."""
        self._playlist_by_id = {}
        for item in self.player.playlist:
            if isinstance(item, PlaylistItem) and item.id:
                self._playlist_by_id[item.id] = item
            elif isinstance(item, dict):
                tid = item.get("id")
                if tid:
                    self._playlist_by_id[tid] = item

    def on_add_files(self, button):
        dialog = Gtk.FileDialog(title="Add Files")
        dialog.set_modal(True)
        audio_filter = Gtk.FileFilter()
        audio_filter.set_name("Audio files")
        audio_filter.add_mime_type("audio/*")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(audio_filter)
        dialog.set_filters(filters)
        dialog.open_multiple(self.player, None, self._on_files_selected)

    def _on_files_selected(self, dialog, result):
        files = dialog.open_multiple_finish(result)
        file_paths = [file.get_path() for file in files if file.get_path() is not None]
        if file_paths:
            self.player.add_to_playlist(file_paths)
        else:
            self.player.set_status_message("No files were selected.")

    def on_add_folder(self, button):
        dialog = Gtk.FileDialog(title="Add Music Folder")
        dialog.set_modal(True)
        dialog.select_folder(self.player, None, self._on_folder_selected)

    def _on_folder_selected(self, dialog, result):
        folder = dialog.select_folder_finish(result)
        if folder:
            folder_path = folder.get_path()
            if folder_path:
                self.player.add_folder_to_playlist(folder_path)
            else:
                self.player.set_status_message("Folder selection did not return a valid path.")

    def on_remove(self, button):
        position = self.selection_model.get_selected()
        if position != Gtk.INVALID_LIST_POSITION:
            item = self.selection_model.get_item(position)
            if item is not None:
                stored_string = item.get_string()
                if "|" in stored_string:
                    track_title, track_id = stored_string.rsplit("|", 1)
                else:
                    track_title = stored_string
                    track_id = hashlib.md5(track_title.encode()).hexdigest()[:16]
                idx = self._playlist_by_id.get(track_id)
                removed_index = None
                if idx is not None:
                    # Find actual index in playlist
                    for i, pitem in enumerate(self.player.playlist):
                        if (isinstance(pitem, PlaylistItem) and pitem.id == track_id) or \
                           (isinstance(pitem, dict) and pitem.get("id") == track_id):
                            self.player.playlist.pop(i)
                            removed_index = i
                            break
                if removed_index is not None and removed_index == getattr(
                    self.player, "current_track", -1
                ):
                    self.player.current_track = -1
                self.player._invalidate_playlist_index()
                self._rebuild_playlist_index()
                self.player._update_playlist_display()
                self.player._schedule_playlist_save()
                self.update_statistics()

    def on_clear(self, button):
        while self.playlist_store.get_n_items() > 0:
            self.playlist_store.remove(0)
        self.player.playlist.clear()
        self.player.current_track = -1
        self.player._invalidate_playlist_index()
        self._rebuild_playlist_index()
        self.player._schedule_playlist_save()
        self.update_statistics()

    def _on_factory_setup(self, factory, list_item):
        label = Gtk.Label()
        label.set_halign(Gtk.Align.START)
        list_item.set_child(label)

    def _on_factory_bind(self, factory, list_item):
        label = list_item.get_child()
        item = list_item.get_item()
        if item is not None:
            text = item.get_string() if hasattr(item, "get_string") else str(item)
            if "|" in text:
                text = text.rsplit("|", 1)[0]
            label.set_text(text)

    def on_selection_changed(self, selection, position, n_items):
        selected_pos = selection.get_selected()
        if selected_pos != Gtk.INVALID_LIST_POSITION:
            item = self.selection_model.get_item(selected_pos)
            if item is not None:
                stored_string = item.get_string()
                if "|" in stored_string:
                    track_title, track_id = stored_string.rsplit("|", 1)
                else:
                    track_title = stored_string
                    track_id = hashlib.md5(track_title.encode()).hexdigest()[:16]
                # Use O(1) lookup instead of linear scan
                for i, pitem in enumerate(self.player.playlist):
                    if (isinstance(pitem, PlaylistItem) and pitem.id == track_id) or \
                       (isinstance(pitem, dict) and pitem.get("id") == track_id):
                        self.player.current_track = i
                        break

    def on_search_changed(self, entry):
        search_text = (entry.get_text() or "").strip().lower()
        # Skip if same as last search
        if search_text == self._last_search_text:
            return
        self._last_search_text = search_text

        if self._search_debounce_id is not None:
            GLib.source_remove(self._search_debounce_id)
            self._search_debounce_id = None

        def apply_filter():
            if not search_text:
                self.filter_model.set_filter(None)
            else:
                # Reuse filter function if possible
                if self._filter_func_ref is None:
                    self._filter_func_ref = Gtk.CustomFilter.new(self._filter_func)
                self.filter_model.set_filter(self._filter_func_ref)
            self._search_debounce_id = None
            return False

        # Increase debounce from 120ms to 200ms
        self._search_debounce_id = GLib.timeout_add(200, apply_filter)

    def _filter_func(self, item, search_text):
        if item is None:
            return False
        raw = item.get_string() if hasattr(item, "get_string") else str(item)
        title, track_id = (*raw.rsplit("|", 1), "")[:2] if "|" in raw else (raw, "")
        terms = [t for t in search_text.split() if t]
        if not terms:
            return True
        haystacks = [title.lower()]
        if track_id:
            pitem = self._playlist_by_id.get(track_id)
            if pitem is not None:
                haystacks.append((pitem.path or "").lower())
        for term in terms:
            if not any(term in h for h in haystacks):
                return False
        return True

    def on_clear_search(self, button):
        self.search_entry.set_text("")
        self.filter_model.set_filter(None)

    def on_shuffle(self, button):
        if self.player.playlist:
            current_item = self._current_playlist_item()
            random.shuffle(self.player.playlist)
            self.player._invalidate_playlist_index()
            self._rebuild_playlist_index()
            self.player._update_playlist_display()
            self.player.save_playlist()
            self._restore_current_track(current_item)

    def on_sort_changed(self, dropdown, pspec):
        selected = dropdown.get_selected()
        if selected == 0:
            return
        current_item = self._current_playlist_item()
        if selected == 1:
            self.player.playlist.sort(key=lambda x: (getattr(x, "title", "") or "").lower())
        elif selected == 2:
            self.player.playlist.sort(key=lambda x: (getattr(x, "path", "") or "").lower())
        elif selected == 3:
            self.player.playlist.sort(key=lambda x: getattr(x, "duration", 0) or 0)
        self.player._invalidate_playlist_index()
        self._rebuild_playlist_index()
        self.player._update_playlist_display()
        self.player.save_playlist()
        self._restore_current_track(current_item)

    def _current_playlist_item(self):
        """Return the playlist item at the current track index, if any."""
        idx = getattr(self.player, "current_track", -1)
        playlist = getattr(self.player, "playlist", None) or []
        if 0 <= idx < len(playlist):
            return playlist[idx]
        return None

    def _restore_current_track(self, item) -> None:
        """Point ``current_track`` back at ``item`` after a playlist reorder."""
        if item is None:
            return
        item_id = getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None)
        if not item_id:
            return
        for i, pitem in enumerate(self.player.playlist):
            pid = getattr(pitem, "id", None) or (
                pitem.get("id") if isinstance(pitem, dict) else None
            )
            if pid == item_id:
                self.player.current_track = i
                return

    def update_statistics(self):
        total_tracks = len(self.player.playlist)
        total_duration = 0
        for item in self.player.playlist:
            if isinstance(item, PlaylistItem):
                dur = item.duration or 0
            elif isinstance(item, dict):
                dur = item.get("duration", 0) or 0
            else:
                continue
            if dur > 0:
                total_duration += int(dur)
        self._rebuild_playlist_index()
        if total_duration > 0:
            hours = total_duration // 3600
            minutes = (total_duration % 3600) // 60
            if hours:
                duration_text = f"{hours}h {minutes}m"
            else:
                duration_text = f"{minutes}m"
            stats_text = f"{total_tracks} tracks • {duration_text}"
        else:
            stats_text = f"{total_tracks} tracks"
        self.stats_label.set_text(stats_text)

    def on_import_m3u(self, button):
        dialog = Gtk.FileDialog(title="Import M3U Playlist")
        dialog.set_modal(True)
        m3u_filter = Gtk.FileFilter()
        m3u_filter.set_name("M3U playlist files")
        m3u_filter.add_mime_type("audio/x-mpegurl")
        m3u_filter.add_pattern("*.m3u")
        m3u_filter.add_pattern("*.M3U")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(m3u_filter)
        dialog.set_filters(filters)
        dialog.open(self.player, None, self._on_m3u_import_selected)

    def _on_m3u_import_selected(self, dialog, result):
        file = dialog.open_finish(result)
        if file:
            filepath = file.get_path()
            if filepath:
                self._import_m3u_file(filepath)

    def _import_m3u_file(self, filepath):
        imported_files = read_m3u_paths(filepath)
        if imported_files:
            self.player.add_to_playlist(imported_files)

    def on_export_m3u(self, button):
        if not self.player.playlist:
            return
        dialog = Gtk.FileDialog(title="Export M3U Playlist")
        dialog.set_modal(True)
        m3u_filter = Gtk.FileFilter()
        m3u_filter.set_name("M3U playlist files")
        m3u_filter.add_mime_type("audio/x-mpegurl")
        m3u_filter.add_pattern("*.m3u")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(m3u_filter)
        dialog.set_filters(filters)
        dialog.set_initial_name("playlist.m3u")
        dialog.save(self.player, None, self._on_m3u_export_selected)

    def _on_m3u_export_selected(self, dialog, result):
        file = dialog.save_finish(result)
        if file:
            filepath = file.get_path()
            if filepath:
                if not filepath.lower().endswith(".m3u"):
                    filepath += ".m3u"
                self._export_m3u_file(filepath)

    def _export_m3u_file(self, filepath):
        write_m3u_lines(filepath, self.player.playlist)

    def on_remove_duplicates(self, button):
        if not self.player.playlist:
            return
        seen_paths = set()
        duplicates_removed = 0
        unique_playlist = []
        for item in self.player.playlist:
            norm_path = os.path.realpath(item.path)
            if norm_path in seen_paths:
                duplicates_removed += 1
                continue
            seen_paths.add(norm_path)
            unique_playlist.append(item)
        if duplicates_removed > 0:
            self.player.playlist = unique_playlist
            self.player._invalidate_playlist_index()
            self._rebuild_playlist_index()
            self.player._update_playlist_display()
            self.player.save_playlist()

    def on_row_activated(self, column_view, position):
        item = self.selection_model.get_item(position)
        if item is None:
            return
        stored_string = item.get_string()
        if "|" in stored_string:
            track_title, track_id = stored_string.rsplit("|", 1)
        else:
            track_title = stored_string
            track_id = hashlib.md5(track_title.encode()).hexdigest()[:16]
        # Use O(1) lookup instead of linear scan
        for i, pitem in enumerate(self.player.playlist):
            if (isinstance(pitem, PlaylistItem) and pitem.id == track_id) or \
               (isinstance(pitem, dict) and pitem.get("id") == track_id):
                self.player.play_track(i)
                break

    def on_fix_tags(self, button):
        if not hasattr(self.player, "playlist") or not self.player.playlist:
            self.player.set_status_message("No tracks in playlist to fix")
            return
        if not METADATA_FIXER_AVAILABLE:
            self.player.set_status_message("Metadata fixer not available")
            return
        self._run_fix_tags()

    def _run_fix_tags(self):
        if not self.player.playlist:
            return
        file_paths = [
            item.path
            for item in self.player.playlist
            if hasattr(item, "path") and item.path and os.path.exists(item.path)
        ]
        if not file_paths:
            self.player.set_status_message("No valid files to fix")
            return
        self.player.set_status_message(f"Fixing metadata for {len(file_paths)} tracks...")

        def fix_in_background():
            fixer = MetadataFixer()
            results = fixer.fix_files_threaded(file_paths, metadata=None, cleanup_options=None)
            successful = sum(1 for r in results.values() if r.get("success", False))
            failed = len(results) - successful
            GLib.idle_add(self._on_fix_tags_complete, successful, failed)

        thread = threading.Thread(target=fix_in_background, daemon=True)
        thread.start()

    def _on_fix_tags_complete(self, successful: int, failed: int):
        if getattr(self, "_fix_tags_in_progress", False):
            return
        self._fix_tags_in_progress = True
        GLib.idle_add(self._on_fix_tags_complete_impl, successful, failed)

    def _on_fix_tags_complete_impl(self, successful: int, failed: int):
        self._fix_tags_in_progress = False
        self.player.set_status_message(f"Fixed {successful} tracks, {failed} failed")
        if hasattr(self.player, "_update_playlist_display"):
            self.player._update_playlist_display()
