"""The Gtk.Application subclass and the process entry point."""

from __future__ import annotations

import os
import sys

from .constants import THEME_CSS_FILE
from .deps import Gdk, Gio, GLib, Gst, Gtk
from .window import WinampWindow


class LinAmpApp(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="org.example.linamp.xmms")
        self.win = None
        Gst.init(None)
        settings = Gtk.Settings.get_default()
        if settings:
            scheme = getattr(Gtk, "InterfaceColorScheme", None)
            if scheme and settings.find_property("gtk-interface-color-scheme"):
                settings.set_property("gtk-interface-color-scheme", scheme.DARK)
            elif settings.find_property("gtk-application-prefer-dark-theme"):
                settings.set_property("gtk-application-prefer-dark-theme", True)

    def do_activate(self):
        if not self.win:
            self.win = WinampWindow(application=self, title="LinAmp")
            self.win.present()
            self.win.set_visible(True)
            self.win.grab_focus()
            GLib.timeout_add(500, self.apply_dark_theme_css)
        else:
            self.win.present()

    def apply_dark_theme_css(self):
        css_file = str(THEME_CSS_FILE)
        css_provider = Gtk.CssProvider()
        css_provider.load_from_path(css_file)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(), css_provider, Gtk.STYLE_PROVIDER_PRIORITY_USER
        )
        return False

    def do_shutdown(self):
        if hasattr(self, "win") and self.win is not None:
            self.win.full_cleanup()
        Gtk.Application.do_shutdown(self)

    def do_startup(self):
        Gtk.Application.do_startup(self)
        self.setup_menu()
        resize_action = Gio.SimpleAction.new("resize_to_cover_art", None)
        resize_action.connect("activate", self.on_resize_to_cover_art)
        self.add_action(resize_action)
        self.set_accels_for_action("app.resize_to_cover_art", ["<Control>r"])
        if len(sys.argv) > 1:
            if not self.win:
                self.win = WinampWindow(application=self, title="LinAmp")
            self.win.add_to_playlist(sys.argv[1:])
            if self.win.playlist:
                self.win.cleanup_invalid_tracks()
                if self.win.playlist:
                    self.win.play_track(0)

    def setup_menu(self):
        open_action = Gio.SimpleAction.new("open", None)
        open_action.connect("activate", self.on_open)
        open_folder_action = Gio.SimpleAction.new("open_folder", None)
        open_folder_action.connect("activate", self.on_open_folder)
        add_to_playlist_action = Gio.SimpleAction.new("add_to_playlist", None)
        add_to_playlist_action.connect("activate", self.on_add_to_playlist)
        add_folder_to_playlist_action = Gio.SimpleAction.new("add_folder_to_playlist", None)
        add_folder_to_playlist_action.connect("activate", self.on_add_folder_to_playlist)
        cleanup_action = Gio.SimpleAction.new("cleanup_invalid", None)
        cleanup_action.connect("activate", self.on_cleanup_invalid)
        quit_action = Gio.SimpleAction.new("quit", None)
        quit_action.connect("activate", lambda *_: self.quit())
        self.add_action(open_action)
        self.add_action(open_folder_action)
        self.add_action(add_to_playlist_action)
        self.add_action(add_folder_to_playlist_action)
        self.add_action(cleanup_action)
        self.add_action(quit_action)
        menu = Gio.Menu()
        file_menu = Gio.Menu()
        file_menu.append("Open File", "app.open")
        file_menu.append("Open Folder", "app.open_folder")
        file_menu.append("Add to Playlist", "app.add_to_playlist")
        file_menu.append("Add Folder to Playlist", "app.add_folder_to_playlist")
        file_menu.append("Cleanup Invalid Tracks", "app.cleanup_invalid")
        file_menu.append("Resize Window to Cover Art", "app.resize_to_cover_art")
        file_menu.append("Quit", "app.quit")
        menu.append_submenu("File", file_menu)
        self.set_menubar(menu)

    def on_open(self, action, param):
        dialog = Gtk.FileChooserNative.new(
            title="Open Audio File", parent=self.win, action=Gtk.FileChooserAction.OPEN
        )
        audio_filter = Gtk.FileFilter()
        audio_filter.set_name("Audio files")
        audio_filter.add_mime_type("audio/*")
        dialog.add_filter(audio_filter)
        dialog.connect("response", self.on_file_chooser_response)
        dialog.set_visible(True)

    def on_add_to_playlist(self, action, param):
        dialog = Gtk.FileChooserNative.new(
            title="Add to Playlist",
            parent=self.win,
            action=Gtk.FileChooserAction.OPEN,
            select_multiple=True,
        )
        audio_filter = Gtk.FileFilter()
        audio_filter.set_name("Audio files")
        audio_filter.add_mime_type("audio/*")
        dialog.add_filter(audio_filter)
        dialog.connect("response", self.on_add_to_playlist_response)
        dialog.set_visible(True)

    def on_open_folder(self, action, param):
        dialog = Gtk.FileChooserNative.new(
            title="Open Music Folder", parent=self.win, action=Gtk.FileChooserAction.SELECT_FOLDER
        )
        dialog.connect("response", self.on_folder_chooser_response)
        dialog.set_visible(True)

    def on_add_folder_to_playlist(self, action, param):
        dialog = Gtk.FileChooserNative.new(
            title="Add Music Folder to Playlist",
            parent=self.win,
            action=Gtk.FileChooserAction.SELECT_FOLDER,
        )
        dialog.connect("response", self.on_add_folder_to_playlist_response)
        dialog.set_visible(True)

    def on_file_chooser_response(self, dialog, response):
        if response == Gtk.ResponseType.OK:
            file = dialog.get_property("file")
            if file:
                self.win.playlist.clear()
                self.win._invalidate_playlist_index()
                if hasattr(self.win, "playlist_tab"):
                    store = self.win.playlist_tab.playlist_store
                    while store.get_n_items() > 0:
                        store.remove(0)
                self._handle_playlist_addition(file.get_path(), is_folder=False)
                self.win.play_track(0)
        dialog.destroy()

    def on_add_to_playlist_response(self, dialog, response):
        if response == Gtk.ResponseType.OK:
            files = dialog.get_files()
            if files:
                for file in files:
                    file_path = file.get_path()
                    if file_path and os.path.isfile(file_path):
                        self.win.add_to_playlist([file_path])
                self.win.play_track(0)
        dialog.destroy()

    def on_folder_chooser_response(self, dialog, response):
        if response == Gtk.ResponseType.OK:
            folder = dialog.get_file()
            if folder:
                folder_path = folder.get_path()
                if folder_path:
                    self.win.playlist.clear()
                    self.win._invalidate_playlist_index()
                    if hasattr(self.win, "playlist_tab"):
                        store = self.win.playlist_tab.playlist_store
                        while store.get_n_items() > 0:
                            store.remove(0)
                    self._handle_playlist_addition(folder_path, is_folder=True)
        dialog.destroy()

    def on_add_folder_to_playlist_response(self, dialog, response):
        if response == Gtk.ResponseType.OK:
            folder = dialog.get_file()
            if folder:
                self._handle_playlist_addition(folder.get_path(), is_folder=True)
        dialog.destroy()

    def _handle_playlist_addition(self, file_path, is_folder=False):
        if not file_path or not hasattr(self, "win") or not self.win:
            return False
        was_empty = not self.win.playlist
        if is_folder:
            self.win.add_folder_to_playlist(file_path)
        else:
            self.win.add_to_playlist([file_path])
        if was_empty and self.win.playlist:
            self.win.play_track(0)
        return True

    def on_cleanup_invalid(self, action, param):
        if hasattr(self, "win") and self.win:
            self.win.cleanup_invalid_tracks()

    def on_resize_to_cover_art(self, action, param):
        if hasattr(self.win, "cover_art_widget") and self.win.cover_art_widget:
            self.win.resize_to_cover_art()


def main() -> None:
    app = LinAmpApp()
    app.run(sys.argv)
