"""Shared widgets: the icon button factory and the cover art view."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..coverart import CoverArt
from ..deps import Gdk, GdkPixbuf, GLib, Gtk


def create_icon_button(text: str, icon_name: str | None = None) -> Gtk.Button:
    button = Gtk.Button()
    if icon_name:
        content_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        icon = Gtk.Image.new_from_icon_name(icon_name)
        icon.set_pixel_size(14)
        content_box.append(icon)
        label = Gtk.Label(label=text)
        content_box.append(label)
        button.set_child(content_box)
    else:
        button.set_label(text)
    return button


class CoverArtWidget(Gtk.Box):
    def __init__(self, player: Any, size: int = 150) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.player = player
        self.size = size
        self.current_cover_path = None
        self.loading = False
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="CoverArtWidget")
        self.set_size_request(size, size)
        self.set_halign(Gtk.Align.CENTER)
        self.set_valign(Gtk.Align.CENTER)
        self.image = Gtk.Image()
        self.image.set_size_request(size, size)
        self.image.set_halign(Gtk.Align.CENTER)
        self.image.set_valign(Gtk.Align.CENTER)
        self.image.set_icon_size(Gtk.IconSize.LARGE)
        self.append(self.image)
        self.show_placeholder()

    def show_placeholder(self) -> None:
        self.image.set_from_icon_name("folder-music")
        self.image.set_pixel_size(self.size // 2)
        self.loading = False

    def _create_texture_from_bytes(self, bytes_data: bytes) -> Gdk.Texture | None:
        """Create a Gdk.Texture from raw image bytes, scaled to self.size."""
        loader = GdkPixbuf.PixbufLoader()
        loader.write(bytes_data)
        loader.close()
        pixbuf = loader.get_pixbuf()
        if not pixbuf:
            return None
        width = pixbuf.get_width()
        height = pixbuf.get_height()
        if width > height:
            scale = self.size / width
        else:
            scale = self.size / height
        new_width = int(width * scale)
        new_height = int(height * scale)
        scaled = pixbuf.scale_simple(new_width, new_height, GdkPixbuf.InterpType.BILINEAR)
        data = scaled.save_to_bufferv("png", [], [])
        if not data:
            return None
        return Gdk.Texture.new_from_bytes(GLib.Bytes.new(data))

    def load_cover(self, cover_art: CoverArt | None, audio_path: str = "") -> None:
        if not cover_art:
            self.show_placeholder()
            return
        if cover_art.path and os.path.exists(cover_art.path):
            try:
                texture = Gdk.Texture.new_from_filename(cover_art.path)
                self.image.set_from_paintable(texture)
                self.image.set_size_request(self.size, self.size)
                self.image.set_halign(Gtk.Align.FILL)
                self.image.set_valign(Gtk.Align.FILL)
                self.current_cover_path = cover_art.path
            except GLib.GError:
                if cover_art.bytes_data:
                    texture = self._create_texture_from_bytes(cover_art.bytes_data)
                    if texture:
                        self.image.set_from_paintable(texture)
                        self.current_cover_path = audio_path
                    else:
                        self.show_placeholder()
                else:
                    self.show_placeholder()
        elif cover_art.bytes_data:
            texture = self._create_texture_from_bytes(cover_art.bytes_data)
            if texture:
                self.image.set_from_paintable(texture)
                self.current_cover_path = audio_path
            else:
                self.show_placeholder()
        else:
            self.show_placeholder()

    def update_settings(self) -> None:
        settings = self.player.settings
        if hasattr(settings, "cover_art_shape"):
            self._apply_shape()

    def _apply_shape(self) -> None:
        settings = self.player.settings
        if not hasattr(settings, "cover_art_shape"):
            return
        shape = settings.cover_art_shape.lower()
        for css_class in ["cover-square", "cover-circle", "cover-rounded"]:
            self.remove_css_class(css_class)
        if shape == "circle":
            self.add_css_class("cover-circle")
        elif shape == "rounded":
            self.add_css_class("cover-rounded")
        elif shape == "square":
            self.add_css_class("cover-square")
        self.queue_draw()

    def set_size(self, size: int) -> None:
        if size < 50:
            size = 50
        elif size > 800:
            size = 800
        self.size = size
        self.set_size_request(size, size)
        self.image.set_size_request(size, size)

    def load_cover_from_path(self, cover_path: str | Path) -> None:
        if not cover_path or not os.path.exists(cover_path):
            self.show_placeholder()
            return
        texture = Gdk.Texture.new_from_filename(str(cover_path))
        self.image.set_from_paintable(texture)
        self.image.set_size_request(self.size, self.size)
        self.image.set_halign(Gtk.Align.FILL)
        self.image.set_valign(Gtk.Align.FILL)
        self.current_cover_path = str(cover_path)

    def clear(self) -> None:
        self.current_cover_path = None
        self.show_placeholder()

    def cleanup(self) -> None:
        if hasattr(self, "executor") and self.executor:
            if not getattr(self.executor, "_shutdown", False):
                self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None
