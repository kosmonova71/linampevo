"""Transport controls, seek bar, volume, and the cover art view."""

from __future__ import annotations

import threading

from ..deps import GLib, Gst, Gtk, Pango
from ..deps import get_default_screen_size as _get_default_screen_size
from .widgets import CoverArtWidget


class PlayerTab(Gtk.Box):
    def __init__(self, player):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.player = player
        self.render_lock = threading.RLock()
        self._frame_counter = 0
        self._last_beat_time = 0
        self._beat_times = []
        self.add_css_class("player-tab")
        self.set_spacing(8)
        self.set_margin_top(8)
        self.set_margin_bottom(8)
        self.set_margin_start(8)
        self.set_margin_end(8)
        top_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.append(top_row)
        self.cover_art_widget = CoverArtWidget(player, size=200)
        self.cover_art_widget.add_css_class("player-cover")
        if player.settings.show_cover_art:
            top_row.append(self.cover_art_widget)
        info_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        info_box.set_hexpand(True)
        top_row.append(info_box)
        self.track_label = Gtk.Label(label="No track playing")
        self.track_label.set_ellipsize(Pango.EllipsizeMode.END)
        self.track_label.add_css_class("track-label")
        self.track_label.set_halign(Gtk.Align.START)
        info_box.append(self.track_label)
        self.time_label = Gtk.Label(label="--:-- / --:--")
        self.time_label.add_css_class("time-label")
        self.time_label.set_halign(Gtk.Align.START)
        info_box.append(self.time_label)
        seek_adjustment = Gtk.Adjustment(
            value=0, lower=0, upper=100, step_increment=0.1, page_increment=1, page_size=0
        )
        self.progress = Gtk.Scale(
            orientation=Gtk.Orientation.HORIZONTAL, adjustment=seek_adjustment
        )
        self.progress.set_hexpand(True)
        self.progress.set_size_request(-1, 20)
        self.progress.add_css_class("seek-bar")
        self.progress.set_draw_value(False)
        self.progress_controller = Gtk.GestureClick()
        self.progress_controller.connect("pressed", self.on_progress_pressed)
        self.progress.add_controller(self.progress_controller)
        info_box.append(self.progress)
        main_controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        main_controls.set_halign(Gtk.Align.CENTER)
        main_controls.set_homogeneous(True)
        self.append(main_controls)
        self.prev_btn = self._create_icon_button("media-skip-backward-symbolic", "Previous")
        self.play_btn = self._create_icon_button("media-playback-start-symbolic", "Play")
        self.pause_btn = self._create_icon_button("media-playback-pause-symbolic", "Pause")
        self.stop_btn = self._create_icon_button("media-playback-stop-symbolic", "Stop")
        self.next_btn = self._create_icon_button("media-skip-forward-symbolic", "Next")
        for btn in [self.prev_btn, self.play_btn, self.pause_btn, self.stop_btn, self.next_btn]:
            btn.add_css_class("control-button")
            btn.set_size_request(32, 32)
            main_controls.append(btn)
        self.play_btn.add_css_class("suggested-action")
        self.prev_btn.connect("clicked", self.player.on_prev)
        self.play_btn.connect("clicked", self.player.on_play)
        self.pause_btn.connect("clicked", self.player.on_pause)
        self.stop_btn.connect("clicked", self.player.on_stop)
        self.next_btn.connect("clicked", self.player.on_next)
        secondary_controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=2)
        secondary_controls.set_halign(Gtk.Align.CENTER)
        self.append(secondary_controls)
        self.repeat_btn = self._create_icon_button("media-playlist-repeat-symbolic", "Repeat")
        self.shuffle_btn = self._create_icon_button("media-playlist-shuffle-symbolic", "Shuffle")
        self.beat_btn = self._create_icon_button("process-working-symbolic", "Beat")
        self.download_btn = self._create_icon_button("folder-download-symbolic", "Download")
        self.autonext_btn = self._create_icon_button("go-next-symbolic", "Auto Next")
        for btn in [
            self.repeat_btn,
            self.shuffle_btn,
            self.beat_btn,
            self.download_btn,
            self.autonext_btn,
        ]:
            btn.add_css_class("icon-button")
            btn.set_size_request(24, 24)
            secondary_controls.append(btn)
        self.repeat_btn.connect("clicked", lambda b: self.player.toggle_repeat_mode())
        self.shuffle_btn.connect("clicked", lambda b: self.player.toggle_shuffle_mode())
        self.beat_btn.connect("clicked", lambda b: self.player.toggle_beat_aware())
        self.download_btn.connect("clicked", lambda b: self.player.toggle_auto_download_covers())
        self.autonext_btn.connect("clicked", lambda b: self.player.toggle_auto_play_next())
        vol_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        vol_row.set_halign(Gtk.Align.CENTER)
        self.append(vol_row)
        self.volume_button = self._create_volume_button()
        self.volume_button.add_css_class("volume-button")
        self.volume_button.set_size_request(24, 24)
        vol_row.append(self.volume_button)
        lower, upper = 0, 100
        value = max(lower, min(upper, 70))
        vol_adjustment = Gtk.Adjustment(
            value=value, lower=lower, upper=upper, step_increment=1, page_increment=10, page_size=0
        )
        self.volume_scale = Gtk.Scale(
            orientation=Gtk.Orientation.HORIZONTAL, adjustment=vol_adjustment
        )
        self.volume_scale.set_hexpand(True)
        self.volume_scale.set_draw_value(False)
        self.volume_scale.set_size_request(150, 20)
        self.volume_scale.connect("value-changed", self.player.on_volume_changed)
        vol_row.append(self.volume_scale)
        self.volume_label = Gtk.Label(label="70%")
        self.volume_label.add_css_class("volume-label")
        self.volume_label.set_size_request(35, -1)
        vol_row.append(self.volume_label)
        self.update_playback_state(False)

    def _create_volume_button(self):
        button = Gtk.Button()
        button.add_css_class("icon-button")
        button.set_tooltip_text("Mute/Unmute (M)")
        button.set_size_request(32, 28)
        button.connect("clicked", self.player.on_mute_toggle)
        self._update_volume_button_icon()
        return button

    def _update_volume_button_icon(self):
        if not hasattr(self, "volume_button") or not hasattr(self, "player") or not self.player:
            return
        player_obj = self.player.player if hasattr(self.player, "player") else self.player
        if not hasattr(player_obj, "get_property"):
            return
        current_volume = player_obj.get_property("volume")
        if current_volume == 0 or getattr(self.player, "_is_muted", False):
            icon_name = "audio-volume-muted-symbolic"
        elif current_volume < 0.3:
            icon_name = "audio-volume-low-symbolic"
        elif current_volume < 0.7:
            icon_name = "audio-volume-medium-symbolic"
        else:
            icon_name = "audio-volume-high-symbolic"
        icon = Gtk.Image.new_from_icon_name(icon_name)
        icon.set_pixel_size(16)
        self.volume_button.set_child(icon)

    def _create_icon_button(self, icon_name, tooltip_text):
        button = Gtk.Button()
        button.add_css_class("icon-button")
        button.set_tooltip_text(tooltip_text)
        button.set_size_request(36, 32)
        icon_image = Gtk.Image.new_from_icon_name(icon_name)
        icon_image.set_pixel_size(16)
        button.set_child(icon_image)
        return button

    def update_playback_state(self, is_playing):
        if is_playing:
            self.play_btn.set_visible(False)
            self.pause_btn.set_visible(True)
            self.add_css_class("playing")
        else:
            self.play_btn.set_visible(True)
            self.pause_btn.set_visible(False)
            self.remove_css_class("playing")

    def on_progress_pressed(self, gesture, n_press, x, y):
        width = self.progress.get_width()
        ratio = x / width if width > 0 else 0
        ratio = max(0.0, min(1.0, ratio))
        player_obj = self.player
        if hasattr(player_obj, "player"):
            player_obj = player_obj.player
        duration = player_obj.query_duration(Gst.Format.TIME)[1]
        if duration > 0:
            seek_time = int(ratio * duration)
            player_obj.seek(
                1.0,
                Gst.Format.TIME,
                Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
                Gst.SeekType.SET,
                seek_time,
                Gst.SeekType.NONE,
                0,
            )

    def update_track_info(self, title, artist="", album=""):
        if not title:
            title = "No track playing"
            artist = ""
        screen_size = _get_default_screen_size()
        if screen_size is not None:
            screen_width, screen_height = screen_size
            if hasattr(self, "get_surface") and self.get_surface():
                x, y = self.get_surface().get_position()
                max_x = screen_width - 450
                max_y = screen_height - 350
                if x > max_x or y > max_y or x < 0 or y < 0:
                    new_x = max(50, min(x, max_x))
                    new_y = max(50, min(y, max_y))
                    self.move(new_x, new_y)
        if artist and album:
            info_text = (
                f"<b>{GLib.markup_escape_text(title)}</b>\n"
                f"<small>{GLib.markup_escape_text(artist)} • {GLib.markup_escape_text(album)}</small>"
            )
        elif artist:
            info_text = (
                f"<b>{GLib.markup_escape_text(title)}</b>\n"
                f"<small>{GLib.markup_escape_text(artist)}</small>"
            )
        else:
            info_text = f"<b>{GLib.markup_escape_text(title)}</b>"
        self.track_label.set_markup(info_text)
        self.track_label.set_tooltip_text(f"{title}\n{artist}{' • ' + album if album else ''}")
        window = self.get_root()
        if window:
            window.set_title(f"{title} - LinAmp")
