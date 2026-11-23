"""GStreamer playback engine.

Everything here is UI-free logic that drives ``Gst`` elements and reacts to
bus messages. It is mixed into the window rather than composed with it, so
the engine can reach window state (``self.player``, ``self.playlist``,
``self.player_tab``) without an import cycle. Nothing in this module may
import ``window`` or ``app``.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from functools import lru_cache

from .constants import (
    BEAT_DETECTION_INTERVAL,
    BEAT_HISTORY_SIZE,
    DEFAULT_VOLUME,
)
from .deps import NUMPY_AVAILABLE, Gio, GLib, Gst, np
from .stations import RadioStation


# Cache FFT windows to avoid repeated allocations
@lru_cache(maxsize=16)
def _get_hanning_window(size: int) -> np.ndarray:
    """Get cached Hanning window for given FFT size."""
    return np.hanning(size).astype(np.float32)


def _get_fft_size(n: int) -> int:
    """Get optimal FFT size for given sample count."""
    fft_size = max(512, 1 << (n - 1).bit_length())
    return min(fft_size, 16384)


class AudioEngine:
    """Playback, bus handling, stream recovery, and beat detection."""

    def _set_player_state_thread_safe(self, state):
        return self.player.set_state(state)

    def _set_player_property_thread_safe(self, property_name, value):
        self.player.set_property(property_name, value)
        return True

    def setup_player(self):
        self.player = Gst.ElementFactory.make("playbin", "player")
        if not self.player:
            GLib.idle_add(self.set_status_message, "GStreamer initialization failed")
            return
        self.playing = False
        self.equalizer = None
        self.audio_sink = None
        self.audio_convert = None
        self.audio_resample = None
        self._audio_sample_rate = 44100.0
        self._audio_channels = 2
        self._rendering_enabled = True
        self.system_audio_pipeline = None
        self.system_audio_probe_id = None
        audio_sink = self._build_visualizer_audio_sink()
        if not audio_sink:
            audio_sink = Gst.ElementFactory.make("autoaudiosink", "audio_sink")
        if not audio_sink:
            audio_sink = Gst.ElementFactory.make("pulsesink", "audio_sink")
        if not audio_sink:
            audio_sink = Gst.ElementFactory.make("alsasink", "audio_sink")
        if audio_sink:
            self.player.set_property("audio-sink", audio_sink)
            self._setup_equalizer()
        self.player.set_property("volume", DEFAULT_VOLUME)
        self._song_finished_id = self.player.connect("about-to-finish", self.on_song_finished)
        self._setup_player_bus()
        GLib.idle_add(self._setup_system_audio_capture)

    def _setup_equalizer(self):
        self.equalizer = Gst.ElementFactory.make("equalizer-10bands", "equalizer")
        if not self.equalizer:
            return False
        self.player.set_property("audio-filter", self.equalizer)
        for i in range(10):
            self.equalizer.set_property(f"band{i}", 0.0)
        return True

    def _build_visualizer_audio_sink(self):
        """Build an audio-sink bin that delivers a known format to the visualizer.
        The visualizer's pad-probe reads buffer bytes as ``np.float32`` and runs
        an FFT to derive bass/mid/treb bands. That assumption only holds if the
        sink actually emits float32 stereo at a known sample rate. A bare
        ``autoaudiosink`` negotiates whatever the system sink wants (often S16LE
        integer PCM), so the float32 reinterpretation produces garbage and the
        visuals stop tracking the music.
        Force ``F32LE, 2ch, 44100 Hz`` so the probe's interpretation is correct.
        Falls back to ``None`` (caller falls back to a plain autoaudiosink) if any
        element is unavailable.
        """
        convert = Gst.ElementFactory.make("audioconvert", "vis_convert")
        resample = Gst.ElementFactory.make("audioresample", "vis_resample")
        capsfilter = Gst.ElementFactory.make("capsfilter", "vis_caps")
        sink = Gst.ElementFactory.make("autoaudiosink", "vis_sink")
        if not (convert and resample and capsfilter and sink):
            return None
        caps = Gst.Caps.from_string(
            "audio/x-raw, format=F32LE, rate=44100, channels=2, layout=interleaved"
        )
        capsfilter.set_property("caps", caps)
        audio_bin = Gst.Bin.new("vis_audio_sink")
        for element in (convert, resample, capsfilter, sink):
            audio_bin.add(element)
        if not (convert.link(resample) and resample.link(capsfilter) and capsfilter.link(sink)):
            return None
        sink_pad = convert.get_static_pad("sink")
        audio_bin.add_pad(Gst.GhostPad.new("sink", sink_pad))
        self.audio_convert = convert
        self.audio_resample = resample
        self.audio_sink = audio_bin
        self._audio_sample_rate = 44100.0
        self._audio_channels = 2
        return audio_bin

    def _setup_player_bus(self):
        bus = self.player.get_bus()
        if bus:
            if self._main_bus_signal_watch_attached:
                bus.remove_signal_watch()
                self._main_bus_signal_watch_attached = False
            if hasattr(self, "_bus_watch_id") and self._bus_watch_id is not None:
                bus.disconnect(self._bus_watch_id)
                self._bus_watch_id = None
            bus.add_signal_watch()
            self._main_bus_signal_watch_attached = True
            self._bus_watch_id = bus.connect("message", self.on_bus_message)

    def _setup_system_audio_capture(self):
        """Set up system audio capture for real-time visualization.
        Called via GLib.idle_add - must return False.
        """
        self.system_audio_pipeline = Gst.Pipeline.new("system_audio_capture")
        source = Gst.ElementFactory.make("pulsesrc", "system_audio_source")
        if not source:
            print(
                "System Audio: pulsesrc unavailable; continuing without system audio visualization."
            )
            self.system_audio_pipeline = None
            return False
        monitor_device = os.environ.get("LINAMP_PULSE_MONITOR", "default")
        source.set_property("device", monitor_device)
        audioconvert = Gst.ElementFactory.make("audioconvert", "system_audio_convert")
        audioresample = Gst.ElementFactory.make("audioresample", "system_audio_resample")
        capsfilter = Gst.ElementFactory.make("capsfilter", "system_audio_caps")
        capsfilter.set_property(
            "caps", Gst.Caps.from_string("audio/x-raw,format=F32LE,channels=2,rate=44100")
        )
        if not all([audioconvert, audioresample, capsfilter]):
            print(
                "System Audio: missing audio processing elements; continuing without system audio visualization."
            )
            self.system_audio_pipeline = None
            return False
        self.system_audio_pipeline.add(source)
        self.system_audio_pipeline.add(audioconvert)
        self.system_audio_pipeline.add(audioresample)
        self.system_audio_pipeline.add(capsfilter)
        if not source.link(audioconvert):
            print("System Audio: could not link source to converter")
            self.system_audio_pipeline.set_state(Gst.State.NULL)
            self.system_audio_pipeline = None
            return False
        if not audioconvert.link(audioresample):
            print("System Audio: could not link converter to resample")
            self.system_audio_pipeline.set_state(Gst.State.NULL)
            self.system_audio_pipeline = None
            return False
        if not audioresample.link(capsfilter):
            print("System Audio: could not link resample to capsfilter")
            self.system_audio_pipeline.set_state(Gst.State.NULL)
            self.system_audio_pipeline = None
            return False
        sink_pad = capsfilter.get_static_pad("src")
        if sink_pad:
            self.system_audio_probe_id = sink_pad.add_probe(
                Gst.PadProbeType.BUFFER, self._system_audio_probe_callback, None
            )
        state_ret = self.system_audio_pipeline.set_state(Gst.State.PLAYING)
        if state_ret == Gst.StateChangeReturn.FAILURE:
            self.system_audio_pipeline.set_state(Gst.State.NULL)
            self.system_audio_pipeline = None
            self.system_audio_probe_id = None
            print(
                "System Audio: capture pipeline unavailable (PulseAudio/monitor device "
                "not present); continuing without system audio visualization."
            )
            return False
        self._system_audio_buffer = np.array([], dtype=np.float32)
        self._system_audio_consume_id = GLib.timeout_add(50, self._system_audio_consume)
        return False

    def _system_audio_probe_callback(self, pad, info, user_data):
        """Callback for system audio capture - optimized with numpy."""
        if not self._rendering_enabled:
            return Gst.PadProbeReturn.OK

        buffer = info.get_buffer()
        if buffer:
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                data = map_info.data
                if data:
                    # Convert directly to numpy array (F32LE stereo -> mono)
                    audio_data = np.frombuffer(data, dtype=np.float32)
                    # Downmix stereo to mono by averaging channels
                    if audio_data.size >= 2:
                        audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                    # Limit buffer size
                    if audio_data.size > 4096:
                        audio_data = audio_data[:4096]
                    # Append to buffer
                    if self._system_audio_buffer.size == 0:
                        self._system_audio_buffer = audio_data
                    else:
                        self._system_audio_buffer = np.concatenate(
                            [self._system_audio_buffer, audio_data]
                        )
                    # Keep buffer bounded
                    if self._system_audio_buffer.size > 65536:
                        self._system_audio_buffer = self._system_audio_buffer[-65536:]
                buffer.unmap(map_info)
        return Gst.PadProbeReturn.OK

    def _system_audio_consume(self):
        """Process stored system audio samples into frequency bands.
        Runs on the GLib main loop at ~20 Hz (50ms interval).
        """
        buffer = getattr(self, "_system_audio_buffer", None)
        if buffer is None or buffer.size == 0:
            return True
        if not hasattr(self, "visualizer_tab"):
            return True

        viz = getattr(self.visualizer_tab, "visualizer", None)
        if viz is None or not hasattr(viz, "update_audio"):
            return True

        # Get samples and clear buffer
        x = self._system_audio_buffer
        self._system_audio_buffer = np.array([], dtype=np.float32)

        # Delegate to the visualizer tab's shared analysis (spectral-flux
        # beat detection, adaptive band levels, downsampled spectrum) so
        # system audio reacts identically to file playback.
        compute = getattr(self.visualizer_tab, "_compute_audio_levels", None)
        if NUMPY_AVAILABLE and x.size >= 2 and compute is not None:
            audio_levels = compute(x, 44100.0, 1, 0.05)
            viz.update_audio(
                audio_levels,
                fft_samples=audio_levels.get("spectrum"),
                sample_rate=44100.0,
            )
            return True

        audio_levels = {
            "bass": 0.0,
            "bass_att": 0.0,
            "mid": 0.0,
            "mid_att": 0.0,
            "treb": 0.0,
            "treb_att": 0.0,
            "beat": 0.0,
            "beat_phase": 0.0,
            "rms": 0.0,
            "peak": 0.0,
            "dt": 0.0,
            "spectrum": None,
        }
        if x.size > 0:  # crude fallback when shared analysis is unavailable
            avg_level = (
                float(np.mean(np.abs(x)))
                if NUMPY_AVAILABLE
                else sum(abs(s) for s in x) / max(len(x), 1)
            )
            level = min(1.0, avg_level * 2.0)
            audio_levels.update(
                {
                    "bass": level,
                    "bass_att": level,
                    "mid": level,
                    "mid_att": level,
                    "treb": level,
                    "treb_att": level,
                    "beat": min(1.0, avg_level * 3.0),
                    "rms": avg_level,
                    "peak": (
                        float(np.max(np.abs(x)))
                        if NUMPY_AVAILABLE
                        else max(abs(s) for s in x)
                    ),
                }
            )
        viz.update_audio(audio_levels)
        return True

    def cleanup_timers(self):
        if hasattr(self, "_active_timers"):
            for timer_id in list(self._active_timers):
                self._safe_source_remove(timer_id)
            self._active_timers.clear()
        timer_attributes = [
            "beat_detection_timer",
            "update_display_timer",
            "auto_save_timer",
            "_system_audio_consume_id",
        ]
        for timer_attr in timer_attributes:
            if hasattr(self, timer_attr):
                timer_id = getattr(self, timer_attr)
                if timer_id:
                    try:
                        GLib.source_remove(timer_id)
                    except GLib.Error:
                        logging.getLogger(__name__).debug(
                            "Failed to remove timer %s: %s", timer_attr, sys.exc_info()[1]
                        )
                    setattr(self, timer_attr, 0)

    def cleanup_pipeline(self, player):
        if player is None:
            return
        bus = player.get_bus()
        if bus:
            if getattr(self, "_main_bus_signal_watch_attached", False):
                try:
                    bus.remove_signal_watch()
                except (TypeError, GLib.Error):
                    logging.getLogger(__name__).debug(
                        "Failed to remove signal watch: %s", sys.exc_info()[1]
                    )
                self._main_bus_signal_watch_attached = False
            if getattr(self, "_bus_watch_id", None) is not None:
                try:
                    bus.disconnect(self._bus_watch_id)
                except (TypeError, GLib.Error):
                    logging.getLogger(__name__).debug(
                        "Failed to disconnect bus watch: %s", sys.exc_info()[1]
                    )
                self._bus_watch_id = None
        if getattr(self, "_song_finished_id", None) is not None:
            try:
                player.disconnect(self._song_finished_id)
            except (TypeError, GLib.Error):
                logging.getLogger(__name__).debug(
                    "Failed to disconnect song_finished signal: %s", sys.exc_info()[1]
                )
            self._song_finished_id = None
        if (
            getattr(self, "audio_probe_id", None) is not None
            and getattr(self, "_probe_pad", None) is not None
        ):
            try:
                self._probe_pad.remove_probe(self.audio_probe_id)
            except (TypeError, GLib.Error):
                logging.getLogger(__name__).debug("Failed to remove probe: %s", sys.exc_info()[1])
            self.audio_probe_id = None
            self._probe_pad = None
            self._probe_sink = None
        for attr in ("audio_convert", "audio_resample", "audio_sink"):
            elem = getattr(self, attr, None)
            if elem is not None:
                try:
                    elem.set_state(Gst.State.NULL)
                except (AttributeError, GLib.Error):
                    logging.getLogger(__name__).debug(
                        "Failed to set %s state to NULL: %s", attr, sys.exc_info()[1]
                    )
                setattr(self, attr, None)
        if getattr(self, "equalizer", None) is not None:
            try:
                self.equalizer.set_state(Gst.State.NULL)
            except (AttributeError, GLib.Error):
                logging.getLogger(__name__).debug(
                    "Failed to set equalizer state to NULL: %s", sys.exc_info()[1]
                )
            self.equalizer = None

    def play_file(self, filepath, title=None):
        if not os.path.exists(filepath):
            return False
        if not hasattr(self, "player") or not self.player:
            return False
        if hasattr(self.player_tab, "track_label"):
            GLib.idle_add(
                self.player_tab.track_label.set_text, f"Loading: {os.path.basename(filepath)}..."
            )
        uri = Gio.File.new_for_path(os.path.abspath(filepath)).get_uri()
        self._current_podcast_episode = None
        self._current_podcast_feed = None
        self.player.set_state(Gst.State.NULL)
        self.player.set_property("uri", uri)
        state_change = self.player.set_state(Gst.State.PLAYING)
        if state_change == Gst.StateChangeReturn.FAILURE:
            GLib.idle_add(self.set_status_message, "Failed to start playback")
            return False
        GLib.timeout_add(100, self._verify_playback_state, filepath, title)
        return True

    def _verify_playback_state(self, stream_id, title):
        if not hasattr(self, "player") or not self.player:
            return False
        success, current_state, _pending_state = self.player.get_state(Gst.CLOCK_TIME_NONE)
        if not success:
            return False
        if current_state == Gst.State.PLAYING:
            self.playing = True
            is_radio = getattr(self, "_is_radio_stream", False)
            if is_radio and self._radio_stream_uri == stream_id:
                track_name = self._radio_stream_title or title or "Radio Stream"
                GLib.idle_add(self.player_tab.track_label.set_text, track_name)
                GLib.idle_add(self.set_title, f"LinAmp - {track_name}")
                GLib.idle_add(self.radio_tab._update_live_indicator)
            else:
                track_name = title or os.path.basename(stream_id)
                GLib.idle_add(self.player_tab.track_label.set_text, track_name)
                GLib.idle_add(self.set_title, f"LinAmp - {track_name}")
            if (
                hasattr(self, "radio_tab")
                and hasattr(self.radio_tab, "_is_playing_radio")
                and self.radio_tab._is_playing_radio
            ):
                if (
                    hasattr(self.radio_tab, "_radio_buffer_timer")
                    and self.radio_tab._radio_buffer_timer
                ):
                    GLib.source_remove(self.radio_tab._radio_buffer_timer)
                    self.radio_tab._radio_buffer_timer = 0
                self.radio_tab._is_playing_radio = False
                self.radio_tab.live_indicator.set_text("")
            if hasattr(self, "visualizer_tab") and self.visualizer_tab is not None:
                self.visualizer_tab.player = self.player
                if hasattr(self.visualizer_tab, "_rendering_enabled"):
                    self.visualizer_tab._rendering_enabled = True
                if hasattr(self.visualizer_tab, "_reconnect_audio_probe"):
                    self.visualizer_tab._reconnect_audio_probe()
            return False
        if current_state in [Gst.State.READY, Gst.State.PAUSED]:
            self.player.seek_simple(
                Gst.Format.TIME, Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT, 0
            )
            GLib.idle_add(self.player.set_state, Gst.State.PLAYING)
            self.playing = True
            if self.beat_aware_enabled:
                self.start_beat_detection()
        return False

    def play_uri(self, uri, title="Radio Stream"):
        if not uri or not RadioStation.validate_url(uri):
            GLib.idle_add(self.set_status_message, "Invalid stream URL")
            return False
        if not hasattr(self, "player") or not self.player:
            return False
        self._is_radio_stream = True
        self._radio_stream_uri = uri
        self._radio_stream_title = title
        self._current_podcast_episode = None
        self._current_podcast_feed = None
        self._radio_start_time = time.time()
        if hasattr(self.player_tab, "track_label"):
            GLib.idle_add(self.player_tab.track_label.set_text, f"Connecting: {title}...")
        self.player.set_state(Gst.State.NULL)
        self.player.set_property("uri", uri)
        state_change = self.player.set_state(Gst.State.PLAYING)
        if state_change == Gst.StateChangeReturn.FAILURE:
            self._is_radio_stream = False
            GLib.idle_add(self.set_status_message, "Failed to connect to stream")
            return False
        GLib.timeout_add(100, self._verify_playback_state, uri, title)
        return True

    def play_podcast_episode(self, episode, feed=None):
        """Stream a podcast episode URL as a regular (non-radio) track."""
        url = getattr(episode, "audio_url", "") or ""
        if not url or not RadioStation.validate_url(url):
            GLib.idle_add(self.set_status_message, "Invalid episode URL")
            return False
        if not hasattr(self, "player") or not self.player:
            return False
        self._is_radio_stream = False
        self._radio_stream_uri = ""
        self._current_podcast_episode = episode
        self._current_podcast_feed = feed
        title = getattr(episode, "title", "") or "Podcast Episode"
        if hasattr(self.player_tab, "track_label"):
            GLib.idle_add(self.player_tab.track_label.set_text, f"Loading: {title}...")
        header = getattr(self, "_ui_header_nowplaying", None)
        if header is not None:
            GLib.idle_add(header.set_text, title)
            self._last_header_text = title
        self.player.set_state(Gst.State.NULL)
        self.player.set_property("uri", url)
        state_change = self.player.set_state(Gst.State.PLAYING)
        if state_change == Gst.StateChangeReturn.FAILURE:
            self._current_podcast_episode = None
            self._current_podcast_feed = None
            GLib.idle_add(self.set_status_message, "Failed to start playback")
            return False
        GLib.timeout_add(100, self._verify_playback_state, url, title)
        position = getattr(episode, "position", 0.0) or 0.0
        if position > 2.0:
            GLib.timeout_add(1500, self._seek_to_position, position)
        return True

    def _mark_podcast_episode_played(self):
        """Flag the finished podcast episode as played and persist state."""
        episode = getattr(self, "_current_podcast_episode", None)
        feed = getattr(self, "_current_podcast_feed", None)
        self._current_podcast_episode = None
        self._current_podcast_feed = None
        if episode is None:
            return
        try:
            episode.played = True
        except Exception:
            pass
        feed_id = getattr(feed, "feed_id", "") if feed is not None else ""
        tab = getattr(self, "podcasts_tab", None)
        if not feed_id or tab is None:
            return
        try:
            if getattr(tab, "selected_feed", None) is not None and (
                tab.selected_feed.feed_id == feed_id
            ):
                # The playing episode object lives in tab.episodes already.
                tab._persist_episode_state()
                GLib.idle_add(tab._populate_episodes)
                return
            from .podcasts import load_feed_episodes

            episodes = load_feed_episodes(feed_id)
            guid = getattr(episode, "guid", "")
            audio_url = getattr(episode, "audio_url", "")
            for ep in episodes:
                if (guid and ep.guid == guid) or (
                    audio_url and ep.audio_url == audio_url
                ):
                    ep.played = True
                    break
            feeds = []
            for f in getattr(tab, "feeds", []) or []:
                feeds.append(f if f.feed_id != feed_id else feed)
            persist = getattr(self, "_save_podcast_snapshot", None)
            if callable(persist):
                persist(feed_id, episodes, feeds)
        except Exception:
            logging.getLogger(__name__).debug(
                "Failed to persist podcast played state", exc_info=True
            )

    def on_pause(self, button):
        if self.playing:
            GLib.idle_add(self.player.set_state, Gst.State.PAUSED)
            self.playing = False
            self.stop_beat_detection()

    def on_play(self, button):
        if self.playing:
            self.on_pause(button)
        else:
            if self.current_track >= 0 and self.current_track < len(self.playlist):
                self.play_track(self.current_track)
            elif self.playlist:
                self.play_track(0)
            else:
                GLib.idle_add(self.set_status_message, "No tracks in playlist - add files first")

    def stop(self):
        self.on_stop(None)

    def on_stop(self, button):
        if hasattr(self, "_cleanup_in_progress") and self._cleanup_in_progress:
            return
        self._is_radio_stream = False
        self._current_podcast_episode = None
        self._current_podcast_feed = None
        if hasattr(self, "_radio_buffer_timer") and self._radio_buffer_timer:
            GLib.source_remove(self._radio_buffer_timer)
            self._radio_buffer_timer = 0
        if hasattr(self, "player") and self.player:
            try:
                if hasattr(self, "_song_finished_id") and self._song_finished_id:
                    self.player.disconnect(self._song_finished_id)
                    self._song_finished_id = None
            except TypeError:
                logging.getLogger(__name__).debug(
                    "Failed to disconnect song_finished signal: %s", sys.exc_info()[1]
                )
            self.player.set_state(Gst.State.NULL)
            self.cleanup_pipeline(self.player)
        self.playing = False
        self.stop_beat_detection()
        if hasattr(self, "player_tab"):
            self.player_tab.time_label.set_text("0:00 / 0:00")
            self.player_tab.progress.set_value(0)
            self.player_tab.track_label.set_text("No track playing")
            if hasattr(self.player_tab, "cover_art_widget"):
                self.player_tab.cover_art_widget.clear()
        self.set_title("LinAmp - XMMS Style")
        self.save_settings_on_stop()

    def on_volume_changed(self, scale):
        volume = scale.get_value() / 100.0
        if hasattr(self, "player_tab") and hasattr(self.player_tab, "volume_label"):
            self.player_tab.volume_label.set_label(f"{int(scale.get_value())}%")
        self.player.set_property("volume", volume)
        if hasattr(self, "_is_muted") and self._is_muted and volume > 0:
            self._is_muted = False
            self._previous_volume = volume
        if hasattr(self, "player_tab") and hasattr(self.player_tab, "_update_volume_button_icon"):
            self.player_tab._update_volume_button_icon()
        self.auto_save_settings()

    def on_mute_toggle(self, button):
        if not hasattr(self, "player") or not self.player:
            return
        current_volume = self.player.get_property("volume")
        if hasattr(self, "_is_muted") and self._is_muted:
            self.player.set_property("volume", self._previous_volume)
            self._is_muted = False
        elif current_volume > 0:
            self._previous_volume = current_volume
            self.player.set_property("volume", 0.0)
            self._is_muted = True
        else:
            self._previous_volume = 1.0
            self.player.set_property("volume", 1.0)
            self._is_muted = False
        if hasattr(self, "player_tab") and hasattr(self.player_tab, "_update_volume_button_icon"):
            self.player_tab._update_volume_button_icon()
        actual_volume = self.player.get_property("volume")
        if hasattr(self, "player_tab") and hasattr(self.player_tab, "volume_scale"):
            self.player_tab.volume_scale.set_value(actual_volume * 100)

    def on_song_finished(self, element):
        GLib.idle_add(self._reset_pipeline_state)

    def _reset_pipeline_state(self):
        if self.player is None:
            self.playing = False
            return
        try:
            current_state = self.player.get_state(Gst.CLOCK_TIME_NONE)[1]
            if current_state == Gst.State.PLAYING:
                return  # Already playing a new track
            self.player.set_state(Gst.State.READY)
        except GLib.Error:
            logging.getLogger(__name__).debug(
                "Failed to set player state to READY: %s", sys.exc_info()[1]
            )
        self.playing = False
        if getattr(self, "_is_radio_stream", False):
            GLib.idle_add(self._radio_auto_reconnect)
            return
        if self.auto_play_next:
            next_index = self.get_next_track_index()
            if next_index is not None:
                self.play_next_track()
            else:
                if hasattr(self.player_tab, "cover_art_widget"):
                    self.player_tab.cover_art_widget.clear()

    def on_bus_message(self, bus, message):
        if message.type == Gst.MessageType.ERROR:
            err, _debug = message.parse_error()
            error_msg = str(err.message).lower()
            if "internal" in error_msg and "stream" in error_msg:
                GLib.idle_add(self.set_status_message, "Internal stream error - recovering...")
                current_uri = self.player.get_property("uri")
                GLib.idle_add(self.player.set_state, Gst.State.NULL)
                GLib.timeout_add(100, self._attempt_recovery, current_uri)
            if getattr(self, "_is_radio_stream", False):
                GLib.idle_add(self._radio_auto_reconnect)
            elif self.playlist and 0 <= self.current_track < len(self.playlist) - 1:
                GLib.idle_add(self.play_track, self.current_track + 1)
            else:
                GLib.idle_add(self.set_status_message, f"Error: {err.message}")
        elif message.type == Gst.MessageType.EOS:
            if getattr(self, "_is_radio_stream", False):
                GLib.idle_add(self._radio_auto_reconnect)
            else:
                GLib.idle_add(self._handle_eos)
        elif message.type == Gst.MessageType.TAG:
            if getattr(self, "_is_radio_stream", False):
                tag_list = message.parse_tag()
                if tag_list:
                    for key in ["title", "StreamTitle", "icy-name", "icy-url"]:
                        success, value = tag_list.get_string(key)
                        if success and value:
                            GLib.idle_add(self._on_radio_tag, key, value)
                            break
        return Gst.BusSyncReply.PASS

    def _handle_eos(self):
        if getattr(self, "_is_radio_stream", False):
            GLib.idle_add(self._radio_auto_reconnect)
            return
        if getattr(self, "_current_podcast_episode", None) is not None:
            self._mark_podcast_episode_played()
            self.on_stop(None)
            return
        self.player.set_state(Gst.State.READY)
        self.playing = False
        if self.auto_play_next:
            if self.repeat_mode == "one":
                self.play_track(self.current_track)
            elif self.repeat_mode == "all":
                self.play_next_track()
            else:
                next_index = self.get_next_track_index()
                if next_index is not None:
                    self.play_track(next_index)
                else:
                    self.on_stop(None)
        else:
            self.on_stop(None)

    def _on_state_changed(self, bus, message):
        if message.src == self.player:
            _old_state, new_state, _pending_state = message.parse_state_changed()
            if new_state == Gst.State.PLAYING:
                self.playing = True
            elif new_state in [Gst.State.PAUSED, Gst.State.NULL]:
                self.playing = False
        return Gst.BusSyncReply.PASS

    def _attempt_recovery(self, uri):
        if not uri:
            GLib.idle_add(self.set_status_message, "Recovery failed - trying next track")
            return False
        self.player.set_property("uri", uri)
        GLib.idle_add(self.player.set_state, Gst.State.PLAYING)
        GLib.timeout_add(500, self._check_recovery_success)
        return False

    def _check_recovery_success(self):
        if not hasattr(self, "player") or not self.player:
            GLib.idle_add(self.set_status_message, "No player available")
            return False
        success, current_state, _pending = self.player.get_state(Gst.CLOCK_TIME_NONE)
        if not success:
            return False
        if current_state == Gst.State.PLAYING:
            self.playing = True
            return False
        # Still not playing — try again
        GLib.timeout_add(500, self._check_recovery_success)
        return False

    def _radio_auto_reconnect(self):
        if not getattr(self, "_is_radio_stream", False):
            return False
        uri = getattr(self, "_radio_stream_uri", None)
        if not uri:
            return False
        current_uri = self.player.get_property("uri") if self.player else None
        if current_uri == uri:
            return False
        GLib.idle_add(self.set_status_message, "Reconnecting to stream...")

        def _reconnect():
            self.play_uri(uri, getattr(self, "_radio_stream_title", "Radio Stream"))
            return False

        GLib.timeout_add(2000, _reconnect)
        return False

    def _on_radio_tag(self, key, value):
        if hasattr(self, "player_tab") and hasattr(self.player_tab, "track_label"):
            display = value
            if key == "StreamTitle" and " - " in value:
                display = value
            elif key == "icy-name" and value:
                display = f"🎙 {value}"
            GLib.idle_add(self.player_tab.track_label.set_text, display)
            GLib.idle_add(self.set_title, f"LinAmp - {display}")
        if hasattr(self, "header_nowplaying") and self.header_nowplaying:
            GLib.idle_add(self.header_nowplaying.set_text, value)
        return False

    def start_beat_detection(self):
        if not self.beat_aware_enabled or not self.playing:
            return
        # Only start if visualizer tab is not handling beats
        if not (hasattr(self, "visualizer_tab") and self.visualizer_tab and hasattr(self.visualizer_tab, "_beat_smooth")):
            self.beat_detection_timer = GLib.timeout_add(BEAT_DETECTION_INTERVAL, self.detect_beat)
            self._active_timers.add(self.beat_detection_timer)

    def stop_beat_detection(self):
        self._safe_source_remove(self.beat_detection_timer)
        self.beat_detection_timer = 0

    def detect_beat(self):
        """Lightweight beat detection for non-visualizer UI effects."""
        if not self.beat_aware_enabled or not self.playing:
            return False
        # Visualizer tab handles beat detection - this is a fallback
        if hasattr(self, "visualizer_tab") and self.visualizer_tab and hasattr(self.visualizer_tab, "_beat_smooth"):
            return True
        try:
            current_time = time.time()
            if current_time - self.last_beat_time < 0.1:
                return False
            audio_levels = getattr(self, "audio_levels", None)
            if not audio_levels:
                return False
            bass = audio_levels.get("bass", 0.0) if isinstance(audio_levels, dict) else 0.0
            peak = audio_levels.get("peak", 0.0) if isinstance(audio_levels, dict) else 0.0
            if hasattr(self, "_last_avg_level"):
                level_increase = bass - self._last_avg_level
                if (level_increase > 0.15 and peak > 0.7) or peak > 0.95:
                    self.last_beat_time = current_time
                    self._last_avg_level = bass
                    if hasattr(self, "_last_beat_timestamp"):
                        beat_interval = current_time - self._last_beat_timestamp
                        if 0.2 <= beat_interval <= 2.0:
                            self.beat_interval_history.append(beat_interval)
                            if len(self.beat_interval_history) > BEAT_HISTORY_SIZE:
                                self.beat_interval_history.pop(0)
                    self._last_beat_timestamp = current_time
                    return True
                self._last_avg_level = bass
            return False
        except (TypeError, ValueError, ZeroDivisionError):
            return False

    def on_beat_detected(self, bpm):
        pass  # Hook for beat-reactive UI effects; currently no-op

    def _resume_last_played(self):
        if not self.settings.last_played_track:
            return
        track_index = None
        for i, item in enumerate(self.playlist):
            if item.path == self.settings.last_played_track:
                track_index = i
                break
        if track_index is not None and self.play_track(track_index):
            if self.settings.last_played_position > 2.0:
                GLib.timeout_add(1000, self._seek_to_position, self.settings.last_played_position)
        else:
            pass

    def _seek_to_position(self, position):
        if hasattr(self, "player") and self.player and self.playing:
            seek_pos = int(position * Gst.SECOND)
            self.player.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH, seek_pos)
        return False
