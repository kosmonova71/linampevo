"""Milkdrop-style visualizer tab driven by live GStreamer audio."""

from __future__ import annotations

import math
import threading
import time

from ..deps import EVO_AVAILABLE, EvoConfig, EvoRenderer, GLib, Gst, Gtk, np

# Cache FFT windows to avoid repeated allocations
_hanning_window_cache: dict[int, np.ndarray] = {}


def _get_hanning_window(size: int) -> np.ndarray:
    """Get cached Hanning window for given FFT size."""
    if size not in _hanning_window_cache:
        _hanning_window_cache[size] = np.hanning(size).astype(np.float32)
    return _hanning_window_cache[size]


# Pre-computed FFT sizes (powers of 2)
_FFT_SIZES = [512, 1024, 2048, 4096, 8192, 16384]

# Analysis bands (Hz): sub, bass, low-mid, presence, high, air.
# The three shader bands are RMS-combined from these so the split is
# perceptual rather than three arbitrary wide slices.
_BAND_EDGES = (
    (20.0, 90.0),
    (90.0, 300.0),
    (300.0, 1500.0),
    (1500.0, 5000.0),
    (2000.0, 8000.0),
    (8000.0, 16000.0),
)


def _get_fft_size(n: int) -> int:
    """Get optimal FFT size for given sample count."""
    # Find smallest power of 2 >= n, clamped to max 16384
    fft_size = max(512, 1 << (n - 1).bit_length())
    return min(fft_size, 16384)


class EvoVisualizerTab(Gtk.Box):
    """GTK tab that hosts a Milkdrop-style visualizer driven by live GStreamer audio."""

    def __init__(self, player, available_presets=None):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)

        self.player = player
        self.available_presets = available_presets or []

        # Visualizer state
        self.visualizer = None
        self.visualizer_type = None
        self._rendering_enabled = True

        # Audio probe state
        self.audio_probe_id = None
        self._probe_sink = None
        self._probe_pad = None
        self._probe_connecting = False
        self._probe_retry_id = None
        self._probe_retry_count = 0

        # Latest audio buffer (protected by lock)
        self._audio_lock = threading.Lock()
        self._latest_audio = None
        self._latest_audio_seq = 0
        self._last_consumed_audio_seq = 0
        self._audio_channels = 2
        self._audio_sample_rate = 44100.0
        self._audio_dt = 0.0
        self._last_audio_time = time.time()

        # Analysis state (beat / smoothing)
        self._beat_bass_avg = 0.0
        self._beat_bass_history = []
        self._last_beat_time = 0.0
        self._bass_smooth = 0.0
        self._mid_smooth = 0.0
        self._treb_smooth = 0.0
        self._beat_smooth = 0.0
        self._rms_smooth = 0.0
        self._peak_smooth = 0.0

        # Beat state
        self._beat = 0.0
        self._beat_phase = 0.0  # 0.0-1.0 for visualizer beat animation
        self._beat_min_peak = 0.15  # Minimum energy threshold for beat

        # Cached FFT state
        self._cached_fft_size = 0
        self._cached_window = None
        self._cached_freqs = None
        self._band_mask_cache = {}

        # Spectral-flux beat tracking + adaptive per-band ceilings so
        # quiet material still produces visible motion
        self._flux_prev = None
        self._flux_avg = 0.0
        self._band_peak = {"bass": 0.02, "mid": 0.02, "treb": 0.02}

        self._audio_consume_timer_id = None
        self._last_vis_update = 0.0

        # Widget setup
        self.set_visible(True)
        self.set_hexpand(True)
        self.set_vexpand(True)
        self.set_hexpand_set(True)
        self.set_vexpand_set(True)
        self.set_halign(Gtk.Align.FILL)
        self.set_valign(Gtk.Align.FILL)
        self.add_css_class("visualizer-widget")

        use_evo = EVO_AVAILABLE and EvoRenderer is not None
        if use_evo:
            self.visualizer = EvoRenderer(EvoConfig())
            self.visualizer.set_hexpand(True)
            self.visualizer.set_vexpand(True)
            self.visualizer.set_hexpand_set(True)
            self.visualizer.set_vexpand_set(True)
            self.visualizer.set_halign(Gtk.Align.FILL)
            self.visualizer.set_valign(Gtk.Align.FILL)
            self.visualizer.set_visible(True)
            self.visualizer.add_css_class("visualizer-glarea")
            # Wrap the GLArea in an overlay so the Cairo mutation-heatmap
            # DrawingArea has somewhere to draw (a Gtk.GLArea cannot host
            # Cairo children and its render signal passes a GdkGLContext).
            self.append(self.visualizer.build_heatmap_overlay())
            self.visualizer_type = "evo"

            self.visualizer.set_available_milk_files(sorted(self.available_presets))
            self.visualizer._disable_auto_switching = not self.player.settings.milkdrop_auto_switch

            self.connect("map", self._on_map)
            self.update_audio = self.visualizer.update_audio
        else:
            placeholder = Gtk.Label(label="Visualizer not available")
            placeholder.set_hexpand(True)
            placeholder.set_vexpand(True)
            placeholder.set_halign(Gtk.Align.CENTER)
            placeholder.set_valign(Gtk.Align.CENTER)
            placeholder.add_css_class("placeholder-label")
            self.append(placeholder)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_map(self, widget):
        """Trigger a render once the widget is mapped and has a real size."""
        GLib.timeout_add(50, self._delayed_render_trigger)

    def _delayed_render_trigger(self):
        if self.visualizer is None:
            return False
        w = self.visualizer.get_width()
        h = self.visualizer.get_height()
        if w > 10 and h > 10:
            self.visualizer.queue_render()
        return False

    def cleanup(self):
        """Stop all audio probing and rendering. Safe to call multiple times."""
        self._rendering_enabled = False

        if self._audio_consume_timer_id is not None:
            GLib.source_remove(self._audio_consume_timer_id)
            self._audio_consume_timer_id = None

        if self._probe_retry_id is not None:
            GLib.source_remove(self._probe_retry_id)
            self._probe_retry_id = None

        if self.audio_probe_id is not None and self._probe_pad is not None:
            self._probe_pad.remove_probe(self.audio_probe_id)

        self.audio_probe_id = None
        self._probe_sink = None
        self._probe_pad = None
        self._probe_connecting = False
        self._probe_retry_count = 0

    # ------------------------------------------------------------------
    # Probe attachment
    # ------------------------------------------------------------------

    def _get_audio_sink_pad(self):
        """Return (pad, sink) preferring a post-conversion pad named 'vis_caps'."""
        player_obj = getattr(self.player, "player", self.player)
        if not hasattr(player_obj, "get_property"):
            return None, None

        audio_sink = player_obj.get_property("audio-sink")
        if not audio_sink:
            return None, None

        # Prefer the pad after format conversion (F32LE stereo)
        vis_caps = audio_sink.get_by_name("vis_caps")
        if vis_caps is not None:
            src = vis_caps.get_static_pad("src")
            if src:
                return src, audio_sink

        # Fallbacks
        sink_pad = audio_sink.get_static_pad("sink")
        if sink_pad:
            return sink_pad, audio_sink

        pads = audio_sink.get_pads()
        if pads:
            return pads[0], audio_sink

        return None, None

    def _reconnect_audio_probe(self):
        """Attach or re-attach the probe when the sink/pad changes."""
        if self._probe_connecting:
            return

        sink_pad, audio_sink = self._get_audio_sink_pad()
        if sink_pad is None or audio_sink is None:
            if self._probe_retry_id is None:
                self._probe_retry_count = 0
                self._probe_retry_id = GLib.timeout_add(250, self._retry_audio_probe)
            return

        need_reconnect = (
            self.audio_probe_id is None
            or self._probe_sink is not audio_sink
            or self._probe_pad is not sink_pad
        )
        if not need_reconnect:
            return

        self._probe_connecting = True
        try:
            if self.audio_probe_id is not None and self._probe_pad is not None:
                self._probe_pad.remove_probe(self.audio_probe_id)

            self.audio_probe_id = sink_pad.add_probe(
                Gst.PadProbeType.BUFFER,
                self._milkdrop_audio_probe_callback,
                None,
            )
            self._probe_sink = audio_sink
            self._probe_pad = sink_pad
            self._probe_retry_id = None
            self._probe_retry_count = 0

            self._audio_sample_rate = float(getattr(self.player, "_audio_sample_rate", 44100.0))
        finally:
            self._probe_connecting = False

    def _retry_audio_probe(self):
        """Bounded retry until the audio sink becomes available."""
        if self.audio_probe_id is not None:
            self._probe_retry_id = None
            return False

        self._probe_retry_count += 1
        if self._probe_retry_count > 40:
            self._probe_retry_id = None
            return False

        self._reconnect_audio_probe()
        if self.audio_probe_id is not None:
            self._probe_retry_id = None
            return False
        return True

    # ------------------------------------------------------------------
    # Probe callback → analysis → visualizer
    # ------------------------------------------------------------------

    def _milkdrop_audio_probe_callback(self, pad, info, user_data):
        """GStreamer pad probe: capture buffer and schedule analysis on the main loop."""
        if not self._rendering_enabled:
            return Gst.PadProbeReturn.OK

        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.PadProbeReturn.OK

        try:
            audio, rate, channels = self._buffer_to_float32(pad, map_info)
            if audio is None or len(audio) == 0:
                return Gst.PadProbeReturn.OK

            # Convert to float32 array without copy where possible
            audio_array = np.asarray(audio, dtype=np.float32)
            audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=1.0, neginf=-1.0)
            audio_array = np.clip(audio_array, -1.0, 1.0)

            # Quick atomic update without full lock for sequence
            with self._audio_lock:
                self._latest_audio = audio_array
                self._audio_channels = channels
                self._audio_sample_rate = float(rate)
                self._latest_audio_seq = (self._latest_audio_seq + 1) & 0xFFFFFFFF

            now = time.time()
            self._audio_dt = now - self._last_audio_time
            self._last_audio_time = now

            if self.visualizer is not None:
                GLib.idle_add(self._process_and_push_audio)
        finally:
            buffer.unmap(map_info)

        return Gst.PadProbeReturn.OK

    def _process_and_push_audio(self):
        """Consume the latest audio buffer, compute levels, push to visualizer."""
        if not self._rendering_enabled or self.visualizer is None:
            return False

        # Quick check without lock first
        if self._latest_audio_seq == self._last_consumed_audio_seq:
            return False

        with self._audio_lock:
            seq = self._latest_audio_seq
            if seq == self._last_consumed_audio_seq:
                return False
            # Get reference without copy - we'll copy in compute if needed
            audio = self._latest_audio
            if audio is None:
                return False
            rate = self._audio_sample_rate
            channels = self._audio_channels or 1
            dt = self._audio_dt
            self._last_consumed_audio_seq = seq

        # Process without lock held
        levels = self._compute_audio_levels(audio, rate, channels, dt)
        self.visualizer.update_audio(
            levels, fft_samples=levels.get("spectrum"), sample_rate=rate
        )
        return False

    def _compute_audio_levels(self, audio, rate, channels, dt):
        """FFT-based band energy + spectral-flux beat detection + adaptive smoothing."""
        x = np.asarray(audio, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if x.size < 2:
            return self._empty_levels(dt)

        # Downmix to mono in-place where possible
        if channels > 1:
            frames = x.size // channels
            if frames < 1:
                return self._empty_levels(dt)
            # Avoid copy by using reshape view
            xx = x[:frames * channels].reshape(frames, channels)
            # Mean along axis=1 (channels) - creates new array but minimal
            xx = xx.mean(axis=1)
        else:
            xx = x

        nn = xx.size
        if nn < 2:
            return self._empty_levels(dt)

        # Windowed FFT - use cached window and freqs
        fft_size = _get_fft_size(nn)

        # Update cache if FFT size changed
        if fft_size != self._cached_fft_size:
            self._cached_fft_size = fft_size
            self._cached_window = _get_hanning_window(fft_size)
            self._cached_freqs = np.fft.rfftfreq(fft_size, d=1.0 / rate)

        window = self._cached_window
        freqs = self._cached_freqs

        if nn >= fft_size:
            # Use last fft_size samples, multiply by cached window (no copy needed)
            windowed = xx[-fft_size:] * window
        else:
            # Efficient: only reallocate window if size changed
            if self._cached_window is not None and len(self._cached_window) != nn:
                windowed = np.zeros(fft_size, dtype=np.float32)
            else:
                windowed = np.zeros_like(xx, dtype=np.float32)
                windowed.fill(0.0)
            # Apply window to available samples
            windowed[:nn] = xx * self._cached_window[:nn]

        spec = np.abs(np.fft.rfft(windowed)).astype(np.float32)
        # Normalise by the loudest bin so band *shape* is scale-invariant;
        # absolute loudness re-enters through the rms/peak gate below.
        spec_n = spec / (float(spec.max()) + 1e-9) if spec.size else spec

        mask_key = (fft_size, int(rate))
        masks = self._band_mask_cache.get(mask_key)
        if masks is None:
            nyquist = rate * 0.5
            built = []
            for lo, hi in _BAND_EDGES:
                hi_eff = min(hi, nyquist)
                built.append((freqs >= lo) & (freqs < hi_eff) if hi_eff > lo else None)
            masks = built
            self._band_mask_cache[mask_key] = masks

        def band_strength(idx):
            mask = masks[idx]
            if mask is None or not np.any(mask):
                return 0.0
            return float(np.sqrt(np.mean(spec_n[mask] ** 2) + 1e-12))

        sub_s = band_strength(0)
        bass_s = band_strength(1)
        lowmid_s = band_strength(2)
        presence_s = band_strength(3)
        high_s = band_strength(4)
        air_s = band_strength(5)

        bass_shape = math.sqrt(0.55 * bass_s * bass_s + 0.45 * sub_s * sub_s)
        mid_shape = math.sqrt(0.55 * lowmid_s * lowmid_s + 0.45 * presence_s * presence_s)
        treb_shape = math.sqrt(0.65 * high_s * high_s + 0.35 * air_s * air_s)

        rms = float(np.sqrt(np.mean(xx**2)))
        peak = float(np.max(np.abs(xx)))

        # Loudness gate: keeps absolute volume in the signal while the
        # normalised spectrum keeps the band split scale-invariant.
        gate = float(np.clip(rms * 6.0 + peak * 1.5, 0.0, 1.4))

        dt_c = min(0.25, max(0.008, float(dt) if dt and dt > 0.0 else 0.033))

        def band_level(shape, key):
            raw = shape * gate * 1.7
            peak_b = self._band_peak[key]
            if raw > peak_b:
                peak_b = raw
            else:
                peak_b *= math.exp(-dt_c / 6.0)
            peak_b = max(peak_b, 1e-4)
            self._band_peak[key] = peak_b
            rel = raw / (peak_b * 0.6 + 1e-6)
            ad_lvl = float(np.clip((rel ** 0.75) * 1.1, 0.0, 1.8))
            abs_lvl = float(np.clip(raw, 0.0, 1.8))
            return 0.6 * abs_lvl + 0.4 * ad_lvl

        bass_lvl = band_level(bass_shape, "bass")
        mid_lvl = band_level(mid_shape, "mid")
        treb_lvl = band_level(treb_shape, "treb")

        # Spectral flux: positive change in the normalised log-spectrum.
        # Catches snare/hat/vocal onsets that a bass-energy detector misses.
        logspec = np.log1p(spec_n * 6.0)
        lmax = float(logspec.max()) if logspec.size else 0.0
        flux_sig = logspec / (lmax + 1e-9) if lmax > 1e-9 else logspec * 0.0
        n_flux = 96
        if flux_sig.size > n_flux:
            step = flux_sig.size / n_flux
            flux_sig = flux_sig[(np.arange(n_flux) * step).astype(np.int64)]
        if self._flux_prev is not None and flux_sig.shape == self._flux_prev.shape:
            flux = float(np.sum(np.maximum(flux_sig - self._flux_prev, 0.0)))
        else:
            flux = 0.0
        self._flux_prev = flux_sig
        self._flux_avg += (flux - self._flux_avg) * 0.06

        # Downsampled normalised spectrum for downstream DNA analysis.
        n_spec = 256
        if spec_n.size > n_spec:
            step = spec_n.size / n_spec
            spectrum = spec_n[(np.arange(n_spec) * step).astype(np.int64)].astype(np.float32)
        else:
            spectrum = spec_n.astype(np.float32)

        # Beat detector: bass-level rise over its own recent average, or a
        # spectral-flux onset, with a refractory period.
        self._beat_bass_avg = self._beat_bass_avg * 0.9 + bass_lvl * 0.1
        self._beat_bass_history.append(bass_lvl)
        if len(self._beat_bass_history) > 50:
            self._beat_bass_history.pop(0)

        avg_bass = float(np.mean(self._beat_bass_history)) if self._beat_bass_history else 0.0
        now = time.time()
        refractory_ok = now - self._last_beat_time > 0.18
        energy_beat = (
            bass_lvl > avg_bass * 1.3
            and bass_lvl > self._beat_min_peak
            and refractory_ok
        )
        flux_beat = (
            flux > self._flux_avg * 1.8 + 0.10
            and flux > 0.4
            and gate > 0.04
            and refractory_ok
        )
        beat = 1.0 if (energy_beat or flux_beat) else 0.0
        if beat > 0.5:
            self._last_beat_time = now

        # Beat phase for smooth visualizer animation (0.0-1.0), decaying
        # in real time rather than per frame.
        if beat > 0.5:
            self._beat_phase = 1.0
        else:
            self._beat_phase *= math.exp(-dt_c / 0.32)

        # dt-based attack/release envelopes: fast snap up, ~0.3 s release.
        def env(prev, target, tau_att, tau_rel):
            tau = tau_att if target >= prev else tau_rel
            a = 1.0 - math.exp(-dt_c / tau)
            return prev + (target - prev) * a

        bass_s = env(self._bass_smooth, bass_lvl, 0.05, 0.30)
        mid_s = env(self._mid_smooth, mid_lvl, 0.05, 0.30)
        treb_s = env(self._treb_smooth, treb_lvl, 0.05, 0.30)
        beat_s = env(self._beat_smooth, beat, 0.03, 0.26)
        rms_s = env(self._rms_smooth, rms, 0.06, 0.22)
        peak_s = env(self._peak_smooth, peak, 0.06, 0.22)

        self._bass_smooth = bass_s
        self._mid_smooth = mid_s
        self._treb_smooth = treb_s
        self._beat_smooth = beat_s
        self._rms_smooth = rms_s
        self._peak_smooth = peak_s

        return {
            "bass": bass_s,
            "bass_att": bass_lvl,
            "mid": mid_s,
            "mid_att": mid_lvl,
            "treb": treb_s,
            "treb_att": treb_lvl,
            "beat": beat_s,
            "beat_phase": self._beat_phase,
            "rms": rms_s,
            "peak": peak_s,
            "dt": dt,
            "spectrum": spectrum,
        }

    @staticmethod
    def _empty_levels(dt):
        return {
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
            "dt": dt,
            "spectrum": None,
        }

    # ------------------------------------------------------------------
    # Buffer → float32 conversion
    # ------------------------------------------------------------------

    def _buffer_to_float32(self, pad, map_info):
        """Convert a mapped GStreamer buffer to normalized float32 samples.

        Supported formats: F32LE, F64LE, S16LE, S24LE, S24_32LE, S32LE.
        Returns (audio, rate, channels) or (None, rate, channels) on failure.
        """
        rate = float(getattr(self, "_audio_sample_rate", 44100.0))
        channels = max(1, int(getattr(self, "_audio_channels", 2) or 2))
        fmt = None

        if pad is not None:
            caps = pad.get_current_caps()
            if caps is not None and caps.get_size() > 0:
                structure = caps.get_structure(0)
                if structure is not None:
                    fmt = structure.get_string("format")
                    ok, ch = structure.get_int("channels")
                    if ok and ch > 0:
                        channels = ch
                    ok, r = structure.get_int("rate")
                    if ok and r > 0:
                        rate = float(r)

        if fmt is None:
            fmt = "F32LE"

        buf = memoryview(map_info.data)

        try:
            if fmt == "F32LE":
                usable = (len(buf) // 4) * 4
                audio = np.frombuffer(buf[:usable], dtype=np.float32)

            elif fmt == "F64LE":
                usable = (len(buf) // 8) * 8
                # Use view + astype with copy=False when possible
                audio = np.frombuffer(buf[:usable], dtype=np.float64).astype(
                    np.float32, copy=False
                )

            elif fmt == "S16LE":
                usable = (len(buf) // 2) * 2
                # Direct view, then normalize - minimal copy
                audio = np.frombuffer(buf[:usable], dtype=np.int16).astype(
                    np.float32, copy=False
                ) / 32768.0

            elif fmt == "S32LE":
                usable = (len(buf) // 4) * 4
                audio = np.frombuffer(buf[:usable], dtype=np.int32).astype(
                    np.float32, copy=False
                ) / 2147483648.0

            elif fmt == "S24_32LE":
                usable = (len(buf) // 4) * 4
                # Single pass: read int32, shift, normalize
                ints = np.frombuffer(buf[:usable], dtype=np.int32)
                audio = (ints >> 8).astype(np.float32, copy=False) / 8388608.0

            elif fmt == "S24LE":
                sample_count = len(buf) // 3
                if sample_count == 0:
                    return None, rate, channels
                # Reshape to 3-byte groups, then pack into int32
                raw = np.frombuffer(buf[: sample_count * 3], dtype=np.uint8).reshape(
                    sample_count, 3
                )
                # Pack bytes into int32 (little-endian): byte0 + byte1<<8 + byte2<<16
                ints = (
                    raw[:, 0].astype(np.int32)
                    | (raw[:, 1].astype(np.int32) << 8)
                    | (raw[:, 2].astype(np.int32) << 16)
                )
                # Sign-extend 24-bit
                ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
                audio = ints.astype(np.float32, copy=False) / 8388608.0

            else:
                return None, rate, channels

        except (ValueError, BufferError):
            return None, rate, channels

        # Truncate to whole frames - avoid copy if already aligned
        if channels > 1:
            remainder = audio.size % channels
            if remainder:
                audio = audio[:-remainder]

        return audio, rate, channels
