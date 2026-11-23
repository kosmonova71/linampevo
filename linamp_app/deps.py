"""Optional third-party dependency probes and the GTK/GStreamer import surface.

Every other module imports the ``gi`` namespaces and the ``*_AVAILABLE``
flags from here, so the version requirements are declared exactly once.
This module must be imported before any GTK/GStreamer code runs.
"""

from __future__ import annotations

import locale
import logging
import sys

try:
    import gi

    gi.require_version("Gdk", "4.0")
    gi.require_version("Gtk", "4.0")
    gi.require_version("Gst", "1.0")
except ImportError as e:
    raise RuntimeError(
        "GTK4 or GStreamer not installed. Please install them to run this application."
    ) from e
# re-exported for the whole package
from gi.repository import (
    Gdk,
    GdkPixbuf,
    Gio,  # noqa: F401
    GLib,  # noqa: F401
    Gst,  # noqa: F401
    Gtk,
    Pango,  # noqa: F401
)

# Apply locale settings programmatically
try:
    locale.setlocale(locale.LC_ALL, "C")
except locale.Error:
    logging.getLogger(__name__).warning("Failed to set locale to 'C': %s", sys.exc_info()[1])

try:
    from evo_visualizer import EvoConfig, EvoRenderer

    EVO_AVAILABLE = True
except ImportError:
    EVO_AVAILABLE = False
    EvoRenderer = None
    EvoConfig = None
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    from metadata_fixer import MetadataFixer

    METADATA_FIXER_AVAILABLE = True
except ImportError:
    MetadataFixer = None
    METADATA_FIXER_AVAILABLE = False

try:
    from io import BytesIO

    import requests
    from PIL import Image, ImageEnhance

    REQUESTS_AVAILABLE = True
    PIL_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    PIL_AVAILABLE = False
    requests = None
    Image = None
    ImageEnhance = None
    BytesIO = None

try:
    import mutagen
    from mutagen import File
    from mutagen.flac import FLAC
    from mutagen.id3 import ID3
    from mutagen.mp3 import MP3
    from mutagen.mp4 import MP4
except ImportError:
    mutagen = None
    File = None
    MP3 = None
    ID3 = None
    FLAC = None
    MP4 = None


def get_default_screen_size() -> tuple[int, int] | None:
    """Return (width, height) of the default display's bounding box."""
    display = Gdk.Display.get_default()
    if display and hasattr(display, "get_geometry"):
        geometry = display.get_geometry()
        if geometry and geometry.width > 0 and geometry.height > 0:
            return geometry.width, geometry.height
    return None


def require_visualizer_runtime() -> None:
    """Exit early when numpy is required for a visualizer entry point."""
    if not NUMPY_AVAILABLE and any(
        name in sys.argv[0].lower() for name in ("visualizer", "projectm")
    ):
        sys.exit(1)


COVER_ART_DEPENDENCIES = {}

COVER_ART_DEPENDENCIES["requests"] = {
    "available": REQUESTS_AVAILABLE,
    "version": getattr(requests, "__version__", None) if requests else None,
}

COVER_ART_DEPENDENCIES["pillow"] = {
    "available": PIL_AVAILABLE,
    "version": getattr(Image, "__version__", None) if PIL_AVAILABLE else None,
}

if mutagen is not None:
    COVER_ART_DEPENDENCIES["mutagen"] = {
        "available": True,
        "version": getattr(mutagen, "__version__", getattr(mutagen, "version_string", "unknown")),
    }
else:
    COVER_ART_DEPENDENCIES["mutagen"] = {
        "available": False,
        "version": None,
        "error": "mutagen not installed",
    }

COVER_ART_DEPENDENCIES["gtk"] = {
    "available": True,
    "version": getattr(Gtk, "get_major_version", lambda: 4)(),
}

COVER_ART_DEPENDENCIES["gdkpixbuf"] = {
    "available": True,
    "version": getattr(GdkPixbuf, "get_major_version", lambda: 0)(),
}

COVER_DOWNLOAD_AVAILABLE = REQUESTS_AVAILABLE and PIL_AVAILABLE
COVER_ART_AVAILABLE = (
    COVER_DOWNLOAD_AVAILABLE
    and COVER_ART_DEPENDENCIES["gtk"]["available"]
    and COVER_ART_DEPENDENCIES["gdkpixbuf"]["available"]
)
