#!/usr/bin/env python3

"""LinAmp -- XMMS-style GTK4 music player launcher."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the application package importable regardless of the
# directory from which this launcher is executed.
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

try:
    from linamp_app.deps import require_visualizer_runtime
except Exception as exc:
    print(
        f"LinAmp: failed to import dependency checker: {exc}",
        file=sys.stderr,
    )
    raise

try:
    require_visualizer_runtime()
except Exception as exc:
    print(
        f"LinAmp: visualizer runtime check failed: {exc}",
        file=sys.stderr,
    )
    raise

try:
    from linamp_app.app import main
except Exception as exc:
    print(
        f"LinAmp: failed to import application: {exc}",
        file=sys.stderr,
    )
    raise


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nLinAmp: interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception:
        # Deliberately preserve the complete traceback during development.
        print("LinAmp: fatal application error:", file=sys.stderr)
        raise
