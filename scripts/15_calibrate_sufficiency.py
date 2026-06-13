"""Compatibility entrypoint for sufficiency calibration."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT_FOR_SCRIPT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_SCRIPT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_SCRIPT))

from scripts.calibrate_sufficiency import main


if __name__ == "__main__":
    main()
