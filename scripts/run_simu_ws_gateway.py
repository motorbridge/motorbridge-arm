from __future__ import annotations

import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_PKG_ROOT = _ROOT / "rebot_sdk"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from rebot_sdk.web.simu_gateway import main

if __name__ == "__main__":
    raise SystemExit(main())
