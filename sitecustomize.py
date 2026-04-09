from __future__ import annotations

import site
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _ensure_local_vendor_on_path() -> None:
    vendor_path = PROJECT_ROOT / ".vendor"
    vendor_str = str(vendor_path)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)


def _ensure_user_site_on_path() -> None:
    user_site_str = str(Path(site.getusersitepackages()))
    if user_site_str not in sys.path:
        sys.path.append(user_site_str)


_ensure_local_vendor_on_path()
_ensure_user_site_on_path()
