"""Central application-data path resolution."""

import os
from pathlib import Path


def application_data_root() -> Path:
    """Return Faceledger's XDG application-data root without creating it."""

    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg_data_home) if xdg_data_home else Path.home() / ".local" / "share"
    return base / "faceledger"
