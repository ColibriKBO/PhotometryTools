"""
I/O utilities: config loading and FITS image reading.
"""

import json
import logging
from pathlib import Path

import numpy as np
from astropy.io import fits

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load a JSON config file and return it as a dict."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# FITS I/O
# ---------------------------------------------------------------------------

def read_fits(path: Path, extension: int) -> np.ndarray:
    """Read a single FITS extension and return the data as float64."""
    with fits.open(path) as hdul:
        data = hdul[extension].data.astype(np.float64)
    return data


def read_fits_datetime(path: Path, extension: int) -> str | None:
    """Try to read an observation timestamp from the FITS header.

    Attempts keywords in order: DATE-OBS, MJD-OBS (converted to ISO), JD
    (converted to ISO). Returns an ISO-format string or None if unavailable.
    """
    try:
        from astropy.time import Time
        with fits.open(path) as hdul:
            hdr = hdul[extension].header
        if "DATE-OBS" in hdr:
            return str(hdr["DATE-OBS"])
        if "MJD-OBS" in hdr:
            return Time(float(hdr["MJD-OBS"]), format="mjd").isot
        if "JD" in hdr:
            return Time(float(hdr["JD"]), format="jd").isot
    except Exception:
        pass
    return None
