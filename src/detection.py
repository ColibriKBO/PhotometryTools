"""
Source detection using DAOStarFinder.
"""

import logging
import sys

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

log = logging.getLogger(__name__)


def detect_sources(data: np.ndarray, fwhm: float, threshold_sigma: float) -> np.ndarray:
    """Return an Nx2 array of (x, y) pixel positions detected in *data*.

    Parameters
    ----------
    data : ndarray
        2-D science image.
    fwhm : float
        Expected FWHM of point sources in pixels, passed to DAOStarFinder.
    threshold_sigma : float
        Detection threshold expressed as a multiple of the background RMS.

    Returns
    -------
    ndarray, shape (N, 2)
        ``(x, y)`` pixel centroids for all detected sources.
    """
    _, median, std = sigma_clipped_stats(data, sigma=3.0)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)
    sources = daofind(data - median)
    if sources is None or len(sources) == 0:
        log.error("No sources detected in reference image. Check detection parameters.")
        sys.exit(1)
    log.info(f"Detected {len(sources)} sources in reference image.")
    positions = np.column_stack([sources["xcentroid"], sources["ycentroid"]])
    return positions
