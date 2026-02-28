"""
Photometry methods.

Currently implements aperture photometry with local sky subtraction via an
annulus.  Future work may add PSF-fitting routines in this module or as a
separate ``psf.py`` alongside this file.
"""

import logging

import numpy as np
from photutils.aperture import (
    ApertureStats,
    CircularAnnulus,
    CircularAperture,
    aperture_photometry,
)

log = logging.getLogger(__name__)


def measure_aperture_photometry(
    data: np.ndarray,
    positions: np.ndarray,
    aperture_radius: float,
    annulus_r_in: float,
    annulus_r_out: float,
    gain: float,
    bad_mask: np.ndarray | None = None,
) -> dict:
    """Perform aperture photometry with local sky subtraction via an annulus.

    Parameters
    ----------
    data : ndarray
        2-D science image.
    positions : ndarray, shape (N, 2)
        ``(x, y)`` centroid positions for each source.
    aperture_radius : float
        Radius of the circular source aperture in pixels.
    annulus_r_in, annulus_r_out : float
        Inner and outer radii of the sky annulus in pixels.
    gain : float
        Detector gain (e⁻/ADU), used for Poisson noise estimation.
    bad_mask : bool ndarray, optional
        True where pixels are invalid (e.g. outside dither overlap after
        alignment).  Any aperture or annulus overlapping a bad pixel is
        set to NaN in the output.

    Returns
    -------
    dict with keys: flux, flux_err, mag, mag_err, snr, sky_bkg
        Each value is a 1-D ndarray of length N.
    """
    apertures = CircularAperture(positions, r=aperture_radius)
    annuli = CircularAnnulus(positions, r_in=annulus_r_in, r_out=annulus_r_out)

    if bad_mask is not None:
        bad_ap = np.array(
            aperture_photometry(bad_mask.astype(np.float64), apertures)["aperture_sum"]
        )
        bad_ann = np.array(
            aperture_photometry(bad_mask.astype(np.float64), annuli)["aperture_sum"]
        )
        ap_invalid = bad_ap > 0
        ann_invalid = bad_ann > 0
    else:
        ap_invalid = np.zeros(len(positions), dtype=bool)
        ann_invalid = np.zeros(len(positions), dtype=bool)

    data_clean = np.where(bad_mask, 0.0, data) if bad_mask is not None else data

    sky_stats = ApertureStats(data_clean, annuli)
    sky_per_pixel = sky_stats.median
    sky_per_pixel = np.where(ann_invalid, np.nan, sky_per_pixel)

    phot_table = aperture_photometry(data_clean, apertures)
    raw_sum = np.array(phot_table["aperture_sum"])
    raw_sum = np.where(ap_invalid, np.nan, raw_sum)

    aperture_area = apertures.area
    flux = raw_sum - sky_per_pixel * aperture_area
    flux_err = np.sqrt(
        np.abs(flux) / gain + aperture_area * np.abs(sky_per_pixel) / gain
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        mag = np.where(flux > 0, -2.5 * np.log10(flux), np.nan)
        snr = np.where(flux_err > 0, flux / flux_err, np.nan)
        mag_err = np.where(snr > 0, 1.0857 / snr, np.nan)

    mag = np.where(flux <= 0, np.nan, mag)
    mag_err = np.where(flux <= 0, np.nan, mag_err)

    return {
        "flux": flux,
        "flux_err": flux_err,
        "mag": mag,
        "mag_err": mag_err,
        "snr": snr,
        "sky_bkg": sky_per_pixel,
    }
