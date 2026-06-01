"""Tests for aperture photometry edge handling."""

import sys
from pathlib import Path

import numpy as np

# Ensure the project root is on the path so the module can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.photometry import measure_aperture_photometry


def test_in_frame_nan_pixels_do_not_invalidate_source():
    data = np.full((31, 31), 10.0)
    data[15, 16] = np.nan
    data[15, 15] = 1000.0
    positions = np.array([[15.0, 15.0]])

    result = measure_aperture_photometry(
        data,
        positions,
        aperture_radius=3.0,
        annulus_r_in=5.0,
        annulus_r_out=7.0,
        gain=1.0,
        bad_mask=None,
    )

    assert np.isfinite(result["flux"][0])
    assert np.isfinite(result["mag"][0])


def test_edge_mask_still_invalidates_source():
    data = np.full((31, 31), 10.0)
    data[3, 3] = 1000.0
    positions = np.array([[3.0, 3.0]])
    bad_mask = np.zeros_like(data, dtype=bool)
    bad_mask[:5, :] = True

    result = measure_aperture_photometry(
        data,
        positions,
        aperture_radius=3.0,
        annulus_r_in=5.0,
        annulus_r_out=7.0,
        gain=1.0,
        bad_mask=bad_mask,
    )

    assert np.isnan(result["flux"][0])
    assert np.isnan(result["mag"][0])