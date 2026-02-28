"""
Tests for the integer-pixel alignment and aperture re-centroiding pipeline.

Covers:
  - align_image: integer-only shift, no sub-pixel interpolation, NaN border fill
  - recentroid_positions: correct re-centroiding after a known integer shift,
    graceful handling of sources shifted outside the image extent
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# Ensure the project root is on the path so the module can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from aperture_photometry import align_image, recentroid_positions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gaussian_source(cx: float, cy: float, amplitude: float, fwhm: float,
                     shape: tuple[int, int]) -> np.ndarray:
    """Return a 2-D Gaussian PSF centred at (cx, cy) on a zero background."""
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    return amplitude * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))


def make_star_field(sources: list[tuple], shape: tuple[int, int] = (128, 128),
                    fwhm: float = 3.0, seed: int = 0) -> np.ndarray:
    """
    Build a synthetic star-field image.

    Parameters
    ----------
    sources : list of (cx, cy, amplitude) tuples
    shape   : (rows, cols)
    fwhm    : PSF FWHM in pixels
    seed    : RandomState seed for background noise
    """
    rng = np.random.default_rng(seed)
    sky = 10.0
    image = rng.normal(sky, 1.0, shape)
    for cx, cy, amp in sources:
        image += gaussian_source(cx, cy, amp, fwhm, shape)
    return image


# ---------------------------------------------------------------------------
# Tests: integer-shift pixel-value preservation (no interpolation)
# ---------------------------------------------------------------------------

class TestIntegerShiftNoInterpolation:
    """
    Verify that scipy.ndimage.shift at order=0 with an integer offset amounts
    to pure pixel relocation — no blending whatsoever.
    """

    def _apply_integer_shift(self, source, dx, dy):
        """Direct wrapper so tests don't depend on astroalign."""
        from scipy.ndimage import shift as ndi_shift
        shifted = ndi_shift(
            source.astype(np.float64),
            shift=(dy, dx),
            order=0,
            mode="constant",
            cval=np.nan,
        )
        return shifted

    def test_pixel_values_preserved_exact(self):
        """
        Every non-border pixel in the shifted image must equal the corresponding
        pixel in the source — no fractional blending is permitted.
        """
        rng = np.random.default_rng(42)
        source = rng.uniform(0, 1000, (64, 64))
        dx, dy = 7, -3

        shifted = self._apply_integer_shift(source, dx, dy)

        # Interior region (away from the NaN-filled border)
        # shifted[y, x] == source[y - dy, x - dx]  for valid interior coords
        rows = np.arange(max(0, dy), min(64, 64 + dy))
        cols = np.arange(max(0, dx), min(64, 64 + dx))

        for row in rows:
            for col in cols:
                assert shifted[row, col] == source[row - dy, col - dx], (
                    f"Pixel mismatch at ({row},{col}): "
                    f"shifted={shifted[row,col]:.4f}, "
                    f"expected={source[row-dy, col-dx]:.4f}"
                )

    def test_border_filled_with_nan(self):
        """Pixels in the border band introduced by a positive shift must be NaN."""
        source = np.ones((64, 64))
        dx, dy = 5, 3
        shifted = self._apply_integer_shift(source, dx, dy)

        # Top band (rows 0..dy-1) should be NaN
        assert np.all(np.isnan(shifted[:dy, :])), "Top border rows should be NaN"
        # Left band (cols 0..dx-1) should be NaN
        assert np.all(np.isnan(shifted[:, :dx])), "Left border columns should be NaN"

    def test_zero_shift_is_identity(self):
        """A zero shift must return an array equal to the input."""
        source = np.arange(100, dtype=float).reshape(10, 10)
        shifted = self._apply_integer_shift(source, 0, 0)
        np.testing.assert_array_equal(shifted, source)

    def test_pixel_values_are_integers_for_integer_input(self):
        """
        When the source contains integer-valued pixels, the output interior
        must also be integer-valued (no fractional blending artefacts).
        """
        source = np.arange(64 * 64, dtype=float).reshape(64, 64)
        shifted = self._apply_integer_shift(source, 4, 4)
        interior = shifted[4:, 4:]
        # All finite values should round-trip exactly
        assert np.all(interior == np.floor(interior)), \
            "Non-integer values detected in shifted output — interpolation occurred"


# ---------------------------------------------------------------------------
# Tests: align_image return contract
# ---------------------------------------------------------------------------

class TestAlignImage:
    """Tests for align_image using a mocked astroalign transform."""

    def _make_mock_transform(self, tx, ty):
        """Return a mock SimilarityTransform-like object with given translation."""
        t = MagicMock()
        t.translation = (tx, ty)
        return t

    def test_returns_three_tuple_on_success(self):
        """align_image must return (shifted, bad_mask, (dx, dy)) on success."""
        source = make_star_field([(40, 40, 500), (80, 30, 300), (20, 70, 400)])
        target = make_star_field([(43, 43, 500), (83, 33, 300), (23, 73, 400)])

        mock_transform = self._make_mock_transform(3.1, 3.0)

        with patch("astroalign.find_transform", return_value=(mock_transform, None)):
            result = align_image(source, target)

        assert len(result) == 3, "align_image must return a 3-tuple"
        shifted, bad_mask, shift_xy = result
        assert shifted is not None
        assert bad_mask is not None
        assert len(shift_xy) == 2

    def test_integer_shift_rounded_correctly(self):
        """Translation is rounded to the nearest integer, not truncated."""
        source = np.ones((64, 64))
        target = np.ones((64, 64))

        # Fractional translation: 2.7 should round to 3, -1.4 to -1
        mock_transform = self._make_mock_transform(2.7, -1.4)

        with patch("astroalign.find_transform", return_value=(mock_transform, None)):
            _, _, (dx, dy) = align_image(source, target)

        assert dx == 3, f"Expected dx=3, got {dx}"
        assert dy == -1, f"Expected dy=-1, got {dy}"

    def test_returns_none_triple_on_failure(self):
        """When astroalign raises an exception, align_image must return (None, None, None)."""
        source = np.ones((64, 64))
        target = np.ones((64, 64))

        with patch("astroalign.find_transform", side_effect=Exception("no stars")):
            result = align_image(source, target)

        assert result == (None, None, None)

    def test_shifted_image_has_same_shape(self):
        """Output shifted image must have the same shape as the source."""
        source = np.random.default_rng(0).uniform(0, 100, (64, 80))
        mock_transform = self._make_mock_transform(5.0, -2.0)

        with patch("astroalign.find_transform", return_value=(mock_transform, None)):
            shifted, _, _ = align_image(source, source)

        assert shifted.shape == source.shape

    def test_bad_mask_matches_nan_pixels(self):
        """bad_mask must be True exactly where shifted contains NaN."""
        source = np.ones((64, 64))
        mock_transform = self._make_mock_transform(5.0, 3.0)

        with patch("astroalign.find_transform", return_value=(mock_transform, None)):
            shifted, bad_mask, _ = align_image(source, source)

        np.testing.assert_array_equal(bad_mask, ~np.isfinite(shifted))

    def test_no_interpolated_pixel_values(self):
        """
        For an integer shift, every finite output pixel must exactly equal
        a pixel from the input — no interpolated fractional values allowed.
        """
        rng = np.random.default_rng(99)
        source = rng.uniform(0, 1000, (64, 64))
        source_set = set(source.ravel().tolist())

        mock_transform = self._make_mock_transform(6.0, -4.0)
        with patch("astroalign.find_transform", return_value=(mock_transform, None)):
            shifted, bad_mask, _ = align_image(source, source)

        # Every finite pixel in shifted must be from source's original pixel set
        finite_pixels = shifted[~bad_mask].tolist()
        for pix in finite_pixels:
            assert pix in source_set, (
                f"Pixel value {pix} not found in source — sub-pixel interpolation occurred"
            )


# ---------------------------------------------------------------------------
# Tests: recentroid_positions
# ---------------------------------------------------------------------------

class TestRecentroidPositions:
    """Tests for recentroid_positions."""

    def test_recovers_correct_position_after_integer_shift(self):
        """
        Given a star at a known location, an integer-shifted copy of the image,
        and the correct shift_xy, recentroid must recover a position close to
        the expected centroid in the shifted frame.
        """
        shape = (128, 128)
        cx, cy = 60.0, 55.0          # true star position in reference frame
        fwhm = 4.0
        dx, dy = 7, -5              # integer shift applied

        # Reference image: single star
        ref = gaussian_source(cx, cy, 1000.0, fwhm, shape)

        # Shifted image: star moves to (cx+dx, cy+dy)
        from scipy.ndimage import shift as ndi_shift
        shifted = ndi_shift(ref, shift=(dy, dx), order=0, mode="constant", cval=np.nan)

        ref_positions = np.array([[cx, cy]])
        new_pos = recentroid_positions(shifted, ref_positions, (dx, dy),
                                       search_box_radius=8)

        expected_x = cx + dx
        expected_y = cy + dy
        assert new_pos.shape == (1, 2)
        assert np.all(np.isfinite(new_pos[0])), "Re-centroided position should be finite"
        assert abs(new_pos[0, 0] - expected_x) < 1.0, \
            f"x centroid off by {abs(new_pos[0, 0] - expected_x):.2f} px"
        assert abs(new_pos[0, 1] - expected_y) < 1.0, \
            f"y centroid off by {abs(new_pos[0, 1] - expected_y):.2f} px"

    def test_source_outside_image_returns_nan(self):
        """A source shifted beyond the image boundary must return NaN."""
        shape = (64, 64)
        ref_positions = np.array([[5.0, 5.0]])   # near edge
        dx, dy = -20, 0                           # shift left by 20 → off the image

        data = np.ones(shape)
        new_pos = recentroid_positions(data, ref_positions, (dx, dy),
                                       search_box_radius=5)

        assert np.all(np.isnan(new_pos[0])), "Out-of-frame source should be NaN"

    def test_output_shape_matches_input(self):
        """Output array must have the same shape as ref_positions."""
        shape = (128, 128)
        n_sources = 10
        rng = np.random.default_rng(7)
        ref_positions = rng.uniform(20, 108, (n_sources, 2))
        data = make_star_field(
            [(x, y, 500.0) for x, y in ref_positions], shape=shape
        )
        new_pos = recentroid_positions(data, ref_positions, (0, 0), search_box_radius=5)
        assert new_pos.shape == ref_positions.shape

    def test_zero_shift_returns_positions_near_original(self):
        """With shift_xy=(0,0) and a well-isolated star, centroid ≈ original."""
        shape = (128, 128)
        cx, cy = 64.0, 64.0
        fwhm = 4.0
        data = gaussian_source(cx, cy, 1000.0, fwhm, shape)

        ref_positions = np.array([[cx, cy]])
        new_pos = recentroid_positions(data, ref_positions, (0, 0), search_box_radius=8)

        assert np.all(np.isfinite(new_pos[0])), "Centroid should be finite"
        assert abs(new_pos[0, 0] - cx) < 0.5, "x centroid should be close to input"
        assert abs(new_pos[0, 1] - cy) < 0.5, "y centroid should be close to input"

    def test_multiple_sources_some_outside(self):
        """Only in-frame sources should have finite positions; others should be NaN."""
        shape = (64, 64)
        # Two sources: one will shift inside, one outside
        ref_positions = np.array([
            [32.0, 32.0],   # well inside, stays inside with shift (+5, 0)
            [60.0, 32.0],   # after shift (+10, 0): 70 > 64 → NaN
        ])
        dx, dy = 10, 0
        from scipy.ndimage import shift as ndi_shift
        data = make_star_field(
            [(x, y, 500.0) for x, y in ref_positions], shape=shape
        )
        # Simulate integer shift of the data
        shifted = ndi_shift(data, shift=(dy, dx), order=0, mode="constant", cval=np.nan)

        new_pos = recentroid_positions(shifted, ref_positions, (dx, dy), search_box_radius=5)

        assert np.all(np.isfinite(new_pos[0])), "In-frame source should have finite centroid"
        assert np.all(np.isnan(new_pos[1])), "Out-of-frame source should be NaN"

    def test_nan_pixels_in_cutout_do_not_crash(self):
        """NaN pixels inside the search box (e.g. border fill) must not raise."""
        shape = (64, 64)
        data = np.ones(shape)
        # Force NaN in the search box region
        data[28:35, 28:35] = np.nan
        ref_positions = np.array([[30.0, 30.0]])

        # Should return NaN gracefully, not raise
        try:
            new_pos = recentroid_positions(data, ref_positions, (0, 0), search_box_radius=5)
        except Exception as exc:
            pytest.fail(f"recentroid_positions raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# Integration: align_image → recentroid_positions round-trip
# ---------------------------------------------------------------------------

class TestAlignRecentroidRoundTrip:
    """
    Verify that aligning an image and then re-centroiding yields aperture
    positions that are within 1 pixel of the true star locations.
    """

    def test_round_trip_position_accuracy(self):
        """
        Using a mocked transform, check the full align-then-recentroid pipeline
        recovers the shifted star positions to sub-pixel accuracy.
        """
        shape = (128, 128)
        true_cx, true_cy = 64.0, 60.0
        fwhm = 4.0
        dx, dy = 8, -5

        ref_image = gaussian_source(true_cx, true_cy, 1000.0, fwhm, shape)
        ref_positions = np.array([[true_cx, true_cy]])

        mock_transform = MagicMock()
        mock_transform.translation = (float(dx), float(dy))

        with patch("astroalign.find_transform", return_value=(mock_transform, None)):
            shifted, bad_mask, shift_xy = align_image(ref_image, ref_image)

        assert shifted is not None

        new_pos = recentroid_positions(shifted, ref_positions, shift_xy, search_box_radius=8)

        assert np.all(np.isfinite(new_pos[0])), "Round-trip should yield finite position"
        assert abs(new_pos[0, 0] - (true_cx + dx)) < 1.0, \
            f"x position error too large: {abs(new_pos[0, 0] - (true_cx + dx)):.2f} px"
        assert abs(new_pos[0, 1] - (true_cy + dy)) < 1.0, \
            f"y position error too large: {abs(new_pos[0, 1] - (true_cy + dy)):.2f} px"
