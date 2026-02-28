"""
Image alignment utilities.

Alignment is performed via astroalign to find the best-fit affine transform
between two images.  Only the translation component is retained; it is
rounded to the nearest whole pixel and applied with scipy.ndimage.shift at
order=0 (nearest-neighbour), ensuring no pixel blending occurs.

After alignment, recentroid_positions refines source centroids within the
shifted image to correct for the sub-pixel rounding residual.
"""

import logging

import numpy as np

log = logging.getLogger(__name__)


def align_image(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | tuple[None, None, None]:
    """Align *source* onto *target* using an integer-pixel shift.

    ``astroalign`` determines the best-fit affine transform; only the
    translation component is kept, rounded to the nearest whole pixel, and
    applied with ``scipy.ndimage.shift`` at ``order=0`` (nearest-neighbour).
    Because the shift is an exact integer, each output pixel receives the
    value of exactly one input pixel — no blending occurs.

    Returns
    -------
    shifted : ndarray
        Source image shifted onto the target frame.  Border pixels with no
        source data are filled with NaN.
    bad_mask : ndarray of bool
        True where shifted pixels are NaN (no coverage).  Apertures that
        overlap any True pixel will be set to NaN in ``measure_photometry``.
    shift_xy : tuple[int, int]
        The ``(dx, dy)`` integer pixel shift applied (column, row order).
    Returns ``(None, None, None)`` on failure.
    """
    try:
        import astroalign as aa
        from scipy.ndimage import shift as ndi_shift

        transform, _ = aa.find_transform(source, target)
        tx, ty = transform.translation
        dx = int(round(tx))
        dy = int(round(ty))

        shifted = ndi_shift(
            source.astype(np.float64),
            shift=(dy, dx),
            order=0,
            mode="constant",
            cval=np.nan,
        )
        bad_mask = ~np.isfinite(shifted)
        return shifted, bad_mask, (dx, dy)
    except Exception as e:
        log.warning(f"Alignment failed: {e}. Photometry for this image will be NaN.")
        return None, None, None


def recentroid_positions(
    data: np.ndarray,
    ref_positions: np.ndarray,
    shift_xy: tuple[int, int],
    search_box_radius: int = 5,
) -> np.ndarray:
    """Re-centroid sources in an integer-shifted image.

    After ``align_image`` applies an integer shift, stars sit at approximately
    their reference-frame positions.  This function applies a centre-of-mass
    centroid within a small search box around each reference position to
    correct the sub-pixel rounding residual.

    Parameters
    ----------
    data : ndarray
        The shifted image (may contain NaN in border regions).
    ref_positions : ndarray, shape (N, 2)
        Source ``(x, y)`` positions in the reference frame.
    shift_xy : tuple[int, int]
        The ``(dx, dy)`` integer shift used to determine the NaN border.
    search_box_radius : int
        Half-width (pixels) of the centroiding search box about each source.

    Returns
    -------
    ndarray, shape (N, 2)
        Refined ``(x, y)`` positions in *data* coordinates.  Rows are NaN
        for sources that fall inside the NaN border, outside the image
        extent, or where centroiding produces a non-finite result.
    """
    from photutils.centroids import centroid_com

    dx, dy = shift_xy
    ny, nx = data.shape
    new_positions = np.full_like(ref_positions, np.nan)
    r = int(search_box_radius)

    for i, (x, y) in enumerate(ref_positions):
        ix = int(round(x))
        iy = int(round(y))
        if not (0 <= ix < nx and 0 <= iy < ny):
            continue
        if not np.isfinite(data[iy, ix]):
            continue  # inside the NaN border

        x0 = max(0, ix - r)
        x1 = min(nx, ix + r + 1)
        y0 = max(0, iy - r)
        y1 = min(ny, iy + r + 1)

        if x1 <= x0 or y1 <= y0:
            continue

        cutout = data[y0:y1, x0:x1].copy()
        cutout[~np.isfinite(cutout)] = 0.0

        if cutout.sum() <= 0:
            continue

        cx, cy = centroid_com(cutout)
        if not (np.isfinite(cx) and np.isfinite(cy)):
            continue

        new_positions[i] = [x0 + cx, y0 + cy]

    return new_positions
