#!/usr/bin/env python3
"""
Simple aperture photometry on a stack of calibrated FITS images.

Usage:
    python aperture_photometry.py --input <dir> [--config <json>] [--output-dir <dir>] [--plots] [--no-align]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats, aperture_photometry

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def read_fits(path: Path, extension: int) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# Source detection
# ---------------------------------------------------------------------------

def detect_sources(data: np.ndarray, fwhm: float, threshold_sigma: float) -> np.ndarray:
    """Return Nx2 array of (x, y) pixel positions detected in data."""
    _, median, std = sigma_clipped_stats(data, sigma=3.0)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)
    sources = daofind(data - median)
    if sources is None or len(sources) == 0:
        log.error("No sources detected in reference image. Check detection parameters.")
        sys.exit(1)
    log.info(f"Detected {len(sources)} sources in reference image.")
    positions = np.column_stack([sources["xcentroid"], sources["ycentroid"]])
    return positions


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def align_image(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Align source onto target frame using astroalign.

    Returns
    -------
    aligned : ndarray
        Source image warped into the target frame. Pixels with no source data
        (outside the dither overlap region) are filled with NaN.
    bad_mask : ndarray of bool
        True where aligned pixels are NaN (no coverage). Apertures that
        overlap any True pixel will be set to NaN in measure_photometry.
    Returns (None, None) on failure.
    """
    try:
        import astroalign as aa
        aligned, _ = aa.register(source, target, fill_value=np.nan)
        bad_mask = ~np.isfinite(aligned)
        return aligned, bad_mask
    except Exception as e:
        log.warning(f"Alignment failed: {e}. Photometry for this image will be NaN.")
        return None, None


# ---------------------------------------------------------------------------
# Aperture photometry
# ---------------------------------------------------------------------------

def measure_photometry(
    data: np.ndarray,
    positions: np.ndarray,
    aperture_radius: float,
    annulus_r_in: float,
    annulus_r_out: float,
    gain: float,
    bad_mask: np.ndarray | None = None,
) -> dict:
    """
    Perform aperture photometry with local sky subtraction via annulus.

    Parameters
    ----------
    bad_mask : bool array, optional
        True where pixels are invalid (e.g. outside dither overlap after
        alignment). Any aperture or annulus that overlaps a bad pixel is
        set to NaN in the output.

    Returns dict of arrays: flux, flux_err, mag, mag_err, snr, sky_bkg.
    """
    apertures = CircularAperture(positions, r=aperture_radius)
    annuli = CircularAnnulus(positions, r_in=annulus_r_in, r_out=annulus_r_out)

    # Identify apertures/annuli that touch any bad (no-coverage) pixel.
    # We sum the bad_mask through each aperture/annulus; any count > 0 → NaN.
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

    # Replace bad pixels with 0 for photometry sums (we'll NaN-out afterwards)
    data_clean = np.where(bad_mask, 0.0, data) if bad_mask is not None else data

    # Sky background: median in annulus per source
    sky_stats = ApertureStats(data_clean, annuli)
    sky_per_pixel = sky_stats.median  # shape (N,)
    sky_per_pixel = np.where(ann_invalid, np.nan, sky_per_pixel)

    # Raw aperture sums
    phot_table = aperture_photometry(data_clean, apertures)
    raw_sum = np.array(phot_table["aperture_sum"])
    raw_sum = np.where(ap_invalid, np.nan, raw_sum)

    # Sky-subtracted flux
    aperture_area = apertures.area  # scalar (same for all, circular)
    flux = raw_sum - sky_per_pixel * aperture_area

    # Flux error: Poisson noise from source + sky within aperture
    flux_err = np.sqrt(np.abs(flux) / gain + aperture_area * np.abs(sky_per_pixel) / gain)

    # Magnitudes (instrumental, relative — requires positive flux)
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


# ---------------------------------------------------------------------------
# Coverage filter
# ---------------------------------------------------------------------------

def apply_coverage_filter(df: pd.DataFrame, min_fraction: float) -> pd.DataFrame:
    """Drop sources whose valid-measurement fraction across all images is below min_fraction."""
    n_images = df["filename"].nunique()
    valid_counts = df.dropna(subset=["flux"]).groupby("source_id").size()
    valid_fraction = valid_counts / n_images
    keep = valid_fraction[valid_fraction >= min_fraction].index
    n_before = df["source_id"].nunique()
    df = df[df["source_id"].isin(keep)].copy()
    n_after = df["source_id"].nunique()
    log.info(
        f"Coverage filter (>={min_fraction:.0%}): kept {n_after}/{n_before} sources "
        f"({n_before - n_after} removed)."
    )
    return df


# ---------------------------------------------------------------------------
# Temporal SNR
# ---------------------------------------------------------------------------

def compute_temporal_snr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a temporal SNR for each source defined as:
        temporal_snr = median(flux) / std(flux)
    across all valid epochs in the light curve.

    The result is merged back into the per-row DataFrame as a new column
    ``temporal_snr`` so it is available in the output CSV.
    """
    stats = (
        df.dropna(subset=["flux"])
        .groupby("source_id")["flux"]
        .agg(flux_median="median", flux_std="std")
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        stats["temporal_snr"] = stats["flux_median"] / stats["flux_std"]
    df = df.merge(stats[["temporal_snr"]], on="source_id", how="left")
    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_reference_image(
    ref_data: np.ndarray,
    positions: np.ndarray,
    kept_ids: set,
    output_path: Path,
):
    """Save reference image with detected sources overlaid.

    Sources that passed the coverage filter are shown in green;
    sources that were dropped are shown in red.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.visualization import ZScaleInterval

    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(ref_data)

    kept_mask = np.array([i in kept_ids for i in range(len(positions))])
    dropped_mask = ~kept_mask

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(ref_data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")

    if kept_mask.any():
        ax.scatter(
            positions[kept_mask, 0], positions[kept_mask, 1],
            s=80, facecolors="none", edgecolors="limegreen", linewidths=0.8,
            label=f"Passed coverage ({kept_mask.sum()})",
        )
    if dropped_mask.any():
        ax.scatter(
            positions[dropped_mask, 0], positions[dropped_mask, 1],
            s=80, facecolors="none", edgecolors="red", linewidths=0.8,
            label=f"Dropped by coverage filter ({dropped_mask.sum()})",
        )

    ax.set_title("Reference Image — Detected Sources", fontsize=14)
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved reference image plot → {output_path}")


def plot_snr_vs_magnitude(df: pd.DataFrame, output_path: Path):
    """Save SNR vs instrumental magnitude scatter plot (per-source medians)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = df.groupby("source_id").agg(
        mag_median=("mag", "median"),
        snr_median=("snr", "median"),
    ).dropna()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(summary["mag_median"], summary["snr_median"], s=15, alpha=0.7, color="steelblue")
    ax.set_xlabel("Instrumental Magnitude (median over epochs)", fontsize=13)
    ax.set_ylabel("Median Per-Epoch SNR", fontsize=13)
    ax.set_title("Median Per-Epoch SNR vs Instrumental Magnitude", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved SNR vs magnitude plot → {output_path}")


def plot_temporal_snr_vs_magnitude(df: pd.DataFrame, output_path: Path):
    """Save temporal SNR vs instrumental magnitude scatter plot (per-source).

    Temporal SNR = median(flux) / std(flux) across all valid epochs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = (
        df.groupby("source_id")
        .agg(mag_median=("mag", "median"), temporal_snr=("temporal_snr", "first"))
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(summary["mag_median"], summary["temporal_snr"], s=15, alpha=0.7, color="darkorange")
    ax.set_xlabel("Instrumental Magnitude (median)", fontsize=13)
    ax.set_ylabel("Temporal SNR  [median flux / std flux]", fontsize=13)
    ax.set_title("Temporal SNR vs Instrumental Magnitude", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved temporal SNR vs magnitude plot → {output_path}")


def plot_lightcurves(df: pd.DataFrame, output_dir: Path):
    """Save one flux light curve plot per source into output_dir.

    Uses the ``obs_time`` column for the x-axis if valid datetimes are present;
    otherwise falls back to integer epoch index.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    output_dir.mkdir(parents=True, exist_ok=True)

    # Attempt to parse obs_time into datetimes; fall back to epoch index on failure
    df = df.copy()
    use_datetime = False
    if "obs_time" in df.columns:
        df["obs_dt"] = pd.to_datetime(df["obs_time"], errors="coerce")
        valid_times = df["obs_dt"].notna().sum()
        use_datetime = valid_times > 0
        if not use_datetime:
            log.warning("obs_time column present but no parseable datetimes; using epoch index.")

    if not use_datetime:
        filenames = sorted(df["filename"].unique())
        fname_to_idx = {f: i for i, f in enumerate(filenames)}
        df["epoch"] = df["filename"].map(fname_to_idx)

    source_ids = sorted(df["source_id"].unique())
    log.info(f"Saving {len(source_ids)} light curve plots...")

    for sid in source_ids:
        src = df[df["source_id"] == sid].sort_values("obs_dt" if use_datetime else "epoch")
        flux = src["flux"].values
        flux_err = src["flux_err"].values
        xvals = src["obs_dt"].values if use_datetime else src["epoch"].values

        valid = np.isfinite(flux)
        if valid.sum() == 0:
            continue

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.errorbar(
            xvals[valid], flux[valid], yerr=flux_err[valid],
            fmt="o", color="steelblue", ecolor="lightsteelblue",
            capsize=3, markersize=4, linewidth=0.8,
        )

        if use_datetime:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
            fig.autofmt_xdate(rotation=30, ha="right")
            ax.set_xlabel("Observation Time (UTC)", fontsize=12)
        else:
            ax.set_xlabel("Epoch (image index)", fontsize=12)

        ax.set_ylabel("Flux (counts)", fontsize=12)
        x_ref = src["x_ref"].iloc[0]
        y_ref = src["y_ref"].iloc[0]
        ax.set_title(f"Source {sid:04d}  (x={x_ref:.1f}, y={y_ref:.1f})", fontsize=13)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_dir / f"source_{sid:04d}.png", dpi=100)
        plt.close(fig)

    log.info(f"Light curve plots saved to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir = Path(__file__).parent
    default_config = script_dir / "config" / "default_config.json"

    parser = argparse.ArgumentParser(description="Simple aperture photometry on calibrated FITS images.")
    parser.add_argument("--input", required=True, help="Directory containing FITS files.")
    parser.add_argument("--config", default=str(default_config), help="Path to JSON config file.")
    parser.add_argument("--output-dir", default="photometry_output", help="Directory for CSV and plot outputs.")
    parser.add_argument("--plots", action="store_true", help="Generate diagnostic plots.")
    parser.add_argument("--no-align", action="store_true", help="Disable image alignment.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.no_align:
        cfg["align"] = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect and sort FITS files
    input_dir = Path(args.input)
    fits_files = sorted({
        f for pattern in ("*.fits", "*.fit", "*.FITS", "*.FIT")
        for f in input_dir.glob(pattern)
    })
    if not fits_files:
        log.error(f"No FITS files found in {input_dir}")
        sys.exit(1)
    log.info(f"Found {len(fits_files)} FITS files. Reference: {fits_files[0].name}")

    # Detect sources in reference image
    ref_data = read_fits(fits_files[0], cfg["fits_extension"])
    positions = detect_sources(ref_data, cfg["detection_fwhm"], cfg["detection_threshold"])
    source_ids = np.arange(len(positions))

    # Photometry loop
    records = []
    for i, fits_path in enumerate(fits_files):
        log.info(f"[{i+1}/{len(fits_files)}] {fits_path.name}")
        data = read_fits(fits_path, cfg["fits_extension"])
        obs_time = read_fits_datetime(fits_path, cfg["fits_extension"])

        # Align if requested and not the reference
        bad_mask = None
        if cfg["align"] and i > 0:
            aligned, bad_mask = align_image(data, ref_data)
            if aligned is None:
                for sid, (x, y) in zip(source_ids, positions):
                    records.append({
                        "source_id": int(sid), "x_ref": x, "y_ref": y,
                        "filename": fits_path.name, "obs_time": obs_time,
                        "flux": np.nan, "flux_err": np.nan,
                        "mag": np.nan, "mag_err": np.nan,
                        "snr": np.nan, "sky_bkg": np.nan,
                    })
                continue
            data = aligned

        phot = measure_photometry(
            data, positions,
            cfg["aperture_radius"], cfg["annulus_r_in"], cfg["annulus_r_out"], cfg["gain"],
            bad_mask=bad_mask,
        )

        for j, sid in enumerate(source_ids):
            records.append({
                "source_id": int(sid),
                "x_ref": positions[j, 0],
                "y_ref": positions[j, 1],
                "filename": fits_path.name,
                "obs_time": obs_time,
                "flux": phot["flux"][j],
                "flux_err": phot["flux_err"][j],
                "mag": phot["mag"][j],
                "mag_err": phot["mag_err"][j],
                "snr": phot["snr"][j],
                "sky_bkg": phot["sky_bkg"][j],
            })

    df = pd.DataFrame(records)

    # Coverage filter
    df = apply_coverage_filter(df, cfg["coverage_min_fraction"])

    # Temporal SNR per source
    df = compute_temporal_snr(df)

    # Optional: reference image plot (after filter so we can colour-code kept vs dropped)
    if args.plots:
        kept_ids = set(df["source_id"].unique())
        plot_reference_image(ref_data, positions, kept_ids, output_dir / "reference_image_sources.png")

    # Write CSV
    csv_path = output_dir / "photometry.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    log.info(
        f"Wrote {len(df)} rows ({df['source_id'].nunique()} sources × "
        f"{df['filename'].nunique()} images) → {csv_path}"
    )

    # Optional diagnostic plots (post-filter)
    if args.plots:
        plot_snr_vs_magnitude(df, output_dir / "snr_vs_magnitude.png")
        plot_temporal_snr_vs_magnitude(df, output_dir / "temporal_snr_vs_magnitude.png")
        plot_lightcurves(df, output_dir / "lightcurves")


if __name__ == "__main__":
    main()
