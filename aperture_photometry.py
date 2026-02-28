#!/usr/bin/env python3
"""
Simple aperture photometry on a stack of calibrated FITS images.

Usage:
    python aperture_photometry.py --input <dir> [--config <json>] [--output-dir <dir>] [--plots] [--no-align]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.io import load_config, read_fits, read_fits_datetime
from src.detection import detect_sources
from src.alignment import align_image, recentroid_positions
from src.photometry import measure_aperture_photometry
from src.pipeline import apply_coverage_filter, compute_temporal_snr
from src import plots

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


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

    centroid_box_radius = int(cfg.get("centroid_box_radius", 5))

    # Photometry loop
    records = []
    for i, fits_path in enumerate(fits_files):
        log.info(f"[{i+1}/{len(fits_files)}] {fits_path.name}")
        data = read_fits(fits_path, cfg["fits_extension"])
        obs_time = read_fits_datetime(fits_path, cfg["fits_extension"])

        # Align if requested and not the reference
        bad_mask = None
        frame_positions = positions  # default: use reference positions as-is
        if cfg["align"] and i > 0:
            aligned, bad_mask, shift_xy = align_image(data, ref_data)
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

            # Re-centroid in the integer-shifted frame so apertures are
            # precisely centred despite the sub-pixel residual from rounding.
            frame_positions = recentroid_positions(
                data, positions, shift_xy, centroid_box_radius
            )
            n_lost = int(np.sum(~np.all(np.isfinite(frame_positions), axis=1)))
            if n_lost:
                log.debug(
                    f"  {n_lost} source(s) outside image after shift "
                    f"(dx={shift_xy[0]}, dy={shift_xy[1]}) — set to NaN."
                )

        # Separate valid and invalid (NaN) positions before calling photutils.
        valid_mask = np.all(np.isfinite(frame_positions), axis=1)
        nan_result = {k: np.nan for k in ("flux", "flux_err", "mag", "mag_err", "snr", "sky_bkg")}

        if valid_mask.any():
            phot_valid = measure_aperture_photometry(
                data, frame_positions[valid_mask],
                cfg["aperture_radius"], cfg["annulus_r_in"], cfg["annulus_r_out"], cfg["gain"],
                bad_mask=bad_mask,
            )
            # Index into the valid-only results using a running counter
            valid_indices = np.where(valid_mask)[0]
            phot_map = {idx: {k: phot_valid[k][vi] for k in phot_valid}
                        for vi, idx in enumerate(valid_indices)}
        else:
            phot_map = {}

        for j, sid in enumerate(source_ids):
            result = phot_map.get(j, nan_result)
            records.append({
                "source_id": int(sid),
                "x_ref": positions[j, 0],
                "y_ref": positions[j, 1],
                "filename": fits_path.name,
                "obs_time": obs_time,
                "flux": result["flux"],
                "flux_err": result["flux_err"],
                "mag": result["mag"],
                "mag_err": result["mag_err"],
                "snr": result["snr"],
                "sky_bkg": result["sky_bkg"],
            })

    df = pd.DataFrame(records)

    # Coverage filter
    df = apply_coverage_filter(df, cfg["coverage_min_fraction"])

    # Temporal SNR per source
    df = compute_temporal_snr(df)

    # Optional: reference image plot (after filter so we can colour-code kept vs dropped)
    if args.plots:
        kept_ids = set(df["source_id"].unique())
        plots.plot_reference_image(ref_data, positions, kept_ids, output_dir / "reference_image_sources.png")

    # Write CSV
    csv_path = output_dir / "photometry.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    log.info(
        f"Wrote {len(df)} rows ({df['source_id'].nunique()} sources × "
        f"{df['filename'].nunique()} images) → {csv_path}"
    )

    # Optional diagnostic plots (post-filter)
    if args.plots:
        plots.plot_snr_vs_magnitude(df, output_dir / "snr_vs_magnitude.png")
        plots.plot_temporal_snr_vs_magnitude(df, output_dir / "temporal_snr_vs_magnitude.png")
        plots.plot_lightcurves(df, output_dir / "lightcurves")


if __name__ == "__main__":
    main()
