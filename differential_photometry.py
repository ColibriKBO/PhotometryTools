#!/usr/bin/env python3
"""
Differential photometry entry-point script.

Reads a photometry CSV produced by aperture_photometry.py, selects stable
reference stars, builds a synthetic reference light curve, subtracts it from
every source, and writes the results to disk.

Example
-------
    python differential_photometry.py \\
        --input demo_output/photometry.csv \\
        --output-dir demo_output \\
        --plots

    python differential_photometry.py \\
        --input demo_output/photometry.csv \\
        --method weighted_mean \\
        --min-ref-snr 15 \\
        --plots
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.io import load_config
from src.differential import (
    REFERENCE_METHODS,
    compute_differential_lightcurves,
    select_reference_stars,
)
from src.plots import plot_differential_lightcurves

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("differential_photometry")

_DEFAULT_CONFIG = Path(__file__).parent / "config" / "default_config.json"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute differential light curves from an aperture-photometry CSV. "
            "Produces a differential_photometry.csv and optionally PNG plots."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="CSV",
        help="Path to the photometry CSV produced by aperture_photometry.py.",
    )
    parser.add_argument(
        "--config", "-c",
        default=str(_DEFAULT_CONFIG),
        metavar="JSON",
        help="Path to a JSON configuration file.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        metavar="DIR",
        help=(
            "Directory for output files.  Defaults to the parent directory of "
            "the input CSV."
        ),
    )
    parser.add_argument(
        "--method",
        default=None,
        choices=list(REFERENCE_METHODS.keys()),
        metavar="METHOD",
        help=(
            f"Reference-curve construction method. "
            f"Overrides config. Choices: {list(REFERENCE_METHODS.keys())}."
        ),
    )
    parser.add_argument(
        "--min-ref-snr",
        type=float,
        default=None,
        metavar="SNR",
        help=(
            "Minimum temporal SNR for a star to qualify as a reference. "
            "Overrides config."
        ),
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Save differential light curve PNG plots.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    cfg = load_config(args.config)

    if args.method is not None:
        cfg["diff_method"] = args.method
    if args.min_ref_snr is not None:
        cfg["min_reference_snr"] = args.min_ref_snr

    method = cfg.get("diff_method", "weighted_mean")
    min_ref_snr = float(cfg.get("min_reference_snr", 10.0))

    log.info(f"Method: {method}  |  Min reference SNR: {min_ref_snr}")

    # ------------------------------------------------------------------
    # I/O setup
    # ------------------------------------------------------------------
    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        log.error(f"Input file not found: {input_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load photometry
    # ------------------------------------------------------------------
    log.info(f"Loading photometry from {input_path}")
    df = pd.read_csv(input_path)
    log.info(f"  {len(df):,} rows, {df['source_id'].nunique()} sources, "
             f"{df['filename'].nunique()} epochs")

    required_cols = {"source_id", "filename", "mag", "mag_err", "temporal_snr"}
    missing = required_cols - set(df.columns)
    if missing:
        log.error(
            f"Input CSV is missing required columns: {missing}. "
            "Ensure it was produced by aperture_photometry.py."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Select reference stars
    # ------------------------------------------------------------------
    try:
        reference_ids = select_reference_stars(df, min_temporal_snr=min_ref_snr)
    except ValueError as exc:
        log.error(str(exc))
        sys.exit(1)

    # ------------------------------------------------------------------
    # Compute differential light curves
    # ------------------------------------------------------------------
    try:
        diff_df = compute_differential_lightcurves(df, reference_ids, method=method)
    except ValueError as exc:
        log.error(str(exc))
        sys.exit(1)

    # ------------------------------------------------------------------
    # Write output CSV
    # ------------------------------------------------------------------
    output_csv = output_dir / "differential_photometry.csv"
    diff_df.to_csv(output_csv, index=False)
    log.info(f"Wrote differential photometry table → {output_csv}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if args.plots:
        plot_dir = output_dir / "differential_lightcurves"
        log.info(f"Generating plots in {plot_dir} ...")
        plot_differential_lightcurves(diff_df, plot_dir)
        log.info("Plotting complete.")

    log.info("Done.")


if __name__ == "__main__":
    main()
