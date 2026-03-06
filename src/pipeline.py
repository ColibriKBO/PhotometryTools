"""
Pipeline-level processing steps applied to the combined photometry table.

These functions operate on a pandas DataFrame that has one row per
(source, image) pair.  Adding new pipeline steps here (e.g. differential
light curve normalisation, period-folding, outlier rejection) keeps the
main script clean and makes each step independently testable.
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def apply_coverage_filter(df: pd.DataFrame, min_fraction: float) -> pd.DataFrame:
    """Drop sources whose valid-measurement fraction across all images is below
    *min_fraction*.

    Parameters
    ----------
    df : DataFrame
        Full photometry table with columns ``source_id``, ``filename``,
        and ``flux``.
    min_fraction : float
        Minimum fraction of images in which a source must have a finite flux
        measurement to be retained (e.g. 0.5 → at least half the images).

    Returns
    -------
    DataFrame
        Filtered copy of *df* containing only sources that meet the threshold.
    """
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


def compute_temporal_snr(df: pd.DataFrame, window: int = 0) -> pd.DataFrame:
    """Compute a temporal SNR for each source and merge it into the table.

    Temporal SNR is defined as::

        temporal_snr = median(flux) / std(flux)

    Parameters
    ----------
    df : DataFrame
        Photometry table with columns ``source_id``, ``obs_time``, and
        ``flux``.  Rows must already be sorted or will be sorted by
        ``obs_time`` (then ``source_id``) internally.
    window : int
        Rolling-window size (number of frames).
        ``0`` (default) — compute over all available epochs for each source
        (original behaviour).
        ``N > 0`` — compute a rolling median/std over *N* consecutive frames
        per source; edge frames with fewer than *N* neighbours use whatever
        data are available (``min_periods=1``).

    Returns
    -------
    DataFrame
        Copy of *df* with an additional ``temporal_snr`` column.
    """
    if window == 0:
        # Global SNR over the full time series
        stats = (
            df.dropna(subset=["flux"])
            .groupby("source_id")["flux"]
            .agg(flux_median="median", flux_std="std")
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            stats["temporal_snr"] = stats["flux_median"] / stats["flux_std"]
        df = df.merge(stats[["temporal_snr"]], on="source_id", how="left")
    else:
        # Rolling SNR: one value per (source, frame)
        df = df.sort_values(["source_id", "obs_time"]).copy()
        def _rolling_snr(grp: pd.Series) -> pd.Series:
            roll = grp.rolling(window=window, min_periods=1, center=True)
            med = roll.median()
            std = roll.std()
            with np.errstate(invalid="ignore", divide="ignore"):
                return med / std
        df["temporal_snr"] = (
            df.groupby("source_id")["flux"].transform(_rolling_snr)
        )
        log.info(f"Temporal SNR computed with rolling window of {window} frames.")
    return df
