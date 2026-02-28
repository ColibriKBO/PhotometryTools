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


def compute_temporal_snr(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a temporal SNR for each source and merge it into the table.

    Temporal SNR is defined as::

        temporal_snr = median(flux) / std(flux)

    across all valid (finite-flux) epochs for a source.  The result is added
    as a new column ``temporal_snr`` so it is available in the output CSV and
    can be used by downstream steps such as variable-star selection.

    Parameters
    ----------
    df : DataFrame
        Photometry table with columns ``source_id`` and ``flux``.

    Returns
    -------
    DataFrame
        Copy of *df* with an additional ``temporal_snr`` column.
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
