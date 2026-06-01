"""
Differential photometry analysis.

Computes differential light curves by subtracting a synthetic reference curve
(built from an ensemble of stable reference stars) from each target's magnitude
time series.

The reference-curve computation is modular: ``REFERENCE_METHODS`` is a plain
dict that maps method names to callables, following the same pure-function
style used throughout the rest of the package.  Adding a new technique (e.g.
PCA-based) means implementing one new function and registering it in
``REFERENCE_METHODS``.

Typical usage
-------------
>>> df = pd.read_csv("photometry.csv")
>>> ref_ids = select_reference_stars(df, min_temporal_snr=10.0)
>>> diff_df = compute_differential_lightcurves(df, ref_ids, method="weighted_mean")
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reference-star selection
# ---------------------------------------------------------------------------


def select_reference_stars(
    df: pd.DataFrame,
    min_temporal_snr: float = 10.0,
) -> list[int]:
    """Return source IDs whose temporal SNR meets the minimum threshold.

    Parameters
    ----------
    df : DataFrame
        Photometry table produced by ``aperture_photometry.py``.  Must
        contain ``source_id`` and ``temporal_snr`` columns.
    min_temporal_snr : float
        Minimum temporal SNR required to qualify as a reference star.

    Returns
    -------
    list[int]
        Sorted list of qualifying ``source_id`` values.

    Raises
    ------
    ValueError
        If no stars satisfy the threshold.
    """
    if "temporal_snr" not in df.columns:
        raise ValueError(
            "Input DataFrame is missing 'temporal_snr' column. "
            "Run compute_temporal_snr() on the photometry table first."
        )

    # One temporal_snr value per source (same across all rows for that source)
    per_source = (
        df[["source_id", "temporal_snr"]]
        .drop_duplicates(subset="source_id")
        .dropna(subset=["temporal_snr"])
    )

    qualified = per_source[per_source["temporal_snr"] >= min_temporal_snr]
    ref_ids = sorted(qualified["source_id"].tolist())

    if not ref_ids:
        raise ValueError(
            f"No reference stars found with temporal_snr >= {min_temporal_snr}. "
            "Try lowering --min-ref-snr."
        )

    log.info(
        f"Selected {len(ref_ids)} reference stars "
        f"(temporal_snr >= {min_temporal_snr}) "
        f"out of {len(per_source)} total sources."
    )
    return ref_ids


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------


def compute_snr_weights(
    df: pd.DataFrame,
    reference_ids: list[int],
) -> dict[int, float]:
    """Compute SNR-squared normalised weights for the reference ensemble.

    Weight for source *i*: :math:`w_i = \\mathrm{snr}_i^2 / \\sum_j \\mathrm{snr}_j^2`

    Parameters
    ----------
    df : DataFrame
        Photometry table with ``source_id`` and ``temporal_snr`` columns.
    reference_ids : list[int]
        Source IDs that form the reference ensemble.

    Returns
    -------
    dict[int, float]
        Mapping from ``source_id`` to normalised weight summing to 1.
    """
    per_source = (
        df[["source_id", "temporal_snr"]]
        .drop_duplicates(subset="source_id")
        .set_index("source_id")
    )

    snr_values = np.array(
        [per_source.loc[sid, "temporal_snr"] for sid in reference_ids],
        dtype=float,
    )
    snr_sq = snr_values ** 2
    total = snr_sq.sum()
    if total == 0:
        # Fall back to equal weights if all SNRs are zero
        weights = np.ones(len(reference_ids)) / len(reference_ids)
    else:
        weights = snr_sq / total

    return dict(zip(reference_ids, weights))


# ---------------------------------------------------------------------------
# Internal matrix helper (shared by public API and LOO computation)
# ---------------------------------------------------------------------------


def _build_ref_matrices(
    df: pd.DataFrame,
    reference_ids: list[int],
) -> dict:
    """Build the raw numpy matrices needed for the weighted-mean reference.

    Returns a dict with keys:
    - ``epoch_index``: DataFrame(filename, obs_time) — epoch ordering
    - ``col_order``: list[int] — source_id ordering matching matrix columns
    - ``mag_filled``: (n_epochs, n_refs) mag matrix, NaN replaced with 0
    - ``err_filled``: (n_epochs, n_refs) err matrix, NaN replaced with 0
    - ``w_eff``: (n_epochs, n_refs) effective (un-normalised) weights; 0 where
      source is invalid at that epoch
    - ``W_sum``: (n_epochs,) sum of effective weights per epoch
    - ``ref_mag``: (n_epochs,) full weighted-mean reference magnitude
    - ``ref_err``: (n_epochs,) propagated reference error
    """
    weights = compute_snr_weights(df, reference_ids)
    ref_df = df[df["source_id"].isin(reference_ids)].copy()

    # Use filename as the epoch key. Some photometry tables have missing or
    # non-parseable obs_time values, and pivoting on a NaN-valued index would
    # drop every epoch. Filenames are already unique per exposure.
    all_epochs = df[["filename"]].drop_duplicates().set_index("filename").index

    mag_pivot = ref_df.pivot_table(
        index="filename",
        columns="source_id",
        values="mag",
        aggfunc="first",
    ).reindex(all_epochs)
    err_pivot = ref_df.pivot_table(
        index="filename",
        columns="source_id",
        values="mag_err",
        aggfunc="first",
    ).reindex(all_epochs)

    col_order = [sid for sid in reference_ids if sid in mag_pivot.columns]
    mag_matrix = mag_pivot[col_order].values.astype(float)
    err_matrix = err_pivot[col_order].values.astype(float)
    w = np.array([weights[sid] for sid in col_order], dtype=float)

    # Subtract each reference star's temporal median so all curves are in
    # "deviation from mean" space.  Without this, the weighted-mean reference
    # has an absolute level set by the ensemble's average brightness, which
    # differs from every target star's brightness and produces a large,
    # spurious constant offset in the differential light curve.
    star_medians = np.nanmedian(mag_matrix, axis=0)  # (n_refs,)
    mag_matrix = mag_matrix - star_medians[np.newaxis, :]

    valid_mask = np.isfinite(mag_matrix) & np.isfinite(err_matrix)
    # Effective (un-normalised) weights: zero out invalid entries
    w_eff = np.where(valid_mask, w[np.newaxis, :], 0.0)  # (n_epochs, n_refs)
    W_sum = w_eff.sum(axis=1)  # (n_epochs,)
    safe_W = np.where(W_sum > 0, W_sum, np.nan)
    w_norm = w_eff / safe_W[:, np.newaxis]

    mag_filled = np.where(valid_mask, mag_matrix, 0.0)
    err_filled = np.where(valid_mask, err_matrix, 0.0)

    ref_mag = (w_norm * mag_filled).sum(axis=1)
    ref_mag[W_sum == 0] = np.nan

    ref_err = np.sqrt((w_norm ** 2 * err_filled ** 2).sum(axis=1))
    ref_err[W_sum == 0] = np.nan

    epoch_index = mag_pivot.reset_index()[["filename"]].copy()
    if "obs_time" in df.columns:
        obs_lookup = (
            df[["filename", "obs_time"]]
            .drop_duplicates(subset="filename")
            .set_index("filename")
        )
        epoch_index["obs_time"] = epoch_index["filename"].map(obs_lookup["obs_time"])

    return {
        "epoch_index": epoch_index,
        "col_order": col_order,
        "mag_filled": mag_filled,
        "err_filled": err_filled,
        "w_eff": w_eff,
        "W_sum": W_sum,
        "ref_mag": ref_mag,
        "ref_err": ref_err,
    }


# ---------------------------------------------------------------------------
# Leave-one-out RMS helper
# ---------------------------------------------------------------------------


def _compute_loo_rms(mats: dict) -> np.ndarray:
    """Compute the leave-one-out differential RMS for each reference star.

    For each star *i* currently in the ensemble the function:

    1. Rebuilds the reference curve **without** star *i* (LOO).
    2. Computes ``diff_i(e) = mag_i(e) - ref_{-i}(e)`` for every valid epoch.
    3. Returns ``RMS_i = sqrt(mean(diff_i²))``.

    Parameters
    ----------
    mats : dict
        Output of :func:`_build_ref_matrices` for the current ensemble.

    Returns
    -------
    np.ndarray
        Shape ``(n_refs,)``.  ``NaN`` for stars with fewer than 2 valid epochs
        or whose removal empties the ensemble at all epochs.
    """
    n_refs = len(mats["col_order"])
    rms = np.full(n_refs, np.nan)

    ref_mag_safe = np.where(np.isfinite(mats["ref_mag"]), mats["ref_mag"], 0.0)

    for i in range(n_refs):
        w_i = mats["w_eff"][:, i]           # (n_epochs,) effective weight
        valid_i = w_i > 0                    # epochs where star i has a measurement

        if valid_i.sum() < 2:
            continue

        W_loo = mats["W_sum"] - w_i          # (n_epochs,)
        safe_W_loo = np.where(W_loo > 0, W_loo, np.nan)

        # LOO reference: remove star i's contribution from the full sum
        loo_mag = (
            mats["W_sum"] * ref_mag_safe
            - w_i * mats["mag_filled"][:, i]
        ) / safe_W_loo
        loo_mag[W_loo == 0] = np.nan

        # Differential at valid epochs only
        diff = mats["mag_filled"][:, i] - loo_mag
        mask = valid_i & np.isfinite(loo_mag)
        diff_valid = diff[mask]

        if len(diff_valid) < 2:
            continue

        rms[i] = np.sqrt(np.mean(diff_valid ** 2))

    return rms


# ---------------------------------------------------------------------------
# Iterative sigma-clipping reference star selection
# ---------------------------------------------------------------------------


def sigma_clip_reference_stars(
    df: pd.DataFrame,
    candidate_ids: list[int],
    sigma: float = 3.0,
    min_ref_stars: int = 3,
) -> list[int]:
    """Iteratively sigma-clip reference stars by their leave-one-out RMS.

    Starting from *candidate_ids* the algorithm removes one star per
    iteration — the worst offender — and repeats until the ensemble is
    clean (no star exceeds the threshold) or the minimum size is reached.

    Algorithm per iteration
    -----------------------
    1. Build LOO reference matrices for the current ensemble.
    2. Compute per-star leave-one-out differential RMS via
       :func:`_compute_loo_rms`.
    3. Compute a robust threshold::

           threshold = median(RMS) + sigma * 1.4826 * MAD(RMS)

       The 1.4826 factor makes MAD a consistent estimator of σ for Gaussian
       data.
    4. Remove the star with the highest RMS if it exceeds the threshold.
    5. Repeat until no star exceeds the threshold (convergence).

    A full clip log is emitted at ``INFO`` level on completion.

    Parameters
    ----------
    df : DataFrame
        Full photometry table — the same DataFrame passed to
        :func:`compute_differential_lightcurves`.
    candidate_ids : list[int]
        Initial set of candidate reference star IDs (e.g. from
        :func:`select_reference_stars`).
    sigma : float
        Sigma threshold used throughout clipping.  Default ``3.0``.
    min_ref_stars : int
        Minimum number of reference stars to keep.  Clipping stops when
        the ensemble reaches this size.  Default ``3``.

    Returns
    -------
    list[int]
        Sorted list of retained reference star IDs.

    Raises
    ------
    ValueError
        If *candidate_ids* is empty.
    """
    if not candidate_ids:
        raise ValueError("candidate_ids is empty — no reference stars to clip.")

    current_ids = list(candidate_ids)
    clipped_log: list[dict] = []

    log.info(
        f"Sigma-clip: threshold={sigma}σ, "
        f"starting ensemble size={len(current_ids)}."
    )
    iteration = 0

    while True:
        if len(current_ids) <= min_ref_stars:
            log.warning(
                f"Ensemble has reached the minimum size of {min_ref_stars} "
                f"stars; stopping sigma-clipping."
            )
            break

        mats = _build_ref_matrices(df, current_ids)
        rms_arr = _compute_loo_rms(mats)
        col_order = mats["col_order"]

        valid_rms = rms_arr[np.isfinite(rms_arr)]
        if len(valid_rms) < 2:
            log.debug("Fewer than 2 stars with valid RMS — stopping.")
            break

        median_rms = float(np.median(valid_rms))
        mad_rms = float(np.median(np.abs(valid_rms - median_rms)))
        threshold = median_rms + sigma * 1.4826 * mad_rms

        worst_idx = int(np.nanargmax(rms_arr))
        worst_rms = float(rms_arr[worst_idx])
        worst_id = col_order[worst_idx]

        iteration += 1
        log.debug(
            f"  Iter {iteration}: median_RMS={median_rms:.5f}, "
            f"MAD={mad_rms:.5f}, threshold={threshold:.5f}, "
            f"worst=source_{worst_id} RMS={worst_rms:.5f}"
        )

        if worst_rms > threshold:
            current_ids.remove(worst_id)
            entry = {
                "source_id": worst_id,
                "rms": worst_rms,
                "threshold": threshold,
                "median_rms": median_rms,
                "mad_rms": mad_rms,
                "iteration": iteration,
            }
            clipped_log.append(entry)
            log.info(
                f"  [iter {iteration}] Clipped source_{worst_id}: "
                f"RMS={worst_rms:.5f} > threshold={threshold:.5f} "
                f"({sigma}σ).  {len(current_ids)} stars remain."
            )
        else:
            log.info(
                f"  Converged after {iteration - 1} removal(s); "
                f"no star exceeds {sigma}σ."
            )
            break

    # ---- Summary --------------------------------------------------------
    if clipped_log:
        log.info(
            f"Sigma-clip summary: removed {len(clipped_log)} star(s) from "
            f"the initial {len(candidate_ids)}-star ensemble:"
        )
        for e in clipped_log:
            log.info(
                f"  source_{e['source_id']:>6d}: RMS={e['rms']:.5f}, "
                f"threshold={e['threshold']:.5f} "
                f"({sigma}σ, iter {e['iteration']})"
            )
    else:
        log.info(
            "Sigma-clip summary: ensemble already clean — no stars removed."
        )

    log.info(
        f"Final reference ensemble after sigma-clipping: "
        f"{len(current_ids)}/{len(candidate_ids)} stars retained."
    )
    return sorted(current_ids)


# ---------------------------------------------------------------------------
# Strategy: weighted-mean reference curve
# ---------------------------------------------------------------------------


def build_weighted_mean_reference(
    df: pd.DataFrame,
    reference_ids: list[int],
) -> pd.DataFrame:
    """Build a per-epoch weighted-mean reference magnitude curve.

    For each epoch *e* the reference magnitude is:

    .. math::

        m_{\\mathrm{ref}}(e) = \\sum_i w_i \\, m_i(e) \\bigg/ \\sum_{i: \\text{valid}} w_i

    Weights are renormalised per epoch to account for missing observations.
    The propagated uncertainty is:

    .. math::

        \\sigma_{\\mathrm{ref}}(e) =
            \\sqrt{\\sum_i \\left(\\tilde{w}_i \\, \\sigma_{m,i}(e)\\right)^2}

    where :math:`\\tilde{w}_i` are the renormalised epoch weights.

    Parameters
    ----------
    df : DataFrame
        Full photometry table.
    reference_ids : list[int]
        Source IDs that form the reference ensemble.

    Returns
    -------
    DataFrame
        Columns: ``filename``, ``obs_time``, ``ref_mag``, ``ref_mag_err``.
        One row per unique ``filename`` in *df*.
    """
    mats = _build_ref_matrices(df, reference_ids)
    result = mats["epoch_index"].copy()
    result["ref_mag"] = mats["ref_mag"]
    result["ref_mag_err"] = mats["ref_err"]
    log.debug(
        f"Built weighted-mean reference curve over {len(mats['col_order'])} "
        f"reference stars and {len(result)} epochs."
    )
    return result


# ---------------------------------------------------------------------------
# Method registry — extend here to add new strategies
# ---------------------------------------------------------------------------

#: Registry mapping method name → reference-curve builder callable.
#:
#: Each callable must have the signature::
#:
#:     func(df: pd.DataFrame, reference_ids: list[int]) -> pd.DataFrame
#:
#: and return a DataFrame with columns
#: ``filename``, ``obs_time``, ``ref_mag``, ``ref_mag_err``.
REFERENCE_METHODS: dict[str, Callable[[pd.DataFrame, list[int]], pd.DataFrame]] = {
    "weighted_mean": build_weighted_mean_reference,
}


# ---------------------------------------------------------------------------
# Main entry point for the analysis
# ---------------------------------------------------------------------------


def compute_differential_lightcurves(
    df: pd.DataFrame,
    reference_ids: list[int],
    method: str = "weighted_mean",
) -> pd.DataFrame:
    """Compute differential light curves for every source in *df*.

    Uses a **leave-one-out** (LOO) scheme: when the target star is itself a
    member of the reference pool, its own flux is excluded from the reference
    curve computed for that star.  Targets outside the reference pool use the
    full ensemble unchanged.

    For a target *t* and epoch *e*, letting :math:`w_j(e)` be the effective
    (un-normalised, NaN-zeroed) SNR² weight of reference star *j*:

    .. math::

        W(e) &= \\textstyle\\sum_j w_j(e) \\\\
        m_\\text{ref}(e) &= \\frac{\\sum_j w_j(e)\\,m_j(e)}{W(e)} \\\\
        m_\\text{ref,loo}^{(t)}(e) &=
            \\frac{W(e)\\,m_\\text{ref}(e) - w_t(e)\\,m_t(e)}{W(e) - w_t(e)}\\\\
        \\Delta m_t(e) &= m_t(e) - m_\\text{ref,loo}^{(t)}(e)

    For targets not in the reference pool, :math:`m_\\text{ref,loo}^{(t)} = m_\\text{ref}`.

    Parameters
    ----------
    df : DataFrame
        Full photometry table (output of ``aperture_photometry.py``).
    reference_ids : list[int]
        Source IDs forming the reference ensemble.
    method : str
        Key into ``REFERENCE_METHODS`` selecting which strategy to use.

    Returns
    -------
    DataFrame
        Long-form table with columns:
        ``source_id``, ``x_ref``, ``y_ref``, ``filename``, ``obs_time``,
        ``mag``, ``mag_err``, ``ref_mag``, ``ref_mag_err``,
        ``diff_mag``, ``diff_mag_err``, ``is_reference``.

    Raises
    ------
    ValueError
        If *method* is not in ``REFERENCE_METHODS``.
    """
    if method not in REFERENCE_METHODS:
        raise ValueError(
            f"Unknown differential photometry method '{method}'. "
            f"Available: {list(REFERENCE_METHODS.keys())}"
        )

    # Build full reference matrices once (shared by all targets)
    mats = _build_ref_matrices(df, reference_ids)
    epoch_index = mats["epoch_index"]  # DataFrame(filename, obs_time)
    col_order = mats["col_order"]
    sid_to_col = {sid: i for i, sid in enumerate(col_order)}

    ref_id_set = set(reference_ids)
    avail_cols = [
        c for c in ["source_id", "x_ref", "y_ref", "filename", "obs_time", "mag", "mag_err"]
        if c in df.columns
    ]

    pieces = []
    for sid, src_df in df.groupby("source_id"):
        src_rows = src_df[avail_cols].copy()

        if sid in ref_id_set and sid in sid_to_col:
            # --- Leave-one-out correction ---
            # Remove star sid's contribution from the full reference curve.
            col_i = sid_to_col[sid]
            w_i = mats["w_eff"][:, col_i]           # (n_epochs,) effective weight
            W_loo = mats["W_sum"] - w_i              # (n_epochs,)
            safe_W_loo = np.where(W_loo > 0, W_loo, np.nan)

            # Numerator: W*ref_mag - w_i*mag_i  (both already account for NaN→0)
            loo_mag = (
                mats["W_sum"] * np.where(np.isfinite(mats["ref_mag"]), mats["ref_mag"], 0.0)
                - w_i * mats["mag_filled"][:, col_i]
            ) / safe_W_loo
            loo_mag[W_loo == 0] = np.nan

            # Error: sqrt(sum_{j!=i} (w_j/W_loo)^2 * err_j^2)
            sum_w2_err2 = (mats["w_eff"] ** 2 * mats["err_filled"] ** 2).sum(axis=1)
            sum_w2_err2_loo = (
                sum_w2_err2
                - mats["w_eff"][:, col_i] ** 2 * mats["err_filled"][:, col_i] ** 2
            )
            loo_err = np.sqrt(np.maximum(sum_w2_err2_loo, 0.0)) / safe_W_loo
            loo_err[W_loo == 0] = np.nan

            ref_for_target = epoch_index.copy()
            ref_for_target["ref_mag"] = loo_mag
            ref_for_target["ref_mag_err"] = loo_err
        else:
            # Non-reference target: use the full ensemble as-is
            ref_for_target = epoch_index.copy()
            ref_for_target["ref_mag"] = mats["ref_mag"]
            ref_for_target["ref_mag_err"] = mats["ref_err"]

        merged = src_rows.merge(ref_for_target, on="filename", how="left")

        # Subtract this target's temporal median so the differential is in
        # "deviation from mean" space, matching how the reference was built.
        # A non-variable star will produce diff_mag ≈ 0 at all epochs.
        target_median = merged["mag"].median()
        merged["diff_mag"] = (merged["mag"] - target_median) - merged["ref_mag"]
        merged["diff_mag_err"] = np.sqrt(
            merged["mag_err"].fillna(0) ** 2 + merged["ref_mag_err"].fillna(0) ** 2
        )
        bad = merged["mag"].isna() | merged["ref_mag"].isna()
        merged.loc[bad, "diff_mag"] = np.nan
        merged.loc[bad, "diff_mag_err"] = np.nan
        merged["is_reference"] = sid in ref_id_set
        pieces.append(merged)

    result = pd.concat(pieces, ignore_index=True)
    log.info(
        f"Computed differential light curves for {result['source_id'].nunique()} sources "
        f"using method='{method}' (LOO for reference stars), "
        f"{len(reference_ids)} reference stars."
    )
    return result
