"""
Diagnostic plotting routines.

All functions write plot files to disk and return None.  Matplotlib is
imported lazily inside each function so that the rest of the package works
in headless / no-display environments without any extra configuration.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Use a non-interactive backend in every function to avoid display issues
_MPL_BACKEND = "Agg"


def plot_reference_image(
    ref_data: np.ndarray,
    positions: np.ndarray,
    kept_ids: set,
    output_path: Path,
) -> None:
    """Save reference image with detected sources overlaid.

    Sources that passed the coverage filter are shown in green; dropped
    sources are shown in red.

    Parameters
    ----------
    ref_data : ndarray
        2-D reference science image.
    positions : ndarray, shape (N, 2)
        ``(x, y)`` pixel positions of all detected sources.
    kept_ids : set
        Source IDs that survived the coverage filter.
    output_path : Path
        Destination file for the saved plot (PNG recommended).
    """
    import matplotlib
    matplotlib.use(_MPL_BACKEND)
    import matplotlib.pyplot as plt
    from astropy.visualization import ZScaleInterval

    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(ref_data)

    kept_mask = np.array([i in kept_ids for i in range(len(positions))])
    dropped_mask = ~kept_mask

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(
        ref_data, origin="lower", cmap="gray",
        vmin=vmin, vmax=vmax, interpolation="nearest",
    )

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


def plot_snr_vs_magnitude(df: pd.DataFrame, output_path: Path) -> None:
    """Save a per-source median SNR vs instrumental magnitude scatter plot.

    Parameters
    ----------
    df : DataFrame
        Photometry table with columns ``source_id``, ``mag``, and ``snr``.
    output_path : Path
        Destination file for the saved plot.
    """
    import matplotlib
    matplotlib.use(_MPL_BACKEND)
    import matplotlib.pyplot as plt

    summary = df.groupby("source_id").agg(
        mag_median=("mag", "median"),
        snr_median=("snr", "median"),
    ).dropna()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        summary["mag_median"], summary["snr_median"],
        s=15, alpha=0.7, color="steelblue",
    )
    ax.set_xlabel("Instrumental Magnitude (median over epochs)", fontsize=13)
    ax.set_ylabel("Median Per-Epoch SNR", fontsize=13)
    ax.set_title("Median Per-Epoch SNR vs Instrumental Magnitude", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved SNR vs magnitude plot → {output_path}")


def plot_temporal_snr_vs_magnitude(df: pd.DataFrame, output_path: Path) -> None:
    """Save a per-source temporal SNR vs instrumental magnitude scatter plot.

    Temporal SNR = median(flux) / std(flux) across all valid epochs.

    Parameters
    ----------
    df : DataFrame
        Photometry table with columns ``source_id``, ``mag``, and
        ``temporal_snr``.
    output_path : Path
        Destination file for the saved plot.
    """
    import matplotlib
    matplotlib.use(_MPL_BACKEND)
    import matplotlib.pyplot as plt

    summary = (
        df.groupby("source_id")
        .agg(mag_median=("mag", "median"), temporal_snr=("temporal_snr", "first"))
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        summary["mag_median"], summary["temporal_snr"],
        s=15, alpha=0.7, color="darkorange",
    )
    ax.set_xlabel("Instrumental Magnitude (median)", fontsize=13)
    ax.set_ylabel("Temporal SNR  [median flux / std flux]", fontsize=13)
    ax.set_title("Temporal SNR vs Instrumental Magnitude", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved temporal SNR vs magnitude plot → {output_path}")


def plot_lightcurves(df: pd.DataFrame, output_dir: Path) -> None:
    """Save one flux light-curve plot per source into *output_dir*.

    Uses the ``obs_time`` column for the x-axis when valid datetimes are
    present; otherwise falls back to an integer epoch index.

    Parameters
    ----------
    df : DataFrame
        Photometry table with columns ``source_id``, ``filename``,
        ``obs_time``, ``flux``, ``flux_err``, ``x_ref``, ``y_ref``.
    output_dir : Path
        Directory into which ``source_NNNN.png`` files are written
        (created if it does not exist).
    """
    import matplotlib
    matplotlib.use(_MPL_BACKEND)
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    output_dir.mkdir(parents=True, exist_ok=True)

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


def plot_differential_lightcurves(df: pd.DataFrame, output_dir: Path) -> None:
    """Save one differential magnitude light-curve plot per source into *output_dir*.

    The plot shows ``diff_mag ± diff_mag_err`` vs observation time (or epoch
    index).  Reference stars are indicated in the plot subtitle.

    Parameters
    ----------
    df : DataFrame
        Output of :func:`src.differential.compute_differential_lightcurves`.
        Must contain columns ``source_id``, ``filename``, ``obs_time``,
        ``diff_mag``, ``diff_mag_err``, ``is_reference``, plus optionally
        ``x_ref`` and ``y_ref``.
    output_dir : Path
        Directory into which ``source_NNNN_diff.png`` files are written
        (created if it does not exist).
    """
    import matplotlib
    matplotlib.use(_MPL_BACKEND)
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    use_datetime = False
    if "obs_time" in df.columns:
        df["obs_dt"] = pd.to_datetime(df["obs_time"], errors="coerce")
        valid_times = df["obs_dt"].notna().sum()
        use_datetime = valid_times > 0
        if not use_datetime:
            log.warning(
                "obs_time column present but no parseable datetimes; using epoch index."
            )

    if not use_datetime:
        filenames = sorted(df["filename"].unique())
        fname_to_idx = {f: i for i, f in enumerate(filenames)}
        df["epoch"] = df["filename"].map(fname_to_idx)

    source_ids = sorted(df["source_id"].unique())
    log.info(f"Saving {len(source_ids)} differential light curve plots...")

    for sid in source_ids:
        src = df[df["source_id"] == sid].sort_values(
            "obs_dt" if use_datetime else "epoch"
        )
        diff_mag = src["diff_mag"].values
        diff_mag_err = src["diff_mag_err"].values
        xvals = src["obs_dt"].values if use_datetime else src["epoch"].values

        valid = np.isfinite(diff_mag)
        if valid.sum() == 0:
            continue

        is_ref = bool(src["is_reference"].any()) if "is_reference" in src.columns else False

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.errorbar(
            xvals[valid], diff_mag[valid], yerr=diff_mag_err[valid],
            fmt="o", color="mediumseagreen" if is_ref else "steelblue",
            ecolor="lightgreen" if is_ref else "lightsteelblue",
            capsize=3, markersize=4, linewidth=0.8,
        )
        ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

        # Set y-axis limits from the data values only — error bar extents are
        # excluded so that large uncertainties on a few epochs don't collapse
        # the visible range.  Use median ± 4*MAD with a minimum half-range of
        # 0.05 mag so a perfectly flat source still gets sensible axes.
        valid_mags = diff_mag[valid]
        med = np.nanmedian(valid_mags)
        mad = np.nanmedian(np.abs(valid_mags - med))
        half_range = max(4.0 * mad, 0.05)
        ax.set_ylim(med - half_range * 1.5, med + half_range * 1.5)

        # Secondary y-axis: relative flux differential  δF/F = 10^(−Δm/2.5) − 1
        # forward : Δm  → δF/F
        # inverse : δF/F → Δm
        def _mag_to_relflux(dm):
            return np.power(10.0, -np.asarray(dm) / 2.5) - 1.0

        def _relflux_to_mag(f):
            # Protect against log of zero/negative
            fv = np.asarray(f)
            with np.errstate(invalid="ignore", divide="ignore"):
                return -2.5 * np.log10(np.where(fv > -1.0, 1.0 + fv, np.nan))

        ax2 = ax.secondary_yaxis("right", functions=(_mag_to_relflux, _relflux_to_mag))
        ax2.set_ylabel("Relative Flux Differential  (δF/F)", fontsize=12)
        # Format secondary ticks as percentages for readability
        ax2.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v * 100:.2g}%")
        )

        if use_datetime:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
            fig.autofmt_xdate(rotation=30, ha="right")
            ax.set_xlabel("Observation Time (UTC)", fontsize=12)
        else:
            ax.set_xlabel("Epoch (image index)", fontsize=12)

        ax.set_ylabel("Differential Magnitude", fontsize=12)

        x_ref = src["x_ref"].iloc[0] if "x_ref" in src.columns else float("nan")
        y_ref = src["y_ref"].iloc[0] if "y_ref" in src.columns else float("nan")
        ref_label = "  [reference star]" if is_ref else ""
        ax.set_title(
            f"Source {sid:04d}  (x={x_ref:.1f}, y={y_ref:.1f}){ref_label}",
            fontsize=13,
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_dir / f"source_{sid:04d}_diff.png", dpi=100)
        plt.close(fig)

    log.info(f"Differential light curve plots saved to {output_dir}")


def plot_stacked_image(stack: np.ndarray, output_path: Path) -> None:
    """Save a mean-stacked image as a PNG with ZScale stretch.

    Parameters
    ----------
    stack : ndarray
        2-D mean-stacked science image (may contain NaN for uncovered pixels).
    output_path : Path
        Destination PNG file path.
    """
    import matplotlib
    matplotlib.use(_MPL_BACKEND)
    import matplotlib.pyplot as plt
    from astropy.visualization import ZScaleInterval

    interval = ZScaleInterval()
    finite = stack[np.isfinite(stack)]
    if finite.size > 0:
        vmin, vmax = interval.get_limits(finite)
    else:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(
        stack, origin="lower", cmap="gray",
        vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    ax.set_title("Mean-Stacked Image", fontsize=14)
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved mean-stacked image → {output_path}")
