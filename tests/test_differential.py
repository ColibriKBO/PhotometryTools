"""
Tests for src/differential.py

Following the class-based pytest pattern used in test_alignment.py.
Each class targets a single public function; synthetic DataFrames are built
inline so the tests have no file-system dependencies.
"""

import numpy as np
import pandas as pd
import pytest

from src.differential import (
    REFERENCE_METHODS,
    build_weighted_mean_reference,
    compute_differential_lightcurves,
    compute_snr_weights,
    select_reference_stars,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_photometry_df(
    n_sources: int = 4,
    n_epochs: int = 5,
    base_mag: float = 10.0,
    mag_step: float = 1.0,
    temporal_snrs: list[float] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a minimal synthetic photometry DataFrame."""
    rng = np.random.default_rng(seed)
    if temporal_snrs is None:
        temporal_snrs = [float(10 * (i + 1)) for i in range(n_sources)]

    filenames = [f"image_{e:04d}.fits" for e in range(n_epochs)]
    obs_times = [f"2025-01-01T00:{e:02d}:00" for e in range(n_epochs)]

    records = []
    for sid in range(n_sources):
        tsnr = temporal_snrs[sid]
        for e in range(n_epochs):
            mag = base_mag + sid * mag_step + rng.normal(0, 0.01)
            records.append(
                {
                    "source_id": sid,
                    "x_ref": float(50 + sid * 10),
                    "y_ref": float(50 + sid * 10),
                    "filename": filenames[e],
                    "obs_time": obs_times[e],
                    "flux": 10 ** (-mag / 2.5),
                    "flux_err": 0.01,
                    "mag": mag,
                    "mag_err": 0.01,
                    "snr": 100.0,
                    "sky_bkg": 1.0,
                    "temporal_snr": tsnr,
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# TestSelectReferenceStars
# ---------------------------------------------------------------------------


class TestSelectReferenceStars:
    def test_returns_sources_above_threshold(self):
        df = make_photometry_df(
            n_sources=4, temporal_snrs=[5.0, 10.0, 15.0, 20.0]
        )
        ref_ids = select_reference_stars(df, min_temporal_snr=10.0)
        assert set(ref_ids) == {1, 2, 3}

    def test_exact_threshold_is_inclusive(self):
        df = make_photometry_df(n_sources=2, temporal_snrs=[10.0, 9.9])
        ref_ids = select_reference_stars(df, min_temporal_snr=10.0)
        assert ref_ids == [0]

    def test_returns_sorted_list(self):
        df = make_photometry_df(
            n_sources=3, temporal_snrs=[30.0, 20.0, 10.0]
        )
        ref_ids = select_reference_stars(df, min_temporal_snr=10.0)
        assert ref_ids == sorted(ref_ids)

    def test_raises_when_no_stars_qualify(self):
        df = make_photometry_df(n_sources=3, temporal_snrs=[1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="No reference stars found"):
            select_reference_stars(df, min_temporal_snr=100.0)

    def test_raises_on_missing_temporal_snr_column(self):
        df = make_photometry_df(n_sources=2)
        df = df.drop(columns=["temporal_snr"])
        with pytest.raises(ValueError, match="missing 'temporal_snr'"):
            select_reference_stars(df)

    def test_all_sources_qualify(self):
        df = make_photometry_df(n_sources=3, temporal_snrs=[50.0, 60.0, 70.0])
        ref_ids = select_reference_stars(df, min_temporal_snr=10.0)
        assert len(ref_ids) == 3


# ---------------------------------------------------------------------------
# TestComputeSnrWeights
# ---------------------------------------------------------------------------


class TestComputeSnrWeights:
    def test_weights_sum_to_one(self):
        df = make_photometry_df(n_sources=3, temporal_snrs=[10.0, 20.0, 30.0])
        weights = compute_snr_weights(df, reference_ids=[0, 1, 2])
        assert pytest.approx(sum(weights.values()), rel=1e-9) == 1.0

    def test_higher_snr_gets_higher_weight(self):
        df = make_photometry_df(n_sources=2, temporal_snrs=[10.0, 30.0])
        weights = compute_snr_weights(df, reference_ids=[0, 1])
        assert weights[1] > weights[0]

    def test_equal_snr_gives_equal_weights(self):
        df = make_photometry_df(n_sources=3, temporal_snrs=[10.0, 10.0, 10.0])
        weights = compute_snr_weights(df, reference_ids=[0, 1, 2])
        np.testing.assert_allclose(list(weights.values()), [1 / 3, 1 / 3, 1 / 3])

    def test_weight_ratio_matches_snr_squared_ratio(self):
        df = make_photometry_df(n_sources=2, temporal_snrs=[3.0, 4.0])
        weights = compute_snr_weights(df, reference_ids=[0, 1])
        expected_ratio = (3.0 ** 2) / (4.0 ** 2)
        actual_ratio = weights[0] / weights[1]
        assert pytest.approx(actual_ratio, rel=1e-9) == expected_ratio


# ---------------------------------------------------------------------------
# TestBuildWeightedMeanReference
# ---------------------------------------------------------------------------


class TestBuildWeightedMeanReference:
    def test_output_columns(self):
        df = make_photometry_df(n_sources=3, n_epochs=4)
        ref_curve = build_weighted_mean_reference(df, reference_ids=[0, 1, 2])
        for col in ("filename", "obs_time", "ref_mag", "ref_mag_err"):
            assert col in ref_curve.columns, f"Missing column: {col}"

    def test_one_row_per_epoch(self):
        n_epochs = 5
        df = make_photometry_df(n_sources=3, n_epochs=n_epochs)
        ref_curve = build_weighted_mean_reference(df, reference_ids=[0, 1, 2])
        assert len(ref_curve) == n_epochs

    def test_ref_mag_is_weighted_average(self):
        """With a single reference star, ref_mag must equal that star's mag."""
        df = make_photometry_df(n_sources=2, n_epochs=3, temporal_snrs=[20.0, 5.0])
        # Use only source 0 as the reference
        ref_curve = build_weighted_mean_reference(df, reference_ids=[0])
        src0 = df[df["source_id"] == 0].set_index("filename")["mag"]
        for _, row in ref_curve.iterrows():
            expected = src0.loc[row["filename"]]
            assert pytest.approx(row["ref_mag"], rel=1e-6) == expected

    def test_ref_mag_err_propagation(self):
        """With two equal-weight stars the propagated error should be
        sqrt(2) * (0.5 * mag_err), i.e. mag_err / sqrt(2)."""
        df = make_photometry_df(
            n_sources=2, n_epochs=3, temporal_snrs=[10.0, 10.0]
        )
        # Force identical mag_err = 0.02 for clarity
        df["mag_err"] = 0.02
        ref_curve = build_weighted_mean_reference(df, reference_ids=[0, 1])
        # w_i = 0.5 each; err = sqrt((0.5*0.02)^2 + (0.5*0.02)^2) = 0.02/sqrt(2)
        expected_err = 0.02 / np.sqrt(2)
        np.testing.assert_allclose(
            ref_curve["ref_mag_err"].values, expected_err, rtol=1e-5
        )

    def test_nan_in_one_reference_is_handled(self):
        """If one reference source has NaN mag in an epoch, that epoch's
        ref_mag should still be finite (built from the remaining source)."""
        df = make_photometry_df(n_sources=2, n_epochs=3)
        # Inject NaN into source 0 for the first epoch
        mask = (df["source_id"] == 0) & (df["filename"] == "image_0000.fits")
        df.loc[mask, "mag"] = np.nan
        ref_curve = build_weighted_mean_reference(df, reference_ids=[0, 1])
        epoch0 = ref_curve[ref_curve["filename"] == "image_0000.fits"]
        assert np.isfinite(epoch0["ref_mag"].values[0])

    def test_all_nan_epoch_produces_nan(self):
        """If ALL reference sources have NaN in an epoch, ref_mag must be NaN."""
        df = make_photometry_df(n_sources=2, n_epochs=3)
        mask = df["filename"] == "image_0000.fits"
        df.loc[mask, "mag"] = np.nan
        ref_curve = build_weighted_mean_reference(df, reference_ids=[0, 1])
        epoch0 = ref_curve[ref_curve["filename"] == "image_0000.fits"]
        assert np.isnan(epoch0["ref_mag"].values[0])


# ---------------------------------------------------------------------------
# TestComputeDifferentialLightcurves
# ---------------------------------------------------------------------------


class TestComputeDifferentialLightcurves:
    def test_output_columns(self):
        df = make_photometry_df(n_sources=3, n_epochs=4)
        diff = compute_differential_lightcurves(df, reference_ids=[0, 1])
        for col in (
            "source_id", "filename", "obs_time",
            "mag", "mag_err", "ref_mag", "ref_mag_err",
            "diff_mag", "diff_mag_err", "is_reference",
        ):
            assert col in diff.columns, f"Missing column: {col}"

    def test_row_count_matches_input(self):
        df = make_photometry_df(n_sources=4, n_epochs=5)
        diff = compute_differential_lightcurves(df, reference_ids=[0, 1])
        assert len(diff) == len(df)

    def test_diff_mag_formula(self):
        """diff_mag must equal mag - ref_mag row-by-row."""
        df = make_photometry_df(n_sources=3, n_epochs=3)
        diff = compute_differential_lightcurves(df, reference_ids=[0, 1, 2])
        valid = diff["diff_mag"].notna() & diff["ref_mag"].notna() & diff["mag"].notna()
        np.testing.assert_allclose(
            diff.loc[valid, "diff_mag"].values,
            (diff.loc[valid, "mag"] - diff.loc[valid, "ref_mag"]).values,
            rtol=1e-9,
        )

    def test_is_reference_flag(self):
        df = make_photometry_df(n_sources=4, n_epochs=3)
        reference_ids = [0, 1]
        diff = compute_differential_lightcurves(df, reference_ids=reference_ids)
        for sid in reference_ids:
            assert diff[diff["source_id"] == sid]["is_reference"].all()
        for sid in [2, 3]:
            assert not diff[diff["source_id"] == sid]["is_reference"].any()

    def test_loo_reference_star_excludes_itself(self):
        """A reference star's diff_mag must not include its own flux in the
        reference curve.  With only two reference stars and perfectly constant
        magnitudes, each star's differential curve should be flat and its
        ref_mag should equal the OTHER star's mag, not the weighted mean of
        both."""
        # Two reference stars with equal SNR so equal weights, 5 epochs
        df = make_photometry_df(
            n_sources=2, n_epochs=5, temporal_snrs=[20.0, 20.0], seed=0
        )
        # Give each star a perfectly constant mag (no noise) for clarity
        df.loc[df["source_id"] == 0, "mag"] = 10.0
        df.loc[df["source_id"] == 1, "mag"] = 12.0

        diff = compute_differential_lightcurves(df, reference_ids=[0, 1])

        # For source 0: LOO reference = source 1's mag = 12.0
        # diff_mag = 10.0 - 12.0 = -2.0
        src0 = diff[diff["source_id"] == 0]
        np.testing.assert_allclose(
            src0["ref_mag"].dropna().values, 12.0, rtol=1e-6,
            err_msg="Source 0 ref_mag should equal source 1's mag (LOO)",
        )
        np.testing.assert_allclose(
            src0["diff_mag"].dropna().values, -2.0, rtol=1e-6,
        )

        # For source 1: LOO reference = source 0's mag = 10.0
        # diff_mag = 12.0 - 10.0 = +2.0
        src1 = diff[diff["source_id"] == 1]
        np.testing.assert_allclose(
            src1["ref_mag"].dropna().values, 10.0, rtol=1e-6,
            err_msg="Source 1 ref_mag should equal source 0's mag (LOO)",
        )
        np.testing.assert_allclose(
            src1["diff_mag"].dropna().values, 2.0, rtol=1e-6,
        )

    def test_non_reference_target_uses_full_ensemble(self):
        """A target NOT in the reference pool should use the full ensemble
        (no LOO), so its ref_mag equals the weighted mean of all ref stars."""
        df = make_photometry_df(
            n_sources=3, n_epochs=4, temporal_snrs=[20.0, 20.0, 5.0], seed=1
        )
        df.loc[df["source_id"] == 0, "mag"] = 10.0
        df.loc[df["source_id"] == 1, "mag"] = 12.0
        df.loc[df["source_id"] == 2, "mag"] = 15.0  # non-reference target

        diff = compute_differential_lightcurves(df, reference_ids=[0, 1])

        # With equal SNRs (both 20.0), equal weights => ref_mag = (10+12)/2 = 11.0
        src2 = diff[diff["source_id"] == 2]
        np.testing.assert_allclose(
            src2["ref_mag"].dropna().values, 11.0, rtol=1e-6,
        )
        np.testing.assert_allclose(
            src2["diff_mag"].dropna().values, 4.0, rtol=1e-6,
        )

    def test_nan_propagation(self):
        """If mag is NaN for a source in an epoch, diff_mag must also be NaN."""
        df = make_photometry_df(n_sources=3, n_epochs=4)
        mask = (df["source_id"] == 2) & (df["filename"] == "image_0001.fits")
        df.loc[mask, "mag"] = np.nan
        diff = compute_differential_lightcurves(df, reference_ids=[0, 1])
        bad = diff[(diff["source_id"] == 2) & (diff["filename"] == "image_0001.fits")]
        assert np.isnan(bad["diff_mag"].values[0])

    def test_raises_on_unknown_method(self):
        df = make_photometry_df(n_sources=2, n_epochs=2)
        with pytest.raises(ValueError, match="Unknown differential photometry method"):
            compute_differential_lightcurves(df, reference_ids=[0], method="pca")

    def test_reference_methods_registry_contains_weighted_mean(self):
        assert "weighted_mean" in REFERENCE_METHODS
        assert callable(REFERENCE_METHODS["weighted_mean"])
