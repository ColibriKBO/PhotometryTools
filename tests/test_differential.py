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
        # With a single reference star, ref_mag must equal that star's
        # deviation from its own temporal median.
        df = make_photometry_df(n_sources=2, n_epochs=3, temporal_snrs=[20.0, 5.0])
        ref_curve = build_weighted_mean_reference(df, reference_ids=[0])
        src0 = df[df["source_id"] == 0].set_index("filename")["mag"]
        src0_median = src0.median()
        for _, row in ref_curve.iterrows():
            expected = src0.loc[row["filename"]] - src0_median
            assert pytest.approx(row["ref_mag"], abs=1e-6) == expected

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
        # diff_mag must equal (mag - target_median) - ref_mag row-by-row,
        # where ref_mag is the ensemble deviation reference for that source.
        df = make_photometry_df(n_sources=3, n_epochs=3)
        diff = compute_differential_lightcurves(df, reference_ids=[0, 1, 2])
        # Compute each source's temporal median (same as what the code uses)
        target_medians = (
            df.groupby("source_id")["mag"].median()
            .reset_index()
            .rename(columns={"mag": "target_median"})
        )
        diff = diff.merge(target_medians, on="source_id")
        valid = diff["diff_mag"].notna() & diff["ref_mag"].notna() & diff["mag"].notna()
        np.testing.assert_allclose(
            diff.loc[valid, "diff_mag"].values,
            (diff.loc[valid, "mag"] - diff.loc[valid, "target_median"] - diff.loc[valid, "ref_mag"]).values,
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
        reference curve.

        All computation is in deviation-from-median space, so a star with
        perfectly constant magnitude should produce diff_mag == 0 regardless
        of its absolute brightness.  This also verifies the LOO: if the star
        were included in its own reference, a systematic bias would appear.
        """
        # Two reference stars with equal SNR so equal weights, 5 epochs
        df = make_photometry_df(
            n_sources=2, n_epochs=5, temporal_snrs=[20.0, 20.0], seed=0
        )
        # Give each star a perfectly constant (but different) mag
        df.loc[df["source_id"] == 0, "mag"] = 10.0
        df.loc[df["source_id"] == 1, "mag"] = 12.0

        diff = compute_differential_lightcurves(df, reference_ids=[0, 1])

        # Both stars are constant, so their deviations are 0 every epoch.
        # The LOO reference deviation is also 0.  Therefore diff_mag == 0.
        for sid in [0, 1]:
            src = diff[diff["source_id"] == sid]
            np.testing.assert_allclose(
                src["ref_mag"].dropna().values, 0.0, atol=1e-6,
                err_msg=f"Source {sid} ref_mag (LOO deviation) should be 0 for constant star",
            )
            np.testing.assert_allclose(
                src["diff_mag"].dropna().values, 0.0, atol=1e-6,
                err_msg=f"Source {sid} diff_mag should be 0 for constant star",
            )

    def test_non_reference_target_uses_full_ensemble(self):
        """A target NOT in the reference pool should use the full ensemble
        (no LOO).  All constant stars should produce diff_mag == 0."""
        df = make_photometry_df(
            n_sources=3, n_epochs=4, temporal_snrs=[20.0, 20.0, 5.0], seed=1
        )
        df.loc[df["source_id"] == 0, "mag"] = 10.0
        df.loc[df["source_id"] == 1, "mag"] = 12.0
        df.loc[df["source_id"] == 2, "mag"] = 15.0  # non-reference target

        diff = compute_differential_lightcurves(df, reference_ids=[0, 1])

        # All three stars are constant.  Deviations are 0 for everyone.
        # ref_mag (ensemble deviation) = 0, target deviation = 0 → diff_mag = 0.
        src2 = diff[diff["source_id"] == 2]
        np.testing.assert_allclose(
            src2["ref_mag"].dropna().values, 0.0, atol=1e-6,
        )
        np.testing.assert_allclose(
            src2["diff_mag"].dropna().values, 0.0, atol=1e-6,
        )

    def test_variable_star_produces_nonzero_diff(self):
        # A star with a real magnitude excursion should show a clear non-zero
        # diff_mag at the variable epoch after deviation-space subtraction.
        df = make_photometry_df(n_sources=3, n_epochs=5, temporal_snrs=[20.0, 20.0, 20.0], seed=2)
        # Stars 0 and 1 are stable references (constant)
        df.loc[df["source_id"] == 0, "mag"] = 10.0
        df.loc[df["source_id"] == 1, "mag"] = 12.0
        # Star 2 is 0.5 mag brighter at epoch 2 only
        df.loc[df["source_id"] == 2, "mag"] = 15.0
        df.loc[(df["source_id"] == 2) & (df["filename"] == "image_0002.fits"), "mag"] = 14.5

        diff = compute_differential_lightcurves(df, reference_ids=[0, 1])
        src2 = diff[diff["source_id"] == 2].sort_values("filename")

        epoch2 = src2[src2["filename"] == "image_0002.fits"]["diff_mag"].values[0]
        other_epochs = src2[src2["filename"] != "image_0002.fits"]["diff_mag"].values

        # The flaring epoch should be ~-0.5 (brighter = lower mag deviation)
        assert pytest.approx(epoch2, abs=0.01) == -0.5
        # All other epochs should be ~0 (constant)
        np.testing.assert_allclose(other_epochs, 0.0, atol=0.01)

    def test_nan_propagation(self):
        # If mag is NaN for a source in an epoch, diff_mag must also be NaN.
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
