# PhotometryTools

A command-line tool for aperture photometry on stacks of calibrated astronomical FITS images taken with the Colibri Telescope Array. Given a directory of images, it detects all point sources in a reference frame, optionally aligns subsequent frames to it, performs sky-subtracted aperture photometry on every source in every image, and writes a CSV table of light curves ready for further analysis.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Command-Line Reference](#command-line-reference)
5. [Configuration Reference](#configuration-reference)
6. [Output Files](#output-files)
7. [Photometry Procedure — Detailed Description](#photometry-procedure--detailed-description)
8. [Differential Photometry Procedure — Detailed Description](#differential-photometry-procedure--detailed-description)

---

## Requirements

| Package | Purpose |
|---|---|
| `numpy` | Array operations |
| `pandas` | Tabular output (CSV) |
| `astropy` | FITS I/O, time conversion, visualisation scaling |
| `photutils` | Source detection (`DAOStarFinder`), aperture photometry, centroiding |
| `astroalign` | Image-to-image alignment (transform estimation) |
| `scipy` | Integer-pixel image shifting (no interpolation) |
| `matplotlib` | Diagnostic plots |

Python **3.10+** is required (uses `X | Y` union type hints).

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/PhotometryTools.git
cd PhotometryTools

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Minimal run — source detection + photometry, no plots
python aperture_photometry.py \
    --input demo_data/CN2025A-65/2025-05-06/calibrated-bsub-science-images/

# With diagnostic plots and a custom output directory
python aperture_photometry.py \
    --input  demo_data/CN2025A-65/2025-05-06/calibrated-bsub-science-images/ \
    --output-dir demo_output \
    --plots

# Skip image alignment (if frames are already on the same pixel grid)
python aperture_photometry.py \
    --input demo_data/CN2025A-65/2025-05-06/calibrated-bsub-science-images/ \
    --no-align \
    --plots

# Use a custom configuration file
python aperture_photometry.py \
    --input /path/to/fits/ \
    --config /path/to/my_config.json \
    --output-dir /path/to/results/ \
    --plots
```

### Differential photometry

```bash
# Minimal run — weighted-mean reference, no plots
python differential_photometry.py \
    --input demo_output/photometry.csv

# With diagnostic plots and an explicit output directory
python differential_photometry.py \
    --input  demo_output/photometry.csv \
    --output-dir demo_output \
    --plots

# Override the reference-star SNR threshold and construction method
python differential_photometry.py \
    --input demo_output/photometry.csv \
    --method weighted_mean \
    --min-ref-snr 15 \
    --plots

# Disable iterative sigma-clipping of reference stars
python differential_photometry.py \
    --input demo_output/photometry.csv \
    --sigma-clip 0
```

---

## Command-Line Reference

```
python aperture_photometry.py --input <dir> [OPTIONS]
```

| Flag | Required | Default | Description |
|---|---|---|---|
| `--input <dir>` | Yes | — | Directory containing FITS files (`*.fits`, `*.fit`, `*.FITS`, `*.FIT`). Files are sorted alphabetically; the first file becomes the reference image. |
| `--config <json>` | No | `config/default_config.json` | Path to a JSON configuration file. See [Configuration Reference](#configuration-reference). |
| `--output-dir <dir>` | No | `photometry_output` | Directory where `photometry.csv` and (optionally) plots are written. Created automatically if it does not exist. |
| `--plots` | No | off | When set, generates diagnostic images: `reference_image_sources.png`, `snr_vs_magnitude.png`, `temporal_snr_vs_magnitude.png`, and per-source light curve PNGs under `lightcurves/`. |
| `--no-align` | No | off | Disables image alignment. Overrides `align: true` in the config file. |

### `differential_photometry.py`

```
python differential_photometry.py --input <csv> [OPTIONS]
```

| Flag | Required | Default | Description |
|---|---|---|---|
| `--input <csv>` | Yes | — | Path to `photometry.csv` produced by `aperture_photometry.py`. |
| `--config <json>` | No | `config/default_config.json` | Path to a JSON configuration file. |
| `--output-dir <dir>` | No | parent of input CSV | Directory where `differential_photometry.csv` and (optionally) plots are written. |
| `--method <method>` | No | `weighted_mean` (from config) | Reference-curve construction method. Currently only `weighted_mean` is available. |
| `--min-ref-snr <SNR>` | No | `10.0` (from config) | Minimum temporal SNR a star must have to qualify as a reference star. Overrides `min_reference_snr` in the config. |
| `--sigma-clip <σ>` | No | `3.0` (from config) | Sigma threshold for iterative leave-one-out reference-star sigma-clipping. Set to `0` to disable. Overrides `sigma_clip` in the config. |
| `--min-ref-stars <N>` | No | `3` (from config) | Minimum number of reference stars to retain during sigma-clipping. Clipping stops when the ensemble reaches this size. Overrides `min_ref_stars` in the config. |
| `--plots` | No | off | When set, saves differential light-curve PNG plots under `differential_lightcurves/`. |

---

## Configuration Reference

The default configuration is in [`config/default_config.json`](config/default_config.json):

```json
{
    "aperture_radius": 5.0,
    "annulus_r_in": 8.0,
    "annulus_r_out": 12.0,
    "detection_fwhm": 3.0,
    "detection_threshold": 5.0,
    "fits_extension": 0,
    "align": true,
    "coverage_min_fraction": 0.9,
    "gain": 1.0,
    "centroid_box_radius": 5,
    "diff_method": "weighted_mean",
    "min_reference_snr": 10.0,
    "sigma_clip": 3.0,
    "min_ref_stars": 3
}
```

| Key | Type | Description |
|---|---|---|
| `aperture_radius` | float (pixels) | Radius of the circular source aperture used to sum flux. |
| `annulus_r_in` | float (pixels) | Inner radius of the sky background annulus. Must be larger than `aperture_radius` to avoid source contamination. |
| `annulus_r_out` | float (pixels) | Outer radius of the sky background annulus. |
| `detection_fwhm` | float (pixels) | Expected FWHM of point sources passed to `DAOStarFinder`. Should approximately match the PSF width in your images. |
| `detection_threshold` | float (σ) | Detection threshold in units of the image background standard deviation. Only peaks above `threshold × σ` are accepted as sources. |
| `fits_extension` | int | FITS HDU extension index from which pixel data are read (0-indexed). |
| `align` | bool | Whether to align each image to the reference frame before photometry. Can be overridden on the command line with `--no-align`. |
| `coverage_min_fraction` | float (0–1) | Minimum fraction of images in which a source must yield a valid (non-NaN) flux to be retained. E.g. `0.9` requires a source to be measurable in at least 90% of frames. |
| `gain` | float (e⁻/ADU) | Detector gain used for Poisson noise estimation. Set to `1.0` if images are already in electrons. |
| `centroid_box_radius` | int (pixels) | Half-width of the search box used to re-centroid each source in every aligned frame (see Step 3). Increase for undersampled PSFs or large dither residuals; decrease to avoid cross-contamination in crowded fields. |
| `diff_method` | string | Reference-curve construction method used by `differential_photometry.py`. Currently only `"weighted_mean"` is supported. |
| `min_reference_snr` | float | Minimum temporal SNR a star must have to be included in the reference ensemble. |
| `sigma_clip` | float | Sigma threshold for iterative leave-one-out sigma-clipping of reference stars. Set to `0` to disable. |
| `min_ref_stars` | int | Minimum number of reference stars preserved during sigma-clipping. Clipping stops if the ensemble would fall below this count. |

---

## Output Files

### `photometry.csv`

One row per (source, image) pair. Columns:

| Column | Description |
|---|---|
| `source_id` | Integer index assigned to each detected source (0-based, stable across all images). |
| `x_ref` | X pixel coordinate of the source centroid in the reference image. |
| `y_ref` | Y pixel coordinate of the source centroid in the reference image. |
| `filename` | Basename of the FITS file for this measurement. |
| `obs_time` | Observation timestamp parsed from the FITS header (`DATE-OBS`, `MJD-OBS`, or `JD`), in ISO format. `NaN` if unavailable. |
| `flux` | Sky-subtracted source flux in ADU (or electrons if images are gain-corrected). |
| `flux_err` | 1-σ Poisson uncertainty on `flux`. |
| `mag` | Instrumental magnitude: −2.5 log₁₀(flux). No standard zero-point applied. |
| `mag_err` | 1-σ uncertainty on `mag` propagated from `flux_err`. |
| `snr` | Per-epoch signal-to-noise ratio: `flux / flux_err`. |
| `sky_bkg` | Median sky background per pixel estimated from the annulus for this source and epoch. |
| `temporal_snr` | Temporal SNR: median(flux) / std(flux) across all valid epochs for this source. The same value is repeated in every row belonging to a given source. |

### Diagnostic plots (`--plots`)

| File | Description |
|---|---|
| `reference_image_sources.png` | The reference FITS image (Z-scale stretch) with detected sources overlaid. Green circles = sources that passed the coverage filter; red circles = sources that were dropped. |
| `snr_vs_magnitude.png` | Per-epoch median SNR vs median instrumental magnitude for every surviving source. Useful for assessing photometric depth and saturation limits. |
| `temporal_snr_vs_magnitude.png` | Temporal SNR vs median magnitude. Sources significantly above the main locus are candidate variable stars or transients. |
| `lightcurves/source_NNNN.png` | One light-curve plot per source: sky-subtracted flux with 1-σ error bars vs observation time (or epoch index if timestamps are unavailable). |

### `differential_photometry.csv`

One row per (source, image) pair. All columns from `photometry.csv` are preserved; additional columns:

| Column | Description |
|---|---|
| `ref_mag` | Weighted-mean reference magnitude at this epoch, computed from the reference ensemble (with leave-one-out correction when the target is itself a reference star). |
| `ref_mag_err` | 1-σ uncertainty on `ref_mag` propagated from the individual reference-star magnitude errors. |
| `diff_mag` | Differential magnitude: `(mag − median(mag)) − ref_mag`. A stable star produces `diff_mag ≈ 0` at all epochs; a variable star shows a non-zero signal. |
| `diff_mag_err` | 1-σ uncertainty on `diff_mag`: $\sqrt{\sigma_m^2 + \sigma_{\text{ref}}^2}$. |
| `is_reference` | Boolean — `True` if this source is a member of the reference ensemble. |

### Differential photometry diagnostic plots (`--plots`)

| File | Description |
|---|---|
| `differential_lightcurves/source_NNNN.png` | One differential light-curve plot per source: `diff_mag` with 1-σ error bars vs observation time (or epoch index). Reference stars are labelled as such in the plot title. |

---

## Photometry Procedure — Detailed Description

This section gives a complete, step-by-step account of every operation performed by the pipeline so that the reduction strategy can be understood, reproduced, or critically evaluated without reading the source code.

### Step 1 — Input collection

All FITS files matching the extensions `*.fits`, `*.fit`, `*.FITS`, or `*.FIT` in the `--input` directory are collected and **sorted alphabetically by filename**. This ordering defines the time axis for the light curves; filenames that encode a timestamp or sequence number (as is standard observatory convention) therefore sort into chronological order naturally. The **first file** after sorting becomes the reference image for all subsequent steps.

### Step 2 — Source detection in the reference image

The pixel array of the reference image is loaded from the configured FITS HDU extension. A **sigma-clipped background estimate** is computed over the full image using iterative 3-σ rejection to robustly measure the sky `median` and standard deviation `σ`, while suppressing the contribution of bright stars and extended sources.

Source detection is then carried out with **`DAOStarFinder`** (`photutils`), which locates local maxima consistent with a 2-D circular Gaussian PSF of the specified FWHM. Only peaks whose height above the background median exceeds `detection_threshold × σ` are accepted. The output is a catalogue of `(x, y)` pixel centroids, assigned sequential integer `source_id` values starting from 0. **This catalogue is held fixed for all subsequent images** — no re-detection or re-centroiding is performed per frame; the same pixel positions are used throughout.

### Step 3 — Image alignment (per-frame, optional)

For each image after the reference, if `align = true`, the science frame is aligned onto the reference pixel grid. The process has two parts:

#### 3a — Integer-pixel shift (no interpolation)

1. Point sources are identified in both the science and reference frames by **`astroalign`**.
2. A geometric hash of source triangles is built, making the match invariant to rotation, scale, and translation.
3. The best-fit affine transformation is found. **Only the translation component is used**; rotation and scale differences between frames are assumed to be negligible for typical observatory dither patterns.
4. The translation vector is **rounded to the nearest integer pixel**. The image is then shifted by that exact integer offset using `scipy.ndimage.shift` at `order=0` (nearest-neighbour mode).

Because the shift is a whole number of pixels, each output pixel receives the value of exactly one input pixel — **no sub-pixel blending or interpolation is performed**. This avoids correlated noise between adjacent pixels that arises from resampling, and eliminates flux redistribution artefacts near the PSF core.

Pixels that fall **outside the overlap region** after the shift (e.g. sky area covered by one dither position but not another) are filled with **NaN** and tracked in a boolean `bad_mask`. Any aperture or sky annulus that overlaps even a single NaN pixel is flagged as invalid and its photometric measurement is set to NaN for that epoch. If alignment fails entirely (insufficient matched stars), **all measurements for that image are set to NaN** and a warning is emitted to the log.

#### 3b — Per-frame aperture re-centroiding

Rounding the shift to an integer leaves a sub-pixel residual: the source may not fall exactly on the nearest pixel. To account for this, each source is **re-centroided** in the shifted frame:

1. Because the integer shift moves each science-frame pixel at `(x_s, y_s)` to `(x_s + dx, y_s + dy)` ≈ `(x_ref, y_ref)`, sources in the shifted image land at approximately their **reference-frame positions**. The search box is therefore centred directly on `(x_ref, y_ref)`.
2. The pixel at `(x_ref, y_ref)` in the shifted image is checked: if it is NaN (i.e. it falls inside the border region introduced by the integer shift, meaning the science frame did not cover that part of the sky), the source is assigned a NaN position and its measurement is set to NaN for that epoch.
3. For sources in the valid region, a square cutout of half-width `centroid_box_radius` pixels is extracted around `(x_ref, y_ref)`.
4. A **centre-of-mass centroid** (`photutils.centroids.centroid_com`) is computed on the cutout to obtain the refined sub-pixel position.

The re-centroided positions are used **only for that individual frame**; the original reference-frame coordinates (`x_ref`, `y_ref`) are always stored in the output CSV so that sources can be identified consistently across epochs.

### Step 4 — Sky background estimation

For each source position, a **circular annulus** is placed concentrically around the source aperture, with inner radius `annulus_r_in` and outer radius `annulus_r_out` (both in pixels). The **median pixel value** within the annulus is taken as the local sky background per pixel, $B$, for that source and epoch.

Using the median rather than the mean makes the sky estimate robust against faint blended neighbours, cosmic-ray hits, and detector artefacts that may land within the annulus.

### Step 5 — Aperture flux measurement

A **circular aperture** of radius `aperture_radius` pixels is centred on each source position (the centroid determined in Step 2, unchanged across all epochs). The raw sum of all pixel values within the aperture is computed, $S_\text{raw}$.

The **sky-subtracted flux** is then:

$$F = S_\text{raw} - B \times A_\text{ap}$$

where $A_\text{ap} = \pi \, r_\text{ap}^2$ is the geometric area of the aperture in pixels and $B$ is the per-pixel sky background from Step 4. This removes the additive sky signal and any residual pedestal from the source count rate.

### Step 6 — Flux uncertainty

The 1-σ flux uncertainty is estimated from **Poisson statistics**:

$$\sigma_F = \sqrt{\dfrac{|F|}{g} + A_\text{ap} \cdot \dfrac{|B|}{g}}$$

where $g$ is the detector gain in e⁻/ADU. The first term represents Poisson noise from the source photons and the second represents Poisson noise from the sky photons collected within the aperture area. Read noise is not included in the current model; for read-noise-dominated regimes (very faint sources or low sky background) the formal uncertainties will be underestimates.

### Step 7 — Instrumental magnitudes

The sky-subtracted flux is converted to an **instrumental magnitude**:

$$m = -2.5 \log_{10}(F)$$

This is on an arbitrary relative scale with no standard photometric zero-point applied. For differential or relative photometry — comparing a target to field reference stars observed simultaneously — the absolute zero-point cancels and is unimportant.

The magnitude uncertainty is propagated from the flux SNR using the standard small-error approximation:

$$\sigma_m = \frac{2.5}{\ln 10} \cdot \frac{\sigma_F}{F} = \frac{1.0857}{\mathrm{SNR}}$$

Sources with non-positive sky-subtracted flux (saturated cores, blends, or very negative residual backgrounds) are assigned `NaN` for both `mag` and `mag_err`.

### Step 8 — Coverage filter

After all images have been processed, each source is evaluated for **temporal completeness**. The fraction of images in which a source yielded a valid (non-NaN) flux is computed. Any source whose valid-measurement fraction falls below `coverage_min_fraction` is **permanently removed** from the output. This eliminates sources that are:

- Located near chip edges or detector gaps that are not covered by every dither position,
- Consistently lost to alignment failures,
- Affected by bad columns or saturated bleeding trails in a large fraction of frames.

### Step 9 — Temporal SNR

For each surviving source, a **temporal SNR** is computed across all valid epochs:

$$\text{temporal SNR} = \frac{\text{median}(F_t)}{\sigma(F_t)}$$

where the median and standard deviation are taken over the time series $\{F_t\}$ of sky-subtracted fluxes. For a stable (non-variable) source this quantity is set by the photon-noise floor and scales approximately as $\sqrt{N_\text{epochs}}$; a source that varies intrinsically will have a suppressed temporal SNR relative to its brightness. This column is appended to every row in the output CSV and is plotted against magnitude in `temporal_snr_vs_magnitude.png` to support identification of variable sources.

### Step 10 — Output

The complete measurement table is written to `photometry.csv`. If `--plots` is specified, four categories of diagnostic figures are produced:

- **Reference image map** — the reference frame displayed with a Z-scale stretch, with circles marking sources colour-coded by whether they passed (green) or failed (red) the coverage filter.
- **Per-epoch SNR vs magnitude** — identifies the photon-noise floor, bright saturation limit, and any systematic noise floor.
- **Temporal SNR vs magnitude** — variable sources appear as outliers above the expected Poisson noise locus.
- **Individual light curves** — one PNG per source: sky-subtracted flux with 1-σ error bars as a function of ISO observation time (if available in the FITS headers) or integer epoch index (if not).

---

## Differential Photometry Procedure — Detailed Description

This section describes every step performed by `differential_photometry.py`. The script reads the `photometry.csv` produced by `aperture_photometry.py` and reduces the instrumental magnitude time series of each source to a **differential** (relative) light curve that is largely free of systematic trends caused by atmospheric transparency variations, flat-fielding residuals, or other epoch-correlated effects.

### Step 1 — Reference-star selection (SNR filter)

The first pass over the photometry table applies a **temporal SNR threshold** (`min_reference_snr`). Only sources whose `temporal_snr` — the ratio of the median to the standard deviation of the flux time series, computed in `aperture_photometry.py` — meets or exceeds the threshold are admitted to the candidate reference pool. Faint, noisy, or intrinsically variable sources are excluded at this stage because including them in the reference would corrupt the ensemble and degrade the differential correction for all targets.

### Step 2 — Iterative sigma-clipping of reference stars

The SNR cut provides a first approximation to a clean ensemble but cannot on its own identify stars that are marginally variable, blended, or affected by detector artefacts. A second pass uses an **iterative leave-one-out (LOO) sigma-clipping** procedure to remove remaining outliers:

1. For each candidate reference star *i*, a LOO reference curve is built from the remaining ensemble (all stars except *i*).
2. The differential residual of star *i* against its LOO curve is computed at every epoch, and the **RMS** of these residuals is recorded.
3. A robust outlier threshold is computed from the RMS distribution:

$$\text{threshold} = \text{median}(\mathrm{RMS}) + \sigma \times 1.4826 \times \mathrm{MAD}(\mathrm{RMS})$$

The factor 1.4826 makes the MAD a consistent estimator of σ for Gaussian data.

4. The star with the highest RMS is removed if it exceeds the threshold.
5. Steps 1–4 repeat until no star exceeds the threshold (convergence) **or** the ensemble reaches `min_ref_stars`.

A full audit log of every removal — including the star's source ID, its RMS, and the threshold at that iteration — is written to the console at `INFO` level. If `sigma_clip = 0` in the configuration (or `--sigma-clip 0` on the command line), this step is skipped entirely.

### Step 3 — SNR-squared weights

Each star in the final reference ensemble is assigned a **weight proportional to its squared temporal SNR**:

$$w_i = \frac{\mathrm{snr}_i^2}{\displaystyle\sum_j \mathrm{snr}_j^2}$$

Higher-SNR stars contribute more to the reference curve and, because the weights are squared, bright stable stars dominate over many faint ones. Weights are normalised to sum to 1 across the ensemble and are then renormalised **per epoch** to handle missing measurements (NaNs) at individual epochs gracefully, ensuring the reference is always computed from the available stars at each time step.

### Step 4 — Reference-curve construction

For each epoch *e* the **weighted-mean reference magnitude** is:

$$m_{\text{ref}}(e) = \frac{\sum_i w_i(e)\, \delta m_i(e)}{\sum_i w_i(e)}$$

where $\delta m_i(e) = m_i(e) - \mathrm{median}_t(m_i)$ is the magnitude of reference star *i* expressed as a **deviation from its own temporal median**. Subtracting each star's median before combining removes the arbitrary brightness offset between ensemble members, so the resulting reference curve captures only the **common-mode temporal variation** of the sky (atmospheric extinction, transparency fluctuations, etc.) rather than an average absolute brightness.

The propagated uncertainty on the reference curve is:

$$\sigma_{\text{ref}}(e) = \sqrt{\sum_i \tilde{w}_i(e)^2\, \sigma_{m,i}(e)^2}$$

where $\tilde{w}_i(e)$ are the epoch-renormalised weights.

### Step 5 — Leave-one-out correction for reference stars

When the **target star is itself a member of the reference ensemble**, using the full reference curve would introduce a spurious self-correlation: any variation in the target would appear in both the target light curve and the reference, causing the differential signal to be suppressed. This is avoided by applying a **leave-one-out (LOO) correction**: the reference curve used for target *t* has star *t*'s contribution analytically removed without rebuilding the ensemble from scratch:

$$m_{\text{ref,loo}}^{(t)}(e) = \frac{W(e)\, m_{\text{ref}}(e) - w_t(e)\, \delta m_t(e)}{W(e) - w_t(e)}$$

where $W(e) = \sum_j w_j(e)$ is the total effective weight at epoch *e*. If removing star *t* would leave the ensemble empty at a given epoch ($W(e) - w_t(e) = 0$), that epoch is assigned a NaN reference magnitude. Targets that are **not** in the reference pool receive the full (un-modified) reference curve unchanged.

### Step 6 — Differential light curve computation

For every source *t* and epoch *e*, the differential magnitude is:

$$\Delta m_t(e) = \bigl(m_t(e) - \mathrm{median}_e(m_t)\bigr) - m_{\text{ref,loo}}^{(t)}(e)$$

Subtracting the target's own temporal median places it in the same "deviation from mean" space as the reference curve (which was built from per-star medians in Step 4). For a **perfectly stable star**, $\Delta m_t(e) \approx 0$ at all epochs. For a **variable source**, $\Delta m_t(e)$ will show the intrinsic brightness changes after the common-mode atmospheric systematics have been removed.

The uncertainty is quadrature-summed from the per-epoch photometric error and the reference-curve error:

$$\sigma_{\Delta m}(e) = \sqrt{\sigma_{m_t}(e)^2 + \sigma_{\text{ref}}(e)^2}$$

Epochs where either $m_t$ or $m_{\text{ref}}$ is NaN are set to NaN in both `diff_mag` and `diff_mag_err`.

### Step 7 — Output

The differential photometry table is written to `differential_photometry.csv`. It contains all columns from the input `photometry.csv` plus `ref_mag`, `ref_mag_err`, `diff_mag`, `diff_mag_err`, and `is_reference`. If `--plots` is specified, one PNG per source is saved under `differential_lightcurves/`: each plot shows `diff_mag` with 1-σ error bars as a function of observation time (or epoch index if timestamps are unavailable), with reference stars labelled in the plot title.
