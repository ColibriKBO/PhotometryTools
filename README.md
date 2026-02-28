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
    "centroid_box_radius": 5
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

Rounding the shift to an integer leaves a sub-pixel residual: sources in the shifted frame are not guaranteed to lie exactly at the reference-frame positions `(x_ref + dx, y_ref + dy)`. To account for this, each source is **re-centroided** in the shifted frame:

1. The nominal expected position after the integer shift is computed for every source.
2. A square cutout of half-width `centroid_box_radius` pixels is extracted around each expected position.
3. A **centre-of-mass centroid** (`photutils.centroids.centroid_com`) is computed on the cutout to obtain the refined sub-pixel position.
4. Sources whose expected position falls outside the image boundary, or where the centroid is non-finite (e.g. an entirely NaN cutout at a frame edge), are assigned NaN positions and their measurements are set to NaN for that epoch.

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
