# FlowCode plugin for Honeychrome

FlowCode debarcoding with FRET and spillover correction for spectral flow cytometry. The plugin is a Honeychrome tab (`FlowCode`) and a Python port of the FlowCodeUnmix 0.6.0 AF-matched staged pipeline, extended with a through-origin error model, a too-few-tag rescue and a tandem-breakdown check.

Contents

1. [What it is](#1-what-it-is)
2. [Setup](#2-setup)
3. [Using it in the app](#3-using-it-in-the-app)
4. [Outputs](#4-outputs)
5. [Mechanics: debarcoding and FRET spillover correction](#5-mechanics-debarcoding-and-fret-spillover-correction)
6. [Differences from the R package](#6-differences-from-the-r-package)

---

## 1. What it is

A FlowCode panel labels each cell with three epitope tags out of a set of 8-14
(HA, V5, AU1, AU5, S, C, Flag, NWS, VSVg, Ollas), each detected with its own 
fluorophore. A cell's barcode is the set of three tags that are positive. 
Two things make this hard on a spectral cytometer:

- **Debarcoding.** Each cell must be assigned to one of up to C(10, 3) = 120 
combinations from three-out-of-ten positivity calls, with a single threshold per tag.
- **FRET and spillover.** When the three tags sit on the same cell, donor-to-acceptor 
FRET and residual spillover add signal to the other fluorophore channels. The 
error grows with the product of donor and acceptor abundance, so a combination's 
error is not a fixed spectrum.

The plugin does five jobs, one per tab:

| Tab | Job |
|---|---|
| 1. Inputs | Choose the backbone samples (cells that carry every combination), 
the matched unstained sample and its AF library |
| 2. Tags & Combos | Pick the tag channels and define the combinations |
| 3. Gates & Thresholds | Scatter gates on the backbone and a positivity threshold
per tag |
| 4. Debarcode & FRET | Characterise the unmixing error of every combination on 
the backbone and show before/after plots |
| 5. Unmix & Export | Unmix the selected samples with the correction and write FCS files with a `FlowCode` channel and one `FC <name>` channel per combination |

The plugin reuses Honeychrome's AutoSpectral machinery: per-cell autofluorescence 
(AF) unmixing, AF libraries from the AutoSpectral AF tab, and the compiled 
AutoSpectral Optimization kernel for the final unmix.

### Files

| File | Role |
|---|---|
| `flowcode_unmix_tab.py` | The tab (UI and workers). The only file Honeychrome discovers |
| `flowcode_core.py` | Numerics, no Qt: debarcoding, rescues, error models, tandem check, staged unmixing |
| `flowcode_state.py` | Experiment state and persistence |
| `flowcode_plots.py` | Section 4 figures |
| `flowcode_rescue_assess.py` | Optional standalone script that assesses the too-few rescue. Not loaded by the plugin |

---

## 2. Setup

### 2.1 Requirements

- Honeychrome run from Python (the PyPI package or a source clone, Python 3.12 
or higher). The plugin system is not available in the packaged executables.
- No packages beyond Honeychrome's own requirements (numpy, scipy, scikit-learn, 
matplotlib, pyqtgraph, flowkit, flowio).
- In the experiment, before opening the FlowCode tab:
  - The unmixing matrix is calculated (Spectral Process tab). The tab shows a 
  disabled message until it is.
  - At least one unstained sample has an AF profile extracted in the AutoSpectral
  AF tab (at least 2 spectra in the library). The matched unstained sample and 
  the AF profiles used for debarcoding come from there.
  - At least 3 tag channels (4 or more recommended; a warning appears below 4).

Recommended but optional:

- Run **Setup** in the AutoSpectral Optimization tab. It computes the per-cell 
spectral variants used in the final unmix and the unstained-derived positivity 
thresholds used as the rescue noise floor. A banner in sections 1 and 5 reminds 
you until it has been run.
- Build the AutoSpectral Optimization kernel 
(`bundled_plugins/build_autospectral_opt_kernel.py`). Without it the plugin 
shows a banner, falls back to Honeychrome's Python per-cell AF unmixing, and 
cannot use per-cell variants in the final unmix.

### 2.2 Install the plugin files

Honeychrome looks for plugins in `Experiments/plugins` in the user's home folder 
(`~/Experiments/plugins`). It creates the folder on first launch.

1. Copy these four files into `~/Experiments/plugins/`, keeping them in the same 
folder:

   ```
   flowcode_unmix_tab.py
   flowcode_core.py
   flowcode_state.py
   flowcode_plots.py
   ```

   Only `*_tab.py` files are discovered. The tab puts its own folder on `sys.path`
   and imports the other three, so all four must sit side by side. Copy files, not symlinks. `flowcode_rescue_assess.py` is optional and can stay out of the plugins folder.

2. Start Honeychrome from your Python environment:

   ```bash
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   honeychrome                    # PyPI install, or from a source clone after "pip install -e ."
   ```

3. Enable the plugin. Open **Application Configuration**. Each file in 
`~/Experiments/plugins` that ends in `_tab.py` appears as a row, "Tabbed plugin",
with the checkbox **Enable FlowCode** (the name comes from `plugin_name` in the 
file). Tick it and accept. Honeychrome reloads the experiment and a **FlowCode**
tab appears.

4. After updating any of the plugin files later, copy the new versions over the 
old ones and restart Honeychrome (or untick and re-tick the plugin) so the 
modules are re-imported.

If the tab does not appear, check `~/Experiments/honeychrome.log`. A missing 
companion file, or a Python package that is not installed, shows up there as an 
import error.

### 2.3 Optional: the AF library and fluorophore database

- Tag fluorophores must be in the experiment's fluorophore profiles 
(`process['profiles']`).
- The tandem check reads a `tandem.parent` column in 
`src/honeychrome/data/fluorophore_database.csv` (for example, PE-Cy7 → PE). Tags
whose fluorophore class is "Protein tandem" with a parent listed are ticked 
for the check automatically.

---

## 3. Using it in the app

Open the **FlowCode** tab. The five sections are listed down the left side. Each
has a help toggle at the top and a status line: `Done`, `Needs re-run: <reason>`
or `Waiting for section N`. Work through them in order.

### Section 1: Inputs

- **Backbone.** Tick the samples whose cells carry the FlowCode combinations, 
typically pools of transduced cells. Several samples can be ticked and their 
events are pooled (at most 2,000,000 events per file). Every combination needs 
enough bright, barcoded events to be characterised: aim for at least 50 events 
per combination with every tag at least 3 times its threshold. Samples with 
"backbone" or "FRET" in the name are ticked at first. Single-stain controls and 
unstained samples are not listed.
- **Matched unstained.** The unstained sample whose FSC/SSC distribution is 
closest to the pooled backbone, scored by the Bhattacharyya coefficient of the 
scatter histograms (1 = identical). Click a row to choose a different one, or 
**Match again** to recompute.
- **AF library for debarcoding.** The AF profiles extracted from the chosen 
unstained sample in the AutoSpectral AF tab. If none exist, the section shows 
how to extract one and stays incomplete.
- **Combinations CSV** (optional). A FlowCodeUnmix file with an `Id` column and 
three tag columns. Without it, every 3-tag combination of the ticked tags is used.

### Section 2: Tags & Combos

- **Tag channels.** Unmixed channels whose antigen is a known tag are ticked 
automatically (exact match on the canonical name, so `S` and `C` do not match 
CD4 or SSC). Tick or untick others by hand. The **Nearest tag** column is the 
other tag with the most similar reference spectrum; it becomes the default y-axis
of the threshold plot.
- **Combinations.** A table with Include, Name, Tag 1–3 and (after section 4) 
the backbone event count. Without a CSV the plugin lists all C(n, 3) triplets 
with names such as `HA_V5_AU1`. Untick **Include** to drop a combination: its 
events are then reported as unresolved. Names become exported channels `FC <name>`,
so they must be unique, non-empty and free of commas. **Save combinations CSV**
writes the R format.

### Section 3: Gates & Thresholds

**FlowCode scatter gates.** Two gates on the pooled, cleaned backbone (saturated
events are removed first):

- **FlowCode Singlets** (FSC-A × FSC-H): keeps events whose area/height ratio is
below the 85th percentile.
- **FlowCode Cells** (FSC-A × SSC-A): a robust 70% contour around the scatter of
the events most clearly positive for three tags, enlarged 1.5 times about its 
centre so it does not clip barcoded cells.

Drag a gate to edit it, or drag the round handle on FlowCode Cells to scale the
whole gate about its centre. **Use experiment gate** copies the Cells or Singlets
gate from the Raw Data tab. A third plot shows the matched unstained with the 
same Cells gate.

**Confirm gates** unmixes the gated backbone and matched unstained with per-cell
AF (the debarcoding space) and sets a starting threshold per tag (rule in 5.2).

**Thresholds.** Pick a tag in the table. The biplot shows it on x against its 
nearest tag on y, with a movable red line. Drag the line or edit the raw value to
change the threshold; the method column changes to `manual`. 
**Import thresholds CSV** reads a FlowCodeUnmix thresholds file (Fluor, Tag, 
Threshold_Raw), matching rows by tag name and then by fluorophore. Editing a 
threshold keeps section 3 done and marks sections 4 and 5 as needing a re-run.

Advanced settings: backbone cap (2,000,000 events per file), unstained cap 
(100,000 events) and `cell.weight` (on for the ID7000).

### Section 4: Debarcode & FRET

Press **Debarcode and identify FRET spillover**. The plugin debarcodes the gated
backbone, fits the error model of each combination (5.4 and 5.5) and shows:

- **Combo frequencies.** Percentage of barcoded events per combination, with the
untransduced, unresolved and rescued counts underneath.
- **FRET and spillover error.** A heatmap of combinations × fluorophores (plus 
AF): how much each channel gains per unit of the combination's on-tag abundance,
at that combination's median expression, in percent. The combination's own tags 
are outlined. A toggle switches to detector space.
- **Panel expression before and after correction.** One ridge panel per tag, 
uncorrected against corrected, in Overlay, Before or After view. Both use the 
final unmix of section 5.
- **Assess correction.** Linked biplots (uncorrected, corrected) restricted to 
the tag channels. Tick **Draw barcoded events as points** to see the barcoded 
events as larger points. Check that off-tag channels of untransduced and barcoded
cells line up.
- **Fit diagnostics.** A table per combination: confident and fit event counts, 
`kappa` (conditioning), in-span and out-span error, curvature, AF slope and excess.

Advanced settings hold both rescues (switches and thresholds), the fit fold 
(every on-tag at least 3× threshold), the minimum fit events per combination (50)
and the ridge penalty (0). Changing one marks sections 4 and 5 as needing a re-run.

Results are saved with the experiment (`process['flowcode_unmix']` in the `.kit`
file, arrays in `cache/flowcode_spectra.npz`). R-compatible `FlowCode/thresholds.csv`
and `FlowCode/combinations.csv` are written to the experiment folder.

### Section 5: Unmix & Export

- **Samples.** All samples except unstained and single-stain controls are listed
and ticked (tick **Include single-stain controls and unstained samples** to list
those too). The backbone is included. Samples whose AF library has fewer than 2 spectra are greyed out. A warning appears for selected samples that lack the AF profiles the thresholds were set with; those profiles are added to the sample's library for the export only (the experiment's assignments are not changed).
- **Output.** Default folder `<experiment>/FlowCode Unmixed`; sub-folders mirror the raw folder.
- **Use per-cell spectral variants in the final unmix.** On by default when Optimization Setup has computed variants. Off means AF-only per-cell unmixing in the final step.
- **Advanced settings.** Kernel settings (passes, noise floor, alpha and so on, with the same defaults as the Optimization tab), the tag shift minimum, the correction mode (`combo`, `pairwise`, `none`) and the regressor.
- **Tandem tags.** Tick the tags on tandem fluorophores to check for sample-specific breakdown (5.7).

Press **Unmix and export**. With variants on, a probe estimates the run time and asks you to confirm before starting; it is rough, and the real time can differ by a factor of 2 or more.

Section 5 is marked stale ("The spectral model changed since the backbone was characterised") if the spectral model or the saved characterisation no longer matches the current reference spectra. Re-run sections 3 and 4 in that case. A characterisation saved by an older error model is also refused.

---

## 4. Outputs

Per exported sample: `<sample> FlowCode (Unmixed).fcs`, uncompensated, with

- the usual unmixed channels, `AF Abundance` and `AF Index`;
- `FlowCode`: for a barcoded event, the median of the final abundances of its three tags; 0 otherwise;
- `FC <name>`: one channel per included combination, holding the same value for that combination's events and 0 otherwise. Gate on `FC <name>` > 0 to select a combination;
- the keywords `FLOWCODEUNMIX` (plugin version) and `FLOWCODECOMBOS` (`FC <name>=TAG_TAG_TAG;…`).

In the output folder:

- `FlowCode_debarcode_summary.csv`: per sample, total events, untransduced, unresolved, rescued (too many), rescued (too few), pairwise corrected, one count column per combination, and three columns (beta, z, at median) per ticked tandem tag.
- `FlowCode_run.json`: backbone, thresholds, settings, AF profiles per sample, tandem records.

---

## 5. Mechanics: debarcoding and FRET spillover correction

### 5.1 Data spaces

All data are uncompensated. Four spaces are used:

| Space | Definition | Used for |
|---|---|---|
| Raw | Detector values in experiment column order | Gates, the error model and its correction |
| Debarcoding space | Per-cell AF unmixing with the sample's own AF library (no variants) | Thresholds, debarcoding, rescues |
| Fixed basis | Ordinary least squares on `[S; AF.Mean]`, where `S` is the reference spectra of all fluorophores and `AF.Mean` is the max-normalised mean raw spectrum of the gated matched unstained | Error-model regressors |
| Final | The joint kernel on corrected raw, with variants when ticked | Exported abundances |

Per-cell AF unmixing picks, for each event, the AF spectrum from the library that fits best and unmixes with it. Debarcoding uses Honeychrome's Python implementation by default; the kernel agrees on the AF choice for at least 98% of events.

### 5.2 Thresholds

One positivity threshold per tag, in debarcoding space, shared by every sample. The starting value for each tag, computed on the gated backbone with a fixed logicle transform (T = magnitude ceiling, W = 0.5, M = 4.5, A = 0):

- If the tag is bimodal (ΔBIC of a 2-component against a 1-component Gaussian mixture above 10), the crossover of the two mixture components.
- Otherwise the upper 1% tail of the negatives.
- Never below the 99.5th percentile of the gated unstained in the same space.

The result is stored in raw units, so it does not move when you change a display transform.

### 5.3 Debarcoding

An event's tag is *positive* when its abundance in debarcoding space is above that tag's threshold. Each event gets a code:

| Code | Meaning |
|---|---|
| 0 | No positive tag (untransduced) |
| 1 … G | Combination `code − 1` of the design |
| G + 1 | Unresolved: 1 or 2 positive tags, more than 3, or a triplet that is not a combination |

Rules:

1. An event with more than three positive tags keeps only its three strongest when the third-strongest signal over threshold is more than twice the fourth. Otherwise it stays with more than three positives.
2. Only events with exactly three positive tags can match a combination. The positive set is turned into a bitmask and looked up in the combination table.

Two rescues then recover events that a single global threshold mis-calls.

**Too-many-tag rescue.** An event with more than three positives is assigned to the combination whose reference it most resembles.

- Space: logicle-transformed abundances. Each tag is scaled from its noise floor (the Optimization positivity threshold, else the 99.5th percentile of the unstained) to the 99.5th percentile of all events.
- Reference: the median of each combination's confident events. In an exported sample, each tag's reference is shifted by the median transformed offset of the sample's confident events from the backbone's, once a tag has 100 supporting events (`tag_shift_min_n`). The shift is estimated on events inside the FlowCode Cells gate.
- Accept when the best cosine similarity is at least 0.9 and beats the runner-up by 0.5 × (1 − cosine between the best combination and its nearest rival combination).

**Too-few-tag rescue** (no counterpart in the R package; on by default). Thresholds are the same for every sample, so a cell whose three tags are all dimmer in one sample can lose one or two of them below threshold.

- Candidates: unresolved events with one or two positive tags that have no detector at the magnitude ceiling and are not AF-dominated (per-cell AF abundance no larger than the event's largest above-threshold tag, measured from the untransduced median).
- Space: abundance over the sample's untransduced median, in units of (threshold − median). This is scale-free in the cell's expression level, so a uniformly dimmer cell scores as its bright counterparts do.
- Reference: each combination's median over the untransduced median, from the block's own confident events, shrunk towards the backbone's with weight n / (n + 50), and scaled per tag to the sample's expression level.
- Compare each candidate with the combinations that contain *all* of its positive tags by cosine similarity. Accept when the best is at least 0.9, beats the runner-up by 0.5 × (1 − rival cosine), and every on-tag of that combination that is below threshold is at least 2 untransduced SDs and 0.25 × (threshold − median) above the untransduced median.

The main debarcoding is not changed by the rescues: confident barcodes stay as they are.

### 5.4 The error model

For an event, the unmixing error is what remains of the raw spectrum after removing the untransduced background and the fixed-basis reconstruction of its tags:

```
baseline    = per-detector median raw spectrum of the untransduced events
a0          = fixed-basis tag abundances of the baseline
a           = event's fixed-basis tag abundances − a0        (on-tags only)
error y     = raw − baseline − a_on · S_on
```

The baseline carries autofluorescence and non-specific staining that every cell has, so those are not treated as error; a corrected cell lands with the untransduced cells, not at zero.

For each combination, the three on-tag abundances `a1, a2, a3` define six regressors:

```
x = [ a1, a2, a3, a1·a2, a1·a3, a2·a3 ]        (negative abundances count as 0 in the products)
```

The combination's error is fitted by least squares through the origin:

```
y ≈ x · E        E: 6 × detectors
```

Regressors are divided by their threshold scale (t1, t2, t3, t1·t2, t1·t3, t2·t3) before solving, for conditioning, and the result is converted back to per-raw-unit units. FRET grows with donor × acceptor, so the error is convex in abundance. A straight line through the middle of the data would leave a constant negative offset after correction (this was the cause of a constant negative RB613 in combinations carrying C with VSVg or S); the through-origin fit with product terms follows the curve down to the untransduced baseline.

**Fit events.** Confident events (code 1 … G, including rescued ones) whose every on-tag, in the fixed basis, is at least 3 × its threshold (`fit_min_fold`). A combination needs at least 50 such events (`min_cells_per_combo`) to get its own model.

**Pairwise model** (fallback). Fitted on the fit events of all combinations together. For an event with active tag set A:

```
error = Σ_i a_i · c_i  +  Σ_{i<j in A} a_i · a_j · product[i, j]
```

where `c_i` is twice the mean of `linear[i, j]` over the partners j in A that were seen with tag i in the fit events. It serves combinations with fewer than 50 fit events and events without a valid barcode.

Diagnostics per combination (the fit table): `kappa` (condition number of the regressor cross-product), in-span error (the part of the error slope that lies in the span of the fluorophore and AF spectra) and out-span error, curvature (how far the fitted error bends from the best straight line, excluding the AF direction), AF slope against the backbone median and the excess over it.

### 5.5 Applying the correction

For each block of events in an exported sample, `unmix_staged` runs:

1. Per-cell AF unmixing with the sample's own AF library gives the debarcoding space.
2. Debarcode with the backbone thresholds, then both rescues (against the backbone's combination medians and untransduced background).
3. Fixed-basis OLS on `[S; AF.Mean]` gives the regressors `a`, measured from the baseline.
4. Subtract the characterised error from the raw data:
   - barcoded events of a combination with its own model: `raw_corrected = raw − x(a − a0) · E_g`;
   - every other event: `raw_corrected = raw − PairModel(a − a0, active tags)`, where the active tags are the combination's tags for a barcoded event and the above-threshold tags otherwise;
   - the `none` setting leaves raw unchanged, and `pairwise` uses the pairwise model for every event.
5. Optionally subtract the tandem breakdown (5.7).
6. Final unmix of the corrected raw with the sample's AF library (and variants when ticked).
7. `FlowCode` and `FC <name>` channels: median of the final abundances of the event's three tags.

The correction is applied in raw (detector) space, before the final unmix, so AF assignment and variant selection see the corrected spectrum.

### 5.6 What the heatmap and before/after plots show

The heatmap value for combination g and channel f is `100 × mean_k (E_g Uᵀ)[k, f]` over the combination's on-tags k, with `U = (B Bᵀ)⁻¹ B` and `B = [S; AF.Mean]`: the error per unit of on-tag abundance, as it appears in the unmixed channels. The before/after panels run the export's final unmix on the same events without and with the correction, so they show what the exported data will look like.

### 5.7 Tandem tags

A tag on a tandem fluorophore (PE-Cy7, for example) that breaks down in a sample emits part of its signal as its parent (PE). When the parent is also in the panel, the extra signal lands on the parent's marker in every combination that carries the tag, in proportion to the tag's abundance. No per-cell residual shows it, and the backbone's fit cannot see a breakdown that is specific to one sample.

The check uses the barcodes as an internal control. For each ticked tag and sample:

1. Within each combination, an expression index is built from the other on-tags (each as a fold of its combination median). Events are split by the top and bottom quartile of that index, and the top-minus-bottom differences of medians of the parent abundance (dy), the index (dE) and the tag (da) are taken. This removes each combination's mean and is robust to tails.
2. Combinations without the tag (and without another tandem with the same parent) give the shared expression trend γ = median(dy / dE) and the null spread.
3. Each combination with the tag gives β_g = (dy − γ·dE) / da. The statistic β is their median, with a standard error from the null spread (or the tag combinations' own spread, if larger) and the error of γ.
4. The backbone gives the same statistic. The sample's departure from the backbone, paired by combination when at least 5 combinations are shared, is divided by its standard error to give z.
5. A soft threshold decides how much to apply: `w = max(0, 1 − (z_min / z)²)` with `z_min = 3`. Nothing is applied below |z| = 3, and almost all of the difference is applied well above it.
6. The correction `β·w × (tag abundance above baseline) × parent spectrum` is subtracted from the raw data of every event carrying the tag (barcoded events of combinations with the tag, plus other events with the tag above threshold). The parent spectrum comes from the panel, or from the cytometer's reference library when the parent is not in the panel.

At least 5 combinations with and 5 without the tag are required. The backbone statistic is computed in section 4; the per-sample estimate is made once per sample, on at most 2,000,000 gated events, at the start of its export.

### 5.8 Guards

| Change | Effect |
|---|---|
| Backbone, unstained sample, AF profile, tags, included combinations | Sections 3 to 5 need a re-run |
| Gate edit | Debarcoding space and thresholds' automatic values; sections 4 and 5 need a re-run |
| Threshold edit, section 4 setting | Sections 4 and 5 need a re-run |
| Combination rename | Only section 5 |
| Spectral model change | Section 5 is disabled until sections 3 and 4 are re-run |

---

## 6. Differences from the R package

- AF libraries come from the AutoSpectral AF tab (extracted on the whole unstained file), not from `get.af.spectra()` on the FlowCode-gated unstained. `AF.Mean` is still taken from the gated unstained.
- Only the FSC-A/FSC-H singlet pass is applied; the SSC-A/SSC-H pass is not.
- The landmark gate uses scikit-learn's FastMCD, not `MASS::cov.rob("mcd")`.
- The rescue transform is a fixed Honeychrome logicle, not AutoSpectral's cytometer-specific biexponential.
- The error model is a through-origin fit with pair products measured from the untransduced median, with a shared pairwise fallback. It replaces R's centred per-combination fit and global error.
- The too-few-tag rescue, the export AF library (assigned profiles plus the profiles the thresholds were set with) and the tandem-breakdown check have no R counterpart.
- Per-sample threshold overrides are not available.
