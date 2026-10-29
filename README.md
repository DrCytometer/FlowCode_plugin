# FlowCode plugin for Honeychrome

FlowCode debarcoding with FRET and spillover correction for spectral flow cytometry. The plugin is a Honeychrome tab (`FlowCode`) and a Python port of the FlowCodeUnmix 0.6.1 AF-matched staged pipeline, extended with a through-origin error model, a too-few-tag rescue and a tandem-breakdown check.

Contents

1. [What it is](#1-what-it-is)
2. [Setup](#2-setup)
3. [Using it in the app](#3-using-it-in-the-app)
4. [Outputs](#4-outputs)
5. [Mechanics: debarcoding and FRET spillover correction](#5-mechanics-debarcoding-and-fret-spillover-correction)
6. [Differences from the R package](#6-differences-from-the-r-package)

---

## 1. What it is

A FlowCode panel labels each cell with three epitope tags out of a set of ten (HA, V5, AU1, AU5, S, C, Flag, NWS, VSVg, Ollas), each detected with its own fluorophore. A cell's barcode is the set of three tags that are positive. Two things make this hard on a spectral cytometer:

- **Debarcoding.** Each cell must be assigned to one of up to C(10, 3) = 120 combinations from three-out-of-ten positivity calls, with a single threshold per tag.
- **FRET and spillover.** When the three tags sit on the same cell, donor-to-acceptor FRET and residual spillover add signal to the other fluorophore channels. The error grows with the product of donor and acceptor abundance, so a combination's error is not a fixed spectrum.

The plugin does five jobs, one per tab:

| Tab | Job |
|---|---|
| 1. Inputs | Choose the backbone samples (cells that carry every combination), the matched unstained sample and its AF library |
| 2. Tags & Combos | Pick the tag channels and define the combinations |
| 3. Gates & Thresholds | Scatter gates on the backbone and a positivity threshold per tag |
| 4. Debarcode & FRET | Characterise the unmixing error of every combination on the backbone and show before/after plots |
| 5. Unmix & Export | Unmix the selected samples with the correction and write FCS files with a `FlowCode` channel and one `FC <name>` channel per combination |
| 6. Analysis | Differential barcode distributions between sample groups, counted live from the raw files or from the exported files |

The plugin reuses Honeychrome's AutoSpectral machinery: AF profiles from AutoSpectral › Autofluorescence, and the experiment's unmixing method set on the AutoSpectral tab (per-cell AF, the noise model (M2pg) or spectral variants, with Fix My Unmix's spectral corrections when they are in use) for the final unmix.

### Files

| File | Role |
|---|---|
| `flowcode_unmix_tab.py` | The tab (UI and workers). The only file Honeychrome discovers |
| `flowcode_core.py` | Numerics, no Qt: debarcoding, rescues, error models, tandem check, staged unmixing |
| `flowcode_state.py` | Experiment state and persistence |
| `flowcode_plots.py` | Section 4 figures |
| `flowcode_method.py` | The experiment's unmixing method as FlowCode uses it (final unmix, debarcoding space), no Qt |
| `flowcode_export.py` | Reading, FlowCode unmixing and writing of whole samples: the export and the Analysis page's live unmix |
| `flowcode_analysis.py`, `flowcode_stats.py` | Analysis page numerics, no Qt |
| `flowcode_analysis_page.py` | Section 6 (Analysis) |
| `flowcode_rescue_assess.py` | Optional standalone script that assesses the too-few rescue. Not loaded by the plugin and not copied by the installer |

---

## 2. Setup

### 2.1 Requirements

- Honeychrome run from Python 3.12, from a source clone of the `autospectral` branch of [Cytkit/honeychrome](https://github.com/Cytkit/honeychrome). The installer in 2.2 sets this up. The plugin needs the AutoSpectral tab's unmixing method (`controller_components/unmix_method.py`); with an older Honeychrome, including the PyPI package, the tab shows a message instead. The plugin system is not available in the packaged executables.
- No packages beyond Honeychrome's own requirements (numpy, scipy, scikit-learn, matplotlib, pyqtgraph, flowkit, flowio).
- In the experiment, before opening the FlowCode tab:
  - The unmixing matrix is calculated (Spectral Process tab). The tab shows a disabled message until it is.
  - At least one unstained sample has an AF profile extracted in AutoSpectral › Autofluorescence (at least 2 spectra in the library). The matched unstained sample and the AF profiles used for debarcoding come from there.
  - At least 3 tag channels (4 or more recommended; a warning appears below 4).

The order of work (shown, with buttons that open each AutoSpectral page, in the **Before you start** box of section 1):

1. **AF profiles** (AutoSpectral › Autofluorescence): extract them for the matched unstained and assign every sample to export.
2. **Noise model** (AutoSpectral › Noise, recommended): *Model Noise*, then *Use noise model*. It is fitted on the controls only. Model it before section 3 if debarcoding is to use it.
3. **FlowCode sections 1 to 4.**
4. **Spectral corrections** (AutoSpectral › Spectral Correction, optional): *Fix Unmix* with the tag fluorophores in its *Fluorophores kept fixed* list (the button fills the list in). Fix My Unmix reads the raw FlowCode samples, where FRET is still uncorrected; with the tag rows free it would absorb FRET into the tag spectra and the FlowCode correction would remove the same error a second time. If you have performed the single-stained controls per the flow protocol, this step should not be necessary.
5. **Section 5 export**, or **section 6** directly: it can unmix live without exporting.

The compiled kernels make the noise model and spectral variants fast. The installer builds them and checks that they load; in a manual install, build them with `bundled_plugins/build_som_kernel.py`, `build_autospectral_opt_kernel.py` and `build_fix_my_unmix_kernel.py`. Without them the noise model runs in NumPy and the export asks you to confirm an estimated run time first. The Spectral Variants page's positivity thresholds, when computed, are the too-many rescue's noise floor; otherwise it is the 99.5th percentile of the matched unstained.

### 2.2 Install and update

One command installs Honeychrome (`autospectral` branch), the compiled kernels and this plugin, and the same command updates them. Step-by-step instructions for people new to the Terminal, troubleshooting and the list of what goes where are in [INSTALL_GUIDE.md](INSTALL_GUIDE.md). An administrator account is needed the first time.

- **Mac** (Terminal):

  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/DrCytometer/FlowCode_plugin/main/install/install_honeychrome.sh)"
  ```

- **Windows 10/11, x64** (PowerShell):

  ```powershell
  powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/DrCytometer/FlowCode_plugin/main/install/install_honeychrome.ps1 | iex"
  ```

The installer puts everything in `~/Honeychrome`, copies the plugin files into `~/Experiments/plugins`, and adds **Honeychrome** and **Update Honeychrome** icons to the Desktop. It logs to `~/Honeychrome/install.log`.

1. Start Honeychrome with the **Honeychrome** icon. Keep its Terminal or console window open while you work.
2. First time only: open **Application Configuration**. Each file in `~/Experiments/plugins` that ends in `_tab.py` appears as a row, "Tabbed plugin", with the checkbox **Enable FlowCode** (the name comes from `plugin_name` in the file). Tick it and accept. Honeychrome reloads the experiment and a **FlowCode** tab appears.
3. To update: quit Honeychrome, double-click **Update Honeychrome**, wait for **Done.**, then start Honeychrome again. If the icon is missing, run the install command again.

If the tab does not appear, check `~/Experiments/honeychrome.log`. A missing companion file, or a Python package that is not installed, shows up there as an import error.

### 2.3 Manual installation into an existing Honeychrome source clone

For developers who already run Honeychrome from a source clone of the `autospectral` branch. Honeychrome looks for plugins in `Experiments/plugins` in the user's home folder (`~/Experiments/plugins`). It creates the folder on first launch.

1. Copy these nine files into `~/Experiments/plugins/`, keeping them in the same folder:

   ```
   flowcode_unmix_tab.py
   flowcode_core.py
   flowcode_state.py
   flowcode_plots.py
   flowcode_method.py
   flowcode_export.py
   flowcode_analysis.py
   flowcode_stats.py
   flowcode_analysis_page.py
   ```

   Only `*_tab.py` files are discovered. The tab puts its own folder on `sys.path` and imports the others, so all nine must sit side by side. Copy files, not symlinks.

2. Start Honeychrome from your Python environment, in the clone's folder:

   ```bash
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   python -m honeychrome.main     # after "pip install -e ."
   ```

3. Enable the plugin in **Application Configuration** as in 2.2.

4. After updating any of the plugin files later, copy the new versions over the old ones and restart Honeychrome (or untick and re-tick the plugin) so the modules are re-imported.

Do not mix the two methods: the installer overwrites the plugin files in `~/Experiments/plugins` on every run.

### 2.4 Optional: the AF library and fluorophore database

- Tag fluorophores must be in the experiment's fluorophore profiles (`process['profiles']`).
- The tandem check reads a `tandem.parent` column in `src/honeychrome/data/fluorophore_database.csv` (for example, PE-Cy7 → PE). Tags whose fluorophore class is "Protein tandem" with a parent listed are ticked for the check automatically.

---

## 3. Using it in the app

Open the **FlowCode** tab. The six sections are listed down the left side; the main window's sample list is hidden while the tab is open. Each has a help toggle at the top and a status line: `Done`, `Needs re-run: <reason>` or `Waiting for section N`. Sections 3 to 6 split into a row of tabs (e.g. Combo frequencies, Assess correction); the open tab is remembered. Figures have a **Size** slider, **⤢ Pop Out** (a resizable window with the matplotlib toolbar) and **Export Graphic**. Work through them in order.

### Section 1: Inputs

- **Backbone.** Tick the samples whose cells carry the FlowCode combinations, typically pools of transduced cells. Several samples can be ticked and their events are pooled (at most 2,000,000 events per file). Every combination needs enough bright, barcoded events to be characterised: aim for at least 50 events per combination with every tag at least 3 times its threshold. Samples with "backbone" or "FRET" in the name are ticked at first. Single-stain controls and unstained samples are not listed.
- **Matched unstained.** The unstained sample whose FSC/SSC distribution is closest to the pooled backbone, scored by the Bhattacharyya coefficient of the scatter histograms (1 = identical). Click a row to choose a different one, or **Match again** to recompute.
- **AF library for debarcoding.** The AF profiles extracted from the chosen unstained sample in AutoSpectral › Autofluorescence. If none exist, the section shows how to extract one and stays incomplete.
- **Combinations CSV** (optional). A FlowCodeUnmix file with an `Id` column and three tag columns. Without it, every 3-tag combination of the ticked tags is used.
- **No FRET backbone: skip FRET correction (results invalid)**. For samples acquired before FRET backbone controls were run, to test the pipeline. Ticking it asks you to acknowledge, in a pop-up, that the downstream results are untrustworthy (the time is saved with the experiment). The Backbone box becomes *Samples for gates and thresholds*: the ticked barcoded samples set the FlowCode gates, the thresholds and the rescue references, and no FRET error is fitted on them (stained samples would put their marker signal into the error). See [No FRET backbone](#no-fret-backbone) below.

### Section 2: Tags & Combos

- **Tag channels.** Unmixed channels whose antigen is a known tag are ticked automatically (exact match on the canonical name, so `S` and `C` do not match CD4 or SSC). Tick or untick others by hand. The **Nearest tag** column is the other tag with the most similar reference spectrum; it becomes the default y-axis of the threshold plot.
- **Combinations.** A table with Include, Name, Tag 1–3 and (after section 4) the backbone event count. Without a CSV the plugin lists all C(n, 3) triplets with names such as `HA_V5_AU1`. Untick **Include** to drop a combination: its events are then reported as unresolved. Names become exported channels `FC <name>`, so they must be unique, non-empty and free of commas. **Save combinations CSV** writes the R format.

### Section 3: Gates & Thresholds

**FlowCode scatter gates.** Two gates on the pooled, cleaned backbone (saturated events are removed first):

- **FlowCode Singlets** (FSC-A × FSC-H): keeps events whose area/height ratio is below the 85th percentile.
- **FlowCode Cells** (FSC-A × SSC-A): a robust 70% contour around the scatter of the events most clearly positive for three tags, enlarged 1.5 times about its centre so it does not clip barcoded cells.

Drag a gate to edit it, or drag the round handle on FlowCode Cells to scale the whole gate about its centre. **Use experiment gate** copies the Cells or Singlets gate from the Raw Data tab. A third plot shows the matched unstained with the same Cells gate.

**Confirm gates** unmixes the gated backbone and matched unstained with per-cell AF (the debarcoding space) and sets a starting threshold per tag (rule in 5.2).

**Thresholds.** Pick a tag in the table. The biplot shows it on x against its nearest tag on y, with a movable red line. Drag the line or edit the raw value to change the threshold; the method column changes to `manual`. **Import thresholds CSV** reads a FlowCodeUnmix thresholds file (Fluor, Tag, Threshold_Raw), matching rows by tag name and then by fluorophore. Editing a threshold keeps section 3 done and marks sections 4 and 5 as needing a re-run.

Advanced settings: backbone cap (2,000,000 events per file), unstained cap (100,000 events) and *Debarcode with*: per-cell AF (default) or the experiment's unmixing method. Changing it means confirming the gates and running section 4 again.

### Section 4: Debarcode & FRET

Press **Debarcode and identify FRET spillover**. The plugin debarcodes the gated backbone, fits the error model of each combination (5.4 and 5.5) and shows:

- **Combo frequencies.** Percentage of barcoded events per combination, with the untransduced, unresolved and rescued counts underneath.
- **FRET and spillover error.** A heatmap of combinations × fluorophores (plus AF): how much each channel gains per unit of the combination's on-tag abundance, at that combination's median expression, in percent. The combination's own tags are outlined. A toggle switches to detector space.
- **Panel expression before and after correction.** One ridge panel per tag, uncorrected against corrected, in Overlay, Before or After view. Both use the final unmix of section 5.
- **Assess correction.** Linked biplots (uncorrected, corrected) restricted to the tag channels. Tick **Draw barcoded events as points** (off by default, 3 px) to see the barcoded events as larger pink points. Check that off-tag channels of untransduced and barcoded cells line up.
- **Fit diagnostics.** A table per combination: confident and fit event counts, `kappa` (conditioning), in-span and out-span error, curvature, AF slope and excess.

Advanced settings hold both rescues (switches and thresholds), the fit fold (every on-tag at least 3× threshold), the minimum fit events per combination (50) and the ridge penalty (0). Changing one marks sections 4 and 5 as needing a re-run.

Results are saved with the experiment (`process['flowcode_unmix']` in the `.kit` file, arrays in `cache/flowcode_spectra.npz`). R-compatible `FlowCode/thresholds.csv` and `FlowCode/combinations.csv` are written to the experiment folder.

### No FRET backbone

With *No FRET backbone* ticked in section 1:

- **Section 4** debarcodes only (*Debarcode (no FRET correction)*): combination frequencies, the per-combination medians the rescues use and the untransduced baseline. No error model, pairwise model or tandem reference is made; the heatmap, before/after and *Assess correction* tabs say the correction was not performed, and the fit table's *Error* column reads `none`.
- **Sections 5 and 6** leave the raw data unchanged before the final unmix, whatever the *correction* setting (shown disabled), and check no tandem tags. *Fix Unmix (FRET-corrected)* is not available. Each export asks for the acknowledgement again.
- **Warnings.** The status lines of sections 4, 5 and 6 are red with *WARNING: no spillover performed! Results invalid!*. Exported FCS files carry `FLOWCODECORRECTION=NONE` and `FLOWCODEWARNING`; `FlowCode_run.json` has `"fret_correction": false` and the warning; `FlowCode_debarcode_summary.csv` has a last *Warning* column. Section 6 shows the warning in red above its tabs whenever the analysed events are uncorrected (the live unmix with this characterisation, or exported files with `FLOWCODECORRECTION=NONE`, even after the override is switched off), stamps it across every figure, asks for the acknowledgement before exporting the analysis, and writes it across every page of the PDF report, on the PNG figures and at the top of `settings.txt`.
- Switching the override on or off makes sections 4 and 5 need a re-run (sections 1 to 3 are kept). The characterisation identifier differs from that of a corrected run, so files exported one way show as *older characterisation* for the other.

### Section 5: Unmix & Export

- **Samples.** All samples except unstained and single-stain controls are listed and ticked (tick **Include single-stain controls and unstained samples** to list those too). The backbone is included. Samples whose AF library has fewer than 2 spectra are greyed out. A warning appears for selected samples that lack the AF profiles the thresholds were set with; those profiles are added to the sample's library for the export only (the experiment's assignments are not changed).
- **Output.** Default folder `<experiment>/FlowCode Unmixed`; sub-folders mirror the raw folder.
- **Final unmix** (the line at the top): the experiment's method from the AutoSpectral tab, with the spectral corrections of each sample's group when they are in use. It is set there, not here; a change makes section 5 stale. A red note appears when Fix My Unmix has corrected a tag fluorophore.
- **Advanced settings.** The tag shift minimum, the correction mode (`combo`, `pairwise`, `none`) and the regressor.
- **Tandem tags.** Tick the tags on tandem fluorophores to check for sample-specific breakdown (5.7).

Press **Unmix and export**. With a slow method (spectral variants, or the noise model without its compiled kernel) a probe estimates the run time and asks you to confirm before starting; it is rough, and the real time can differ by a factor of 2 or more. When Honeychrome runs one heavy job at a time, the export is refused, with the running job named, while another one holds the application.

### Section 6: Analysis

**Source**: *Live unmix* (default) FlowCode-unmixes each sample's raw file with the current characterisation and the experiment's method, exactly as section 5 would export it, without writing files; the unmixed values are cached in `cache/flowcode_unmixed` (pruned with the App Settings' unmixing cache size), so changing the population later does not unmix again. *Exported files* reads the files section 5 wrote. Counts from the two sources are identical for the same inputs. Saved counts say which source they came from and are marked out of date when the samples, the unmixing, the population or the gating change.

**Populations**: tick one or more gates of the unmixed gating tree; each ticked gate is counted as a population of its own, in one pass over the samples (e.g. CD8 and Tconv, or Naive, Activated and Resident). Counting also sums each combination's marker intensities (every fluorophore of the spectral model except the FlowCode tags, on the experiment's display scale) into `cache/flowcode_analysis/marker_sums.npz` for the marker test.

**Groups**: a table of groups with a sample-name pattern (regular expression, case-insensitive) and a colour each. *Assign by pattern* gives every listed sample the first group whose pattern matches its name. Double-click a colour to change it; group colours are saved with the experiment and used in every figure (volcano direction ramps, heatmap group bars, fold-change bars, PCA, QC, combination plots).

**Statistics with several populations**: *Compare: sample groups* runs the group comparisons within each population (`Pancreas vs Spleen [CD8]`); library sizes and expanded clones are worked out within each population. Tick *Population contrasts* to also test, against the *reference population*, whether a group's change against the reference group differs between populations: `CD8 vs Tconv [Pancreas vs Spleen]` is log2(Pancreas/Spleen) among CD8 minus the same among Tconv. Paired by mouse, each mouse has its own baseline per population and its own group effect shared by the populations, so every mouse is compared with itself (the GLM form of the T cell type scripts' paired t-test of spleen-normalised frequencies). *Compare: populations* tests the populations against the reference population within each sample (paired by sample, split by sample group by default).

**Expanded clones**: the lab R rule. Within each non-reference group, a combination is a clone in a sample when its % of barcoded events exceeds the median + SD of that combination's % over the group's samples in which it was seen, with more than 200 events and 10 %. It is removed for that mouse in every tissue unless it is expanded in more than two mice of that group.

**Markers**: for every combination and marker, the mean intensity of the combination's cells minus a reference in the same sample. The default reference is the *tag model*: per sample and marker, the combinations' means are fitted on their tags (one additive term per tag, weighted by cells, Huber-robust), and each combination is compared with what the other combinations predict for its own tags. A tag fluorophore's spillover into a marker, shared by every combination carrying the tag, is then not taken for an effect (against tag-unbalanced controls it shows as blocks of hits). The control combinations, or every other barcoded cell, can be chosen instead. *Leave out features with high tag spillover* (on by default, limit 0.03 display units): the spillover spread of each combination's tags into each marker at their median brightness, from the reference spectra, the characterisation and the noise model, converted to the display scale at the sample's level of the marker; above the limit the feature is not tested (hatched in the heatmap, listed with its spillover in the table). Each group's (and population's) differences are tested against zero with a moderated t, one value per sample, and combinations are families for the cluster-first FDR. A combination or reference with fewer than the minimum events in a sample has no value there; expanded clones removed from the counts are left out too. The *Markers* menu in Results picks the group whose heatmap (combinations × markers, coloured where significant), volcano and table are shown; it is independent of the *Comparison* menu. Double-click a marker volcano point or a table row for the *Marker* tab: every sample's reference and combination means joined by a line, one column per group, a mouse's samples linked across groups. The export writes `markers.csv` and every group's figures.

**Results**: double-click a combination in the results table, a volcano or the fold-change bars to plot it in every sample (*Combination* tab: a column per group, a line per mouse, expanded clones hollow); right-click for *Open in a window*. Changing the comparison redraws only that comparison's figures.

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
| Debarcoding space | Per-cell AF unmixing with the sample's own AF library (no variants), or the experiment's method (section 3 *Debarcode with*) | Thresholds, debarcoding, rescues |
| Fixed basis | Ordinary least squares on `[S; AF.Mean]`, where `S` is the reference spectra of all fluorophores and `AF.Mean` is the max-normalised mean raw spectrum of the gated matched unstained | Error-model regressors |
| Final | The experiment's method on corrected raw (per-cell AF, the noise model or spectral variants), with the sample's group's corrected spectra | Exported abundances |

Per-cell AF unmixing picks, for each event, the AF spectrum from the library that fits best and unmixes with it (Honeychrome's Python implementation). The debarcoding space and the fixed basis always use the base spectra: the thresholds and the error model are measured against them.

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

- Space: logicle-transformed abundances. Each tag is scaled from its noise floor (the Spectral Variants page's positivity threshold, else the 99.5th percentile of the unstained) to the 99.5th percentile of all events.
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
6. Final unmix of the corrected raw with the sample's AF library, by the experiment's method.
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
| Unmixing method or spectral corrections (AutoSpectral tab) | Section 5 needs a re-run; the section 4 figures are drawn again; with *Debarcode with* the experiment's method, sections 3 to 5 |
| *Debarcode with* | Sections 3 to 5 |
| *No FRET backbone* | Sections 4 and 5 |

---

## 6. Differences from the R package

- AF libraries come from AutoSpectral › Autofluorescence (extracted on the whole unstained file), not from `get.af.spectra()` on the FlowCode-gated unstained. `AF.Mean` is still taken from the gated unstained.
- Only the FSC-A/FSC-H singlet pass is applied; the SSC-A/SSC-H pass is not.
- The landmark gate uses scikit-learn's FastMCD, not `MASS::cov.rob("mcd")`.
- The rescue transform is a fixed Honeychrome logicle, not AutoSpectral's cytometer-specific biexponential.
- The error model is a through-origin fit with pair products measured from the untransduced median, with a shared pairwise fallback. It replaces R's centred per-combination fit and global error.
- The too-few-tag rescue, the export AF library (assigned profiles plus the profiles the thresholds were set with) and the tandem-breakdown check have no R counterpart.
- Per-sample threshold overrides are not available (and should not be necessary).
