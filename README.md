# FlowCode plugin for Honeychrome

FlowCode debarcoding with FRET and spillover correction for spectral flow cytometry, and the analysis of FlowCode knockout screens. The plugin is a Honeychrome tab (`FlowCode`) and a Python port of the FlowCodeUnmix 0.6.0 AF-matched staged pipeline, extended with a through-origin error model, a too-few-tag rescue, a tandem-breakdown check and a per-sample calibration of each combination's correction, followed by differential statistics on the combinations, a marker test, signed pathway and gene-group tests, and a per-guide phenotype-shift test. A cohort acquired over several days, each day with its own controls, is processed day by day and analysed together (see [Cohorts acquired over several days](#cohorts-acquired-over-several-days)).

Contents

1. [What it is](#1-what-it-is)
2. [Setup](#2-setup)
3. [Using it in the app](#3-using-it-in-the-app)
4. [Outputs](#4-outputs)
5. [Mechanics: debarcoding and FRET spillover correction](#5-mechanics-debarcoding-and-fret-spillover-correction)
6. [Mechanics: pathway and gene-group tests](#6-mechanics-pathway-and-gene-group-tests)
7. [Mechanics: phenotype shift and tag spillover correction](#7-mechanics-phenotype-shift-and-tag-spillover-correction)
8. [Differences from the R package](#8-differences-from-the-r-package)

---

## 1. What it is

A FlowCode panel labels each cell with three epitope tags out of a set of ten (HA, V5, AU1, AU5, S, C, Flag, NWS, VSVg, Ollas), each detected with its own fluorophore. A cell's barcode is the set of three tags that are positive. Two things make this hard on a spectral cytometer:

- **Debarcoding.** Each cell must be assigned to one of up to C(10, 3) = 120 combinations from three-out-of-ten positivity calls, with a single threshold per tag.
- **FRET and spillover.** When the three tags sit on the same cell, donor-to-acceptor FRET and residual spillover add signal to the other fluorophore channels. The error grows with the product of donor and acceptor abundance, so a combination's error is not a fixed spectrum.

The plugin does six jobs, one per section:

| Tab | Job |
|---|---|
| 1. Inputs | Choose the backbone samples (cells that carry every combination), the matched unstained sample and its AF library |
| 2. Tags & Combos | Pick the tag channels and define the combinations, for one guide library or several |
| 3. Gates & Thresholds | Scatter gates on the backbone and a positivity threshold per tag |
| 4. Debarcode & FRET | Characterise the unmixing error of every combination on the backbone and show before/after plots |
| 5. Unmix & Export | Unmix the selected samples with the correction and write FCS files with a `FlowCode` channel and one `FC <name>` channel per combination |
| 6. Analysis | Differential barcode distributions between sample groups or populations, marker effects of each combination, pathway and gene-group tests and each guide's phenotype shift, counted live from the raw files or from the exported files, including those of other experiments of the same study |

The plugin reuses Honeychrome's AutoSpectral machinery: AF profiles from AutoSpectral › Autofluorescence, and the experiment's unmixing method set on the AutoSpectral tab (per-cell AF, the noise model (M2pg) or spectral variants, with Fix My Unmix's spectral corrections when they are in use) for the final unmix.

### Files

| File | Role |
|---|---|
| `flowcode_unmix_tab.py` | The tab (UI and workers). The only file Honeychrome discovers |
| `flowcode_core.py` | Numerics, no Qt: debarcoding, rescues, error models, tandem check, staged unmixing |
| `flowcode_state.py` | Experiment state and persistence |
| `flowcode_plots.py` | Section 4 figures |
| `flowcode_figure_panel.py` | A figure with Size, Pop Out and Export Graphic, for sections 4 and 6 |
| `flowcode_method.py` | The experiment's unmixing method as FlowCode uses it (final unmix, debarcoding space), no Qt |
| `flowcode_export.py` | Reading, FlowCode unmixing and writing of whole samples: the export and the Analysis page's live unmix |
| `flowcode_fix_unmix.py` | Fix My Unmix (AutoSpectral › Spectral Correction) on FRET-corrected data, started from section 5, no Qt |
| `flowcode_analysis.py`, `flowcode_stats.py` | Analysis page numerics, no Qt (`flowcode_analysis.py` also links the exported files of other experiments) |
| `flowcode_count_cache.py` | The compact per-sample events *Count events* keeps, so that counting again needs no unmixing, no Qt |
| `flowcode_analysis_page.py` | Section 6 (Analysis) |
| `flowcode_view_stack.py` | The drop-down that chooses section 6's result views |
| `flowcode_gating.py`, `flowcode_gating_view.py` | Section 6's Gating tab: the experiment's gates over FlowCode-unmixed events (numerics and figure, no Qt; the tab) |
| `flowcode_phenotype.py` | Phenotype populations for the Analysis page: definitions from Honeychrome's cell-type database, resolved on the panel and scored on the live-unmixed events, no Qt |
| `flowcode_phenotype_shift.py` | Per-guide phenotype shift: clustering, cluster fingerprints, the agreement test, kept events and saved results, no Qt |
| `flowcode_tag_correction.py` | Data-level correction of combination-specific tag spillover for the phenotype shift, no Qt |
| `flowcode_shift_view.py`, `flowcode_shift_plots.py` | Section 6's *Phenotype shift* tab and its figures |
| `flowcode_pathways.py` | Signed GO pathway tests on the Counts results, and their figures, no Qt |
| `flowcode_go_signed_sets.csv.gz`, `flowcode_mgi_gene_names.csv.gz` | The signed mouse GO Biological Process sets and the MGI gene symbols and synonyms (GO release 2026-07-26, MGI GAF 2026-05-21; CC BY 4.0, Gene Ontology Consortium and MGI) |
| `flowcode_gene_groups.py` | Gene-group tests, the guide allocation and the spider figure, no Qt |
| `flowcode_gene_groups_homing.csv`, `flowcode_gene_groups_tx_otb01.csv`, `flowcode_gene_groups_metab.csv` | The built-in gene-group tables of the lab's libraries (Homing, Treg transcription factors OTB01, Metabolism) |
| `install_honeychrome.sh`, `install_honeychrome.ps1`, `INSTALL_GUIDE.md` | Installers of Honeychrome with the plugin (macOS, Windows) and their step-by-step guide. Not plugin files |

---

## 2. Setup

### 2.1 Requirements

- Honeychrome run from Python (the PyPI package or a source clone, Python 3.12 or higher), in a version with the AutoSpectral tab's unmixing method (`controller_components/unmix_method.py`). With an older Honeychrome the tab shows a message instead. The plugin system is not available in the packaged executables.
- For section 6's Pathways and Gene groups, a Honeychrome whose counts statistics have the calibrated quasi-likelihood test (`ql_common_dispersion` and `ql_comparisons` in `controller_components/differential_stats.py`). With an older one, running statistics with either ticked fails.
- A Honeychrome with the plugins' figure options (`view_components/figure_options.py`) and the cell-type database module (`controller_components/cell_type_panel.py`). The install scripts check every Honeychrome module, function and setting the plugin imports before copying it, and keep the previous plugin when one is missing.
- The phenotype shift clusters with the FlowSOM bundled with Honeychrome (Honeycluster's `flowsom_consensus`); the compiled SOM kernel makes it fast.
- No packages beyond Honeychrome's own requirements (numpy, scipy, pandas, scikit-learn, matplotlib, colorcet, pyqtgraph, flowkit, flowio).
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

Optional: the compiled kernels (`bundled_plugins/build_autospectral_opt_kernel.py`) make the noise model and spectral variants fast; without them the noise model runs in NumPy and the export asks you to confirm an estimated run time first. The Spectral Variants page's positivity thresholds, when computed, are the too-many rescue's noise floor; otherwise it is the 99.5th percentile of the matched unstained.

### 2.2 Install the plugin files

Honeychrome looks for plugins in `Experiments/plugins` in the user's home folder (`~/Experiments/plugins`). It creates the folder on first launch.

The install scripts in this repository (`install_honeychrome.sh` for macOS, `install_honeychrome.ps1` for Windows; steps in `INSTALL_GUIDE.md`) install or update Honeychrome's `autospectral` branch with its compiled kernels, check that the plugin's Honeychrome imports are all present, and copy the plugin files below for you; run the same command again to update. To install the plugin by hand:

1. Copy these 27 files into `~/Experiments/plugins/`, keeping them in the same folder:

   ```
   flowcode_unmix_tab.py
   flowcode_core.py
   flowcode_state.py
   flowcode_plots.py
   flowcode_method.py
   flowcode_export.py
   flowcode_analysis.py
   flowcode_stats.py
   flowcode_count_cache.py
   flowcode_analysis_page.py
   flowcode_view_stack.py
   flowcode_figure_panel.py
   flowcode_fix_unmix.py
   flowcode_gating.py
   flowcode_gating_view.py
   flowcode_phenotype.py
   flowcode_phenotype_shift.py
   flowcode_tag_correction.py
   flowcode_shift_view.py
   flowcode_shift_plots.py
   flowcode_pathways.py
   flowcode_go_signed_sets.csv.gz
   flowcode_mgi_gene_names.csv.gz
   flowcode_gene_groups.py
   flowcode_gene_groups_homing.csv
   flowcode_gene_groups_tx_otb01.csv
   flowcode_gene_groups_metab.csv
   ```

   Only `*_tab.py` files are discovered. The tab puts its own folder on `sys.path` and imports the others, and the data tables (`.csv`, `.csv.gz`) are read from beside the modules, so all of them must sit side by side. Copy files, not symlinks. `flowcode_phenotypes.csv`, from earlier versions, is no longer read (phenotypes come from Honeychrome's cell-type database) and can be deleted.

2. Start Honeychrome from your Python environment:

   ```bash
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   honeychrome                    # PyPI install, or from a source clone after "pip install -e ."
   ```

3. Enable the plugin. Open **Application Configuration**. Each file in `~/Experiments/plugins` that ends in `_tab.py` appears as a row, "Tabbed plugin", with the checkbox **Enable FlowCode** (the name comes from `plugin_name` in the file). Tick it and accept. Honeychrome reloads the experiment and a **FlowCode** tab appears.

4. After updating any of the plugin files later, copy the new versions over the old ones and restart Honeychrome (or untick and re-tick the plugin) so the modules are re-imported.

If the tab does not appear, check `~/Experiments/honeychrome.log`. A missing companion file, or a Python package that is not installed, shows up there as an import error.

### 2.3 Optional: the AF library and fluorophore database

- Tag fluorophores must be in the experiment's fluorophore profiles (`process['profiles']`).
- The tandem check reads a `tandem.parent` column in `src/honeychrome/data/fluorophore_database.csv` (for example, PE-Cy7 → PE). Tags whose fluorophore class is "Protein tandem" with a parent listed are ticked for the check automatically.

---

## 3. Using it in the app

Open the **FlowCode** tab. The six sections are listed down the left side; the main window's sample list is hidden while the tab is open. Each has a help toggle at the top and a status line: `Done`, `Needs re-run: <reason>` or `Waiting for section N`. Sections 3 to 6 split into a row of tabs (e.g. Combo frequencies, Assess correction); the open tab is remembered. In sections 4 and 5 the whole page scrolls: scroll past the controls at the top and the tabs fill the window. Every Advanced settings tab, and the Statistics, inclusion-rule and figure options of section 6, have a **Restore defaults** button (greyed when everything is at its default). Figures have a **Size** slider, **⤢ Pop Out** (a resizable window with the matplotlib toolbar) and **Export Graphic**, and figure options set in the **Options** side panel (grid lines, the title and panel titles, font sizes, the background, and the axis ranges of volcano plots), kept with the experiment and used on screen, in Pop Out, in Export Graphic and in the exported report alike. Work through them in order.

### Section 1: Inputs

- **Backbone.** Tick the samples whose cells carry the FlowCode combinations, typically pools of transduced cells. Several samples can be ticked and their events are pooled (at most 2,000,000 events per file). Every combination needs enough bright, barcoded events to be characterised: aim for at least 50 events per combination with every tag at least 3 times its threshold. Samples with "backbone" or "FRET" in the name are ticked at first. Single-stain controls and unstained samples are not listed.
- **Matched unstained.** The unstained sample whose FSC/SSC distribution is closest to the pooled backbone, scored by the Bhattacharyya coefficient of the scatter histograms (1 = identical). Click a row to choose a different one, or **Match again** to recompute.
- **AF library for debarcoding.** The AF profiles extracted from the chosen unstained sample in the AutoSpectral AF tab. If none exist, the section shows how to extract one and stays incomplete.
- **Combinations CSV** (optional). A FlowCodeUnmix file with an `Id` column and three tag columns. Without it, every 3-tag combination of the ticked tags is used.
- **No FRET backbone: skip FRET correction (results invalid)**. For samples acquired before FRET backbone controls were run, to test the pipeline. Ticking it asks you to acknowledge, in a pop-up, that the downstream results are untrustworthy (the time is saved with the experiment). The Backbone box becomes *Samples for gates and thresholds*: the ticked barcoded samples set the FlowCode gates, the thresholds and the rescue references, and no FRET error is fitted on them (stained samples would put their marker signal into the error). See [No FRET backbone](#no-fret-backbone) below.

### Section 2: Tags & Combos

- **Tag channels.** Unmixed channels whose antigen is a known tag are ticked automatically (exact match on the canonical name, so `S` and `C` do not match CD4 or SSC). Tick or untick others by hand. The **Nearest tag** column is the other tag with the most similar reference spectrum; it becomes the default y-axis of the threshold plot.
- **Combinations.** A table with Include, Name, Tag 1–3 and (after section 4) the backbone event count. Without a CSV the plugin lists all C(n, 3) triplets with names such as `HA_V5_AU1`. Untick **Include** to drop a combination: its events are then reported as unresolved. Names become exported channels `FC <name>`, so they must be unique, non-empty and free of commas. **Save combinations CSV** writes the R format.
- **Several libraries** (Advanced options). For samples transduced with different guide libraries on the same tags. Each library has its own combinations CSV (the section 1 CSV becomes Library 1, and section 1's picker is disabled), and each sample is assigned to one library, or to *Other* for samples that belong to none (FMOs, for example). Debarcoding uses every triplet of every library, so combinations are named by their triplet (`FC AU1_C_FLAG`), and the guide names come from each library's CSV. **Assign by pattern** matches each library's sample-name pattern (a regular expression, case-insensitive) against the sample names; the default pattern finds the library's number, letter or name after `Librar`, directly or after a space or underscore (`Library1`, `Library 1`, `Library_1`, but not `Library10`). It runs once on its own when *Several libraries* is first ticked, for samples without a library. *Label* is the name shown in the results. With *Merge guides with the same name* (on by default), a guide name used by several libraries (a non-targeting control, for example) is one guide in the Analysis. The export is refused for selected samples without a library. In section 6, pairing and expanded clones are worked out within each library, a mouse never spans two libraries, and the pathway and gene-group tests treat each library as its own unit (6.6).

### Section 3: Gates & Thresholds

**FlowCode scatter gates.** Two gates on the pooled, cleaned backbone (saturated events are removed first):

- **FlowCode Singlets** (FSC-A × FSC-H): keeps events whose area/height ratio is below the 85th percentile.
- **FlowCode Cells** (FSC-A × SSC-A): a robust 70% contour around the scatter of the events most clearly positive for three tags, enlarged 1.5 times about its centre so it does not clip barcoded cells.

Drag a gate to edit it, or drag the round handle on FlowCode Cells to scale the whole gate about its centre. **Use experiment gate** copies the Cells or Singlets gate from the Raw Data tab. A third plot shows the matched unstained with the same Cells gate.

**Confirm gates** unmixes the gated backbone and matched unstained with per-cell AF (the debarcoding space) and sets a starting threshold per tag (rule in 5.2).

**Load events** reads the backbone and the matched unstained. Opening the section does not read them again: when an experiment is reopened, click it to see the gates. With section 3 done, it also rebuilds the debarcoding space so the saved gates and thresholds can be viewed and edited; the reload changes neither.

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

When the experiment is reopened, the saved run is shown without the figures that need the gated backbone (panel expression and the *Assess correction* biplots). **Show event figures** reloads and unmixes the gated backbone to rebuild them for the saved run; nothing is saved or changed. Opening the section never re-runs the characterisation.

### No FRET backbone

With *No FRET backbone* ticked in section 1:

- **Section 4** debarcodes only (*Debarcode (no FRET correction)*): combination frequencies, the per-combination medians the rescues use and the untransduced baseline. No error model, pairwise model or tandem reference is made; the heatmap, before/after and *Assess correction* tabs say the correction was not performed, and the fit table's *Error* column reads `none`.
- **Sections 5 and 6** leave the raw data unchanged before the final unmix, whatever the *correction* setting (shown disabled), and check no tandem tags. *Fix Unmix (FRET-corrected)* is not available. Each export asks for the acknowledgement again.
- **Warnings.** The status lines of sections 4, 5 and 6 are red with *WARNING: no spillover performed! Results invalid!*. Exported FCS files carry `FLOWCODECORRECTION=NONE` and `FLOWCODEWARNING`; `FlowCode_run.json` has `"fret_correction": false` and the warning; `FlowCode_debarcode_summary.csv` has a last *Warning* column. Section 6 shows the warning in red above its tabs whenever the analysed events are uncorrected (the live unmix with this characterisation, or exported files with `FLOWCODECORRECTION=NONE`, even after the override is switched off), stamps it across every figure, asks for the acknowledgement before exporting the analysis, and writes it across every page of the PDF report, on the PNG figures and at the top of `settings.txt`.
- Switching the override on or off makes sections 4 and 5 need a re-run (sections 1 to 3 are kept). The characterisation identifier differs from that of a corrected run, so files exported one way show as *older characterisation* for the other.

### Section 5: Unmix & Export

- **Samples.** All samples except unstained and single-stain controls are listed and ticked (tick **Include single-stain controls and unstained samples** to list those too). The backbone is included. Samples whose AF library has fewer than 2 spectra are greyed out. **Group** lists the folders of the listed samples (as in the sample browser, with the number of samples); **Select group** and **Deselect group** tick or untick that folder's samples and keep the rest of the selection. A warning appears for selected samples that lack the AF profiles the thresholds were set with; those profiles are added to the sample's library for the export only (the experiment's assignments are not changed).
- **Output.** Default folder `<experiment>/FlowCode Unmixed`; sub-folders mirror the raw folder.
- **Final unmix** (the line at the top): the experiment's method from the AutoSpectral tab, with the spectral corrections of each sample's group when they are in use. It is set there, not here; a change makes section 5 stale. A red note appears when Fix My Unmix has corrected a tag fluorophore.
- **Advanced settings.** The tag shift minimum, the correction mode (`combo`, `pairwise`, `none`) and the regressor.
- **Tandem tags.** Tick the tags on tandem fluorophores to check for sample-specific breakdown (5.7).
- **Combination calibration.** *Calibrate each combination's correction per sample (out-of-span)*, on by default, rescales each combination's error model by what the sample itself shows (5.8). *Brightness-dependent calibration* (off by default) lets the scale change with tag brightness. The z threshold (3), minimum out-of-span share (0.05) and minimum singlet events per combination (40) are here too. It applies only with the `combo` correction and FRET correction. After an export the result line lists, per sample, the combinations calibrated out of those identifiable, the five largest changes and the combinations that could not be calibrated.

- **FRET correction overrides** (Advanced settings). Replace a combination's FRET/error correction in every exported sample: *Off* removes it and *Scaled* multiplies it (1 is the backbone's model). Either takes the place of that combination's per-sample calibration, and the tandem breakdown is estimated on the overridden correction. They are meant for a backbone whose model does not fit the samples, such as FRET controls from another experiment: a combination whose correction moves a gating marker that the uncorrected data show no sign of needing, or makes a marker depend on tag brightness within the combination. Overrides are kept by the combination's tags, so renaming it keeps them, and are recorded in `FlowCode_run.json` and the `FLOWCODEOVERRIDES` keyword.

Press **Unmix and export**. With a slow method (spectral variants, or the noise model without its compiled kernel) a probe estimates the run time and asks you to confirm before starting; it is rough, and the real time can differ by a factor of 2 or more. When Honeychrome runs one heavy job at a time, the export is refused, with the running job named, while another one holds the application.

### Section 6: Analysis

**Gating** (first tab, view only): the experiment's Unmixed Data plots and gates, drawn with Honeychrome's own plot widget, over the FlowCode-unmixed events of the sample picked in the main window's sample browser (shown on this tab only). Choose the view (*Uncorrected*: the experiment's own unmix of the raw file through the same plan; *FlowCode standard*; *FlowCode calibrated*: with the per-sample combination calibration) and what to highlight in pink (barcoded events, nothing, one combination, or the events of a ticked phenotype with membership at least one half). The table gives each gate's % of its parent, its events and the highlighted events changing side against the uncorrected view, which is unmixed as well for that comparison. Membership is the counts' own (`SampleGating.memberships`, sample overrides included), and the view the counts use reads their live-unmix cache; the other views are cached too. Plots, axes and gates cannot be changed here: edit them in the main window.

**Source**: *Live unmix* (default) FlowCode-unmixes each sample's raw file with the current characterisation and the experiment's method, exactly as section 5 would export it, without writing files; the unmixed values are cached in `cache/flowcode_unmixed` (pruned with the App Settings' unmixing cache size), so changing the population later does not unmix again. *Exported files* reads the files section 5 wrote. Counts from the two sources are identical for the same inputs. Saved counts say which source they came from and are marked out of date when the samples, the unmixing, the population or the gating change. Each exported file is linked to its sample by its `FLOWCODESOURCE` keyword, or by the export's file name; files exported with a different combination table, or with an earlier characterisation, are listed but not used unless ticked.

**Other experiments** (with *Exported files*): the export folders (section 5) of other experiments of the same study, typically the other days of a cohort acquired over several days (see [Cohorts acquired over several days](#cohorts-acquired-over-several-days)). **Add folder…** lists every FlowCode export under the folder. Their files are counted with this experiment's samples, named by their file names (with the acquisition date, else the folder, added when a name is taken), and gated with this experiment's unmixed gates: the template boundaries, since sample overrides exist only for this experiment's samples. A file is used only when it has the same combination table (`FLOWCODECOMBOS`) and every unmixed channel of this experiment; with several libraries, its `FLOWCODELIBRARY` must name a library of this experiment. The Gating tab, *Live unmix* and phenotypes use this experiment's samples only; the phenotype shift uses the linked files, without its spillover correction (which needs *Live unmix*).

**Acquisition date**: when the samples were acquired on more than one day (the files' `$DATE`, read from the raw files with *Live unmix* and from the exported files otherwise), a covariate *acquisition date* is filled in on the Groups tab and ticked under *Adjust for*, so that day-to-day differences in staining, acquisition and FlowCode processing are not taken for group effects. When samples are paired by mouse and each mouse was acquired on one day, the pairing already accounts for the day, and the covariate is left out of the model. When every sample of one group was acquired on one day and another group on another, day and group cannot be told apart, and the statistics say so. Values typed by hand are kept; removing the covariate stops the filling.

**Populations**: tick one or more gates of the unmixed gating tree; each ticked gate is counted as a population of its own, in one pass over the samples (e.g. CD8 and Tconv, or Naive, Activated and Resident). Counting also sums each combination's marker intensities (every fluorophore of the spectral model except the FlowCode tags, on the experiment's display scale) into `cache/flowcode_analysis/marker_sums.npz` for the marker test. Only the gate boundaries come from the experiment: they are applied to the FlowCode-corrected events, with each sample's own gate overrides. To make counting again fast, the first count keeps, for each sample, only what counting uses: the events inside the gate above every counted population that is drawn on scatter, Time or the viability dye (usually the viable gate), with the channels gates are drawn on (and, for barcoded events, the marker channels), in `cache/flowcode_analysis/count_events`. Later counts read only these, so redrawing gates below that gate, or ticking other populations under it, needs no unmixing; a sample is read in full again when its file, the unmixing, that gate or a gate above it changes, or a gate is drawn on a channel that was not kept. Counts with a phenotype ticked read every sample in full.

**Phenotypes**: populations without a gate, listed under the gating tree. A phenotype is a marker definition (`CD4 Treg`: CD3+ CD4+ FoxP3+ CD8- CD19-) from Honeychrome's cell-type database (`drc_cell_type_database.csv`, the definitions Honeycluster and Automated Gating use). Each definition is resolved against the antigens typed into the spectral model through the marker database's synonyms. Only the markers on the panel have a say: a marker the panel lacks, of either sign, is left unchecked and said so, and a marker on two fluorophores leaves the definition unresolved (greyed, with the reason in its tooltip); the panel's viability dye is added as a negative marker when there is one. Definitions that read the same on the panel (alpha-beta and gamma-delta T cells on a panel without the TCRs, for example) are one phenotype, named by its signs and its members; its tooltip says which markers would separate them. Ticked, a phenotype is counted as a population of its own from the live unmix: every event inside the FlowCode Singlets and Cells gates gets, per listed marker, the probability of being positive, a logistic soft threshold in units of the sample's negative SD (the AF library's envelope null variance at the event's AF abundance plus the spillover spread at its abundances, as the AF assignment uses them; one half at 3 SD, 0.12 at 2 SD, 0.88 at 4 SD), and its membership is the product of those probabilities (positives) and their complements (negatives). The counts per combination are the memberships summed, so `in population`, `barcoded` and the counts are fractional; the marker sums are membership-weighted. The probability says "above the negative" only (no high/intermediate levels), scatter has no probability (the FlowCode gates do that), and the markers are taken as independent. Phenotypes need *Live unmix* (exported files carry no negative model), and counts that include one run the Frequencies and Markers tests only: the Counts test needs whole events and is withheld with a note. The resolution of each counted phenotype is written to `settings.txt`. New definitions belong in Honeychrome's cell-type database (canonical marker names, or any synonym the marker database knows). *Split each phenotype by congenic marker* counts each ticked phenotype as its CD45.1+ CD45.2- and CD45.1- CD45.2+ variants when the panel carries both. Pathways and Gene groups are built on the Counts test, so they are not run on counts that include a phenotype.

**Groups**: when no sample has a group, groups and a pairing variable are suggested from the sample names (repeated fields separated by spaces or underscores) and applied; **Undo** removes them and **Suggest groupings…** shows every field found. Covariates are set in their own table. The groups are a table with a sample-name pattern (regular expression, case-insensitive) and a colour each. *Assign by pattern* gives every listed sample the first group whose pattern matches its name. Double-click a colour to change it; group colours are saved with the experiment and used in every figure (volcano direction ramps, heatmap group bars, fold-change bars, PCA, QC, combination plots).

**Statistics**: *Counts* is a negative-binomial GLM per combination with the sample's library size as offset; each sample is weighted by its cell number, so small samples count for less. The *quasi-likelihood F test* (default) fits each comparison on the samples of its two groups, gives each combination a dispersion (its extra sample-to-sample noise) taken from the combinations with similar counts, so that rare combinations are not treated as quieter than they are, and corrects for low counts. On random splits of inbred mice with plenty of events (BATF chimeras, where nothing differed) it gave 3–6 % false positives at p < 0.05 instead of 5 %, and 2–4 % on FlowCode mice with tissue labels swapped within each mouse; with about 100 barcoded cells per sample (e.g. brain) 5 % on average and at most 7 % in one tissue, where a fold-change threshold with TREAT keeps the hits reliable; on gated populations of a human cohort 7–10 % for populations under 100 events, because donors vary more than the model allows for. The *Wald test* uses each combination's own dispersion and is not recommended (21–31 % on the same mice, 10–13 % on the human cohort). *Frequencies* is a moderated linear model on log2 % of the library size (half an event added to every count). *Normalisation* sets the library size: *Robust* (default) rescales the barcoded events by a trimmed ratio of sums, so a few expanded clones do not push every other combination down; *Total* is the barcoded events; *Control* uses the chosen control combinations (names with "non-targeting" are ticked first), without their own expanded clones. Pairing (e.g. by mouse) enters as a blocking term, and *Split by* runs the same comparisons within each level of a covariate. FDR is Benjamini-Hochberg within each comparison; with *TREAT* the p-value tests |log2FC| above the threshold rather than any change.

**Check hits: leave one out** (after a run with significant results) asks whether a hit rests on one mouse. The statistics are run again once without each mouse (each sample when there is no pairing), with library sizes and expanded clones worked out again, and every significant result is judged again in each refit at the current thresholds, with each refit's own FDR. The results table gains *leave one out* (e.g. ⚠ 4/6: significant in 4 of 6 refits), the range of log2FC over the refits and the *most influential* mouse, the one whose removal moves the log2FC most, with the log2FC without it; that mouse is drawn in red when the combination is plotted. ⚠ means the hit lost significance, or changed direction, without some mouse. A hit that changes direction, or shrinks towards zero, without one mouse rests on that mouse: check its samples. A borderline hit lost without any of several mice mostly reflects the loss of power with one mouse fewer. Changing the thresholds judges the refits again without running them; running statistics again clears the check. It takes about one statistics run per mouse.

**Statistics with several populations**: *Compare: sample groups* runs the group comparisons within each population (`Pancreas vs Spleen [CD8]`); library sizes and expanded clones are worked out within each population. Tick *Population contrasts* to also test, against the *reference population*, whether a group's change against the reference group differs between populations: `CD8 vs Tconv [Pancreas vs Spleen]` is log2(Pancreas/Spleen) among CD8 minus the same among Tconv. Paired by mouse, each mouse has its own baseline per population and its own group effect shared by the populations, so every mouse is compared with itself (the GLM form of the T cell type scripts' paired t-test of spleen-normalised frequencies). *Compare: populations* tests the populations against the reference population within each sample (paired by sample, split by sample group by default).

**Expanded clones**: the lab R rule. Within each non-reference group, a combination is a clone in a sample when its % of barcoded events exceeds the median + SD of that combination's % over the group's samples in which it was seen, with more than 200 events and 10 %. It is removed for that mouse in every tissue unless it is expanded in more than two mice of that group.

**Markers**: for every combination and marker, the mean intensity of the combination's cells minus a reference in the same sample. The default reference is the *tag model*: per sample and marker, the combinations' means are fitted on their tags (one additive term per tag, weighted by cells, Huber-robust), and each combination is compared with what the other combinations predict for its own tags. A tag fluorophore's spillover into a marker, shared by every combination carrying the tag, is then not taken for an effect (against tag-unbalanced controls it shows as blocks of hits). The control combinations, or every other barcoded cell, can be chosen instead. *Account for tag spillover* (on by default): the spillover spread of each combination's tags into each marker at their median brightness, from the reference spectra, the characterisation and the noise model, converted to the display scale at the sample's level of the marker, is added to the feature's standard error, SE_total² = SE_moderated² + (k × spillover)², and t, the p-value (TREAT included) and the confidence interval come from SE_total. Every feature stays tested. With *Auto (NON-Targetting)* (default) k is, per population, the smallest value from 0 to 4 in steps of 0.25 at which the control combinations have no significant marker hits; untick it to set k. The value used is shown under the setting and in `settings.txt`. Features whose uncertainty is mostly spillover (k × spillover above the moderated SE) are hatched in the heatmap; the table and `markers.csv` give the spillover, k and both standard errors. Each group's (and population's) differences are tested against zero with a moderated t, one value per sample, and combinations are families for the cluster-first FDR. A combination or reference with fewer than the minimum events in a sample has no value there; expanded clones removed from the counts are left out too. The *Markers* menu in Results picks the group whose heatmap (combinations × markers, coloured where significant), volcano and table are shown; it is independent of the *Comparison* menu. Double-click a marker volcano point or a table row for the *Marker* view: every sample's reference and combination means joined by a line, one column per group, a mouse's samples linked across groups. The export writes `markers.csv` and every group's figures.

**Pathways** (tick *Pathways*; needs the Counts test with the quasi-likelihood F test): do the knockouts of a biological process, taken together, change the counts? Small effects of several guides then add up. Each mouse GO Biological Process term (MGI annotations, bundled with the plugin) is a pathway whose genes carry a sign: +1 when the gene is part of the process or drives it, −1 when it holds the process back. Knocking out a +1 gene lowers the process and knocking out a −1 gene raises it, so if the process matters the two kinds of guide move in opposite directions, and the pathway adds them up instead of letting them cancel. The *score* is the mean of −sign × z over the pathway's guides, z being each guide's Counts result as a z-score: for *Gut vs Spleen*, a positive score means the process promotes accumulation in the gut (knockouts that lower it are depleted there, knockouts that raise it are enriched). An enhancer knockout at z = −2 and an inhibitor knockout at z = +1.5 give +1.75. The table gives the *self-contained* test (is a coordinated signed change happening in these guides at all?), the *competitive* test (signed CAMERA: does the pathway stand out from the other guides of its library?) and the *any-change* test (is anything happening, whatever the direction?); see [Mechanics: pathway and gene-group tests](#6-mechanics-pathway-and-gene-group-tests). A pathway whose guides change strongly against their signs can pass the self-contained test with a score near 0 and a low *agreement* (the share of guides moving the way the score says), so read the score, the agreement and the competitive FDR together. Settings: *min. members* (tested guides a pathway or gene group needs over all libraries; default 3), *rotations* (default 9,999; the smallest p-value is 1/(rotations + 1)) and *GO evidence* (all annotations, electronic IEA included, by default, or experimental evidence only). Guide names are matched to MGI symbols, then to unique MGI synonyms (e.g. Gm5897 is Sult2b1); **Gene names…** lists the matches and sets a gene by hand. NON-Targetting guides are never in a pathway; together they form a check that should not score. Results: *Pathway dot plot* (pathways × comparisons; colour the score, size the competitive FDR, ringed when the self-contained FDR passes), *Pathway table* and *Pathway members* (double-click a dot or a table row to fill it; double-click a guide there to plot it in every sample).

**Gene groups** (tick *Gene groups*; needs the Counts test with the quasi-likelihood F test and a gene-group table): does a family of knockouts, taken together, change the counts, for example whether integrin knockouts as a whole are lost from the gut? In **Gene groups…**, tick a built-in table of the lab's libraries (Homing, Treg transcription factors OTB01, Metabolism) or load a CSV with a `gene` and a `group` column (optional `colour`, #rrggbb, and `area`, which gives groups shades of one colour per area); the dialog shows each combination's group and sets the order and colours of the groups. With no table loaded, ticking *Gene groups* picks the built-in table that covers at least 80 % of the guides, when there is one. The table is kept with the experiment, so a newer built-in table in a later plugin version changes nothing until it is ticked again. A combination's group comes from a gene set by hand in **Gene names…**, then its own name, then its MGI symbol; the others are unassigned and not tested. Every guide counts the same way, so a group's *score* is the mean of −z over its guides: for *Gut vs Spleen*, a positive score means the group's knockouts are depleted in the gut, so the family is needed for the cells to be there. The *competitive* test (CAMERA) is the test of direction; the *any-change* test asks whether anything happens in the family in either direction (one strongly changed guide is enough, and a shift of the whole library can pass it too); the *self-contained* test is listed in the table. Groups use the pathways' *min. members* and *rotations*, and a group may hold more than half of a library. Results: *Guide allocation* (the libraries' guides per group, pooled, shown as soon as a table is loaded), *Gene group spider* (one spoke per group, one line per comparison against the reference group, at most six, **Spider comparisons…** chooses others; outside the dashed ring the knockouts are depleted; a filled point passes the competitive FDR, a ringed one the any-change FDR), *Gene group table* and *Gene group members* (double-click a point or a table row).

**Phenotype shift** (its own tab, after Results): has the mix of phenotypes among a guide's cells changed, beyond what its tags predict in the same mouse? The counts test asks whether a guide has more or fewer cells and the marker test whether a marker is brighter or dimmer; this asks whether the guide's cells have become a different kind of cell. Counting is not needed first: the run reads (with *Live unmix*, unmixes) the samples itself. Tick the populations in Data (each is clustered on its own) and set the sample groups in Groups (each group, a tissue for example, is tested on its own with one sample per mouse); then choose the *Reference* (the tag model by default, as the marker test), the number of *Clusters* (consensus metaclusters of a 10 × 10 FlowSOM grid, 5 to 20, default 20) and the clustering *Channels* (by default the panel's markers without the tags, the viability dye, congenic markers and the channels of the gates that define the populations), and press **Run phenotype shift**. Each guide in each mouse gets a cluster fingerprint (the share of its cells in each cluster), and the test asks whether the mice agree on the direction in which the guide's fingerprint differs from its reference: chance points a different way in each mouse, a real shift the same way in every mouse. Results per population and group: the *shift angle* (the size of the shift, 0° for none, 90° when the guide's cells sit in entirely different clusters), z, p and the Benjamini-Hochberg FDR, and the cluster each guide gains and loses most. Figures: the shift plot, the direction (log2 ratio of the guide's mean share to the reference's per cluster), the *direction map* (guides on the first two principal components of their mean displacements, with arrows showing which clusters each direction moves cells into) and the *summary heatmap* (every guide's shift angle in every group, coloured where significant, following the comparison heatmap's order and rows settings). Double-click a guide in the table, the shift plot or the direction map to show its cluster composition in every mouse beside its reference's. Minimum cells per mouse (10), minimum mice (3) and the FDR (0.1) are settings. **Check calibration** runs the statistics again on 5 shuffles of the guide labels: about 5 % of guides should then pass p < 0.05. With *Correct tag spillover before clustering* (on by default; needs *Live unmix*), each combination's brightness-dependent tag spillover is subtracted from its cells' markers before the populations are gated and clustered, and the coefficients are listed in the *Correction* view; the correction is used by this readout only. The first run keeps the barcoded events and up to 50,000 untransduced events per sample, inside the deepest gate the correction cannot change, in `cache/flowcode_analysis/shift_events` (tens of MB per sample); later runs read only these. Each run is saved when it finishes, shown again when the experiment is reopened, marked *Out of date* when the samples, gates, unmixing, groups or settings have changed, and added to the export when current. Mechanics in [section 7](#7-mechanics-phenotype-shift-and-tag-spillover-correction).

**Saved results**: each statistics run is saved as soon as it finishes, in `cache/flowcode_analysis/stats/` (`data.json` and one CSV per table), and shown again when the experiment is reopened, with the current FDR, |log2FC| and |Δ| thresholds, TREAT, FDR correction and spillover k applied. If the counts or any other Analysis setting (groups, design, tests, normalisation, clone removal, markers, pathways, gene groups, inclusion rules, library setup) has changed since the run, the results are still shown, with an orange *Out of date* note on the Statistics and Results tabs, and the analysis cannot be exported until statistics are run again. Changing a setting during a session clears the results from view, as before; the saved run stays until the next one replaces it.

**Results**: *View* chooses the table or figure shown, listed under the comparison, all comparisons, markers, pathways and gene groups; the arrows beside it step to the previous or next one. The *Comparison heatmap* shows every comparison's significant log2 fold changes from the Counts test (from the Frequencies test when Counts is not run); with both tests run, *Comparison heatmap (frequencies)* shows the same from the Frequencies test. Its order (by effect or alphabetical), its number of rows (1 to 3) and the number of PCA loadings drawn are set above the table, with their own **Restore defaults**. Double-click a combination in the results table, a volcano or the fold-change bars to plot it in every sample (*Combination* view: a column per group, a line per mouse, expanded clones hollow); right-click for *Open in a window*. Changing the comparison redraws only that comparison's figures.

**Export analysis** writes a folder named after the populations: the counts (`counts_all_samples.csv`, `counts_analysed.csv`, `percent_of_barcoded.csv` and the R-style `pct_matrix.csv`), `qc.csv`, `samples.csv`, `results.csv`, `library_sizes.csv`, `clones.csv`, `markers.csv`, the pathway tables (`pathways.csv`, `pathway_libraries.csv`, `pathway_members.csv`, `pathway_controls.csv`, `pathway_gene_names.csv`), the gene-group tables (`gene_groups.csv`, `gene_group_libraries.csv`, `gene_group_members.csv`, `gene_group_assignment.csv`, `gene_group_table.csv`, `guide_allocation.csv`), `settings.txt` (with the GO release and MGI GAF date, and the phenotype-shift settings), when the phenotype shift is current its tables (`shift_results.csv`, `shift_directions.csv`, `shift_cluster_counts.csv`, `shift_cluster_medians.csv`, and `shift_correction.csv`, `shift_correction_eligibility.csv` and `shift_calibration.csv` when they were made), a PNG per figure (with its figure options) and a PDF report. Tables of tests that were not run are left out.

Section 5 is marked stale ("The spectral model changed since the backbone was characterised") if the spectral model or the saved characterisation no longer matches the current reference spectra. Re-run sections 3 and 4 in that case. A characterisation saved by an older error model is also refused.

### Cohorts acquired over several days

FlowCode cohorts are often acquired over several days, each day with its own single-stained controls, unstained samples and FRET backbone. Honeychrome processes one day per experiment, and the Analysis page brings the days together:

1. Make one experiment per day and run, in each, the spectral model, the AutoSpectral steps and FlowCode sections 1 to 4 on that day's controls, with the same tag panel, the same combinations CSV (or the same libraries) and the same unmixed channels.
2. Export every day's samples in section 5.
3. In the experiment chosen for the analysis (any of the days), set section 6's *Source* to *Exported files* and add the other days' export folders under *Other experiments*. Each file is listed with its status; files with another combination table or missing channels are listed but not used.
4. Draw the unmixed gates in that experiment so they suit every day: the other days' files are gated with its template gates.
5. Set the groups and pairing as usual. The *acquisition date* covariate is filled in and adjusted for automatically when the days differ, and left out when pairing by mouse already accounts for it.

Each day's FRET correction is fitted on its own backbone, so the days' remaining differences are what the date covariate absorbs. When a whole group was acquired on one day and the other group on another, the day and the group effects cannot be told apart, and the statistics say so: spread groups across days when designing the experiment.

---

## 4. Outputs

Per exported sample: `<sample> FlowCode (Unmixed).fcs`, uncompensated, with

- the usual unmixed channels, `AF Abundance` and `AF Index`;
- `FlowCode`: for a barcoded event, the median of the final abundances of its three tags; 0 otherwise;
- `FlowCode ID`: 0 untransduced, 1 to n the combination in table order, n + 1 unresolved;
- `FC <name>`: one channel per included combination, holding the same value for that combination's events and 0 otherwise. Gate on `FC <name>` > 0 to select a combination. With several libraries the channels are named by triplet (`FC AU1_C_FLAG`) and each channel's `$PnS` is the sample's library's guide name;
- the keywords `FLOWCODEUNMIX` (plugin version), `FLOWCODECOMBOS` (`FC <name>=TAG_TAG_TAG;…`), `FLOWCODESOURCE` (the experiment sample the file was made from), `FLOWCODECHARACTERISATION` (an identifier of the thresholds, combinations and tag spectra used, which section 6 checks against the current one) and `FLOWCODECALIBRATION`; `FLOWCODEOVERRIDES` (`<name>=<scale>;…`) when FRET correction overrides are set; with several libraries, `FLOWCODELIBRARY`, `FLOWCODELIBRARYLABEL` and `FLOWCODEGUIDES` (`FC <triplet>=<guide>;…`); without FRET correction, `FLOWCODECORRECTION=NONE` and `FLOWCODEWARNING`.

In the output folder:

- `FlowCode_debarcode_summary.csv`: per sample, total events, untransduced, unresolved, rescued (too many), rescued (too few), pairwise corrected, one count column per combination, three columns (beta, z, at median) per ticked tandem tag and, with the calibration, *Calibrated combos* (calibrated / identifiable).
- `FlowCode_run.json`: backbone, thresholds, settings (the calibration settings and FRET correction overrides included), `calibration` (`on`, `brightness` or `off`), AF profiles per sample, tandem records, and per sample `combo_scales`: for each identifiable combination `s`, `slope`, `se`, `z`, `weight`, `applied`, `share`, `n`, plus the combinations `not_identifiable` and with `too_few` events, and the detector `weighting`.
- The FCS keyword `FLOWCODECALIBRATION` (`on`, `brightness` or `off`).

In the experiment's `cache` folder, section 6 keeps `flowcode_unmixed` (live-unmixed values, pruned with the App Settings' unmixing cache size), `flowcode_analysis/count_events` (the events counting uses), `flowcode_analysis/shift_events` (the events the phenotype shift uses), `flowcode_analysis/marker_sums.npz` and the saved statistics and phenotype-shift runs. Deleting these folders loses nothing that running again does not make again; the rest of `cache` holds the characterisation (`flowcode_spectra.npz`) and must be kept.

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

Each sample is debarcoded once, as a whole, before it is unmixed (steps 1–3, in blocks of 2,000,000 events for the unmixing). The rescues therefore take their per-tag shift (from the events inside the FlowCode gates) and their too-few references from the whole sample, which matters where barcoded events are sparse. The calibration (5.8), the tandem estimate (5.7) and every block of the final unmix reuse this debarcoding. Then, for each block of events, `unmix_staged` runs steps 4–8:

1. Per-cell AF unmixing with the sample's own AF library gives the debarcoding space.
2. Debarcode with the backbone thresholds, then both rescues (against the backbone's combination medians and untransduced background).
3. Fixed-basis OLS on `[S; AF.Mean]` gives the regressors `a`, measured from the baseline.
4. Subtract the characterised error from the raw data:
   - barcoded events of a combination with its own model: `raw_corrected = raw − x(a − a0) · E_g`;
   - every other event: `raw_corrected = raw − PairModel(a − a0, active tags)`, where the active tags are the combination's tags for a barcoded event and the above-threshold tags otherwise;
   - the `none` setting leaves raw unchanged, and `pairwise` uses the pairwise model for every event.
5. Rescale each calibrated combination's error by the sample's own scale (5.8).
6. Optionally subtract the tandem breakdown (5.7), estimated on the calibrated correction.
7. Final unmix of the corrected raw with the sample's AF library, by the experiment's method.
8. `FlowCode` and `FC <name>` channels: median of the final abundances of the event's three tags.

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

### 5.8 Per-sample combination calibration

The backbone's error model of a combination need not hold in a sample. On JA_007, in vivo T cells carry less error per unit tag than the backbone for some combinations, so their bright cells are over-corrected and pushed out of CD3. Marker biology and AF can move an event only within the span of the reference spectra and the AF directions; the part of a combination's error outside that span shows only as unmixing residual, so it can be measured in each sample without confusion with biology.

For each sample, at the start of its export (or live unmix), on its FlowCode singlets (at most the tandem's event limit):

1. Detectors are whitened with weights 1 / V(the sample's median raw signal) from the experiment's noise model when it unmixes with M2pg; otherwise 1 / the median raw signal (Poisson-like, floored at 1 % of the largest detector median; recorded as `poisson`).
2. The span is the reference spectra, the sample's AF library mean and its top 3 principal directions. P projects onto its complement. (The whole AF library would span every detector.)
3. y = P(raw) − the median P(raw) of the untransduced singlets.
4. For each combination with its own error model and at least 40 singlet events: x its `combo_features`, e = x · P(E_g). The out-of-span share is Σ‖e‖² / Σ‖x · W · E_g‖²; below 0.05 the combination is *not identifiable* and keeps the backbone's model.
5. s = Σ⟨y, e⟩ / Σ‖e‖², its SE from 100 bootstrap resamples (seeded by the sample), z = (s − 1) / SE, weight `max(0, 1 − (z_min / z)²)` and the applied scale clip(1 + weight · (s − 1), 0, 1.5).
6. With the brightness option, s0 + s1 · u (u the log on-tag fold sum minus its median in the combination), z the bootstrap Mahalanobis distance of (s0 − 1, s1), and the per-event scale clipped to [0, 1.5].
7. `(1 − s) · x · E_g` is added back to every event of the combination (all events, not only singlets). Pairwise-model events are unchanged.

The backbone defines scale 1: its error model is its own, so its scales are 1. Scales are not borrowed across combinations (the over-correction is specific to each combination, not to its tags), so a combination whose error lies inside the span (C_FLAG_HA) stays as the backbone model has it. The calibration is not part of the characterisation and does not apply to section 4's figures.

### 5.9 Guards

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

## 6. Mechanics: pathway and gene-group tests

Both analyses ask whether a set of knockouts, taken together, changes the counts, so that small, consistent effects of several guides add up to something a single guide's test would miss. They start from the Counts test of the statistics run (quasi-likelihood F test) and are made for every comparison of that run.

### 6.1 Per-guide statistic

Each guide's quasi-likelihood t (signed √F, low-count adjusted, on df_total = residual df + d0) is turned into a z-score with the same p-value:

```
z = Φ⁻¹( F_t(t; df_total) )
```

Guides with different degrees of freedom are then on one scale (limma's roast and camera also work on z). The fit behind each t is rebuilt inside the plugin from the inputs the statistics run recorded, with Honeychrome's own counts functions; its residual df + d0 must equal the run's for every guide, or the analysis stops.

### 6.2 The sets

**Pathways.** The bundled table holds one signed set per mouse GO Biological Process term X, built offline from `go-basic.obo` and the MGI GAF:

- genes annotated to X, or to an is_a/part_of descendant of X: +1 (−1 for qualifiers ending in `negative_effect`);
- genes annotated to a term that positively regulates X (or its descendants): +1; one that negatively regulates X: −1; two regulation steps multiply (negative regulation of a negative regulator is +1); plain "regulates" gives no member;
- regulation precedence: a gene's regulation sign replaces its plain-membership sign, and a gene whose regulation signs conflict is dropped from that set;
- NOT annotations are ignored, and sets with more than 2,000 genes are left out.

In the app, each set is cut to the guides that were tested. It needs at least *min. members* tested guides over all libraries (default 3, never fewer than 2) and may hold at most half of a library's tested guides (GO is redundant, and very large sets say little here). Sets with the same signed membership after the cut are one row, listing all their GO names. With *Experimental evidence only*, a gene stays in a set when its best annotation there is experimental; its sign still comes from all its annotations.

**Gene groups.** The groups of the gene-group table, with every sign +1 (a family has no enhancer/inhibitor split). A group may hold the whole library, and each gene belongs to one group only.

**Matching guides to genes.** A combination name is matched, ignoring case, to a gene set by hand in **Gene names…**, then (gene groups only) to a gene of the table, then to an MGI symbol, then to a unique MGI synonym. NON-Targetting combinations are never members. A guide shared by several libraries after identically named guides are merged is not a member either; the merged NON-Targetting guides form the control check instead (6.6).

### 6.3 Score

Each guide contributes −sign × z (pathways) or −z (gene groups): the minus turns a knockout's depletion into evidence that the process, or the family, is needed in the first-named group. The score is the mean contribution over the set's guides. Example for *Gut vs Spleen*: process X promotes gut accumulation; an enhancer knockout (sign +1) is depleted in the gut (z = −2) and contributes +2; an inhibitor knockout (sign −1) is enriched (z = +1.5) and contributes +1.5; score +1.75. *Agreement* is the share of the set's guides whose contribution has the sign of the score.

### 6.4 Self-contained test (rotation)

In words: is a coordinated change happening in these guides at all? The test makes many "nothing is happening" versions of the data by turning each guide's leftover mouse-to-mouse wobble in a random direction, the same direction for every guide, so guides whose wobble rises and falls together (counts are compositional: when some guides gain share, others lose it) keep doing so. The p-value is the share of these versions whose score is at least as far from zero as the real one. It is after limma's roast (Wu et al. 2010).

The mathematics, per library (each library has its own mice, so it is its own rotation unit):

- n samples, design X with p estimable columns (the tested one last); Q is the orthonormal basis of the residual space of X (n − p columns), the same for every guide.
- A guide's wobble is its vector of Pearson residuals r = (y − μ) / √(μ(1 + αμ)) from the quasi-likelihood fit (zero where the guide was not measured, e.g. a removed clone), and e = Qᵀr / |Qᵀr| its direction.
- With D = n − p + 1 and a₀ chosen so that √(D−1)·a₀ / √(1 − a₀²) has, on t with D − 1 df, the same lower-tail probability as z on N(0, 1), the guide is the unit vector v = (a₀, √(1 − a₀²)·e) in R^D.
- A rotation is a random unit vector u, uniform on the sphere, shared by all guides. The guide's rotated z is

  ```
  a  = u · v
  z* = Φ⁻¹( F_t( √(D−1)·a / √(1 − a²); D−1 ) )
  ```

- For any v, a is distributed as the first coordinate of a uniform unit vector, so z* is exactly N(0, 1) under the rotations: every guide's null matches its own test, also for guides with few events. The shared u keeps the guides' correlations (v_g · v_h). The unrotated direction gives back the observed z, so the observed data are one of the rotations, and p = (b + 1) / (R + 1), with b the number of the R rotations whose |score| is at least the observed one ("percentile-matched rotation").

### 6.5 Competitive test (signed CAMERA)

In words: does this set's signed change stand out from the other guides of the same library, allowing for its guides moving together? It is after limma's camera (Wu & Smyth 2012); for gene groups (every sign +1) it is camera with an estimated inter-gene correlation, and it is the test of direction.

In one library with G tested guides, m of them in the set, with knockout signs k = −sign:

```
x_g  = k_g (z_g − mean z of the other guides)
δ    = mean(x)
σ²   = pooled variance of x within the set and of z within the others   (G − 2 df)
VIF  = m × mean_i( mean_g k_g U_gi )²      (U: the guides' residual directions, scaled to mean square 1; at least 1)
t    = δ / √( σ² (VIF/m + mean(k)² / (G − m)) )   on min(n − p, G − 2) df
```

### 6.6 Any-change test, libraries, FDR and the control check

**Any change** (limma's "mixed" alternative): is anything happening in these guides, whatever the direction? The statistic is the mean of z² over the set's guides, compared with its value on the same rotations (one-sided: only large values count). Opposite movements add up instead of cancelling, so a family whose members are needed in different tissues still shows up; one strongly changed guide is enough, so read it with the leading guides.

**Several libraries.** Each library is tested on its own mice and the results are combined by a weighted Stouffer sum, z = Σ √n_l·z_l / √Σ n_l, with n_l the set's guides in library l and z_l = Φ⁻¹(1 − p_l/2) × sign(score_l) (any change: z_l = Φ⁻¹(1 − p_l), one-sided). Mice never carry two libraries, so the libraries are independent. Each library's own result is listed.

**FDR.** Benjamini-Hochberg within each comparison, separately for each test.

**Control check.** The NON-Targetting guides form one more set, rotated over the samples of all their libraries. It should not score; `pathway_controls.csv` gives its result.

### 6.7 Deliberate simplifications

- The rotation does not re-estimate each guide's noise level on every rotation (limma's roast does); that uncertainty enters through each guide's own quasi-likelihood p-value instead.
- A guide shared by several libraries is never a set member (6.2).
- With *Experimental evidence only*, membership follows the gene's best annotation in the set, its sign all of its annotations.
- Each gene belongs to one gene group only.

---

## 7. Mechanics: phenotype shift and tag spillover correction

### 7.1 Cluster fingerprints

The barcoded cells of each analysed population, pooled over the samples (up to 10,000 per sample for training), are clustered on the display scale with FlowSOM: a 10 × 10 self-organising map, merged into *K* consensus metaclusters (default 20). Every barcoded cell is then assigned to a cluster. For guide g in mouse i, the fingerprint p_gi is the share of the guide's cells in each cluster. The reference fingerprint r_gi comes from the same mouse:

| Reference | Definition |
|---|---|
| Tag model (default) | What the guide's three tags predict from the other guides of the mouse: the cluster shares fitted on the tags (one additive term per tag, as the marker test's tag model), within each library. A tag's spillover that moves every combination carrying the tag is then not taken for an effect |
| Control combinations | The pooled control combinations of the sample (a control is compared with the other controls) |
| All other barcoded cells | Every other barcoded cell of the sample |

A guide is tested in a group when at least 3 mice have at least 10 of its cells.

### 7.2 The agreement statistic

In words: in one mouse, with 50–100 cells, chance alone moves a fingerprint by about 11–15°, so one mouse cannot show a shift. But chance points a different way in each mouse, while a real shift points the same way in every mouse; the test asks whether the mice agree.

For guide g in mouse i, the displacement over the K clusters is

```
d_i = (p_gi − r_gi) / (2 √r̄)
```

with r̄ the pooled cluster shares of the population and sample group, so that every cluster is on the same noise scale and a rare cluster is not drowned by a large one. The agreement between mice is

```
U = mean over pairs of different mice (i ≠ j) of d_i · d_j
```

about 0 when the mice point in unrelated directions, positive when they agree. Its spread under no effect is estimated from the products themselves, as if each mouse's sign were flipped at random; z = U / SD is read against a t distribution with (mice − 1) degrees of freedom, one-sided (only agreement is a shift). P-values are adjusted by Benjamini-Hochberg within each population and sample group. With few mice the smallest possible p is limited: z is at most √(pairs of mice), so the smallest p is about 0.017 with 5 mice, 0.006 with 6, 0.002 with 7 and 0.0006 with 8.

The shift angle 2·asin(√|U| / 2), signed like U, is the size: 0° for no shift, negative when the mice disagree, near 90° when the cells have moved to entirely different clusters. The direction is each cluster's mean displacement, shown as the log2 ratio of the guide's mean share to the reference's.

Calibration: on JA_007 label shuffles the test gave 2.7–3.6 % at p < 0.05, and detected 10 % of a guide's cells moving into one cluster about 80 % of the time (5–7 mice, 50–115 cells per guide per mouse). **Check calibration** repeats the shuffle check on the experiment's own data.

### 7.3 Tag spillover correction

In words: a guide's tags cannot change where truly negative cells sit on a marker. If the negative cells of one combination move up or down a marker with how bright that combination's tags are, the movement is spillover or leftover FRET error, not biology. It is measured per combination and marker from the combination's own cells and subtracted cell by cell; cells without tags are not changed.

For one sample group, combination and marker, pooled over the group's mice (up to 2,000 of the combination's cells per mouse in the *fit gate*, by default the viable gate), the marker value y is modelled on the same six tag terms as the backbone's error model (5.4), with a_k the tag's abundance divided by its debarcoding threshold:

```
y = c0 + b1·a1 + b2·a2 + b3·a3 + b12·a1·a2 + b13·a1·a3 + b23·a2·a3
```

- **Robust fit.** Tukey biweight (tuning 4.685): cells more than 4.685 robust SDs from the current fit get no weight, so on a marker whose main population is negative the fit follows the negatives and ignores the positives. It starts from a Theil-Sen line through the densest points of ten brightness bins, so a bin whose main peak happens to be the positives cannot tilt it; a small ridge (0.001) keeps the correlated product terms stable. At least 200 cells are needed.
- **Only where there is evidence.** The coefficients are estimated once per group, so their estimation noise would move a combination's cells the same way in every mouse, which the shift test would read as a consistent shift. A combination's marker is corrected only when its six tag terms are jointly significant (F test of the weighted residual sum of squares against an intercept alone, p below 0.01 by default; 1.0–2.5 % false corrections at p < 0.01 on synthetic negatives without tag dependence).
- **Which markers.** A marker is corrected only when at least 30 % of the untransduced cells of the *eligibility gate* are negative by the plugin's negative model (phenotype positivity below one half); otherwise its main peak may be the positives, and moving it would erase real expression. The eligibility gate is by default the deepest gate above the populations that still holds 80 % of the barcoded cells of the fit gate (T cells, for example).
- **Anchored markers.** Markers whose negatives are better seen elsewhere (FoxP3 and Helios by default, absent from CD8 T cells) are fitted on the combination's cells in the *anchor gate* (by default the CD8 T-cell gate) and applied everywhere; anchoring is off without such a gate.
- **Applying it.** Only the six tag terms are subtracted (c0 is kept), so a cell without tags is unchanged. The coefficients are estimated in a first pass over the samples and applied in a second, before the populations are gated and clustered.

### 7.4 Kept events

The phenotype shift needs only a small part of each sample: the barcoded events and some untransduced ones inside the *selection gate*, the deepest gate above every analysed population (and the fit, eligibility and anchor gates) that is drawn only on channels the correction never changes (scatter, the viability dye, tags). The correction can move an event across a gate drawn on a marker (FoxP3 decides who is a Treg), but not across these, so every event that can end up in a population after correction is kept: the barcoded events, up to 50,000 random untransduced ones (the 30 % check needs no more) and, for each per-sample axis (Time), the event with the largest value, so that per-sample gates are scaled as on the whole sample. They are made again only when the unmixing, the sample's file, or the selection gate and the gates above it change.

### 7.5 Deliberate simplifications

- One robust correction fit per sample group, combination and marker, pooled over the group's samples (no per-sample term), applied only with evidence.
- Eligibility is judged once per group from the untransduced cells' negative share, with a fixed 30 % rule.
- Anchoring is a list of markers fitted on one gate, and is off when that gate is missing.
- The clusters are fitted once per population over all groups, so a phenotype present in only one tissue shares its clusters with the others.

---

## 8. Differences from the R package

- AF libraries come from the AutoSpectral AF tab (extracted on the whole unstained file), not from `get.af.spectra()` on the FlowCode-gated unstained. `AF.Mean` is still taken from the gated unstained.
- Only the FSC-A/FSC-H singlet pass is applied; the SSC-A/SSC-H pass is not.
- The landmark gate uses scikit-learn's FastMCD, not `MASS::cov.rob("mcd")`.
- The rescue transform is a fixed Honeychrome logicle, not AutoSpectral's cytometer-specific biexponential.
- The error model is a through-origin fit with pair products measured from the untransduced median, with a shared pairwise fallback. It replaces R's centred per-combination fit and global error.
- The too-few-tag rescue, the export AF library (assigned profiles plus the profiles the thresholds were set with) and the tandem-breakdown check have no R counterpart.
- Per-sample threshold overrides are not available.
