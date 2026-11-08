"""
flowcode_analysis_page.py
-------------------------
Page 6 of the FlowCode plugin: differential analysis of FlowCode barcode
distributions between sample groups or between populations, in populations
gated on the FlowCode-unmixed data, and of the combinations' marker
intensities.

Sections
    Gating      the experiment's unmixed gates over one sample's FlowCode-
                unmixed events (uncorrected, standard or calibrated), view
                only (flowcode_gating_view.py)
    Data        source (live unmix of the raw files, or the files section 5
                exported), linked samples, populations (each ticked unmixed
                gate), combinations and inclusion rules, counting
    Groups      sample groups, pairing and covariates (suggested from the
                sample names when nothing is assigned)
    Statistics  tests (counts, frequencies, markers), what is compared
                (sample groups within each population, with optional
                population contrasts, or populations within each sample),
                pairing, split, thresholds
    Results     results table and figures per comparison, comparison
                heatmap, sample PCA, QC, and the marker tabs of one group
    Phenotype shift
                whether each guide's cells move to other phenotypes
                (cluster fingerprints against the tag model), with its own
                settings, run and figures (flowcode_shift_view.py)
    Export      CSV tables, figures and a PDF report in a folder named after
                the populations

With several populations each row of the counts is one population of one
sample (``flowcode_analysis.unit_id``); groups, pairing and covariates stay
per sample and are carried to each of its populations.

With several libraries (section 2) the counts kept per triplet are named by
each sample's library from the current setup, NaN where a guide is not in
the sample's library (``flowcode_analysis.library_features``). Samples
assigned to Other, and samples that need no library (backbones), are
unticked by default and have no guides; samples without a library are left
out of the tables and statistics are refused until they are assigned or
unticked. Pairing levels are qualified by library, Library is offered
under Split by, and the statistics work out library sizes, clones and group
sizes within each library (flowcode_stats.py).

With *Live unmix* the page FlowCode-unmixes each sample's raw file as
section 5 would export it (flowcode_export.live_sample_events, cached in
``cache/flowcode_unmixed``); with *Exported files* it reads the files
section 5 wrote. Either way it applies the experiment's unmixed gate
boundaries to those events (see flowcode_analysis.py).

Numerics are in flowcode_analysis.py and flowcode_stats.py; this module
only holds widgets. Not named ``*_tab.py``, so the plugin loader does not
load it as a tab of its own.
"""

from __future__ import annotations

import hashlib
import json
import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QRadioButton, QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem,
    QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

import honeychrome.settings as hc_settings
from honeychrome.controller_components import differential_stats as ds
from honeychrome.view_components import suggest_groupings_dialog as sgd
from honeychrome.view_components.copyable_table_widget import CopyableTableWidget
from honeychrome.view_components.help_toggle_widget import WheelBlocker

import flowcode_analysis as fa
import flowcode_count_cache as fcnt
import flowcode_figure_panel as fcfp
import flowcode_gating_view as fcgv
import flowcode_gene_groups as fgg
import flowcode_pathways as fpw
import flowcode_shift_view as fshv
import flowcode_stats as fst
import flowcode_view_stack as fvs
from flowcode_state import NO_FRET_WARNING, OTHER_LIBRARY, library_label

logger = logging.getLogger(__name__)

SECTION = 6
TITLE = 'Analysis'
ANALYSIS_SUBDIR = 'FlowCode Analysis'

_ERROR_STYLE = 'color: #d62728;'
_DONE_STYLE = 'color: #2ca02c;'
_WARN_STYLE = 'color: #cc6600;'
_HINT_STYLE = 'color: grey; font-style: italic;'
_INVALID_STYLE = 'color: #d62728; font-weight: bold;'

CONTRAST_MODES = {'reference': 'Each group against a reference', 'pairwise': 'Every pair of groups'}
COMPARE = {'groups': 'Sample groups (within each population)', 'populations': 'Populations (within each sample)'}
MARKER_REFERENCES = {'tags': 'Tag model (other combinations, same tags)', 'controls': 'Control combinations',
                     'all': 'All other barcoded cells'}
FDR_SCOPES = {'per_comparison': 'Within each comparison', 'global': 'Pooled over all comparisons'}
COUNT_TESTS = {'ql': 'Quasi-likelihood F test (default)', 'wald': 'Wald test (not recommended)'}
NORMALISATIONS = {'robust': 'Robust (trimmed; clones cannot shift it)', 'total': 'Total barcoded events',
                  'control': 'Control combinations'}
SAMPLE_COLUMN_MAX_WIDTH = 280
COVARIATE_COLUMN_MIN_WIDTH = 110
HEATMAP_ORDERS = {'effect': 'By effect', 'alphabetical': 'Alphabetical'}
# Figures over every comparison, kept when only the chosen comparison changes.
SUMMARY_FIGURES = ('comparison_heatmap', 'comparison_heatmap_freq', 'sample_pca', 'qc')
# Marker test tabs, drawn for the chosen marker group.
MARKER_FIGURES = ('marker_table', 'marker_heatmap', 'marker_volcano', 'marker_combination')
# Pathway tabs, after the marker tabs: the dot plot over every comparison,
# and the chosen comparison's table and chosen pathway's members.
PATHWAY_FIGURES = ('pathway_dot_plot', 'pathway_table', 'pathway_members')
PATHWAY_EVIDENCE = {'all': 'All annotations (IEA included)', 'experimental': 'Experimental evidence only'}
# Gene-group tabs, after the pathway tabs: the guide allocation and the
# spider over every comparison, the chosen comparison's table and the chosen
# group's guides.
GENE_GROUP_FIGURES = ('guide_allocation', 'gene_group_spider', 'gene_group_table', 'gene_group_members')
# Settings put back by the Restore defaults buttons. The design (groups,
# reference, pairing, covariates, split, chosen markers and controls), the
# pathway gene names set by hand and the gene-group table are kept.
STATS_DEFAULT_KEYS = ('run_counts', 'run_freq', 'run_markers', 'counts_method', 'normalisation', 'remove_clones',
                      'clone_min_count', 'clone_min_pct', 'clone_max_levels', 'marker_reference',
                      'marker_spillover', 'marker_spillover_k', 'marker_min_events', 'marker_threshold',
                      'pval_threshold', 'fc_threshold', 'use_treat', 'fdr_scope', 'run_pathways',
                      'pathway_min_size', 'pathway_rotations', 'pathway_evidence', 'run_gene_groups')
# Statistics settings applied to stored results without running them again.
THRESHOLD_KEYS = ('pval_threshold', 'fc_threshold', 'use_treat', 'fdr_scope', 'marker_threshold',
                  'marker_spillover_k')
RULE_DEFAULT_KEYS = ('min_combo_events', 'min_sample_events')
FIGURE_DEFAULT_KEYS = ('heatmap_order', 'heatmap_rows', 'n_loadings')
RESTORE_TOOLTIP = 'Put these settings back to their default values.'
# Settings a statistics run does not depend on: figure options, export
# folder, group colours and patterns, and the thresholds, which are applied
# again to saved results.
STATS_INPUT_EXCLUDED = ('heatmap_order', 'heatmap_rows', 'n_loadings', 'export_root', 'group_colors',
                        'group_patterns', 'counts', 'gene_group_file', 'gene_group_areas', 'gene_group_colours',
                        'gene_group_order', 'gene_group_comparisons', 'shift') + THRESHOLD_KEYS
RESTORING_TEXT = 'Reading the saved results.'
STALE_TEXT = ('Out of date: these saved results were made with other counts or settings. Run statistics again '
              'to update them.')
LOO_TOOLTIP = ('Run the statistics again without each mouse in turn (each sample when unpaired) and show, for every '
               'significant result, in how many of these refits it stays significant and which mouse moves its '
               'log2FC most. Takes about one statistics run per mouse.')
LOO_COLUMNS = ('leave one out', 'LOO log2FC', 'LOO most influential')


def _loo_display(loo: pd.DataFrame) -> pd.DataFrame:
    """The leave-one-out columns of the results table as text
    (``LOO_COLUMNS``): the flag with significant refits / refits, the
    log2FC range over the refits and the most influential level with the
    log2FC without it."""
    out = pd.DataFrame(index=loo.index)
    out[LOO_COLUMNS[0]] = [f'{flag} {int(sig)}/{int(n)}'.strip()
                           for flag, sig, n in zip(loo['loo_flag'], loo['loo_significant'], loo['loo_refits'])]
    out[LOO_COLUMNS[1]] = [f'{a:+.2f} to {b:+.2f}' if np.isfinite(a) else ''
                           for a, b in zip(loo['loo_logFC_min'], loo['loo_logFC_max'])]
    out[LOO_COLUMNS[2]] = [f'{name} ({v:+.2f})' if name else ''
                           for name, v in zip(loo['loo_influential'], loo['loo_influential_logFC'])]
    return out


def _figure_set(key: str) -> str:
    if (key in SUMMARY_FIGURES or str(key).startswith('sample_pca ')
            or key in ('pathway_dot_plot', 'guide_allocation', 'gene_group_spider')):
        return 'summary'
    if key in MARKER_FIGURES:
        return 'markers'
    return 'comparison'


def _figure_rank(key) -> int:
    """View order: the comparison's figures, its combination plot, the
    figures over every comparison, the marker views, the pathway views, then
    the gene-group views."""
    if key == 'combination':
        return 1
    if key in PATHWAY_FIGURES:
        return 4 + PATHWAY_FIGURES.index(key)
    if key in GENE_GROUP_FIGURES:
        return 4 + len(PATHWAY_FIGURES) + GENE_GROUP_FIGURES.index(key)
    return {'comparison': 0, 'summary': 2, 'markers': 3}[_figure_set(key)] if key else 0


def _figure_section(key) -> str:
    """The heading a view is listed under in the results drop-down."""
    if key in PATHWAY_FIGURES:
        return 'Pathways'
    if key in GENE_GROUP_FIGURES:
        return 'Gene groups'
    return ('This comparison', 'This comparison', 'All comparisons', 'Markers')[_figure_rank(key)]

HELP = (
    '<p>Tests whether FlowCode combinations (barcodes, i.e. guides) are distributed differently '
    'between groups of samples or between populations, and whether they change the cells\' markers.</p>'
    '<p><b>Source.</b> <i>Live unmix</i> (default) FlowCode-unmixes each sample\'s raw file with the '
    'current characterisation and the experiment\'s unmixing method, exactly as section 5 would export '
    'it, without writing files; the unmixed values are cached in the experiment\'s '
    '<tt>cache/flowcode_unmixed</tt> folder, so changing the population later does not unmix again. '
    '<i>Exported files</i> reads the files section 5 wrote: each sample is linked to its file by the '
    '<tt>FLOWCODESOURCE</tt> keyword, or by the export\'s file name, and files exported with a different '
    'combination table, or with an earlier characterisation, are listed but not used unless ticked.</p>'
    '<p><b>Other experiments.</b> When a cohort is acquired over several days, each day with its own '
    'controls, process each day in its own experiment (sections 1 to 5) and export its samples. Then, in one '
    'of these experiments, choose <i>Exported files</i> and add the other days\' export folders under '
    '<i>Other experiments</i>. Their files are counted with this experiment\'s samples, named by their file '
    'names, and gated with this experiment\'s unmixed gates (the template boundaries: sample overrides exist '
    'only for this experiment\'s samples, and the Gating tab shows only its samples). A file is used only when '
    'it has the same combination table (<tt>FLOWCODECOMBOS</tt>) and every unmixed channel of this '
    'experiment; with several libraries, its <tt>FLOWCODELIBRARY</tt> must name a library of this experiment. '
    'Live unmix and phenotypes do not use linked files.</p>'
    '<p><b>Acquisition date.</b> When the samples were acquired on more than one day (the files\' '
    '<tt>$DATE</tt>), a covariate <i>acquisition date</i> is filled in on the Groups tab and ticked under '
    '<i>Adjust for</i>, so that day-to-day differences in staining, acquisition and FlowCode processing are '
    'not taken for group effects. When samples are paired by mouse and each mouse was acquired on one day, '
    'the pairing already accounts for the day and the covariate is left out of the model. When every sample '
    'of a group was acquired on one day and another group on another, day and group cannot be told apart and '
    'the statistics say so. Values typed by hand are kept; removing the covariate stops the filling.</p>'
    '<p><b>Data.</b> Tick the populations in the unmixed gating tree; each ticked gate is counted as a population of '
    'its own (e.g. CD8 and Tconv, or Naive, Activated and Resident). Only the '
    'gate <i>boundaries</i> come from the experiment: they are applied to the FlowCode-corrected '
    'events, with each sample\'s own gate overrides, so the population is defined on the data being '
    'analysed. Gates whose edge lies on a tag channel or a FRET-affected marker can catch slightly '
    'different cells than in the main window; the events in the population are listed per sample so this '
    'can be checked. Unstained samples, single-stain controls (also copies of them elsewhere in the raw '
    'folder, matched by file name) and bead samples are left out unless their <i>Include</i> tick box is '
    'ticked. <b>Phenotypes</b> are populations without a gate: a marker definition (CD3+ CD4+ FoxP3+ CD8-, '
    'from Honeychrome\'s cell-type database) resolved '
    'against the antigens of the spectral model, where every event inside the FlowCode gates gets a '
    'membership between 0 and 1 (each marker is a soft threshold at 3 SD of the sample\'s negative, from the '
    'AF library\'s envelope and the spillover spread) and the counts are memberships summed. Only the '
    'markers on the panel have a say: a marker the panel lacks is left unchecked, and the viability dye is '
    'added when there is one. Definitions that read the same on this panel (for example alpha-beta and '
    'gamma-delta T cells without the TCRs) are one phenotype, named by its signs and its members; the tooltip '
    'says which markers would separate them. <i>Split by congenic marker</i> counts each '
    'ticked phenotype as its CD45.1+ CD45.2- and CD45.1- CD45.2+ variants when the panel carries both. '
    'Phenotypes need <i>Live unmix</i>, and their fractional counts take the Frequencies test only; the Gating '
    'tab can highlight a phenotype\'s events (membership at least one half) over the gates. '
    '<b>Select all</b> and <b>Deselect all</b> tick or untick every sample in the table. '
    '<b>Count events</b> gates every sample and counts its events per combination in each population, '
    'and sums each combination\'s marker intensities for the marker test; the counts are kept with the '
    'experiment and the marker sums in its <tt>cache/flowcode_analysis</tt> folder. To make counting again fast, '
    'the first count keeps, for each sample, only what counting uses: the events inside the gate above every '
    'counted population that is drawn on scatter, Time or the viability dye (usually the viable gate), with the '
    'channels gates are drawn on (and, for barcoded events, the marker channels), in '
    '<tt>cache/flowcode_analysis/count_events</tt>. Later counts read only these, so redrawing gates below that '
    'gate or ticking other populations under it needs no unmixing; a sample is read in full again when its file, '
    'the unmixing, that gate or a gate above it changes, or a gate is drawn on a channel not kept. Counts with '
    'phenotypes ticked read every sample in full.</p>'
    '<p><b>Inclusion rules</b> (applied in this order): unticked combinations are left out; combinations '
    'with fewer events than the minimum, summed over all samples, are left out; samples with fewer '
    'barcoded events than the minimum are left out. Frequencies are among barcoded cells of the '
    'population, over the remaining combinations.</p>'
    '<p><b>Groups.</b> When no sample has a group, groups and a pairing variable are suggested from the '
    'sample names (repeated fields separated by spaces or underscores) and applied; <b>Undo</b> removes '
    'them. <b>Suggest groupings</b> shows every field found. Each group can have a sample-name pattern '
    '(a regular expression, case-insensitive): <b>Assign by pattern</b> gives every listed sample the first '
    'group whose pattern matches its name. Double-click a group\'s colour to change it; the colours are used '
    'in every figure.</p>'
    '<p><b>Statistics.</b> <i>Counts</i>: a negative-binomial GLM per combination with the sample\'s '
    'library size as offset; each sample is weighted by its cell number, so small samples count for '
    'less. The <i>quasi-likelihood F test</i> (default) fits each comparison on the samples of its two groups, '
    'and gives each combination a dispersion (its extra sample-to-sample noise) taken from the combinations '
    'with similar counts, so rare combinations are not treated as quieter than they are. On random splits of '
    'inbred mice with plenty of events (BATF chimeras, where nothing differed) it found 3–6 % false positives '
    'at p < 0.05 instead of 5 %, and 2–4 % on FlowCode mice with tissue labels swapped within each mouse; on '
    'gated populations of a human cohort it found 7–10 % for populations under 100 events, because donors '
    'vary more than the model allows for. The <i>Wald test</i> uses each combination\'s own dispersion and is '
    'not recommended: 21–31 % on the same mice and 10–13 % on the human cohort. With few barcoded cells per '
    'sample (about 100, e.g. brain) the quasi-likelihood test stays close to 5 % (5 % on average, at most 7 % '
    'in one tissue); a fold-change threshold with TREAT keeps the hits reliable. <i>Frequencies</i>: a moderated '
    'linear model on log2 % of the library size (half an event added to every count). '
    '<i>Normalisation</i> sets the library size: <i>Robust</i> (default) rescales the barcoded events by a '
    'trimmed ratio of sums, so a few expanded clones do not push every other combination down; '
    '<i>Total</i> is the barcoded events; <i>Control</i> uses the chosen control combinations (names with '
    '"non-targeting" are ticked first), without their own expanded clones. <i>Remove expanded clones</i> '
    'applies the lab R rule in the non-reference groups: a combination is a clone in a sample when its % of '
    'barcoded events is above the median + SD of that combination\'s % over the group\'s samples in which it was '
    'seen, with more than 200 events and 10 %; it is removed for that mouse in every tissue, unless it is '
    'expanded in more than two mice. Library sizes and clones are worked out within each population. '
    'Pairing (e.g. by mouse) enters as a blocking term. <i>Split by</i> runs the same '
    'comparisons within each level of a covariate (e.g. COPD vs healthy within each tissue). FDR is '
    'Benjamini-Hochberg within each comparison by default, as in the lab\'s R scripts; with <i>TREAT</i> '
    'the p-value tests |log2FC| above the threshold rather than a non-zero change.</p>'
    '<p><b>Check hits: leave one out</b> (after a run with significant results) asks whether a hit rests on '
    'one mouse. The statistics are run again once without each mouse (each sample when there is no pairing), '
    'with library sizes and expanded clones worked out again, and every significant result is judged again in '
    'each refit at the current thresholds. The results table gains three columns: <i>leave one out</i> '
    '(e.g. ⚠ 4/6: significant in 4 of 6 refits), the range of log2FC over the refits, and the <i>most '
    'influential</i> mouse, the one whose removal moves the log2FC most, with the log2FC without it; that mouse '
    'is drawn in red when the combination is plotted. ⚠ means the hit lost significance, or changed direction, '
    'without some mouse. A hit that changes direction, or shrinks towards zero, without one mouse rests on that '
    'mouse: check its samples. A borderline hit lost without any of several mice mostly reflects the loss of '
    'power with one mouse fewer, not one animal driving it. Changing the thresholds judges the refits again '
    'without running them; running statistics again clears the check. It takes about one statistics run per '
    'mouse.</p>'
    '<p><b>Several populations.</b> <i>Compare: sample groups</i> runs the group comparisons within each '
    'population (labelled e.g. <i>Pancreas vs Spleen [CD8]</i>). <i>Population contrasts</i> also test, for '
    'each population against the <i>reference population</i>, whether a group\'s change against the reference '
    'group differs between them: <i>CD8 vs Tconv [Pancreas vs Spleen]</i> is the log2 fold change '
    '(Pancreas/Spleen among CD8) minus the same among Tconv, with each mouse compared with itself when '
    'paired by mouse (the spleen-normalised comparison of the lab\'s T cell type scripts). '
    '<i>Compare: populations</i> compares the populations with the reference population, paired by sample '
    'and split by sample group by default (e.g. Resident vs Naive within each tissue).</p>'
    '<p><b>Markers.</b> For each combination and marker, the mean intensity of the combination\'s cells minus '
    'a reference in the same sample, on the experiment\'s display scale, tested against zero within each group '
    '(and population) with a moderated t, one value per sample. The default reference is the <i>tag model</i>: '
    'what the combination\'s own tags predict from the other combinations of the sample, so a tag fluorophore\'s '
    'spillover into a marker, shared by every combination with that tag, is not taken for an effect; the control '
    'combinations or all other barcoded cells can be chosen instead. <i>Account for tag spillover</i>: the '
    'spillover spread of the combination\'s tags into the marker (spectral model, characterisation and noise '
    'model, on the display scale) is added to the feature\'s standard error, '
    'SE<sub>total</sub>² = SE<sub>moderated</sub>² + (k × spillover)², and t, the p-value (TREAT included) and '
    'the confidence interval come from SE<sub>total</sub>. Every feature stays tested; where the spillover can '
    'move the marker, the test trusts it less. With <i>Auto (NON-Targetting)</i>, k is the smallest value '
    '(0 to 4, steps of 0.25) at which the control combinations have no significant marker hits, found per '
    'population; untick it to set k yourself. Features whose uncertainty is mostly spillover (k × spillover '
    'above the moderated SE) are hatched in the heatmap. Combinations are families (cluster-first FDR). '
    'Expanded clones removed from the counts are also left out here.</p>'
    '<p><b>Pathways.</b> Asks whether the knockouts of a biological process, taken together, change the '
    'readout, so that small effects of several guides add up. Each GO Biological Process term (mouse, MGI '
    'annotations, bundled with the plugin) is a pathway whose genes carry a sign: <i>+1</i> when the gene is part '
    'of the process or drives it, <i>−1</i> when it holds the process back. Knocking out a +1 gene lowers the '
    'process and knocking out a −1 gene raises it, so if the process matters the two kinds of guide move in '
    'opposite directions, and the pathway adds them up instead of letting them cancel. Each guide\'s Counts '
    'result (the QL test, which Pathways needs) is turned into a z-score with the same p-value, and the '
    '<i>score</i> is the mean of −sign × z over the pathway\'s guides: for <i>Gut vs Spleen</i>, a positive score '
    'means the process promotes accumulation in the gut (knockouts that lower it are depleted there, knockouts '
    'that raise it are enriched). For example, an enhancer knockout at z = −2 and an inhibitor knockout at '
    'z = +1.5 give a score of +1.75. Two tests per pathway: the <i>self-contained</i> test asks whether anything '
    'is changing in the pathway\'s guides at all (a rotation test: many "nothing happening" versions of the data '
    'are made by re-mixing each guide\'s mouse-to-mouse noise the same way for every guide, so guides that vary '
    'together keep doing so); the <i>competitive</i> test (signed CAMERA) asks whether the pathway stands out '
    'from the other guides of its library, allowing for correlation between its guides. A pathway whose guides '
    'change strongly but in directions that contradict their signs can pass the self-contained test with a '
    'score near 0 and low <i>agreement</i> (the share of guides moving the way the score says), so read the score, '
    'the agreement and the competitive FDR together. Several libraries are tested one by one (their mice are '
    'different) and combined; each library\'s result is listed. Guide names are matched to MGI symbols, then to '
    'MGI synonyms (e.g. Gm5897 is Sult2b1); <b>Gene names…</b> lists the matches and lets you set a gene by '
    'hand. Non-targeting guides are never in a pathway; together they form a check that should not score. '
    '<i>Pathway dot plot</i>: pathways × comparisons, colour the score, size the competitive FDR, ringed when the '
    'self-contained FDR passes. Double-click a dot, or a row of the <i>Pathway table</i>, to show that pathway\'s '
    'guides in <i>Pathway members</i>; double-click a guide there to plot it in every sample.</p>'
    '<p><b>Gene groups.</b> Asks whether a family of knockouts, taken together, changes the readout: for '
    'example whether integrin knockouts as a whole are lost from the gut. In <b>Gene groups…</b>, tick one of the '
    'built-in tables of the lab\'s libraries (Homing, Treg transcription factors OTB01, Metabolism) or load a CSV '
    'of your own with a <i>gene</i> and a <i>group</i> column (optional <i>colour</i>, #rrggbb, and <i>area</i>, '
    'which gives more than eight groups shades of one colour per area); the dialog shows the group of every '
    'combination, and there you can change the order and colours of the groups. With no table loaded, ticking '
    '<b>Gene groups</b> picks the built-in table that covers at least 80% of the guides, when there is one. The '
    'table is kept with the experiment, so a newer built-in table in a later plugin version changes nothing until '
    'you tick it again. Combinations are matched by their own name, then by MGI gene '
    'symbol (a gene set by hand in <b>Gene names…</b> is used first); the others are unassigned and not tested. '
    'Every guide counts the same way, so a group\'s <i>score</i> is the mean of −z over its guides: for <i>Gut vs '
    'Spleen</i>, a positive score means the group\'s knockouts are depleted in the gut, so the family is needed '
    'for the cells to be there. Three tests per group: the <i>competitive</i> test (CAMERA) asks whether the '
    'family moves one way more than the rest of its library, and is the test of direction; the <i>any-change</i> '
    'test asks whether anything happens in the family in either direction (opposite changes add up, so a family '
    'whose members are needed in different tissues still shows up; one strongly changed guide is enough, and a '
    'shift of the whole library, such as the other guides gaining share when many are lost, can pass it too); the '
    '<i>self-contained</i> test of the score (as for pathways) is listed in the table. Groups need the same minimum '
    'number of tested guides as pathways and use the same rotations; a group may hold more than half of a library. '
    'Mice never carry two libraries, so when a family sits mostly in one library its result comes from that '
    'library\'s mice. <i>Guide allocation</i>: the libraries\' guides per group, pooled (shown as soon as a table '
    'is loaded). <i>Gene group spider</i>: one spoke per group, one line per comparison (the comparisons against '
    'the reference group, at most six; <b>Spider comparisons…</b> chooses others), coloured as the comparison\'s '
    'first group; outside the dashed ring the knockouts are depleted; a filled point passes the competitive FDR, a '
    'ringed one the any-change FDR. Double-click a point, or a row of the <i>Gene group table</i>, to show that '
    'group\'s guides in <i>Gene group members</i>.</p>'
    '<p><b>Phenotype shift</b> (tab after Results): whether the mix of phenotypes among a guide\'s cells differs '
    'from what its tags predict in the same mouse, how large the shift is (the shift angle) and towards which '
    'clusters, for each ticked population and sample group; it has its own settings, run, help and saved '
    'results, and is added to the export when current.</p>'
    '<p><b>Gating</b> (first tab, view only): the experiment\'s Unmixed Data plots and gates over the '
    'FlowCode-unmixed events of the sample picked in the sample browser, gated exactly as the counts (sample '
    'overrides included), in three views: <i>Uncorrected</i> (the experiment\'s own unmix of the raw file), '
    '<i>FlowCode standard</i> and <i>FlowCode calibrated</i> (with the per-sample combination calibration). '
    'Highlight the barcoded events or one combination (red); the table gives each gate\'s % of its parent '
    'and how many highlighted events change side against the uncorrected view, which is unmixed too for that '
    'comparison (both are cached). Gates are edited in the main window.</p>'
    '<p><b>Results.</b> Each statistics run is saved when it finishes and shown again when the experiment is '
    'reopened, with the current thresholds. It is marked <i>Out of date</i> (and cannot be exported) when the counts '
    'or other settings have changed since; run statistics again to update it. <i>View</i> chooses the table or '
    'figure shown, listed under the comparison, all comparisons, markers, pathways and gene groups; the arrows '
    'step to the previous or next one. The <i>Comparison heatmap</i> shows every comparison\'s significant '
    'log2 fold changes from the Counts test (from the Frequencies test when Counts is not run); with both tests '
    'run, <i>Comparison heatmap (frequencies)</i> shows the same from the Frequencies test. With the marker test, the <i>Markers</i> menu chooses the group whose combination × '
    'marker heatmap, volcano and table are shown (independent of the <i>Comparison</i> menu). Double-click a '
    'marker volcano point or a marker table row to plot that combination and marker in every sample '
    '(<i>Marker</i> view: each sample\'s reference and combination means joined by a line).</p>'
    '<p><b>Export</b> writes a folder named after the populations with the counts, frequencies (also as '
    'an R-style <tt>pct_matrix.csv</tt>), QC, samples, results, marker results, settings, a PNG per figure and '
    'one PDF report.</p>'
)


def _no_wheel(*widgets):
    for w in widgets:
        w.installEventFilter(WheelBlocker(w))
        w.setFocusPolicy(Qt.FocusPolicy.ClickFocus)


def _int_spin(lo, hi) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(lo, hi)
    _no_wheel(spin)
    return spin


def _float_spin(lo, hi, decimals, step) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    _no_wheel(spin)
    return spin


def _label(text: str = '', style: str = _HINT_STYLE) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet(style)
    return label


def _item(text, editable: bool = False, data=None) -> QTableWidgetItem:
    item = QTableWidgetItem('' if text is None else str(text))
    flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
    if editable:
        flags |= Qt.ItemIsEditable
    item.setFlags(flags)
    if data is not None:
        item.setData(Qt.UserRole, data)
    return item


def _gene_group_state(table: pd.DataFrame, file_name: str) -> dict:
    """The Gene groups dialog's state for a table read from *file_name*."""
    return {'rows': [[g, grp] for g, grp in zip(table['gene'], table['group'])], 'file': file_name,
            'areas': {g: a for g, a in fgg.group_areas(table).items() if a},
            'colours': fgg.table_colours(table), 'order': fgg.group_order(table)}


def _swatch_icon(colour: str) -> QIcon:
    pixmap = QPixmap(12, 12)
    pixmap.fill(QColor(colour))
    return QIcon(pixmap)


def _number(value, digits: int = 1) -> str:
    if value is None:
        return ''
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        return ''
    return f'{int(v):,}' if digits == 0 else f'{v:.{digits}f}'


class AnalysisPage(QWidget):
    """The Analysis page; ``plugin`` is the FlowCode PluginWidget."""

    def __init__(self, plugin, parent=None):
        super().__init__(parent)
        self.plugin = plugin
        self.section = SECTION
        self._filling = False
        self.links = None
        self._links_key = None
        self._linked_names = {}
        self.counts = None
        self.meta = None
        self.qc = None
        self.filtered = None
        self.stats = None
        self.figures = {}
        self._auto_groupings_key = None
        self._auto_groupings_undo = None
        self._hierarchy_key = None
        self._phenotype_key = None
        self._phenotypes = {}
        self._phenotype_aliases = {}
        self._count_token = None
        self._combination = None
        self._markers_key = ()
        self._markers = None
        self._marker_feature = None
        self._library_key = None
        self._left_out = {}
        self._needing = None
        self._cov_start = 2
        self._stats_checked = False
        self._stats_stale = False
        self._loo_inputs, self._loo_labels = None, {}
        self._pathway_set = None
        self._group_set = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        head = QWidget()
        self.body = QVBoxLayout(head)
        self.body.setAlignment(Qt.AlignTop)
        title = QLabel(f'{SECTION}. {TITLE}')
        title.setStyleSheet(hc_settings.heading_style)
        self.body.addWidget(title)
        self.status = _label('')
        self.status.setTextFormat(Qt.RichText)
        self.body.addWidget(self.status)
        self.no_fret_label = _label('', _INVALID_STYLE)
        self.no_fret_label.setVisible(False)
        self.body.addWidget(self.no_fret_label)
        self.body.addWidget(fcfp.scrolling_help(HELP))
        outer.addWidget(head)
        self.section_tabs = QTabWidget()
        self.section_tabs.setDocumentMode(True)
        outer.addWidget(self.section_tabs, stretch=1)

        self.gating_view = fcgv.GatingSection(self)
        self.section_tabs.addTab(self.gating_view, 'Gating')
        self._build_data(self._add_section('Data'))
        self._build_groups(self._add_section('Groups'))
        self._build_stats(self._add_section('Statistics'))
        self._build_results(self._add_section('Results', scroll=False))
        self.shift_view = fshv.ShiftSection(self)
        self.section_tabs.addTab(self.shift_view, 'Phenotype shift')
        self._build_export(self._add_section('Export'))
        titles = [self.section_tabs.tabText(i) for i in range(self.section_tabs.count())]
        opened = str(plugin.qsettings.value(f'open_section_{SECTION}_title', 'Data') or 'Data')
        self.section_tabs.setCurrentIndex(titles.index(opened) if opened in titles else titles.index('Data'))
        self.section_tabs.currentChanged.connect(
            lambda i: plugin.qsettings.setValue(f'open_section_{SECTION}_title', self.section_tabs.tabText(i)))
        self.section_tabs.currentChanged.connect(self._on_section_changed)

    def _on_section_changed(self, _index):
        """The Gating tab follows the main window's sample browser: show the
        browser there only, and draw the selected sample."""
        update = getattr(self.plugin, 'update_sample_browser_visibility', None)
        if update is not None:
            update()
        if self.gating_view.is_shown():
            self.gating_view.on_shown()

    def reset_session(self):
        """Forget everything held for the previous experiment."""
        self.links = None
        self._links_key = None
        self._linked_names = {}
        self.counts = self.meta = self.qc = self.filtered = None
        self.stats = None
        self.figures = {}
        self._auto_groupings_key = None
        self._auto_groupings_undo = None
        self._hierarchy_key = None
        self._phenotype_key = None
        self._phenotypes = {}
        self._phenotype_aliases = {}
        self._combination = None
        self._markers_key = ()
        self._markers = None
        self._marker_feature = None
        self._library_key = None
        self._left_out = {}
        self._needing = None
        self._stats_checked = False
        self._pathway_set = None
        self._group_set = None
        self._set_stats_stale(False)
        self._loo_inputs, self._loo_labels = None, {}
        self.loo_status.setText('')
        self._clear_figure_tabs()
        self._show_auto_note(None)
        self.shift_view.reset_session()

    # -- frame ---------------------------------------------------------------

    def _add_section(self, title: str, scroll: bool = True) -> QVBoxLayout:
        """A new tab, scrolling on its own (or, without *scroll*, filling the
        tab); returns its layout."""
        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setSpacing(8)
        if not scroll:
            self.section_tabs.addTab(content, title)
            return lay
        lay.setAlignment(Qt.AlignTop)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        area.setWidget(content)
        self.section_tabs.addTab(area, title)
        return lay

    def set_status(self, kind: str, text: str):
        uncorrected = self.uncorrected()
        if kind == 'done' and uncorrected:
            kind, text = 'invalid', f'{text} (no FRET correction)'
        style = {'done': _DONE_STYLE, 'error': _ERROR_STYLE, 'stale': _WARN_STYLE,
                 'invalid': _INVALID_STYLE}.get(kind, _HINT_STYLE)
        self.status.setStyleSheet(style)
        self.status.setText(text)
        self.no_fret_label.setText(f'{NO_FRET_WARNING} The samples were debarcoded without FRET correction '
                                   '("No FRET backbone" in section 1, or files exported that way).'
                                   if uncorrected else '')
        self.no_fret_label.setVisible(uncorrected)

    def uncorrected(self) -> bool:
        """True when the analysed events have no FRET correction: the live
        unmix with a characterisation made without it, or an exported file
        written without it."""
        if self.is_live():
            fc = self.plugin.saved_spectra()
            return bool(self.plugin.state.no_fret) or (fc is not None and not fc.fret_corrected)
        files = (self.links or {}).get('files') or {}
        return any((files.get(key) or {}).get('uncorrected') for key in self.used_files())

    def _stamped(self, fig, make=None):
        """(*fig*, *make*) with the no-FRET warning across the figures when
        the analysed events have no FRET correction."""
        if not self.uncorrected():
            return fig, make
        stamped_make = None if make is None else (lambda: fst.stamp_invalid(make()))
        return fst.stamp_invalid(fig), stamped_make

    # -- state ------------------------------------------------------------------

    @property
    def settings(self) -> dict:
        """The page's saved settings (``state.analysis``), defaults filled in."""
        state = self.plugin.state
        if not isinstance(state.analysis, dict) or set(fa.ANALYSIS_DEFAULTS) - set(state.analysis):
            state.analysis = fa.analysis_settings(state.analysis)
        return state.analysis

    def _changed(self):
        self.plugin.save_state()
        bus = self.plugin.bus
        if bus is not None:
            bus.autoSaveRequested.emit()
        self._update_default_buttons()

    def _at_defaults(self, keys) -> bool:
        s = self.settings
        return all(s.get(key) == fa.ANALYSIS_DEFAULTS[key] for key in keys)

    def _update_default_buttons(self):
        for button, keys in ((self.stats_defaults_btn, STATS_DEFAULT_KEYS),
                             (self.rule_defaults_btn, RULE_DEFAULT_KEYS),
                             (self.figure_defaults_btn, FIGURE_DEFAULT_KEYS)):
            button.setEnabled(not self._at_defaults(keys))

    def _restore_defaults(self, keys) -> list:
        """Put the settings in *keys* back to their defaults; returns those
        that changed."""
        s = self.settings
        changed = [key for key in keys if s.get(key) != fa.ANALYSIS_DEFAULTS[key]]
        for key in changed:
            s[key] = deepcopy(fa.ANALYSIS_DEFAULTS[key])
        if changed:
            self._changed()
        return changed

    def _restore_rule_defaults(self):
        if not self._restore_defaults(RULE_DEFAULT_KEYS):
            return
        s = self.settings
        self._filling = True
        try:
            self.min_combo_spin.setValue(int(s['min_combo_events']))
            self.min_sample_spin.setValue(int(s['min_sample_events']))
        finally:
            self._filling = False
        self._apply_filters()
        self._invalidate_stats()

    def _restore_stats_defaults(self):
        changed = self._restore_defaults(STATS_DEFAULT_KEYS)
        if not changed:
            return
        self._fill_stats_choices()
        if set(changed) - set(THRESHOLD_KEYS):
            self._invalidate_stats()
        else:
            self._reapply_thresholds()
        self._update_run_button()

    def _restore_figure_defaults(self):
        if self._restore_defaults(FIGURE_DEFAULT_KEYS):
            self._show_results()

    def _controller(self):
        return self.plugin.controller

    def figure_options_store(self):
        """The experiment's figure options (figure_options), or None."""
        from honeychrome.view_components import figure_options as fo
        return fo.store_for(self.plugin.controller)

    @staticmethod
    def _report_figure_key(name: str) -> str:
        """The figure options key of a report figure: the key of the same
        figure on the Analysis page."""
        fixed = {'QC': 'qc', 'Comparison heatmap': 'comparison_heatmap',
                 'Comparison heatmap (frequencies)': 'comparison_heatmap_freq',
                 'Phenotype shift summary heatmap': 'shift_summary_heatmap',
                 'Pathway dot plot': 'pathway_dot_plot', 'Guide allocation': 'guide_allocation',
                 'Gene group spider': 'gene_group_spider'}
        if name in fixed:
            return fixed[name]
        if name.startswith('Sample PCA'):
            return 'sample_pca'
        tail = name.rsplit(' - ', 1)[-1]
        if name.startswith('Markers '):
            return f'marker_{tail}'
        if tail.startswith('pathway '):
            return 'pathway_members'
        if tail.startswith('gene group '):
            return 'gene_group_members'
        return tail.replace(' ', '_')

    def _display_names(self) -> dict:
        """{sample: name} of the experiment's samples and of the files
        linked from other experiments."""
        names = dict(self._controller().experiment.samples.get('all_samples', {}) or {})
        names.update(self._linked_names)
        return names

    def _name(self, key: str) -> str:
        return self._display_names().get(key, Path(str(key)).stem)

    def export_folder(self) -> Path:
        folder = self.settings.get('folder')
        if folder:
            return Path(folder)
        last = (self.plugin.state.last_export or {}).get('folder')
        if last:
            return Path(last)
        return self.plugin.export_folder()

    def analysis_root(self) -> Path:
        root = self.settings.get('export_root')
        return Path(root) if root else Path(self._controller().experiment_dir) / ANALYSIS_SUBDIR

    def population_name(self) -> str:
        """The ticked populations, joined (what the next count uses)."""
        pops = list(self.settings.get('gates') or []) + list(self.phenotype_names())
        return ', '.join(pops) if pops else fa.ALL_EVENTS

    def counted_populations(self) -> list:
        """The populations of the saved counts."""
        return fa.saved_records(self.settings.get('counts') or {})[0]

    def counted_phenotypes(self) -> dict:
        """``{population: resolution summary}`` of the phenotype populations among the saved counts."""
        return dict((self.settings.get('counts') or {}).get('phenotypes') or {})

    # -- phenotypes -------------------------------------------------------------

    def _panel(self):
        """The experiment's labelled fluorophores and their markers (``flowcode_phenotype.Panel``)."""
        import flowcode_method as fcm
        import flowcode_phenotype as fp
        c = self._controller()
        return fp.panel_from_spectral_model(c.experiment.process.get('spectral_model') or [], fcm.fcc_fluor_names(c))

    def resolved_phenotypes(self) -> dict:
        """``{phenotype key: Resolved | Unresolved}`` of the cell-type
        database on this panel, definitions the panel cannot tell apart
        grouped into one phenotype (``flowcode_phenotype.phenotype_groups``);
        resolved again when the panel's fluorophores or antigens, or the
        definitions, change."""
        import flowcode_phenotype as fp
        try:
            panel = self._panel()
            definitions = fp.load_definitions()
            key = (panel.fluorophores, tuple(sorted(panel.antigens.items())), tuple(definitions))
            if key != self._phenotype_key:
                self._phenotype_key = key
                self._phenotypes, self._phenotype_aliases = fp.phenotype_groups(definitions, panel)
            return dict(self._phenotypes)
        except Exception as exc:
            logger.warning('FlowCode analysis: phenotype definitions could not be resolved: %s', exc)
            return {}

    def congenic_split(self) -> bool:
        """Whether each phenotype is counted as its CD45.1 and CD45.2 variants."""
        return bool(self.settings.get('congenic_split'))

    def phenotype_populations(self, keys=None) -> dict:
        """``{population name: Resolved}`` of the resolved definitions among
        *keys* (default: the ticked ones), in definition order; with the
        congenic split, each becomes its CD45.1 and CD45.2 variants when the
        panel carries both markers."""
        import flowcode_phenotype as fp
        resolved = self.resolved_phenotypes()
        chosen = set(self._current_keys(self.settings.get('phenotypes') if keys is None else keys))
        split = self.congenic_split()
        panel = self._panel() if split else None
        out = {}
        for key, res in resolved.items():
            if key not in chosen or not isinstance(res, fp.Resolved):
                continue
            variants = fp.split_congenic(res, panel) if split else {}
            if variants:
                out.update(variants)
            else:
                out[res.definition.cell_type] = res
        return out

    def _current_keys(self, keys) -> list:
        """*keys* as keys of the current phenotypes: a definition now grouped
        with others, or one renamed, is carried over to its phenotype."""
        aliases = self._phenotype_aliases
        return list(dict.fromkeys(aliases.get(k, k) for k in (keys or [])))

    def ticked_phenotypes(self) -> dict:
        """``{population name: Resolved}`` of the ticked, resolved definitions (split when asked)."""
        return self.phenotype_populations()

    def phenotype_names(self) -> list:
        return list(self.ticked_phenotypes())

    def _on_congenic_split(self, checked: bool):
        if self._filling:
            return
        self.settings['congenic_split'] = bool(checked)
        self._changed()
        self._fill_phenotype_list()
        self._check_counts_current()
        self.refresh_status()

    def _on_phenotype_item_changed(self, _item):
        if self._filling:
            return
        chosen = [self.phenotype_list.item(i).data(Qt.UserRole) for i in range(self.phenotype_list.count())
                  if self.phenotype_list.item(i).checkState() == Qt.Checked]
        self.settings['phenotypes'] = chosen
        self._changed()
        self._check_counts_current()
        self.refresh_status()

    def _fill_phenotype_list(self):
        import flowcode_phenotype as fp
        resolved = self.resolved_phenotypes()
        chosen = [k for k in self._current_keys(self.settings.get('phenotypes')) if k in resolved]
        if chosen != list(self.settings.get('phenotypes') or []):
            self.settings['phenotypes'] = chosen
        filling, self._filling = self._filling, True
        try:
            self.congenic_check.setChecked(self.congenic_split())
            self.phenotype_list.clear()
            for key, res in resolved.items():
                d = res.definition
                it = QListWidgetItem(d.key if len(d.members) > 1 else f'{d.key}: {d.text()}')
                it.setData(Qt.UserRole, key)
                if isinstance(res, fp.Resolved):
                    it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                    it.setCheckState(Qt.Checked if key in chosen else Qt.Unchecked)
                    tip = res.summary() + f' ({d.source})'
                    if d.separated_by:
                        tip += ('\nThis panel cannot tell these cell types apart; staining for '
                                + ', '.join(d.separated_by) + ' would.')
                    it.setToolTip(tip)
                else:
                    it.setFlags(Qt.ItemIsUserCheckable)
                    it.setCheckState(Qt.Unchecked)
                    it.setToolTip(f'Unresolved: {res.reason} ({d.source})')
                self.phenotype_list.addItem(it)
            ticked = self.ticked_phenotypes()
            if not resolved:
                self.phenotype_note.setText('No phenotype definition could be resolved: give the spectral model\'s '
                                            'fluorophores their antigens.')
            elif not self.is_live() and ticked:
                self.phenotype_note.setStyleSheet(_WARN_STYLE)
                self.phenotype_note.setText('Phenotypes are counted from the live unmix only: choose Live unmix '
                                            'as the source.')
            else:
                import flowcode_phenotype as fp
                self.phenotype_note.setStyleSheet(_HINT_STYLE)
                notes = [f'{name}: {res.summary()}' for name, res in ticked.items()]
                if self.congenic_split() and ticked and not any(
                        m in fp.CONGENIC_MARKERS for res in ticked.values() for m, _f, _s in res.markers):
                    notes.insert(0, 'Not split: the panel does not carry both CD45.1 and CD45.2.')
                self.phenotype_note.setText('; '.join(notes))
        finally:
            self._filling = filling

    def counted_population(self) -> str:
        """The populations of the saved counts, joined (what results describe)."""
        pops = self.counted_populations()
        return ', '.join(pops) if pops else self.population_name()

    def several_populations(self) -> bool:
        return len(self.counted_populations()) > 1

    def soft_counts(self) -> bool:
        """Whether the saved counts include a phenotype population (fractional counts)."""
        return bool(self.counted_phenotypes())

    def run_counts(self) -> bool:
        """The Counts test as it will run: ticked, and the counts are whole events."""
        return bool(self.settings['run_counts']) and not self.soft_counts()

    def unit_sample(self, unit: str) -> str:
        """The sample key of a counts row (a population of a sample)."""
        return fa.unit_parts(unit)[0]

    def unit_population(self, unit: str) -> str:
        pops = self.counted_populations()
        return fa.unit_parts(unit, pops[0] if pops else '')[1]

    def unit_labels(self) -> dict:
        """{counts row: displayed name}: the sample's name, with its
        population when several were counted."""
        if self.counts is None:
            return self._display_names()
        several = self.several_populations()
        return {u: self._name(self.unit_sample(u)) + (f' [{self.unit_population(u)}]' if several else '')
                for u in self.counts.index}

    def _current_design(self):
        try:
            return self.plugin.design()
        except Exception:
            return None

    # -- several libraries --------------------------------------------------------

    def multi_library(self) -> bool:
        return bool(self.plugin.state.multi_library)

    def library_of(self, sample: str) -> str | None:
        """The sample's library, Other, or None when it needs one and has
        none. A sample that needs no library (a backbone) is Other unless
        assigned. Always None with one library."""
        if not self.multi_library():
            return None
        if fa.is_linked(sample):
            # A linked file carries its library (FLOWCODELIBRARY); it counts
            # only when this experiment has a library of that name.
            lib = (((self.links or {}).get('files') or {}).get(sample) or {}).get('library')
            return lib if lib in self.library_labels() else None
        lib = self.plugin.state.library_of(sample)
        if lib is None and sample not in self._library_samples():
            return OTHER_LIBRARY
        return lib

    def _library_samples(self) -> set:
        """Samples that need a library (section 2), kept until the next refresh."""
        if self._needing is None:
            self._needing = set(self.plugin.library_samples())
        return self._needing

    def library_labels(self) -> dict:
        """{library: the name shown in results (its label, else its name)}."""
        return {str(lib.get('name', '')): library_label(lib) for lib in self.plugin.state.libraries}

    def feature_map(self) -> dict | None:
        """``flowcode_analysis.library_feature_map`` with several libraries, else None."""
        return fa.library_feature_map(self.plugin.state) if self.multi_library() else None

    def feature_libraries(self) -> dict:
        """{feature: [libraries carrying it]}."""
        out = {}
        for lib, fmap in (self.feature_map() or {}).items():
            for feature in fmap.values():
                owners = out.setdefault(feature, [])
                if lib not in owners:
                    owners.append(lib)
        return out

    def _library_setup_key(self):
        """What the counts tables depend on in the library setup (None with one library)."""
        if not self.multi_library():
            return None
        return repr((self.feature_map(), sorted(self.plugin.state.sample_libraries.items()),
                     sorted(self._library_samples())))

    def _default_use(self, key: str, info: dict) -> bool:
        """A linked sample is used by default when its file is ok and, with
        several libraries, it is not assigned to Other."""
        return info['status'] in fa.USABLE_BY_DEFAULT and self.library_of(key) != OTHER_LIBRARY

    def _left_out_note(self) -> str:
        """The samples left out of the counts for having no library."""
        other = [self._name(s) for s, lib in self._left_out.items() if lib == OTHER_LIBRARY]
        none = [self._name(s) for s, lib in self._left_out.items() if lib != OTHER_LIBRARY and not fa.is_linked(s)]
        linked = [self._name(s) for s, lib in self._left_out.items() if lib != OTHER_LIBRARY and fa.is_linked(s)]
        parts = []
        if other:
            parts.append('Samples assigned to Other have no guides and were left out: ' + ', '.join(other) + '.')
        if none:
            parts.append('Samples without a library were left out (assign them in section 2, Tags & Combos): '
                         + ', '.join(none) + '.')
        if linked:
            parts.append('Linked files whose library (FLOWCODELIBRARY) is not a library of this experiment were '
                         'left out: ' + ', '.join(linked) + '.')
        return ' '.join(parts)

    def status_summary(self) -> tuple:
        """(kind, text) for the page's status line."""
        links = self.links
        if links is None:
            return 'pending', 'Not scanned yet.'
        n_ok = sum(1 for v in links['files'].values() if v['status'] in fa.USABLE_BY_DEFAULT)
        if self.is_live():
            problem = self._live_problem()
            if problem:
                return 'pending', problem
            if n_ok == 0:
                return 'pending', 'No sample has a raw file and an AF library of 2 or more spectra.'
            parts = [f'Live unmix: {n_ok} sample(s)']
        elif n_ok == 0:
            return 'pending', f'No exported files found in {self.export_folder()}. Export samples in section 5.'
        else:
            parts = [f'{n_ok} linked file(s)']
        if self.counts is not None:
            parts.append(f'{self.counts.shape[0]} counted')
        if self.stats is not None:
            parts.append(f"{len(self.stats['comparisons'])} comparison(s)")
        return 'done', ', '.join(parts) + f' — population: {self.counted_population()}'

    # =====================================================================
    # Data
    # =====================================================================

    def _build_data(self, lay):
        row = QHBoxLayout()
        row.addWidget(QLabel('Source:'))
        self.source_combo = QComboBox()
        self.source_combo.addItem('Live unmix (raw files, as section 5 would export them)', fa.SOURCE_LIVE)
        self.source_combo.addItem('Exported files (written by section 5)', fa.SOURCE_EXPORTED)
        self.source_combo.setToolTip('Live unmix FlowCode-unmixes each raw file with the current '
                                     'characterisation and the experiment\'s unmixing method, without writing '
                                     'files. Exported files reads what section 5 wrote.')
        _no_wheel(self.source_combo)
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        row.addWidget(self.source_combo, stretch=1)
        lay.addLayout(row)
        self.source_note = _label('')
        self.source_note.setWordWrap(True)
        lay.addWidget(self.source_note)

        box = QGroupBox('Exported files')
        self.export_box = box
        bl = QVBoxLayout(box)
        row = QHBoxLayout()
        self.folder_label = _label('', '')
        row.addWidget(self.folder_label, stretch=1)
        for text, slot in (('Choose…', self._on_choose_folder), ('Default', self._on_default_folder),
                           ('Rescan', self.rescan)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        bl.addLayout(row)
        linked_row = QHBoxLayout()
        linked_label = QLabel('Other experiments:')
        linked_label.setAlignment(Qt.AlignTop)
        linked_row.addWidget(linked_label)
        self.linked_list = QListWidget()
        self.linked_list.setMaximumHeight(80)
        self.linked_list.setToolTip('Export folders (section 5) of other experiments of the same study, e.g. '
                                    'samples acquired on other days with their own controls. Their files are '
                                    'counted with this experiment\'s samples, gated with this experiment\'s '
                                    'unmixed gates.')
        linked_row.addWidget(self.linked_list, stretch=1)
        linked_buttons = QVBoxLayout()
        for text, slot in (('Add folder…', self._on_add_linked), ('Remove', self._on_remove_linked)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            linked_buttons.addWidget(b)
        linked_buttons.addStretch()
        linked_row.addLayout(linked_buttons)
        bl.addLayout(linked_row)
        self.link_note = _label('')
        self.link_note.setWordWrap(True)
        bl.addWidget(self.link_note)
        lay.addWidget(box)

        top = QHBoxLayout()
        pop_box = QGroupBox('Populations (unmixed gates and phenotypes)')
        pl = QVBoxLayout(pop_box)
        self.gate_tree = self._make_gate_tree()
        pl.addWidget(self.gate_tree)
        self.phenotype_list = QListWidget()
        self.phenotype_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.phenotype_list.setMinimumHeight(90)
        self.phenotype_list.setToolTip('Tick a marker definition to count it as a population of its own, '
                                       'scored on the live-unmixed events inside the FlowCode gates. '
                                       'Greyed definitions have a positive marker the panel lacks.')
        self.phenotype_list.itemChanged.connect(self._on_phenotype_item_changed)
        pl.addWidget(self.phenotype_list)
        self.congenic_check = QCheckBox('Split each phenotype by congenic marker (CD45.1+ CD45.2- and CD45.1- CD45.2+)')
        self.congenic_check.setToolTip('Each ticked phenotype is counted twice, as "<name> CD45.1" and "<name> '
                                       'CD45.2", when the panel carries both markers.')
        self.congenic_check.toggled.connect(self._on_congenic_split)
        pl.addWidget(self.congenic_check)
        self.phenotype_note = _label('')
        self.phenotype_note.setWordWrap(True)
        pl.addWidget(self.phenotype_note)
        top.addWidget(pop_box, stretch=1)

        rules = QGroupBox('Combinations and inclusion rules')
        rl = QVBoxLayout(rules)
        self.combo_list = QListWidget()
        self.combo_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.combo_list.setMinimumHeight(140)
        self.combo_list.itemChanged.connect(self._on_combo_item_changed)
        rl.addWidget(self.combo_list)
        form = QFormLayout()
        self.min_combo_spin = _int_spin(0, 1_000_000)
        self.min_combo_spin.setToolTip('Combinations with fewer events than this, summed over all samples, '
                                       'are left out (barcodes that failed or were barely transduced).')
        self.min_sample_spin = _int_spin(0, 1_000_000)
        self.min_sample_spin.setToolTip('Samples with fewer barcoded events than this in the population '
                                        'are left out.')
        form.addRow('Min. events per combination:', self.min_combo_spin)
        form.addRow('Min. barcoded events per sample:', self.min_sample_spin)
        self.min_combo_spin.valueChanged.connect(lambda v: self._on_rule('min_combo_events', int(v)))
        self.min_sample_spin.valueChanged.connect(lambda v: self._on_rule('min_sample_events', int(v)))
        rl.addLayout(form)
        self.rule_defaults_btn = QPushButton('Restore defaults')
        self.rule_defaults_btn.setToolTip(RESTORE_TOOLTIP + ' (minimum events per combination and per sample)')
        self.rule_defaults_btn.clicked.connect(self._restore_rule_defaults)
        rule_row = QHBoxLayout()
        rule_row.addWidget(self.rule_defaults_btn)
        rule_row.addStretch()
        rl.addLayout(rule_row)
        top.addWidget(rules, stretch=1)
        lay.addLayout(top)

        run = QHBoxLayout()
        self.count_btn = QPushButton('Count events')
        self.count_btn.clicked.connect(self.start_count)
        run.addWidget(self.count_btn)
        self.count_status = _label('')
        run.addWidget(self.count_status, stretch=1)
        lay.addLayout(run)
        self.filter_note = _label('')
        lay.addWidget(self.filter_note)

        include_row = QHBoxLayout()
        select_all_btn = QPushButton('Select all')
        select_all_btn.setToolTip('Tick every readable sample in the table below for counting.')
        select_all_btn.clicked.connect(lambda: self._set_all_use(True))
        deselect_all_btn = QPushButton('Deselect all')
        deselect_all_btn.setToolTip('Untick every sample in the table below.')
        deselect_all_btn.clicked.connect(lambda: self._set_all_use(False))
        include_row.addWidget(select_all_btn)
        include_row.addWidget(deselect_all_btn)
        self.include_checks = {}
        for key, text in (('unstained', 'Include unstained samples'),
                          ('single_stain', 'Include single-stained controls'),
                          ('beads', 'Include bead samples')):
            chk = QCheckBox(text)
            chk.toggled.connect(lambda checked, key=key: self._on_include_toggled(key, checked))
            include_row.addWidget(chk)
            self.include_checks[key] = chk
        include_row.addStretch()
        self.qc_pop_label = QLabel('Population shown:')
        self.qc_pop_combo = QComboBox()
        self.qc_pop_combo.setToolTip('The population whose counts the table shows.')
        _no_wheel(self.qc_pop_combo)
        self.qc_pop_combo.currentIndexChanged.connect(lambda _i: None if self._filling else self._fill_data())
        include_row.addWidget(self.qc_pop_label)
        include_row.addWidget(self.qc_pop_combo)
        lay.addLayout(include_row)

        self.sample_table = QTableWidget(0, 11)
        self.sample_table.setHorizontalHeaderLabels([
            'Use', 'Sample', 'File status', 'Events', 'In population', 'Barcoded', '% barcoded',
            'Untransduced', 'Unresolved', 'Gini', 'Top combination (%)'])
        self.sample_table.verticalHeader().setVisible(False)
        self.sample_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.sample_table.horizontalHeader().setStretchLastSection(False)
        self.sample_table.setTextElideMode(Qt.ElideMiddle)
        self.sample_table.setWordWrap(False)
        self.sample_table.setMinimumHeight(260)
        self.sample_table.itemChanged.connect(self._on_use_changed)
        lay.addWidget(self.sample_table)

    def _make_gate_tree(self):
        tree = QTreeWidget()
        tree.setHeaderHidden(True)
        tree.setMinimumHeight(160)
        tree.setToolTip('Tick the populations to analyse. Each ticked gate is counted as a population of its '
                        'own; ticking a gate does not tick the gates below it.')
        tree.itemChanged.connect(self._on_gate_item_changed)
        return tree

    def source(self) -> str:
        value = self.settings.get('source')
        return value if value in fa.SOURCES else fa.SOURCE_LIVE

    def is_live(self) -> bool:
        return self.source() == fa.SOURCE_LIVE

    def _on_source_changed(self, _index):
        if self._filling:
            return
        value = self.source_combo.currentData()
        if value == self.source():
            return
        self.settings['source'] = value
        self._changed()
        self.rescan()

    def _on_choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Folder of exported FlowCode files',
                                                  str(self.export_folder()))
        if folder:
            self.settings['folder'] = folder
            self._changed()
            self.rescan()

    def _on_default_folder(self):
        self.settings['folder'] = None
        self._changed()
        self.rescan()

    def _on_add_linked(self):
        folder = QFileDialog.getExistingDirectory(self, "Export folder of another experiment's FlowCode files",
                                                  str(self.export_folder().parent))
        if not folder:
            return
        folders = list(self.settings.get('linked_folders') or [])
        if folder in folders:
            return
        if Path(folder).resolve() == Path(self.export_folder()).resolve():
            QMessageBox.information(self, 'Other experiments', "This is this experiment's own export folder.")
            return
        self.settings['linked_folders'] = folders + [folder]
        self._changed()
        self.rescan()

    def _on_remove_linked(self):
        folders = list(self.settings.get('linked_folders') or [])
        row = self.linked_list.currentRow()
        if not 0 <= row < len(folders):
            return
        del folders[row]
        self.settings['linked_folders'] = folders
        self._changed()
        self.rescan()

    def _on_rule(self, key, value):
        if self._filling:
            return
        self.settings[key] = value
        self._changed()
        self._apply_filters()
        self._invalidate_stats()

    def _on_gate_item_changed(self, _item, _column=0):
        if self._filling:
            return
        names = []

        def walk(item):
            if item.checkState(0) == Qt.Checked:
                names.append(item.text(0))
            for i in range(item.childCount()):
                walk(item.child(i))

        for i in range(self.gate_tree.topLevelItemCount()):
            walk(self.gate_tree.topLevelItem(i))
        self._on_gates_changed(names)

    def _on_gates_changed(self, names):
        self.settings['gates'] = list(names)
        self._changed()
        self._check_counts_current()
        self.refresh_status()

    def _on_combo_item_changed(self, item):
        if self._filling:
            return
        excluded = [self.combo_list.item(i).data(Qt.UserRole) for i in range(self.combo_list.count())
                    if self.combo_list.item(i).checkState() != Qt.Checked]
        self.settings['exclude_combos'] = excluded
        self._changed()
        self._apply_filters()
        self._invalidate_stats()

    def _on_include_toggled(self, key: str, checked: bool):
        """Offer or withdraw a sample type in the sample table."""
        if self._filling:
            return
        self.settings['include'] = {**(self.settings.get('include') or {}), key: bool(checked)}
        self._changed()
        self.rescan()

    def _on_use_changed(self, item):
        if self._filling or item.column() != 0:
            return
        key = item.data(Qt.UserRole)
        self.settings['use'][key] = item.checkState() == Qt.Checked
        self._changed()
        self._check_counts_current()
        self._fill_groups()
        self.refresh_status()

    def _set_all_use(self, ticked: bool):
        """Tick or untick every readable sample in the sample table for counting."""
        if self.links is None:
            return
        use = self.settings.setdefault('use', {})
        for key, info in self.links['files'].items():
            if info['path'] is not None and info['status'] != fa.LINK_UNREADABLE:
                use[key] = ticked
        self._changed()
        self._check_counts_current()
        self._fill_data()
        self._fill_groups()
        self.refresh_status()

    # -- linking ----------------------------------------------------------------

    def rescan(self):
        """Re-link the export folder's files to the experiment's samples."""
        self._links_key = None
        self.refresh()

    def _live_problem(self) -> str | None:
        """Why the live unmix cannot run now, or None."""
        if self.plugin.section_status(4)[0] != 'done':
            return 'Live unmix needs the backbone characterised (section 4).'
        try:
            self.plugin.export_spectra()
        except ValueError as exc:
            return str(exc)
        return None

    def _scan_links(self):
        c = self._controller()
        keys = self.plugin.analysis_sample_keys(self.settings.get('include'))
        if self.is_live():
            af_ok = self.plugin.live_af_ok(keys)
            key = ('live', tuple(keys), tuple(sorted(af_ok.items())))
            if key == self._links_key and self.links is not None:
                return
            self._links_key = key
            self.links = fa.live_links(keys, c.experiment_dir, af_ok)
            self._linked_names = {}
            for k, date in fa.file_dates({k: v['path'] for k, v in self.links['files'].items() if v['path']}).items():
                self.links['files'][k]['date'] = date
            self._update_date_covariate()
            return
        folder = self.export_folder()
        design = self._current_design()
        combos_kw = None
        if design is not None:
            import flowcode_core as fcc
            combos_kw = fcc.combos_keyword(design)
        # The characterisation as an export writes it now: named and ordered
        # by the current combination table (with several libraries, by triplet).
        try:
            characterisation = self.plugin.export_spectra().characterisation_id()
        except ValueError:
            characterisation = None
        libraries = {k: self.library_of(k) for k in keys} if self.multi_library() else None
        key = ('exported', str(folder), tuple(keys), combos_kw, characterisation,
               tuple(sorted((libraries or {}).items(), key=lambda kv: kv[0])),
               tuple(self.settings.get('linked_folders') or []))
        if key == self._links_key and self.links is not None:
            return
        self._links_key = key
        self._linked_names = {}
        raw_subdir = c.experiment.settings['raw']['raw_samples_subdirectory']
        self.links = fa.link_exports(folder, keys, raw_subdir, self._display_names(),
                                     self.plugin.export_name_suffix(), combos_kw, characterisation, libraries)
        self._link_other_experiments(combos_kw)
        self._update_date_covariate()

    def _link_other_experiments(self, combos_kw):
        """Add the exported files in the other experiments' folders
        (``linked_folders``) to the links, each named by its file, with its
        acquisition date (else its folder) added when the name is taken."""
        own_folder = Path(self.export_folder()).resolve()
        folders = [f for f in self.settings.get('linked_folders') or [] if f and Path(f).resolve() != own_folder]
        if not folders:
            return
        files = self.links['files']
        own = [info['path'] for info in files.values() if info.get('path')]
        required = list(self._controller().experiment.settings['unmixed']['event_channels_pnn'])
        linked = fa.link_folders(folders, combos_kw, required, self.plugin.export_name_suffix(), exclude=own)
        taken = set(self._display_names().values())
        for key, info in linked['files'].items():
            name = info['name']
            if name in taken:
                name = f"{name} ({info.get('date') or Path(info['folder']).name})"
            taken.add(name)
            self._linked_names[key] = name
        files.update(linked['files'])
        linked_paths = {Path(info['path']).resolve() for info in linked['files'].values()}
        self.links['unlinked'] = ([p for p in self.links['unlinked'] if Path(p).resolve() not in linked_paths]
                                  + linked['unlinked'])

    def _update_date_covariate(self):
        """Fill the acquisition-date covariate from the files' $DATE when
        the samples were acquired on more than one day. Values typed by hand
        are kept. A newly made covariate is ticked under Adjust for (unless it
        is the pairing or split); removing it stops the automatic filling."""
        s = self.settings
        if not s.get('date_covariate', True) or self.links is None:
            return
        dates = {k: info['date'] for k, info in self.links['files'].items() if info.get('path') and info.get('date')}
        if len(set(dates.values())) < 2:
            return
        covariates = s.setdefault('covariates', {})
        created = fa.DATE_COVARIATE not in covariates
        values = covariates.setdefault(fa.DATE_COVARIATE, {})
        added = {k: d for k, d in dates.items() if not str(values.get(k) or '').strip()}
        if not added and not created:
            return
        values.update(added)
        in_design = (s.get(self._design_key('pairing')), s.get(self._design_key('split_by')))
        if created and fa.DATE_COVARIATE not in in_design and fa.DATE_COVARIATE not in (s.get('adjust') or []):
            s['adjust'] = list(s.get('adjust') or []) + [fa.DATE_COVARIATE]
        self._changed()

    def used_files(self) -> dict:
        """{sample: path} of the linked files ticked for use, in experiment order."""
        if self.links is None:
            return {}
        use = self.settings.get('use') or {}
        out = {}
        for key, info in self.links['files'].items():
            if info['path'] is None:
                continue
            if use.get(key, self._default_use(key, info)):
                out[key] = info['path']
        return out

    # -- counting ---------------------------------------------------------------

    def _gating_snapshot(self):
        return fa.SampleGating.from_controller(self._controller())

    def _counts_signature(self, snapshot=None, phenotypes: dict | None = None) -> str:
        snapshot = snapshot or self._gating_snapshot()
        files = self.used_files()
        unmixing = self.plugin.live_unmixing_signature(list(files)) if self.is_live() else None
        phenotypes = self.ticked_phenotypes() if phenotypes is None else phenotypes
        return fa.counts_signature(files, self.settings.get('gates') or [], snapshot.signature(), unmixing,
                                   [res.signature_parts() for res in phenotypes.values()])

    def start_count(self):
        files = self.used_files()
        if not files:
            self.count_status.setText('No samples are ticked for use.')
            return
        gates = list(self.settings.get('gates') or [])
        phenotypes = self.ticked_phenotypes()
        snapshot = self._gating_snapshot()
        populations = (gates + list(phenotypes)) or [fa.ALL_EVENTS]
        markers = self.plugin.analysis_marker_channels()
        cache = fa.marker_cache_path(self._controller().experiment_dir)
        if self.is_live():
            self._start_live_count(files, gates, snapshot, populations, markers, cache, phenotypes)
            return
        if phenotypes:
            self.count_status.setText('Phenotypes are counted from the live unmix only: choose Live unmix as the '
                                      'source, or untick them.')
            return
        signature = fa.counts_signature(files, gates, snapshot.signature())
        folder, fluors, viability = self._compact_inputs()
        source_keys = {key: json.dumps([str(path), fa.file_signature(path)]) for key, path in files.items()}

        def work(progress):
            records, errors, read = {p: {} for p in populations}, {}, 0
            for i, (key, path) in enumerate(files.items()):
                progress(i, len(files), Path(path).name)
                try:
                    counted, full = fcnt.count_sample(key, lambda path=path: fa.read_exported_events(path), snapshot,
                                                      gates, markers, folder, source_keys[key], fluors, viability)
                    read += int(full)
                    for pop in populations:
                        records[pop][key] = counted[pop]
                except Exception as exc:
                    errors[key] = str(exc) or type(exc).__name__
            progress(len(files), len(files), '')
            fa.save_marker_sums(cache, signature, records, populations)
            return {'records': records, 'errors': errors, 'signature': signature, 'populations': populations,
                    'source': fa.SOURCE_EXPORTED, 'read_full': read}

        if self.plugin._start_task('analysis_count', work):
            self.count_btn.setEnabled(False)
            self.count_status.setText(f'Counting {len(files)} file(s)…')

    def _start_live_count(self, files: dict, gates: list, snapshot, populations: list, markers: list, cache,
                          phenotypes: dict | None = None):
        """FlowCode-unmix each sample's raw file (or read its cached unmix),
        gate and score it and count it, holding the heavy-job gate when
        there is one. *phenotypes* is ``{population: Resolved}``; each
        sample's negative model comes from its own AF library."""
        import flowcode_export as fce
        import flowcode_phenotype as fp
        import flowcode_state as fcs

        keys = list(files)
        phenotypes = dict(phenotypes or {})
        try:
            inputs = self.plugin.live_unmix_inputs(keys)
        except ValueError as exc:
            self.count_status.setText(f'Cannot unmix: {exc}')
            return
        gate_records = [self.plugin.state.gates[n] for n in fcs.GATE_NAMES if n in self.plugin.state.gates]
        fluor_names = list(inputs['layout'].fluor_names)

        def sample_phenotypes(key):
            if not phenotypes:
                return None
            model = fp.sample_negative_model(inputs['af_by_sample'][key], fluor_names, inputs['plan'].state, key)
            if model is None:
                raise ValueError('no negative model: the sample has no AF library and the experiment no noise model')
            return {name: (res, model) for name, res in phenotypes.items()}
        gate = getattr(self._controller(), 'job_gate', None)
        token = None
        if gate is not None:
            token = gate.try_acquire('FlowCode live unmix')
            if token is None:
                busy = getattr(gate, 'busy_text', None)
                self.count_status.setText(busy() if callable(busy) else 'Another job is running.')
                return
        unmixing = inputs['signature']
        signature = fa.counts_signature(files, gates, snapshot.signature(), unmixing,
                                        [res.signature_parts() for res in phenotypes.values()])
        cache_keys = {key: fce.live_cache_key(unmixing, key, fce.file_signature(path)) for key, path in files.items()}
        folder, fluors, viability = self._compact_inputs()

        def unmixed(key):
            events = fce.live_sample_events(
                inputs['layout'], inputs['af_by_sample'][key], inputs['unmix_fn'], key, inputs['pnn'],
                inputs['n_extra'], cache_key=cache_keys[key], max_cache_bytes=inputs['max_cache_bytes'])
            return np.asarray(events, dtype=np.float64), list(inputs['pnn'])

        def work(progress):
            records, errors, read = {p: {} for p in populations}, {}, 0
            for i, key in enumerate(keys):
                progress(i, len(keys), Path(key).stem)
                try:
                    if phenotypes:
                        events, pnn = unmixed(key)
                        counted = fa.count_populations(events, pnn, key, snapshot, populations, markers,
                                                       phenotypes=sample_phenotypes(key), gate_records=gate_records)
                        read += 1
                        del events
                    else:
                        counted, full = fcnt.count_sample(key, lambda key=key: unmixed(key), snapshot, gates, markers,
                                                          folder, cache_keys[key], fluors, viability)
                        read += int(full)
                    for pop in populations:
                        records[pop][key] = counted[pop]
                except Exception as exc:
                    logger.exception('FlowCode live unmix failed for %s', key)
                    errors[key] = str(exc) or type(exc).__name__
            progress(len(keys), len(keys), '')
            fa.save_marker_sums(cache, signature, records, populations)
            return {'records': records, 'errors': errors, 'signature': signature, 'populations': populations,
                    'phenotypes': {name: res.summary() for name, res in phenotypes.items()},
                    'source': fa.SOURCE_LIVE, 'method': inputs['plan'].label, 'read_full': read}

        if self.plugin._start_task('analysis_count', work):
            self._count_token = token
            self.count_btn.setEnabled(False)
            self.count_status.setText(f'Unmixing and counting {len(keys)} sample(s) ({inputs["plan"].label})…')
        elif token is not None:
            token.release()

    def _compact_inputs(self) -> tuple:
        """(folder, fluorophores, viability dyes) for the compact count cache
        (``flowcode_count_cache``), on the main thread."""
        panel = self._panel()
        return (fcnt.cache_folder(self._controller().experiment_dir), list(panel.fluorophores),
                list(panel.viability))

    def _release_count_token(self):
        token, self._count_token = self._count_token, None
        if token is not None:
            token.release()

    def on_count_finished(self, result):
        self._release_count_token()
        self.count_btn.setEnabled(True)
        self.settings['counts'] = {'signature': result['signature'], 'records': result['records'],
                                   'errors': result['errors'], 'populations': result['populations'],
                                   'phenotypes': dict(result.get('phenotypes') or {}),
                                   'source': result.get('source', fa.SOURCE_EXPORTED),
                                   'method': result.get('method'),
                                   'time': datetime.now().strftime('%Y-%m-%d %H:%M')}
        self._changed()
        self._load_counts()
        self._invalidate_stats()
        errors = result['errors']
        what = 'sample(s), unmixed live' if result.get('source') == fa.SOURCE_LIVE else 'file(s)'
        n_counted = len(next(iter(result['records'].values()), {}))
        text = f"Counted {n_counted} {what} in {', '.join(result['populations'])}."
        if 'read_full' in result:
            text += (f" {result['read_full']} read in full, the others from the kept events "
                     '(cache/flowcode_analysis/count_events).')
        if errors:
            text += ' Could not count: ' + '; '.join(f'{self._name(k)} ({m})' for k, m in errors.items())
        self.count_status.setText(text)
        self.refresh()

    def on_count_error(self, message: str):
        self._release_count_token()
        self.count_btn.setEnabled(True)
        self.count_status.setText(f'Counting failed: {message}')

    def _load_counts(self):
        """Tables from the saved counts (no file is read)."""
        populations, records = fa.saved_records(self.settings.get('counts') or {})
        if not any(records.values()):
            self.counts = self.meta = self.qc = self.filtered = None
            return
        design = self._current_design()
        order = list(design.combo_names) if design is not None else None
        self.counts, self.meta = fa.units_table(records, populations, combo_order=order)
        self._left_out = {}
        self._library_key = self._library_setup_key()
        if self.multi_library():
            samples = dict.fromkeys(self.meta['sample'])
            self.counts, self.meta, self._left_out = fa.library_features(
                self.counts, self.meta, {k: self.library_of(k) for k in samples}, self.feature_map())
            if self.counts.empty:
                self.counts = self.meta = self.qc = None
                self._apply_filters()
                return
        self.qc = fa.qc_table(self.counts, self.meta, self.unit_labels())
        self._apply_filters()

    def _apply_filters(self):
        left_out = self._left_out_note()
        if self.counts is None:
            self.filtered = None
            self.filter_note.setText(left_out)
            return
        s = self.settings
        self.filtered = fa.filter_counts(self.counts, s.get('exclude_combos') or [],
                                         int(s.get('min_combo_events', fa.DEFAULT_MIN_COMBO_EVENTS)),
                                         int(s.get('min_sample_events', fa.DEFAULT_MIN_SAMPLE_EVENTS)))
        if left_out:
            self.filtered.notes.insert(0, left_out)
        self.filter_note.setText(' '.join(self.filtered.notes) or
                                 f'{self.filtered.counts.shape[0]} sample(s) and '
                                 f'{self.filtered.counts.shape[1]} combination(s) analysed.')
        if hasattr(self, 'list_controls'):
            self._fill_controls()

    def _check_counts_current(self):
        """Say when the saved counts no longer match the files, gates or gating."""
        saved = self.settings.get('counts') or {}
        if not saved.get('records'):
            self.count_status.setText('Not counted yet.')
            return
        try:
            current = self._counts_signature()
        except Exception:
            current = None
        when = saved.get('time', '')
        if current != saved.get('signature'):
            self.count_status.setStyleSheet(_WARN_STYLE)
            self.count_status.setText(f"Counts from {when} ({self.counted_population()}) are out of date: "
                                      'the samples, the source, the unmixing, the population or the gating '
                                      'changed. Count again.')
        else:
            self.count_status.setStyleSheet(_HINT_STYLE)
            self.count_status.setText(f"Counts from {when}: {self.counted_population()}.")

    # -- data tables ------------------------------------------------------------

    def _fill_data(self):
        self._filling = True
        try:
            s = self.settings
            live = self.is_live()
            self.source_combo.setCurrentIndex(max(self.source_combo.findData(self.source()), 0))
            self.export_box.setVisible(not live)
            if live:
                summary = self.plugin.method_summary()
                problem = self._live_problem()
                self.source_note.setText(problem or (
                    f"Each sample is unmixed from its raw file with the {summary.get('label') or 'per-cell AF'} "
                    "and the FlowCode correction, as section 5 would export it. Unmixed values are cached in "
                    "the experiment's cache/flowcode_unmixed folder."))
            else:
                self.source_note.setText('')
            self.folder_label.setText(str(self.export_folder()))
            self.linked_list.clear()
            for folder in s.get('linked_folders') or []:
                self.linked_list.addItem(folder)
            links = self.links or {'files': {}, 'unlinked': []}
            notes = []
            n_linked = sum(1 for k in links['files'] if fa.is_linked(k))
            if s.get('linked_folders'):
                notes.append(f'{n_linked} file(s) linked from other experiments, gated with this experiment\'s '
                             'unmixed gates (without per-sample overrides).')
            if links['unlinked']:
                notes.append(f"{len(links['unlinked'])} file(s) in the folder match no sample: "
                             + ', '.join(Path(p).name for p in links['unlinked'][:5])
                             + ('…' if len(links['unlinked']) > 5 else ''))
            self.link_note.setText(' '.join(notes))
            self.min_combo_spin.setValue(int(s.get('min_combo_events', fa.DEFAULT_MIN_COMBO_EVENTS)))
            self.min_sample_spin.setValue(int(s.get('min_sample_events', fa.DEFAULT_MIN_SAMPLE_EVENTS)))
            include = s.get('include') or {}
            for key, chk in self.include_checks.items():
                chk.setChecked(bool(include.get(key)))

            names = []
            if self.counts is not None:
                names = list(self.counts.columns)
            elif self.multi_library():
                names = list(dict.fromkeys(f for fmap in self.feature_map().values() for f in fmap.values()))
            else:
                design = self._current_design()
                names = list(design.combo_names) if design is not None else []
            excluded = set(s.get('exclude_combos') or [])
            totals = self.counts.sum(axis=0) if self.counts is not None else None
            self.combo_list.clear()
            for name in names:
                text = name if totals is None else f'{name}  ({int(totals.get(name, 0)):,} events)'
                it = QListWidgetItem(text)
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Unchecked if name in excluded else Qt.Checked)
                it.setData(Qt.UserRole, name)
                self.combo_list.addItem(it)

            files = links['files']
            use = s.get('use') or {}
            pops = self.counted_populations()
            several = len(pops) > 1
            shown_pop = self.qc_pop_combo.currentData()
            self.qc_pop_combo.clear()
            for pop in pops:
                self.qc_pop_combo.addItem(pop, pop)
            if shown_pop in pops:
                self.qc_pop_combo.setCurrentIndex(pops.index(shown_pop))
            shown_pop = self.qc_pop_combo.currentData() or (pops[0] if pops else '')
            self.qc_pop_label.setVisible(several)
            self.qc_pop_combo.setVisible(several)
            multi = self.multi_library()
            lib_labels = self.library_labels()
            headers = (['Use', 'Sample'] + (['Library'] if multi else [])
                       + ['File status', 'Events', 'In population', 'Barcoded', '% barcoded', 'Untransduced',
                          'Unresolved'] + (['Out of library'] if multi else []) + ['Gini', 'Top combination (%)'])
            self.sample_table.setColumnCount(len(headers))
            self.sample_table.setHorizontalHeaderLabels(headers)
            self.sample_table.setRowCount(len(files))
            for r, (key, info) in enumerate(files.items()):
                chk = _item('', data=key)
                chk.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                usable = info['path'] is not None and info['status'] != fa.LINK_UNREADABLE
                if not usable:
                    chk.setFlags(Qt.ItemIsUserCheckable)
                ticked = usable and use.get(key, self._default_use(key, info))
                chk.setCheckState(Qt.Checked if ticked else Qt.Unchecked)
                self.sample_table.setItem(r, 0, chk)
                name = _item(self._name(key))
                name.setToolTip(info['path'] or key)
                self.sample_table.setItem(r, 1, name)
                lib = self.library_of(key)
                status = _item(info['status'] + (f" — {info['note']}" if info['note'] else ''))
                unit = fa.unit_id(key, shown_pop, several)
                qc = self.qc.loc[unit] if self.qc is not None and unit in self.qc.index else None
                cells = [''] * (9 if multi else 8)
                if qc is not None:
                    top = (f"{qc['top_combination']} ({_number(qc['top_share_pct'])})"
                           if qc['top_combination'] else '')
                    cells = ([_number(qc['events'], 0), _number(qc['in_population'], 0),
                              _number(qc['barcoded'], 0), _number(qc['pct_barcoded']),
                              _number(qc['untransduced'], 0), _number(qc['unresolved'], 0)]
                             + ([_number(qc['out_of_library'], 0)] if multi else [])
                             + [_number(qc['gini'], 2), top])
                row = ([_item(lib_labels.get(lib, lib) if lib else '—')] if multi else []) + [status]
                row += [_item(text) for text in cells]
                for j, item in enumerate(row, start=2):
                    self.sample_table.setItem(r, j, item)
            # Sized once after filling; the sample name is capped (and elided
            # in the middle) so the QC columns stay in view.
            self.sample_table.resizeColumnsToContents()
            self.sample_table.setColumnWidth(1, min(self.sample_table.columnWidth(1), SAMPLE_COLUMN_MAX_WIDTH))
        finally:
            self._filling = False

    def _fill_gate_tree(self):
        gating = self._controller().unmixed_gating
        try:
            hierarchy = gating.get_gate_hierarchy(output='dict')
        except Exception:
            hierarchy = None
        known = set(self._gating_snapshot().gate_names())
        kept = [g for g in (self.settings.get('gates') or []) if g in known]
        if kept != list(self.settings.get('gates') or []):
            self.settings['gates'] = kept
        key = repr(hierarchy)
        filling, self._filling = self._filling, True
        try:
            if key != self._hierarchy_key:
                self._hierarchy_key = key
                self.gate_tree.clear()

                def add(node, parent):
                    for child in node.get('children', []) or []:
                        item = QTreeWidgetItem(parent, [child['name']])
                        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                        item.setCheckState(0, Qt.Unchecked)
                        add(child, item)

                if hierarchy:
                    add(hierarchy, self.gate_tree.invisibleRootItem())
                self.gate_tree.expandToDepth(1)

            def sync(item):
                checked = item.text(0) in kept
                item.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
                if checked:
                    parent = item.parent()
                    while parent is not None:
                        parent.setExpanded(True)
                        parent = parent.parent()
                for i in range(item.childCount()):
                    sync(item.child(i))

            for i in range(self.gate_tree.topLevelItemCount()):
                sync(self.gate_tree.topLevelItem(i))
        finally:
            self._filling = filling

    # =====================================================================
    # Groups
    # =====================================================================

    def _build_groups(self, lay):
        top = QHBoxLayout()
        gbox = QGroupBox('Groups')
        gl = QVBoxLayout(gbox)
        self.group_defs = QTableWidget(0, 3)
        self.group_defs.setHorizontalHeaderLabels(['Group', 'Sample-name pattern (regex)', 'Colour'])
        header = self.group_defs.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        self.group_defs.setColumnWidth(0, 160)
        self.group_defs.setColumnWidth(2, 60)
        self.group_defs.verticalHeader().setVisible(False)
        self.group_defs.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.group_defs.setSelectionMode(QAbstractItemView.SingleSelection)
        self.group_defs.setMaximumHeight(170)
        self.group_defs.setToolTip(
            'Double-click a name or pattern to edit it, or a colour to change it. The colours are used '
            'in every figure (volcano direction ramps, heatmap group bars, PCA, QC and per-combination '
            'plots).')
        self.group_defs.itemChanged.connect(self._on_group_def_changed)
        self.group_defs.cellDoubleClicked.connect(self._on_group_def_double_clicked)
        gl.addWidget(self.group_defs)
        brow = QHBoxLayout()
        assign = QPushButton('Assign by pattern')
        assign.setToolTip(
            'Match each group\'s pattern (a regular expression, case-insensitive) against the sample '
            'names below; the first matching group, in the order above, wins. Every sample listed is '
            'reassigned, so a sample matching no pattern loses its group.')
        for b, slot in ((QPushButton('Add group'), self._on_add_group),
                        (QPushButton('Remove group'), self._on_remove_group),
                        (assign, self._on_assign_by_pattern),
                        (QPushButton('Suggest groupings…'), self._on_suggest_groupings)):
            b.clicked.connect(slot)
            brow.addWidget(b)
        for text, step, tip in (('▲', -1, 'Move the selected group up.'),
                                ('▼', 1, 'Move the selected group down.')):
            b = QPushButton(text)
            b.setFixedWidth(32)
            b.setToolTip(tip + ' The group order is the order used in the Reference list, '
                         'comparisons between every pair (the earlier group is the baseline) '
                         'and every figure.')
            b.clicked.connect(lambda _c=False, s=step: self._on_move_group(s))
            brow.addWidget(b)
        brow.addStretch()
        gl.addLayout(brow)
        top.addWidget(gbox, stretch=2)

        cbox = QGroupBox('Covariates')
        cl = QVBoxLayout(cbox)
        cl.addWidget(_label('Per-sample values, e.g. mouse (for pairing), tissue (to split by) or sex (to adjust '
                            'for). Edit them in the table below.', ''))
        for pair in ((('Add covariate…', self._on_add_covariate), ('Remove covariate…', self._on_remove_covariate)),
                     (('Import table…', self._on_import_table), ('Export table…', self._on_export_table))):
            crow = QHBoxLayout()
            for text, slot in pair:
                b = QPushButton(text)
                b.clicked.connect(slot)
                crow.addWidget(b)
            crow.addStretch()
            cl.addLayout(crow)
        cl.addStretch()
        top.addWidget(cbox, stretch=1)
        lay.addLayout(top)

        auto = QHBoxLayout()
        self.auto_label = _label('')
        self.undo_btn = QPushButton('Undo')
        self.undo_btn.setToolTip('Remove the groups and pairing applied from the sample names.')
        self.undo_btn.clicked.connect(self._undo_auto_groupings)
        auto.addWidget(self.auto_label, stretch=1)
        auto.addWidget(self.undo_btn)
        self.auto_label.setVisible(False)
        self.undo_btn.setVisible(False)
        lay.addLayout(auto)
        self.group_count_label = _label('')
        lay.addWidget(self.group_count_label)
        self.samples_table = QTableWidget(0, 2)
        self.samples_table.verticalHeader().setVisible(False)
        self.samples_table.setTextElideMode(Qt.ElideMiddle)
        self.samples_table.setWordWrap(False)
        self.samples_table.setMinimumHeight(300)
        self.samples_table.itemChanged.connect(self._on_group_cell_changed)
        lay.addWidget(self.samples_table)

    def group_samples(self) -> list:
        """Samples shown in the Groups table: those ticked for use, or the
        counted samples when counts exist."""
        if self.meta is not None:
            return list(dict.fromkeys(self.meta['sample']))
        return list(self.used_files())

    def group_colours(self) -> dict:
        """{group: colour} for every group; groups without one get the next
        unused colour, which is saved."""
        s = self.settings
        if fa.resolve_group_colours(s):
            self._changed()
        return {g: s['group_colors'][g] for g in s.get('group_names') or []}

    def _fill_groups(self):
        self._fill_group_defs()
        self._fill_samples_table()

    def _fill_group_defs(self):
        s = self.settings
        colours = self.group_colours()
        patterns = s.get('group_patterns') or {}
        self._filling = True
        try:
            self.group_defs.setRowCount(len(s.get('group_names') or []))
            for r, g in enumerate(s.get('group_names') or []):
                self.group_defs.setItem(r, 0, _item(g, editable=True, data=g))
                self.group_defs.setItem(r, 1, _item(patterns.get(g, ''), editable=True, data=g))
                swatch = _item('', data=g)
                swatch.setBackground(QColor(colours[g]))
                self.group_defs.setItem(r, 2, swatch)
        finally:
            self._filling = False

    def _fill_samples_table(self):
        s = self.settings
        keys = self.group_samples()
        covs = list(s.get('covariates') or {})
        names = s.get('group_names') or []
        icons = {g: _swatch_icon(c) for g, c in self.group_colours().items()}
        table = self.samples_table
        multi = self.multi_library()
        fixed = ['Sample', 'Group'] + (['Library (section 2)'] if multi else [])
        self._cov_start = len(fixed)
        lib_labels = self.library_labels()
        self._filling = True
        table.setUpdatesEnabled(False)
        try:
            header = table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.Interactive)
            header.setStretchLastSection(False)
            table.setColumnCount(len(fixed) + len(covs))
            table.setHorizontalHeaderLabels(fixed + covs)
            table.setRowCount(len(keys))
            for r, key in enumerate(keys):
                item = _item(self._name(key), data=key)
                item.setToolTip(str(key))
                table.setItem(r, 0, item)
                combo = QComboBox()
                combo.addItem('—', '')
                for g in names:
                    combo.addItem(icons[g], g, g)
                combo.setCurrentIndex(max(combo.findData((s.get('sample_groups') or {}).get(key, '')), 0))
                combo.currentIndexChanged.connect(lambda _i, k=key, c=combo: self._set_group(k, c.currentData()))
                _no_wheel(combo)
                table.setCellWidget(r, 1, combo)
                if multi:
                    lib = self.library_of(key)
                    item = _item(lib_labels.get(lib, lib) if lib else '—', data=key)
                    item.setToolTip('Assigned in section 2 (Tags & Combos).')
                    table.setItem(r, 2, item)
                for j, cov in enumerate(covs, start=self._cov_start):
                    table.setItem(r, j, _item((s['covariates'].get(cov) or {}).get(key, ''),
                                              editable=True, data=key))
            # Widths from font metrics: the sample column is capped (names are
            # elided in the middle) so groups and covariates stay readable.
            fm = table.fontMetrics()
            longest = max((fm.horizontalAdvance(self._name(k)) for k in keys), default=0)
            table.setColumnWidth(0, max(120, min(longest + 24, SAMPLE_COLUMN_MAX_WIDTH)))
            group_w = max([fm.horizontalAdvance(g) for g in names] + [fm.horizontalAdvance('Group')]) + 64
            table.setColumnWidth(1, max(group_w, 110))
            if multi:
                table.setColumnWidth(2, max(fm.horizontalAdvance(fixed[2]) + 32, COVARIATE_COLUMN_MIN_WIDTH))
            for j, cov in enumerate(covs, start=self._cov_start):
                values = (s['covariates'].get(cov) or {}).values()
                widest = max([fm.horizontalAdvance(str(v)) for v in values] + [0]) + 24
                table.setColumnWidth(j, max(COVARIATE_COLUMN_MIN_WIDTH, fm.horizontalAdvance(cov) + 32,
                                            min(widest, SAMPLE_COLUMN_MAX_WIDTH)))
        finally:
            table.setUpdatesEnabled(True)
            self._filling = False
        self._update_group_counts()

    def _update_group_counts(self):
        s = self.settings
        keys = self.group_samples()
        n = {g: 0 for g in s.get('group_names') or []}
        unassigned = 0
        for k in keys:
            g = (s.get('sample_groups') or {}).get(k, '')
            if g in n:
                n[g] += 1
            else:
                unassigned += 1
        parts = [f'{g}: {c}' for g, c in n.items()] + [f'No group: {unassigned}']
        self.group_count_label.setText('  |  '.join(parts))

    def _set_group(self, key, group):
        if self._filling:
            return
        groups = self.settings.setdefault('sample_groups', {})
        if group:
            groups[key] = group
        else:
            groups.pop(key, None)
        self._changed()
        self._update_group_counts()
        self._invalidate_stats()
        self._fill_stats_choices()

    def _on_group_cell_changed(self, item):
        if self._filling or item.column() < self._cov_start:
            return
        cov = self.samples_table.horizontalHeaderItem(item.column()).text()
        key = item.data(Qt.UserRole)
        value = item.text().strip()
        values = self.settings['covariates'].setdefault(cov, {})
        if value:
            values[key] = value
        else:
            values.pop(key, None)
        self._changed()
        self._invalidate_stats()

    def _on_group_def_changed(self, item):
        """A group renamed (column 0) or its pattern edited (column 1)."""
        if self._filling:
            return
        s = self.settings
        old = item.data(Qt.UserRole)
        if old not in (s.get('group_names') or []):
            return
        if item.column() == 1:
            s.setdefault('group_patterns', {})[old] = item.text().strip()
            self._changed()
            return
        if item.column() != 0:
            return
        new = item.text().strip()
        if new == old:
            return
        if not new or new in s['group_names']:
            QMessageBox.warning(self, 'Rename group', 'A group name must be new and not blank.')
            self._fill_group_defs()
            return
        s['group_names'][s['group_names'].index(old)] = new
        for table in ('group_patterns', 'group_colors'):
            values = s.setdefault(table, {})
            if old in values:
                values[new] = values.pop(old)
        s['sample_groups'] = {k: (new if g == old else g) for k, g in (s.get('sample_groups') or {}).items()}
        if s.get('reference') == old:
            s['reference'] = new
        self._changed()
        self._fill_groups()
        self._invalidate_stats()
        self._fill_stats_choices()

    def _on_group_def_double_clicked(self, row: int, col: int):
        if col != 2:
            return
        item = self.group_defs.item(row, 2)
        name = item.data(Qt.UserRole) if item is not None else None
        if name not in (self.settings.get('group_names') or []):
            return
        current = self.group_colours()[name]
        colour = QColorDialog.getColor(QColor(current), self, f"Colour for group '{name}'")
        if colour.isValid() and colour.name() != current:
            self.settings['group_colors'][name] = colour.name()
            self._changed()
            self._fill_groups()
            if self.stats is not None:
                self._show_results()

    def _on_assign_by_pattern(self):
        s = self.settings
        keys = self.group_samples()
        names = {k: self._name(k) for k in keys}
        try:
            groups, matched = fa.assign_by_pattern(keys, names, list(s.get('group_names') or []),
                                                   s.get('group_patterns') or {})
        except ValueError as exc:
            QMessageBox.warning(self, 'Assign by pattern', f'Invalid pattern for {exc}')
            return
        if not matched:
            QMessageBox.information(self, 'Assign by pattern', 'Enter a sample-name pattern for at least one group.')
            return
        sample_groups = s.setdefault('sample_groups', {})
        for k in keys:
            if k in groups:
                sample_groups[k] = groups[k]
            else:
                sample_groups.pop(k, None)
        self._show_auto_note(None)
        self._changed()
        self._fill_samples_table()
        self._fill_stats_choices()
        self._invalidate_stats()
        self.group_count_label.setText(
            self.group_count_label.text() + '  —  matched: '
            + ', '.join(f'{g} {n}' for g, n in matched.items()))

    def _on_add_group(self):
        name, ok = QInputDialog.getText(self, 'Add group', 'Group name:')
        name = (name or '').strip()
        if ok and name and name not in self.settings['group_names']:
            self.settings['group_names'].append(name)
            self._changed()
            self._fill_groups()
            self._fill_stats_choices()

    def _on_move_group(self, step: int):
        """Move the group selected in the Groups table *step* rows up (-1)
        or down (+1)."""
        names = self.settings.get('group_names') or []
        row = self.group_defs.currentRow()
        new = row + step
        if row < 0 or not 0 <= new < len(names):
            return
        names.insert(new, names.pop(row))
        self._changed()
        self._fill_groups()
        self._fill_stats_choices()
        self._invalidate_stats()
        self.group_defs.setCurrentCell(new, 0)

    def _on_remove_group(self):
        """Remove the group selected in the Groups table (or chosen from a
        list when none is selected); its samples lose their group."""
        names = list(self.settings['group_names'])
        if not names:
            return
        row = self.group_defs.currentRow()
        item = self.group_defs.item(row, 0) if row >= 0 else None
        name = item.data(Qt.UserRole) if item is not None else None
        if name not in names:
            name, ok = QInputDialog.getItem(self, 'Remove group', 'Group:', names, editable=False)
            if not ok:
                return
        if name:
            s = self.settings
            s['group_names'].remove(name)
            s['sample_groups'] = {k: g for k, g in s['sample_groups'].items() if g != name}
            for table in ('group_patterns', 'group_colors'):
                (s.get(table) or {}).pop(name, None)
            self._changed()
            self._fill_groups()
            self._fill_stats_choices()
            self._invalidate_stats()

    def _on_add_covariate(self):
        name, ok = QInputDialog.getText(self, 'Add covariate', 'Covariate name (e.g. mouse, sex):')
        name = (name or '').strip()
        if ok and name and name not in self.settings['covariates']:
            self.settings['covariates'][name] = {}
            self._changed()
            self._fill_groups()
            self._fill_stats_choices()

    def _on_remove_covariate(self):
        names = list(self.settings['covariates'])
        if not names:
            return
        name, ok = QInputDialog.getItem(self, 'Remove covariate', 'Covariate:', names, editable=False)
        if ok and name:
            self._drop_covariate(name)

    def _drop_covariate(self, name):
        s = self.settings
        s['covariates'].pop(name, None)
        if name == fa.DATE_COVARIATE:
            s['date_covariate'] = False
        if s.get('pairing') == name:
            s['pairing'] = ''
        if s.get('split_by') == name:
            s['split_by'] = ''
        s['adjust'] = [c for c in s.get('adjust') or [] if c != name]
        self._changed()
        self._fill_groups()
        self._fill_stats_choices()
        self._invalidate_stats()

    def _on_import_table(self):
        path, _f = QFileDialog.getOpenFileName(self, 'Import groups and covariates', '',
                                               'Tables (*.csv *.tsv *.txt)')
        if not path:
            return
        self.plugin._ensure_bundled_on_path()
        import ag_results
        try:
            sep = '\t' if path.lower().endswith(('.tsv', '.txt')) else ','
            table = ag_results.read_sample_table(pd.read_csv(path, sep=sep, dtype=str), self.group_samples())
        except Exception as exc:
            QMessageBox.warning(self, 'Import table', f'Could not read the table: {exc}')
            return
        if self.multi_library():
            # The libraries are assigned in section 2; an exported table's column is not a covariate.
            table['covariates'].pop('library', None)
        s = self.settings
        for g in dict.fromkeys(table['groups'].values()):
            if g not in s['group_names']:
                s['group_names'].append(g)
        s['sample_groups'].update(table['groups'])
        for name, values in table['covariates'].items():
            s['covariates'].setdefault(name, {}).update(values)
        self._changed()
        self._fill_groups()
        self._fill_stats_choices()
        self._invalidate_stats()
        if table['unmatched']:
            QMessageBox.information(self, 'Import table', 'Rows matching no sample: '
                                    + ', '.join(table['unmatched'][:10]))

    def _on_export_table(self):
        path, _f = QFileDialog.getSaveFileName(self, 'Export groups and covariates', 'flowcode_samples.csv',
                                               'CSV (*.csv)')
        if path:
            self.sample_table_frame().to_csv(path)

    def sample_table_frame(self) -> pd.DataFrame:
        """One row per sample: name, group, the library (with several
        libraries), then each covariate."""
        s = self.settings
        keys = self.group_samples()
        df = pd.DataFrame(index=pd.Index(keys, name='sample'))
        df['name'] = [self._name(k) for k in keys]
        df['group'] = [(s.get('sample_groups') or {}).get(k, '') for k in keys]
        if self.multi_library():
            labels = self.library_labels()
            df['library'] = [labels.get(self.library_of(k), self.library_of(k) or '') for k in keys]
        for cov, values in (s.get('covariates') or {}).items():
            df[cov] = [values.get(k, '') for k in keys]
        return df

    # -- suggested groupings ------------------------------------------------------

    def _groupings_result(self, keys):
        return ds.suggest_groupings_from_names(keys, self._display_names())

    def _on_suggest_groupings(self):
        keys = self.group_samples()
        if len(keys) < 2:
            QMessageBox.information(self, 'Suggest Groupings', 'Need at least two samples to detect a pattern.')
            return
        result = self._groupings_result(keys)
        if not result['suggestions']:
            QMessageBox.information(self, 'Suggest Groupings', sgd.no_suggestion_message(result))
            return
        chosen = sgd.choose_groupings(self, result)
        if chosen:
            self.apply_groupings(chosen, keys)
            self._show_auto_note(None)

    def maybe_auto_suggest_groupings(self):
        """Apply groupings found in the sample names when none of the samples
        has a group: the detected group, and the detected pairing variable as
        Pair by. Runs once per sample list and never overwrites assignments;
        the note above the table says what was applied and offers Undo."""
        s = self.settings
        keys = self.group_samples()
        if len(keys) < 2 or any((s.get('sample_groups') or {}).get(k) in (s.get('group_names') or [])
                                for k in keys):
            return
        if tuple(keys) == self._auto_groupings_key:
            return
        self._auto_groupings_key = tuple(keys)
        chosen = sgd.automatic_groupings(self._groupings_result(keys))
        if not chosen:
            return
        from copy import deepcopy
        self._auto_groupings_undo = {k: deepcopy(s.get(k)) for k in
                                     ('group_names', 'group_patterns', 'sample_groups', 'covariates', 'pairing',
                                      'reference')}
        self.apply_groupings(chosen, keys)
        self._show_auto_note(sgd.describe_groupings(chosen))

    def _show_auto_note(self, text):
        if text is None:
            self._auto_groupings_undo = None
        self.auto_label.setText(text or '')
        self.auto_label.setVisible(text is not None)
        self.undo_btn.setVisible(text is not None and self._auto_groupings_undo is not None)

    def _undo_auto_groupings(self):
        if self._auto_groupings_undo is None:
            return
        self.settings.update(self._auto_groupings_undo)
        self._show_auto_note(None)
        self._changed()
        self._fill_groups()
        self._fill_stats_choices()
        self._invalidate_stats()

    def apply_groupings(self, chosen: list, keys: list):
        """Write chosen groupings: a 'group' choice replaces the groups and the
        listed samples' assignments; the others become covariates, and a
        'pairing' choice is selected under Pair by."""
        s = self.settings
        for choice in chosen:
            if choice['role'] == 'group':
                names = sorted(set(choice['values'].values()))
                s['group_names'] = names
                s['group_patterns'] = {n: (s.get('group_patterns') or {}).get(n, '') for n in names}
                groups = s.setdefault('sample_groups', {})
                for k in keys:
                    if k in choice['values']:
                        groups[k] = choice['values'][k]
                    else:
                        groups.pop(k, None)
            else:
                name = choice['name']
                if name in s['covariates']:
                    name = f'{name}_2'
                s['covariates'][name] = dict(choice['values'])
                if choice['role'] == 'pairing':
                    s['pairing'] = name
        self._changed()
        self._fill_groups()
        self._fill_stats_choices()
        self._invalidate_stats()

    # =====================================================================
    # Statistics
    # =====================================================================

    def _build_stats(self, lay):
        row = QHBoxLayout()
        left = QWidget()
        left.setMaximumWidth(480)
        form = QFormLayout(left)
        self.stats_defaults_btn = QPushButton('Restore defaults')
        self.stats_defaults_btn.setToolTip(
            'Put the tests, normalisation, clone, marker and threshold settings back to their default values. '
            'Groups, reference, pairing, covariates, split and the chosen markers and control combinations '
            'are kept.')
        self.stats_defaults_btn.clicked.connect(self._restore_stats_defaults)
        defaults_row = QHBoxLayout()
        defaults_row.addWidget(self.stats_defaults_btn)
        defaults_row.addStretch()
        form.addRow(defaults_row)
        self.chk_counts = QCheckBox('Counts (negative binomial GLM, library-size offset)')
        self.chk_freq = QCheckBox('Frequencies (moderated linear model on log2 % of library size)')
        self.chk_markers = QCheckBox('Markers (each combination against reference cells of its sample)')
        self.chk_markers.setToolTip(
            'For every combination and marker: the mean intensity of the combination\'s cells minus that of '
            'the reference cells in the same sample (on the experiment\'s display scale), tested against '
            'zero within each group with a moderated t. Combinations are families: a combination is '
            'screened over all its markers first (cluster-first FDR).')
        form.addRow(self.chk_counts)
        self.counts_note = _label('The Counts test needs whole events; the counts include a phenotype population '
                                  '(memberships summed), so only the Frequencies and Markers tests run.')
        self.counts_note.setWordWrap(True)
        self.counts_note.setVisible(False)
        form.addRow(self.counts_note)
        form.addRow(self.chk_freq)
        form.addRow(self.chk_markers)
        self.cb_count_test = QComboBox()
        for key, text in COUNT_TESTS.items():
            self.cb_count_test.addItem(text, key)
        self.cb_count_test.setToolTip(
            'Quasi-likelihood F: each comparison fitted on its two groups, negative-binomial dispersions taken '
            'from combinations with similar counts, then a moderated quasi-dispersion per combination; 2–6 % '
            'false positives at p < 0.05 on inbred mice (about 5 % with 100 cells per sample), more on varied '
            'donors (7–10 %). Wald: each combination\'s own dispersion and a z test; not recommended (21–31 % '
            'on the same mice).')
        self.cb_norm = QComboBox()
        for key, text in NORMALISATIONS.items():
            self.cb_norm.addItem(text, key)
        self.cb_norm.setToolTip(
            'What each sample\'s counts are compared against. Total: its barcoded events, so one expanded '
            'clone pushes every other combination down. Robust: barcoded events rescaled by a trimmed ratio '
            'of sums that a few dominant combinations cannot move. Control: the chosen control combinations '
            '(e.g. non-targeting guides), after removing their own expanded clones.')
        self.list_controls = QListWidget()
        self.list_controls.setMaximumHeight(90)
        self.list_controls.setToolTip('Control combinations for Control normalisation.')
        self.lbl_controls = QLabel('Control combinations:')
        form.addRow('Counts test:', self.cb_count_test)
        form.addRow('Normalisation:', self.cb_norm)
        form.addRow(self.lbl_controls, self.list_controls)
        self.chk_clones = QCheckBox('Remove expanded clones')
        self.chk_clones.setToolTip(
            'The lab R rule: in each non-reference group, a combination whose % of barcoded in a sample is '
            'above the median + SD of its % over the group\'s samples (those in which it was seen), with more '
            'than the minimum events and %, is removed for that mouse (pairing level) in every tissue, unless '
            'it is expanded in more than the given number of mice. Found clones are listed in clones.csv on '
            'export.')
        self.sb_clone_count = _int_spin(0, 1000000)
        self.sb_clone_pct = _float_spin(0.0, 100.0, 1, 1.0)
        self.sb_clone_levels = _int_spin(0, 1000)
        self.sb_clone_levels.setToolTip('A combination expanded in more pairing levels (mice) than this is '
                                        'kept: a shared expansion is treated as an effect, not a clone.')
        clone_row = QHBoxLayout()
        for text, w in (('min. events', self.sb_clone_count), ('min. %', self.sb_clone_pct),
                        ('keep if in more than', self.sb_clone_levels)):
            clone_row.addWidget(QLabel(text))
            clone_row.addWidget(w)
        clone_row.addStretch()
        form.addRow(self.chk_clones)
        form.addRow(clone_row)
        self.cb_marker_ref = QComboBox()
        for key, text in MARKER_REFERENCES.items():
            self.cb_marker_ref.addItem(text, key)
        self.cb_marker_ref.setToolTip(
            'What each combination is compared with in its sample. Tag model (recommended): what its own tags '
            'predict from the other combinations of the sample, so a tag fluorophore\'s spillover into a marker, '
            'shared by every combination carrying that tag, is not taken for an effect. Control combinations: '
            'those ticked for Control normalisation (their tags may not be balanced). All other barcoded cells.')
        self.chk_marker_spill = QCheckBox('Account for tag spillover')
        self.chk_marker_spill.setToolTip(
            'For each combination and marker, the spillover spread of the combination\'s tags into the marker at '
            'their median brightness (from the spectral model, the characterisation and the noise model), on the '
            'display scale at the sample\'s level of the marker, is added to the standard error: '
            'SE_total² = SE_moderated² + (k × spillover)². Every feature is tested; features whose uncertainty is '
            'mostly spillover are hatched in the heatmap.')
        self.chk_marker_k_auto = QCheckBox('Auto (NON-Targetting)')
        self.chk_marker_k_auto.setToolTip(
            'k is the smallest value (0 to 4, steps of 0.25) at which the control combinations (those ticked for '
            'Control normalisation) have no significant marker hits, per population. Untick to set k.')
        self.sb_marker_spill = _float_spin(0.0, 10.0, 2, 0.25)
        self.sb_marker_spill.setToolTip('k: how many spillover spreads are added to the standard error.')
        self.list_markers = QListWidget()
        self.list_markers.setMaximumHeight(110)
        self.list_markers.setToolTip('Markers to test.')
        self.sb_marker_events = _int_spin(1, 100000)
        self.sb_marker_events.setToolTip('A combination, or its reference, with fewer events than this in a '
                                         'sample has no marker value there.')
        self.sb_marker_delta = _float_spin(0.0, 1.0, 3, 0.01)
        self.sb_marker_delta.setToolTip('|Δ| threshold in transformed units (the display scale, 0 to 1).')
        self.lbl_markers = _label('')
        marker_row = QHBoxLayout()
        for text, w in (('min. events', self.sb_marker_events), ('|Δ| threshold', self.sb_marker_delta)):
            marker_row.addWidget(QLabel(text))
            marker_row.addWidget(w)
        marker_row.addStretch()
        spill_row = QHBoxLayout()
        spill_row.addWidget(self.chk_marker_spill)
        spill_row.addWidget(QLabel('k'))
        spill_row.addWidget(self.chk_marker_k_auto)
        spill_row.addWidget(self.sb_marker_spill)
        spill_row.addStretch()
        self.marker_widgets = [self.cb_marker_ref, self.list_markers, self.sb_marker_events, self.sb_marker_delta,
                               self.chk_marker_spill, self.chk_marker_k_auto, self.sb_marker_spill]
        form.addRow('Marker reference:', self.cb_marker_ref)
        form.addRow('Markers:', self.list_markers)
        form.addRow(marker_row)
        form.addRow(spill_row)
        self.lbl_marker_k = _label('')
        form.addRow(self.lbl_marker_k)
        form.addRow(self.lbl_markers)
        self.chk_pathways = QCheckBox('Pathways (signed GO Biological Process sets, from the Counts test)')
        self.chk_pathways.setToolTip(
            'Tests whether the knockouts of a process, taken together and with their signs (+1 part of or driving '
            'the process, -1 holding it back), change the counts. Needs the Counts test with the quasi-likelihood '
            'F test.')
        self.sb_pathway_size = _int_spin(fpw.MIN_SIZE_FLOOR, 50)
        self.sb_pathway_size.setToolTip('A pathway or gene group needs at least this many tested guides, over all '
                                        'libraries.')
        self.sb_pathway_rotations = _int_spin(999, 99999)
        self.sb_pathway_rotations.setSingleStep(1000)
        self.sb_pathway_rotations.setToolTip(
            'Number of "nothing happening" versions of the data per library. The smallest p-value is '
            '1/(rotations + 1).')
        self.cb_pathway_evidence = QComboBox()
        for key, text in PATHWAY_EVIDENCE.items():
            self.cb_pathway_evidence.addItem(text, key)
        self.cb_pathway_evidence.setToolTip(
            'Which GO annotations make a gene a member: all of them (electronic annotations, IEA, included), or '
            'only those with experimental evidence.')
        self.btn_pathway_names = QPushButton('Gene names…')
        self.btn_pathway_names.setToolTip('The MGI gene matched to each combination name; set a gene by hand.')
        self.btn_pathway_names.clicked.connect(self._on_pathway_names)
        pathway_row = QHBoxLayout()
        for text, w in (('min. members', self.sb_pathway_size), ('rotations', self.sb_pathway_rotations)):
            pathway_row.addWidget(QLabel(text))
            pathway_row.addWidget(w)
        pathway_row.addWidget(self.btn_pathway_names)
        pathway_row.addStretch()
        self.pathway_widgets = [self.sb_pathway_size, self.sb_pathway_rotations, self.cb_pathway_evidence]
        self.lbl_pathways = _label('')
        form.addRow(self.chk_pathways)
        form.addRow(pathway_row)
        form.addRow('GO evidence:', self.cb_pathway_evidence)
        form.addRow(self.lbl_pathways)
        self.chk_gene_groups = QCheckBox('Gene groups (families of guides from a table, from the Counts test)')
        self.chk_gene_groups.setToolTip(
            'Tests whether the knockouts of each gene group (e.g. integrins, AP-1 factors), taken together, change '
            'the counts. Uses the min. members and rotations above. Needs the Counts test with the quasi-likelihood '
            'F test and a gene-group table.')
        self.btn_gene_groups = QPushButton('Gene groups…')
        self.btn_gene_groups.setToolTip('Load the gene-group table; see each combination\'s group; order and colour '
                                        'the groups.')
        self.btn_gene_groups.clicked.connect(self._on_gene_groups)
        self.btn_spider_comparisons = QPushButton('Spider comparisons…')
        self.btn_spider_comparisons.setToolTip(
            f'The comparisons drawn on the gene group spider (at most {fgg.SPIDER_COMPARISONS}); by default those '
            'against the reference group.')
        self.btn_spider_comparisons.clicked.connect(self._on_spider_comparisons)
        group_row = QHBoxLayout()
        group_row.addWidget(self.btn_gene_groups)
        group_row.addWidget(self.btn_spider_comparisons)
        group_row.addStretch()
        self.lbl_gene_groups = _label('')
        form.addRow(self.chk_gene_groups)
        form.addRow(group_row)
        form.addRow(self.lbl_gene_groups)
        self.cb_compare = QComboBox()
        for key, text in COMPARE.items():
            self.cb_compare.addItem(text, key)
        self.cb_compare.setToolTip('Sample groups: e.g. tissues against the spleen, within each population. '
                                   'Populations: e.g. CD8 against Tconv cells, paired by sample, within each '
                                   'sample group. Needs several populations counted.')
        self.cb_ref_pop = QComboBox()
        self.cb_ref_pop.setToolTip('The population the others are compared with.')
        self.chk_pop_contrasts = QCheckBox('Population contrasts')
        self.chk_pop_contrasts.setToolTip(
            'Also test whether each group\'s change against the reference group differs between each '
            'population and the reference population, e.g. whether a guide is more enriched in the pancreas '
            '(against the spleen) among CD8 than among Tconv cells. Paired by mouse when a pairing is chosen.')
        self.cb_mode = QComboBox()
        for key, text in CONTRAST_MODES.items():
            self.cb_mode.addItem(text, key)
        self.cb_reference = QComboBox()
        self.cb_pairing = QComboBox()
        self.cb_split = QComboBox()
        self.cb_split.setToolTip('Run the comparisons within each level of this covariate, '
                                 'e.g. COPD vs healthy within each tissue.')
        self.list_adjust = QListWidget()
        self.list_adjust.setMaximumHeight(80)
        self.list_adjust.setToolTip('Covariates to adjust for (numbers as continuous terms, anything else as '
                                    'categories).')
        form.addRow('Compare:', self.cb_compare)
        form.addRow('Comparisons:', self.cb_mode)
        form.addRow('Reference group:', self.cb_reference)
        form.addRow('Reference population:', self.cb_ref_pop)
        form.addRow(self.chk_pop_contrasts)
        form.addRow('Pair samples by:', self.cb_pairing)
        form.addRow('Split by:', self.cb_split)
        form.addRow('Adjust for:', self.list_adjust)
        self.sb_p = _float_spin(0.0001, 0.5, 4, 0.01)
        self.sb_fc = _float_spin(0.0, 10.0, 2, 0.1)
        self.chk_treat = QCheckBox('TREAT: test |log2FC| above the threshold')
        self.cb_fdr = QComboBox()
        for key, text in FDR_SCOPES.items():
            self.cb_fdr.addItem(text, key)
        form.addRow('FDR threshold:', self.sb_p)
        form.addRow('|log2FC| threshold:', self.sb_fc)
        form.addRow(self.chk_treat)
        form.addRow('FDR correction:', self.cb_fdr)
        for w in (self.cb_mode, self.cb_reference, self.cb_pairing, self.cb_split, self.cb_fdr,
                  self.cb_count_test, self.cb_norm, self.sb_clone_count, self.sb_clone_pct,
                  self.sb_clone_levels, self.cb_compare, self.cb_ref_pop, self.cb_marker_ref,
                  self.sb_marker_events, self.sb_marker_delta, self.sb_marker_spill, self.cb_pathway_evidence):
            _no_wheel(w)
        row.addWidget(left)
        right = QVBoxLayout()
        self.run_btn = QPushButton('Run statistics')
        self.run_btn.clicked.connect(self.start_stats)
        right.addWidget(self.run_btn)
        self.stats_status = _label('')
        right.addWidget(self.stats_status)
        self.loo_btn = QPushButton('Check hits: leave one out')
        self.loo_btn.setToolTip(LOO_TOOLTIP)
        self.loo_btn.setEnabled(False)
        self.loo_btn.clicked.connect(self.start_leave_one_out)
        right.addWidget(self.loo_btn)
        self.loo_status = _label('')
        right.addWidget(self.loo_status)
        right.addStretch()
        row.addLayout(right, stretch=1)
        lay.addLayout(row)

        self.chk_counts.toggled.connect(lambda v: self._on_stat_setting('run_counts', bool(v), refit=True))
        self.chk_freq.toggled.connect(lambda v: self._on_stat_setting('run_freq', bool(v), refit=True))
        self.cb_count_test.currentIndexChanged.connect(
            lambda _i: self._on_stat_setting('counts_method', self.cb_count_test.currentData(), refit=True))
        self.cb_norm.currentIndexChanged.connect(
            lambda _i: self._on_stat_setting('normalisation', self.cb_norm.currentData(), refit=True))
        self.list_controls.itemChanged.connect(self._on_controls_changed)
        self.chk_clones.toggled.connect(lambda v: self._on_stat_setting('remove_clones', bool(v), refit=True))
        self.sb_clone_count.valueChanged.connect(
            lambda v: self._on_stat_setting('clone_min_count', int(v), refit=True))
        self.sb_clone_pct.valueChanged.connect(
            lambda v: self._on_stat_setting('clone_min_pct', float(v), refit=True))
        self.sb_clone_levels.valueChanged.connect(
            lambda v: self._on_stat_setting('clone_max_levels', int(v), refit=True))
        self.cb_mode.currentIndexChanged.connect(
            lambda _i: self._on_stat_setting('contrast_mode', self.cb_mode.currentData(), refit=True))
        self.cb_reference.currentIndexChanged.connect(
            lambda _i: self._on_stat_setting('reference', self.cb_reference.currentData() or '', refit=True))
        self.cb_pairing.currentIndexChanged.connect(
            lambda _i: self._on_stat_setting(self._design_key('pairing'), self.cb_pairing.currentData() or '',
                                             refit=True))
        self.cb_split.currentIndexChanged.connect(
            lambda _i: self._on_stat_setting(self._design_key('split_by'), self.cb_split.currentData() or '',
                                             refit=True))
        self.cb_compare.currentIndexChanged.connect(
            lambda _i: self._on_stat_setting('compare', self.cb_compare.currentData(), refit=True))
        self.cb_ref_pop.currentIndexChanged.connect(
            lambda _i: self._on_stat_setting('reference_population', self.cb_ref_pop.currentData() or '',
                                             refit=True))
        self.chk_pop_contrasts.toggled.connect(
            lambda v: self._on_stat_setting('population_contrasts', bool(v), refit=True))
        self.chk_markers.toggled.connect(lambda v: self._on_stat_setting('run_markers', bool(v), refit=True))
        self.cb_marker_ref.currentIndexChanged.connect(
            lambda _i: self._on_stat_setting('marker_reference', self.cb_marker_ref.currentData(), refit=True))
        self.list_markers.itemChanged.connect(self._on_markers_changed)
        self.sb_marker_events.valueChanged.connect(
            lambda v: self._on_stat_setting('marker_min_events', int(v), refit=True))
        self.sb_marker_delta.valueChanged.connect(lambda v: self._on_stat_setting('marker_threshold', float(v)))
        self.chk_marker_spill.toggled.connect(
            lambda v: self._on_stat_setting('marker_spillover', bool(v), refit=True))
        self.chk_marker_k_auto.toggled.connect(
            lambda v: self._on_stat_setting('marker_spillover_k', 'auto' if v else float(self.sb_marker_spill.value())))
        self.sb_marker_spill.valueChanged.connect(
            lambda v: self._on_stat_setting('marker_spillover_k', float(v)))
        self.chk_pathways.toggled.connect(lambda v: self._on_stat_setting('run_pathways', bool(v), refit=True))
        self.sb_pathway_size.valueChanged.connect(
            lambda v: self._on_stat_setting('pathway_min_size', int(v), refit=True))
        self.sb_pathway_rotations.valueChanged.connect(
            lambda v: self._on_stat_setting('pathway_rotations', int(v), refit=True))
        self.cb_pathway_evidence.currentIndexChanged.connect(
            lambda _i: self._on_stat_setting('pathway_evidence', self.cb_pathway_evidence.currentData(), refit=True))
        self.chk_gene_groups.toggled.connect(self._on_gene_groups_ticked)
        self.list_adjust.itemChanged.connect(self._on_adjust_changed)
        self.sb_p.valueChanged.connect(lambda v: self._on_stat_setting('pval_threshold', float(v)))
        self.sb_fc.valueChanged.connect(lambda v: self._on_stat_setting('fc_threshold', float(v)))
        self.chk_treat.toggled.connect(lambda v: self._on_stat_setting('use_treat', bool(v)))
        self.cb_fdr.currentIndexChanged.connect(
            lambda _i: self._on_stat_setting('fdr_scope', self.cb_fdr.currentData()))

    def compare_mode(self) -> str:
        """'populations' when populations are compared (several counted),
        else 'groups'."""
        if self.settings.get('compare') == 'populations' and self.several_populations():
            return 'populations'
        return 'groups'

    def _design_key(self, key: str) -> str:
        """The setting holding the pairing or split of the current comparison
        kind (populations keep their own)."""
        if self.compare_mode() == 'populations':
            return {'pairing': 'population_pairing', 'split_by': 'population_split'}[key]
        return key

    def _marker_data(self):
        """The marker sums of the saved counts, or None."""
        saved = self.settings.get('counts') or {}
        key = saved.get('signature')
        if self._markers_key != key:
            self._markers_key = key
            self._markers = (fa.load_marker_sums(fa.marker_cache_path(self._controller().experiment_dir), key)
                             if key else None)
        return self._markers

    def marker_channels(self) -> list:
        """The markers ticked for the marker test (all of them by default)."""
        data = self._marker_data()
        if data is None:
            return []
        chosen = self.settings.get('marker_channels')
        return [c for c in data['channels'] if chosen is None or c in chosen]

    def _fill_stats_choices(self):
        s = self.settings
        self._filling = True
        try:
            self.chk_counts.setChecked(bool(s['run_counts']))
            self.chk_freq.setChecked(bool(s['run_freq']))
            self.chk_markers.setChecked(bool(s['run_markers']))
            soft = self.soft_counts()
            self.chk_counts.setEnabled(not soft)
            self.counts_note.setVisible(soft)
            self.cb_count_test.setCurrentIndex(max(self.cb_count_test.findData(s['counts_method']), 0))
            self.cb_norm.setCurrentIndex(max(self.cb_norm.findData(s['normalisation']), 0))
            self._fill_controls()
            self.chk_clones.setChecked(bool(s['remove_clones']))
            self.sb_clone_count.setValue(int(s['clone_min_count']))
            self.sb_clone_pct.setValue(float(s['clone_min_pct']))
            self.sb_clone_levels.setValue(int(s['clone_max_levels']))
            self._fill_markers()
            self.chk_pathways.setChecked(bool(s['run_pathways']))
            self.sb_pathway_size.setValue(int(s['pathway_min_size']))
            self.sb_pathway_rotations.setValue(int(s['pathway_rotations']))
            self.cb_pathway_evidence.setCurrentIndex(max(self.cb_pathway_evidence.findData(s['pathway_evidence']), 0))
            self.chk_gene_groups.setChecked(bool(s['run_gene_groups']))
            self._update_count_controls()
            pops = self.counted_populations()
            several = len(pops) > 1
            compare = self.compare_mode()
            self.cb_compare.setCurrentIndex(max(self.cb_compare.findData(compare), 0))
            self.cb_compare.setEnabled(several)
            self.cb_ref_pop.clear()
            for pop in pops:
                self.cb_ref_pop.addItem(pop, pop)
            if s.get('reference_population') not in pops:
                s['reference_population'] = pops[-1] if several else ''
            self.cb_ref_pop.setCurrentIndex(max(self.cb_ref_pop.findData(s['reference_population']), 0))
            self.cb_mode.setCurrentIndex(max(self.cb_mode.findData(s['contrast_mode']), 0))
            self.cb_reference.clear()
            for g in s.get('group_names') or []:
                self.cb_reference.addItem(g, g)
            if s.get('reference') not in (s.get('group_names') or []):
                s['reference'] = (s.get('group_names') or [''])[0]
            self.cb_reference.setCurrentIndex(max(self.cb_reference.findData(s['reference']), 0))
            self._update_design_controls()
            covs = list(s.get('covariates') or {})
            extra = {'pairing': [], 'split_by': []}
            if compare == 'populations':
                extra = {'pairing': [('Sample (its populations)', fa.PAIR_BY_SAMPLE)],
                         'split_by': [('Sample group', fa.SPLIT_BY_GROUP)]}
            if self.multi_library():
                extra['split_by'].append(('Library (section 2)', fa.SPLIT_BY_LIBRARY))
            for combo, key, none_text in ((self.cb_pairing, 'pairing', 'Not paired'),
                                          (self.cb_split, 'split_by', 'Not split')):
                combo.clear()
                combo.addItem(none_text, '')
                for text, data in extra[key]:
                    combo.addItem(text, data)
                for c in covs:
                    combo.addItem(c, c)
                combo.setCurrentIndex(max(combo.findData(s.get(self._design_key(key)) or ''), 0))
            self.list_adjust.clear()
            in_design = (s.get(self._design_key('pairing')), s.get(self._design_key('split_by')))
            for c in covs:
                it = QListWidgetItem(c)
                if c in in_design:
                    it.setFlags(Qt.ItemIsUserCheckable)
                    it.setCheckState(Qt.Unchecked)
                    it.setToolTip('Already in the design (Pair samples by or Split by).')
                else:
                    it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                    it.setCheckState(Qt.Checked if c in (s.get('adjust') or []) else Qt.Unchecked)
                self.list_adjust.addItem(it)
            self.sb_p.setValue(float(s['pval_threshold']))
            self.sb_fc.setValue(float(s['fc_threshold']))
            self.chk_treat.setChecked(bool(s['use_treat']))
            self.cb_fdr.setCurrentIndex(max(self.cb_fdr.findData(s['fdr_scope']), 0))
        finally:
            self._filling = False
        self._update_run_button()

    def _update_design_controls(self):
        """Enable the controls that apply to the current comparison kind."""
        s = self.settings
        several = self.several_populations()
        groups = self.compare_mode() == 'groups'
        reference = s['contrast_mode'] == 'reference'
        self.cb_reference.setEnabled(groups and reference)
        self.cb_ref_pop.setEnabled(several and reference and (not groups or bool(s['population_contrasts'])))
        self.chk_pop_contrasts.setChecked(bool(s['population_contrasts']))
        self.chk_pop_contrasts.setEnabled(several and groups and reference)

    def _fill_markers(self):
        """The marker list and reference choice, from the saved marker sums."""
        s = self.settings
        data = self._marker_data()
        channels = list(data['channels']) if data is not None else []
        chosen = s.get('marker_channels')
        labels = self.plugin.channel_labels()
        filling, self._filling = self._filling, True
        try:
            self.list_markers.clear()
            for c in channels:
                it = QListWidgetItem(labels.get(c, c))
                it.setData(Qt.UserRole, c)
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if chosen is None or c in chosen else Qt.Unchecked)
                self.list_markers.addItem(it)
            has_controls = bool([c for c in (s.get('control_combos') or [])
                                 if self.filtered is not None and c in self.filtered.counts.columns])
            item = self.cb_marker_ref.model().item(self.cb_marker_ref.findData('controls'))
            if item is not None:
                item.setEnabled(has_controls)
            ref = s['marker_reference']
            if ref == 'controls' and not has_controls:
                ref = 'all'
            self.cb_marker_ref.setCurrentIndex(max(self.cb_marker_ref.findData(ref), 0))
            self.sb_marker_events.setValue(int(s['marker_min_events']))
            self.sb_marker_delta.setValue(float(s['marker_threshold']))
            self.chk_marker_spill.setChecked(bool(s['marker_spillover']))
            auto = s['marker_spillover_k'] == 'auto'
            self.chk_marker_k_auto.setChecked(auto)
            self.sb_marker_spill.setValue(fst.SPILLOVER_K_NO_CONTROLS if auto else float(s['marker_spillover_k']))
        finally:
            self._filling = filling
        if self.counts is not None and data is None:
            self.lbl_markers.setText('No marker intensities with these counts: count again to collect them.')
        elif s['marker_reference'] == 'controls' and not has_controls:
            self.lbl_markers.setText('No control combinations ticked: markers are compared with all other '
                                     'barcoded cells.')
        else:
            self.lbl_markers.setText('')

    def _on_markers_changed(self, _item):
        if self._filling:
            return
        ticked = [self.list_markers.item(i).data(Qt.UserRole) for i in range(self.list_markers.count())
                  if self.list_markers.item(i).checkState() == Qt.Checked]
        self.settings['marker_channels'] = (None if len(ticked) == self.list_markers.count() else ticked)
        self._changed()
        self._invalidate_stats()

    def _fill_controls(self):
        """The control-combination list: the analysed combinations, ticked as
        saved, or the non-targeting ones the first time."""
        s = self.settings
        combos = list(self.filtered.counts.columns) if self.filtered is not None else []
        if s.get('control_combos') is None and combos:
            s['control_combos'] = fst.control_candidates(combos)
        filling, self._filling = self._filling, True
        try:
            self.list_controls.clear()
            for c in combos:
                it = QListWidgetItem(c)
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if c in (s.get('control_combos') or []) else Qt.Unchecked)
                self.list_controls.addItem(it)
        finally:
            self._filling = filling

    def _update_count_controls(self):
        s = self.settings
        control = s['normalisation'] == 'control'
        self.lbl_controls.setVisible(control)
        self.list_controls.setVisible(control)
        for w in (self.sb_clone_count, self.sb_clone_pct, self.sb_clone_levels):
            w.setEnabled(bool(s['remove_clones']))
        for w in self.marker_widgets:
            w.setEnabled(bool(s['run_markers']))
        spill = bool(s['run_markers']) and bool(s['marker_spillover'])
        self.chk_marker_k_auto.setEnabled(spill)
        self.sb_marker_spill.setEnabled(spill and s['marker_spillover_k'] != 'auto')
        blocked = self.pathways_blocked()
        either = bool(s['run_pathways'] or s['run_gene_groups']) and blocked is None
        self.sb_pathway_size.setEnabled(either)
        self.sb_pathway_rotations.setEnabled(either)
        self.cb_pathway_evidence.setEnabled(bool(s['run_pathways']) and blocked is None)
        self.lbl_pathways.setText(f'Pathways need {blocked}.' if s['run_pathways'] and blocked else '')
        self.lbl_gene_groups.setText(self.gene_group_status())

    def pathways_blocked(self) -> str | None:
        """What the pathway analysis is missing (the Counts test with the
        quasi-likelihood F test), or None."""
        if not self.run_counts():
            return 'the Counts test (whole events, Counts ticked)'
        if self.settings['counts_method'] != 'ql':
            return 'the quasi-likelihood F test (Counts test)'
        return None

    def pathway_matches(self, overrides: dict | None = None) -> pd.DataFrame:
        """The MGI gene of each analysed combination (``flowcode_pathways.match_features``)."""
        symbols, synonyms, _meta = fpw.bundled_names()
        combos = list(self.filtered.counts.columns) if self.filtered is not None else []
        names = self.settings.get('pathway_names') if overrides is None else overrides
        return fpw.match_features(combos, symbols, synonyms, names)

    def _on_pathway_names(self):
        """A table of the combinations, the gene each is matched to and how,
        with a column to set the gene by hand."""
        try:
            matches = self.pathway_matches(overrides={})
        except Exception as exc:
            QMessageBox.warning(self, 'Gene names', f'The bundled MGI gene names could not be read: {exc}')
            return
        overrides = dict(self.settings.get('pathway_names') or {})
        dialog = QDialog(self)
        dialog.setWindowTitle('Pathway gene names')
        lay = QVBoxLayout(dialog)
        lay.addWidget(_label('Each combination is matched to an MGI symbol (ignoring case), then to an MGI synonym '
                             'that belongs to one gene only. Type a gene symbol in the last column to set it by hand '
                             '(leave it empty to use the match). Non-targeting combinations are never in a pathway.'))
        table = QTableWidget(len(matches), 4)
        table.setHorizontalHeaderLabels(['Combination', 'Matched gene', 'How', 'Set by hand'])
        for r, (_i, row) in enumerate(matches.iterrows()):
            table.setItem(r, 0, _item(row['feature']))
            table.setItem(r, 1, _item(row['gene']))
            table.setItem(r, 2, _item(row['match']))
            table.setItem(r, 3, _item(overrides.get(str(row['feature']), ''), editable=True))
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(table)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        lay.addWidget(buttons)
        dialog.resize(620, 520)
        if dialog.exec() != QDialog.Accepted:
            return
        chosen = {table.item(r, 0).text(): table.item(r, 3).text().strip() for r in range(table.rowCount())}
        chosen = {k: v for k, v in chosen.items() if v}
        if chosen != overrides:
            self.settings['pathway_names'] = chosen
            self._changed()
            self._invalidate_stats()

    # -- gene groups ----------------------------------------------------------

    def gene_groups_blocked(self) -> str | None:
        """What the gene-group analysis is missing, or None."""
        if not self.settings.get('gene_group_table'):
            return 'a gene-group table (Gene groups…)'
        return self.pathways_blocked()

    def gene_group_frame(self) -> pd.DataFrame | None:
        """The loaded gene-group table (gene, group, colour, area), or None."""
        s = self.settings
        rows = s.get('gene_group_table') or []
        if not rows:
            return None
        areas = s.get('gene_group_areas') or {}
        return pd.DataFrame([(str(g), str(grp), '', str(areas.get(grp, ''))) for g, grp in rows],
                            columns=list(fgg.GROUP_COLUMNS))

    def gene_group_order(self) -> list:
        """The groups in the order chosen in the Gene groups dialog, then any
        others in table order."""
        table = self.gene_group_frame()
        if table is None:
            return []
        groups = fgg.group_order(table)
        chosen = [g for g in self.settings.get('gene_group_order') or [] if g in groups]
        return chosen + [g for g in groups if g not in chosen]

    def gene_group_colours(self) -> dict:
        """{group: colour}: set by hand or in the table, else ``fgg.group_colours``."""
        return fgg.group_colours(self.gene_group_order(), self.settings.get('gene_group_areas') or {},
                                 self.settings.get('gene_group_colours') or {})

    def library_guides(self) -> dict:
        """{library label: its combination names} with several libraries,
        else {'': the design's combinations}."""
        if self.multi_library():
            return {library_label(lib): [str(c[0]) for c in lib.get('combos') or []]
                    for lib in self.plugin.state.libraries}
        design = self._current_design()
        if design is not None:
            return {'': [str(n) for n in design.combo_names]}
        return {'': [str(c) for c in self.filtered.counts.columns] if self.filtered is not None else []}

    def gene_group_allocation(self) -> pd.DataFrame | None:
        """Guides per group from the library setup (``fgg.allocation``), or None."""
        table = self.gene_group_frame()
        if table is None:
            return None
        symbols, synonyms, _meta = fpw.bundled_names()
        return fgg.allocation(self.library_guides(), table, symbols, synonyms,
                              self.settings.get('pathway_names') or {}, self.gene_group_order())

    def gene_group_status(self) -> str:
        """The line under the gene-group controls."""
        s = self.settings
        rows = s.get('gene_group_table') or []
        text = (f"Table: {self.gene_group_source()} ({len(rows)} genes, "
                f"{len(dict.fromkeys(r[1] for r in rows))} groups)." if rows else 'No gene-group table loaded.')
        if s['run_gene_groups']:
            blocked = self.gene_groups_blocked()
            if blocked:
                text += f' Gene groups need {blocked}.'
        return text

    def gene_group_source(self, file_name: str | None = None, rows: list | None = None) -> str:
        """The table's file name, with the built-in table's name when it is
        one (and a note when the installed built-in table has changed since
        it was picked)."""
        s = self.settings
        file_name = s.get('gene_group_file') if file_name is None else file_name
        rows = s.get('gene_group_table') if rows is None else rows
        names = dict((f, n) for n, f in fgg.builtin_tables())
        table = (pd.DataFrame([(g, grp, '', '') for g, grp in rows], columns=list(fgg.GROUP_COLUMNS))
                 if rows else None)
        try:
            kind = fgg.builtin_of(file_name or '', table)
        except (fgg.GroupTableError, OSError, UnicodeDecodeError):
            kind = None
        if kind == 'same':
            return f'{names[file_name]} (built in)'
        if kind == 'older':
            return f'{names[file_name]} (built in, an earlier version)'
        return file_name or 'loaded'

    def best_builtin_table(self) -> tuple | None:
        """``fgg.best_builtin`` over the experiment's guides, or None."""
        names = list(dict.fromkeys(n for guides in self.library_guides().values() for n in guides))
        if not names:
            return None
        try:
            symbols, synonyms, _meta = fpw.bundled_names()
            return fgg.best_builtin(names, symbols, synonyms, self.settings.get('pathway_names') or {})
        except (fgg.GroupTableError, OSError, UnicodeDecodeError):
            return None

    def _on_gene_groups_ticked(self, value: bool):
        """Gene groups ticked with no table loaded: pick the built-in table
        that covers the experiment's guides (``fgg.best_builtin``)."""
        if value and not self._filling and not self.settings.get('gene_group_table'):
            best = self.best_builtin_table()
            if best is not None:
                self.apply_gene_groups(_gene_group_state(fgg.builtin_table(best[1]), best[1]))
        self._on_stat_setting('run_gene_groups', bool(value), refit=True)

    def _on_gene_groups(self):
        """Pick a built-in gene-group table or load one, see each
        combination's group and how it was matched, and order and colour the
        groups."""
        s = self.settings
        state = {'rows': [list(r) for r in s.get('gene_group_table') or []],
                 'file': s.get('gene_group_file') or '',
                 'areas': dict(s.get('gene_group_areas') or {}),
                 'colours': dict(s.get('gene_group_colours') or {}),
                 'order': self.gene_group_order()}
        if not state['rows']:
            best = self.best_builtin_table()
            if best is not None:
                state.update(_gene_group_state(fgg.builtin_table(best[1]), best[1]))
        dialog = QDialog(self)
        dialog.setWindowTitle('Gene groups')
        lay = QVBoxLayout(dialog)
        lay.addWidget(_label('Tick a built-in table, or load a CSV with a gene and a group column (optional colour, '
                             '#rrggbb, and area). Each combination is matched by its own name, then by MGI gene '
                             'symbol; a gene set by hand in Gene names… is used first. Double-click a colour to '
                             'change it; Up and Down set the order of the spokes and slices.'))
        builtin_row = QHBoxLayout()
        builtin_row.addWidget(QLabel('Built-in tables:'))
        builtin_group = QButtonGroup(dialog)
        for name, file_name in fgg.builtin_tables():
            radio = QRadioButton(name)
            radio.setProperty('table_file', file_name)
            radio.setToolTip(f'The lab\'s {name} gene groups ({file_name}, installed with the plugin).')
            builtin_group.addButton(radio)
            builtin_row.addWidget(radio)
        builtin_row.addStretch()
        lay.addLayout(builtin_row)
        top = QHBoxLayout()
        load_btn = QPushButton('Load CSV…')
        clear_btn = QPushButton('Remove table')
        file_lbl = _label('')
        top.addWidget(load_btn)
        top.addWidget(clear_btn)
        top.addWidget(file_lbl, stretch=1)
        lay.addLayout(top)
        groups_table = QTableWidget(0, 4)
        groups_table.setHorizontalHeaderLabels(['Group', 'Guides', 'Colour', 'Area'])
        groups_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        groups_table.setSelectionMode(QAbstractItemView.SingleSelection)
        move = QHBoxLayout()
        up_btn, down_btn = QPushButton('Up'), QPushButton('Down')
        move.addWidget(up_btn)
        move.addWidget(down_btn)
        move.addStretch()
        combos_table = QTableWidget(0, 4)
        combos_table.setHorizontalHeaderLabels(['Combination', 'Gene', 'Group', 'How'])
        lay.addWidget(groups_table, stretch=1)
        lay.addLayout(move)
        lay.addWidget(combos_table, stretch=2)

        def frame():
            if not state['rows']:
                return None
            return pd.DataFrame([(g, grp, '', state['areas'].get(grp, '')) for g, grp in state['rows']],
                                columns=list(fgg.GROUP_COLUMNS))

        def fill():
            table = frame()
            source = self.gene_group_source(state['file'], state['rows'])
            file_lbl.setText(f"{source}: {len(state['rows'])} genes, {len(state['order'])} groups"
                             if table is not None else 'No table loaded.')
            builtin_group.setExclusive(False)
            for radio in builtin_group.buttons():
                radio.setChecked(table is not None and radio.property('table_file') == state['file']
                                 and source.endswith('(built in)'))
            builtin_group.setExclusive(True)
            colours = fgg.group_colours(state['order'], state['areas'], state['colours'])
            counts = {}
            assignment = pd.DataFrame(columns=['feature', 'gene', 'group', 'match'])
            if table is not None:
                try:
                    symbols, synonyms, _meta = fpw.bundled_names()
                    alloc = fgg.allocation(self.library_guides(), table, symbols, synonyms,
                                           s.get('pathway_names') or {}, state['order'])
                    counts = dict(zip(alloc['group'], alloc['guides']))
                    names = list(dict.fromkeys(n for guides in self.library_guides().values() for n in guides))
                    assignment = fgg.assign_groups(names, table, symbols, synonyms, s.get('pathway_names') or {})
                except Exception as exc:
                    file_lbl.setText(f'The bundled MGI gene names could not be read: {exc}')
            groups_table.setRowCount(len(state['order']))
            for r, g in enumerate(state['order']):
                groups_table.setItem(r, 0, _item(g))
                groups_table.setItem(r, 1, _item(str(int(counts.get(g, 0)))))
                swatch = _item(colours.get(g, ''))
                swatch.setBackground(QColor(colours.get(g, '#ffffff')))
                groups_table.setItem(r, 2, swatch)
                groups_table.setItem(r, 3, _item(state['areas'].get(g, '')))
            groups_table.resizeColumnsToContents()
            combos_table.setRowCount(len(assignment))
            for r, (_i, row) in enumerate(assignment.iterrows()):
                for j, c in enumerate(('feature', 'gene', 'group', 'match')):
                    combos_table.setItem(r, j, _item(row[c]))
            combos_table.resizeColumnsToContents()

        def load():
            path, _f = QFileDialog.getOpenFileName(dialog, 'Gene-group table', str(self.analysis_root()),
                                                   'CSV files (*.csv);;All files (*)')
            if not path:
                return
            try:
                table = fgg.read_group_table(path)
            except (fgg.GroupTableError, OSError, UnicodeDecodeError) as exc:
                QMessageBox.warning(dialog, 'Gene groups', str(exc))
                return
            state.update(_gene_group_state(table, Path(path).name))
            fill()

        def pick_builtin(radio):
            file_name = str(radio.property('table_file'))
            try:
                table = fgg.builtin_table(file_name)
            except (fgg.GroupTableError, OSError, UnicodeDecodeError) as exc:
                QMessageBox.warning(dialog, 'Gene groups', str(exc))
                fill()
                return
            state.update(_gene_group_state(table, file_name))
            fill()

        def clear():
            state.update(rows=[], file='', areas={}, colours={}, order=[])
            fill()

        def shift(step):
            r = groups_table.currentRow()
            if r < 0 or not 0 <= r + step < len(state['order']):
                return
            order = state['order']
            order[r], order[r + step] = order[r + step], order[r]
            fill()
            groups_table.selectRow(r + step)

        def pick_colour(row, col):
            if col != 2 or not 0 <= row < len(state['order']):
                return
            group = state['order'][row]
            current = fgg.group_colours(state['order'], state['areas'], state['colours']).get(group, '#999999')
            colour = QColorDialog.getColor(QColor(current), dialog, f'Colour of {group}')
            if colour.isValid():
                state['colours'][group] = colour.name()
                fill()

        load_btn.clicked.connect(load)
        builtin_group.buttonClicked.connect(pick_builtin)
        clear_btn.clicked.connect(clear)
        up_btn.clicked.connect(lambda: shift(-1))
        down_btn.clicked.connect(lambda: shift(1))
        groups_table.cellDoubleClicked.connect(pick_colour)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        lay.addWidget(buttons)
        fill()
        dialog.resize(720, 640)
        self._gene_group_dialog = dialog
        if dialog.exec() != QDialog.Accepted:
            return
        self.apply_gene_groups(state)

    def apply_gene_groups(self, state: dict):
        """Keep the gene groups chosen in the dialog: a new table (genes or
        groups) invalidates the statistics; order and colours only redraw."""
        s = self.settings
        rows_changed = [list(r) for r in state['rows']] != [list(r) for r in s.get('gene_group_table') or []]
        s['gene_group_table'] = [list(r) for r in state['rows']]
        s['gene_group_file'] = state['file']
        s['gene_group_areas'] = dict(state['areas'])
        s['gene_group_colours'] = {g: c for g, c in state['colours'].items() if g in state['order']}
        s['gene_group_order'] = list(state['order'])
        self._changed()
        self._update_count_controls()
        if rows_changed and self.stats is not None and s['run_gene_groups']:
            self._invalidate_stats()
        else:
            self._show_results()

    def _on_spider_comparisons(self):
        """Choose the comparisons drawn on the gene group spider."""
        result = (self.stats or {}).get('gene_groups')
        if not result:
            QMessageBox.information(self, 'Spider comparisons', 'Run statistics with Gene groups ticked first.')
            return
        comparisons = list(self.stats['comparisons'])
        shown = set(self.spider_comparisons())
        dialog = QDialog(self)
        dialog.setWindowTitle('Spider comparisons')
        lay = QVBoxLayout(dialog)
        lay.addWidget(_label(f'Tick the comparisons to draw (at most {fgg.SPIDER_COMPARISONS}; the first ones ticked '
                             'are drawn). Untick all to go back to those against the reference group.'))
        listing = QListWidget()
        for c in comparisons:
            item = QListWidgetItem(c)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if c in shown else Qt.Unchecked)
            listing.addItem(item)
        lay.addWidget(listing)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        lay.addWidget(buttons)
        self._spider_dialog = dialog
        if dialog.exec() != QDialog.Accepted:
            return
        chosen = [listing.item(i).text() for i in range(listing.count())
                  if listing.item(i).checkState() == Qt.Checked]
        self.set_spider_comparisons(chosen)

    def set_spider_comparisons(self, chosen: list):
        """Keep *chosen* as the spider's comparisons (empty: the default) and
        redraw the figures over every comparison."""
        self.settings['gene_group_comparisons'] = list(chosen)
        self._changed()
        if self.stats is not None:
            self._shown_figure = self.figure_tabs.tabText(self.figure_tabs.currentIndex())
            self._clear_figure_tabs(tuple(k for k in self.figures if _figure_set(k) != 'summary'))
            self.start_figures(summary=True, comparison=False, markers=False)

    def spider_comparisons(self) -> list:
        """The comparisons drawn on the spider (``fgg.spider_comparisons``)."""
        if self.stats is None:
            return []
        return fgg.spider_comparisons(list(self.stats['comparisons']), self.settings.get('gene_group_comparisons'),
                                      self.stats.get('pairs'), self.settings.get('reference'))

    def _on_controls_changed(self, _item):
        if self._filling:
            return
        self.settings['control_combos'] = [
            self.list_controls.item(i).text() for i in range(self.list_controls.count())
            if self.list_controls.item(i).checkState() == Qt.Checked]
        self._changed()
        self._fill_markers()
        self._invalidate_stats()

    def _on_adjust_changed(self, _item):
        if self._filling:
            return
        self.settings['adjust'] = [self.list_adjust.item(i).text() for i in range(self.list_adjust.count())
                                   if self.list_adjust.item(i).checkState() == Qt.Checked
                                   and self.list_adjust.item(i).flags() & Qt.ItemIsEnabled]
        self._changed()
        self._invalidate_stats()

    def _on_stat_setting(self, key, value, refit: bool = False):
        if self._filling:
            return
        self.settings[key] = value
        self._changed()
        if key in ('contrast_mode', 'population_contrasts'):
            self._update_design_controls()
        if key in ('normalisation', 'remove_clones', 'run_markers', 'marker_spillover', 'marker_spillover_k',
                   'run_counts', 'counts_method', 'run_pathways', 'run_gene_groups'):
            self._update_count_controls()
        if key in ('pairing', 'split_by', 'population_pairing', 'population_split', 'compare'):
            s = self.settings
            in_design = (s.get(self._design_key('pairing')), s.get(self._design_key('split_by')))
            s['adjust'] = [c for c in s.get('adjust') or [] if c not in in_design]
            self._fill_stats_choices()
        if refit:
            self._invalidate_stats()
        else:
            self._reapply_thresholds()
        self._update_run_button()

    def _threshold_settings(self) -> dict:
        """The settings ``_thresholded`` applies."""
        return {k: self.settings.get(k) for k in THRESHOLD_KEYS + ('marker_spillover',)}

    @staticmethod
    def _thresholded(stats: dict, s: dict) -> dict:
        """*stats* with the thresholds in *s* applied (no refit)."""
        stats = fst.reapply_thresholds(stats, s['pval_threshold'], s['fc_threshold'], s['fdr_scope'], s['use_treat'])
        if stats.get('markers') is not None:
            stats['markers'] = fst.reapply_marker_thresholds(
                stats['markers'], s['pval_threshold'], s['marker_threshold'], s['fdr_scope'],
                s['use_treat'], s['marker_spillover_k'] if s['marker_spillover'] else None)
        return stats

    def _reapply_thresholds(self):
        """Apply the current thresholds to the stored results and redraw."""
        if self.stats is None:
            return
        self.stats = self._thresholded(self.stats, self._threshold_settings())
        self._after_thresholds()

    def _after_thresholds(self):
        """Show thresholded results: the marker k, the leave-one-out line and the figures."""
        if self.stats.get('markers') is not None:
            self._show_marker_k()
        if not self.plugin.task_running('analysis_loo'):
            self.loo_status.setText(self.leave_one_out_text())
        self._update_loo_button()
        self._show_results()

    def marker_k_text(self) -> str:
        """The spillover k used by the marker results, per population, e.g.
        'k = 0.75 (calibrated on 4 control combinations, CD8)'; '' without
        spillover inflation."""
        markers = (self.stats or {}).get('markers') or {}
        k_map = markers.get('spillover_k')
        if not k_map:
            return ''
        controls = markers.get('spillover_controls') or {}
        parts = []
        for pop, k in k_map.items():
            name = pop or self.counted_population()
            if markers.get('spillover_k_auto'):
                parts.append(f'k = {k:g} (calibrated on {controls.get(pop, 0)} control combinations, {name})')
            else:
                parts.append(f'k = {k:g} ({name})')
        return '; '.join(parts)

    def _show_marker_k(self):
        self.lbl_marker_k.setText(self.marker_k_text())

    def _update_run_button(self):
        s = self.settings
        problem = None
        unassigned = sum(lib != OTHER_LIBRARY for lib in self._left_out.values())
        if unassigned:
            problem = (f'{unassigned} counted sample(s) have no library: assign them in section 2 (Tags & Combos), '
                       'or untick them (Data).')
        elif self.filtered is None or self.filtered.counts.empty:
            problem = 'Count events first (Data).'
        elif not (self.run_counts() or s['run_freq'] or s['run_markers']):
            problem = ('Tick Frequencies or Markers: the counts include a phenotype population, so the Counts '
                       'test cannot run.' if s['run_counts'] else 'Tick at least one test.')
        elif self.compare_mode() == 'groups' and len(s.get('group_names') or []) < 2:
            problem = 'Assign at least two groups (Groups).'
        elif s['run_markers'] and not (self.run_counts() or s['run_freq']) and self._marker_data() is None:
            problem = 'No marker intensities with these counts: count again.'
        self.run_btn.setEnabled(problem is None and not self.plugin.task_running('analysis_stats'))
        if problem and self.stats is None:
            self.stats_status.setText(problem)
        self._update_loo_button()

    def _update_loo_button(self):
        """The leave-one-out check is offered for current (not out of date)
        results with at least one significant counts or frequency result."""
        stats = self.stats or {}
        hits = any(stats.get(k) is not None and bool(stats[k]['significant'].astype(bool).any())
                   for k in ('counts', 'freq'))
        busy = self.plugin.task_running('analysis_stats') or self.plugin.task_running('analysis_loo')
        self.loo_btn.setEnabled(hits and not self._stats_stale and self.filtered is not None and not busy)

    def _invalidate_stats(self):
        if self.stats is not None:
            self.stats = None
            self.figures = {}
            self._set_stats_stale(False)
            self.stats_status.setText('Settings changed: run statistics again.')
            self.loo_status.setText('')
            self._show_results()
        self._show_marker_k()
        self._update_run_button()

    def _stats_inputs(self) -> str:
        """Digest of what a statistics run depends on: the counts (by their
        signature), the library setup and the Analysis settings other than
        ``STATS_INPUT_EXCLUDED``."""
        s = self.settings
        inputs = {k: v for k, v in s.items() if k not in STATS_INPUT_EXCLUDED}
        inputs['counts_signature'] = (s.get('counts') or {}).get('signature')
        inputs['library_setup'] = self._library_setup_key()
        text = json.dumps(inputs, sort_keys=True, default=str)
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _set_stats_stale(self, stale: bool):
        self._stats_stale = bool(stale)
        self.stale_label.setVisible(self._stats_stale)

    def _restore_saved_stats(self):
        """Once per experiment, when the page is first refreshed with counts:
        the last statistics run (saved when it was made) with the current
        thresholds applied, marked out of date when the counts or settings
        it was run with have changed since."""
        if self._stats_checked:
            return
        self._stats_checked = True
        if self.stats is not None or self.filtered is None:
            return
        path = fst.stats_path(self._controller().experiment_dir)
        settings = self._threshold_settings()
        thresholded = self._thresholded

        def work(_progress):
            saved = fst.load_stats(path)
            if saved is None:
                return None
            stats, inputs, when = saved
            return {'stats': thresholded(stats, settings), 'inputs': inputs, 'when': when, 'settings': settings}

        if self.plugin._start_task('analysis_restore', work):
            self.stats_status.setText(RESTORING_TEXT)
        else:
            self._stats_checked = False

    def on_restore_finished(self, result):
        """The saved run, read and thresholded on a worker, is shown unless
        statistics were run meanwhile; it is marked out of date when the
        counts or settings it was run with have changed since."""
        if self.stats_status.text() == RESTORING_TEXT:
            self.stats_status.setText('')
        if result is None or self.stats is not None or self.filtered is None:
            return
        stats = result['stats']
        self.stats = stats
        self._set_stats_stale(result['inputs'] != self._stats_inputs())
        n_comparisons = len(stats.get('comparisons') or [])
        text = (f"Saved results ({result['when']}): {n_comparisons} comparison(s), "
                f"{len(stats.get('samples') or [])} sample(s).")
        if self._stats_stale:
            text += f'<br><span style="{_WARN_STYLE}">{STALE_TEXT}</span>'
        self.stats_status.setText(text)
        self._show_marker_k()
        if result['settings'] != self._threshold_settings():
            self._reapply_thresholds()
        else:
            self._after_thresholds()

    def on_restore_error(self, message):
        logger.warning('FlowCode analysis: could not read the saved statistics: %s', message)
        if self.stats_status.text() == RESTORING_TEXT:
            self.stats_status.setText('')
        self._update_loo_button()

    def _covariate_frame(self, names: list) -> pd.DataFrame | None:
        covs = self.settings.get('covariates') or {}
        names = [n for n in names if n in covs]
        if not names:
            return None
        units = list(self.filtered.counts.index)
        return pd.DataFrame({n: [covs[n].get(self.unit_sample(u), '') for u in units] for n in names}, index=units)

    def _unit_factor(self, key: str, pairing: bool = False) -> dict | None:
        """{counts row: value} of a pairing or split choice: a covariate (the
        row's sample's value), the sample itself, the sample's group or its
        library. With several libraries a covariate used for *pairing* is
        qualified by library (``<library>·<mouse>``), so that mice of
        different libraries never share a level."""
        if not key:
            return None
        units = list(self.filtered.counts.index)
        if key == fa.PAIR_BY_SAMPLE:
            return {u: self.unit_sample(u) for u in units}
        if key == fa.SPLIT_BY_GROUP:
            groups = self.settings.get('sample_groups') or {}
            return {u: groups.get(self.unit_sample(u), '') for u in units}
        if key == fa.SPLIT_BY_LIBRARY:
            if not self.multi_library():
                return None
            labels = self.library_labels()
            return {u: labels.get(self.library_of(self.unit_sample(u)), '') for u in units}
        values = (self.settings.get('covariates') or {}).get(key) or {}
        out = {u: values.get(self.unit_sample(u), '') for u in units}
        if pairing and self.multi_library():
            out = {u: f'{self.library_of(self.unit_sample(u))}·{v}' if v else '' for u, v in out.items()}
        return out

    def _stats_call(self) -> dict:
        """
        The arguments of a statistics run from the current settings:
        {'counts', 'denominator', 'groups', 'kwargs'} for
        ``fst.run_statistics``, with the 'pairing', 'split', 'split_key',
        'sample_groups', 'population', 'several' and 'unit_library' they
        were built from. Raises RuntimeError when adjustment covariate values
        are missing.
        """
        s = self.settings
        units = list(self.filtered.counts.index)
        several = self.several_populations()
        pops = self.counted_populations()
        population = {u: self.unit_population(u) for u in units}
        sample_groups = {u: (s.get('sample_groups') or {}).get(self.unit_sample(u), '') for u in units}
        pair_key, split_key = s.get(self._design_key('pairing')), s.get(self._design_key('split_by'))
        pairing, split = self._unit_factor(pair_key, pairing=True), self._unit_factor(split_key)
        adjust = [c for c in s.get('adjust') or [] if c not in (pair_key, split_key)]
        unit_library = {u: self.library_of(self.unit_sample(u)) for u in units} if self.multi_library() else None
        if unit_library is not None:
            design_libraries = dict(libraries=unit_library, feature_libraries=self.feature_libraries(),
                                    library_labels=self.library_labels())
        else:
            design_libraries = {}
        if self.compare_mode() == 'populations':
            groups, order, reference = dict(population), pops, s.get('reference_population') or ''
            design = dict(strata=population, within_strata=False, clone_groups=sample_groups, clone_reference=None)
        else:
            groups, order, reference = sample_groups, list(s.get('group_names') or []), s.get('reference') or ''
            design = dict(strata=population if several else None,
                          population_contrasts=(s.get('reference_population') or None)
                          if several and s['population_contrasts'] else None)
        grouped = [u for u in units if groups.get(u)]
        covariates = self._covariate_frame(adjust)
        if covariates is not None:
            missing = ds.missing_covariate_values(covariates, grouped, list(covariates.columns))
            if missing:
                labels = self.unit_labels()
                raise RuntimeError('Covariate values are missing: '
                                   + '; '.join(f"{c}: {', '.join(labels.get(k, k) for k in ks[:5])}"
                                               for c, ks in missing.items()))
        kwargs = dict(group_order=order, contrast_mode=s['contrast_mode'],
                      reference=reference, pairing=pairing, covariates=covariates, split_by=split,
                      pval_threshold=float(s['pval_threshold']), fc_threshold=float(s['fc_threshold']),
                      use_treat=bool(s['use_treat']), fdr_scope=s['fdr_scope'],
                      run_counts=self.run_counts(), run_freq=bool(s['run_freq']),
                      counts_method=s['counts_method'], normalisation=s['normalisation'],
                      control_combos=list(s.get('control_combos') or []),
                      clone_filter=({'min_count': int(s['clone_min_count']),
                                     'min_pct': float(s['clone_min_pct']),
                                     'max_levels': int(s['clone_max_levels'])}
                                    if s['remove_clones'] else None),
                      allow_no_comparisons=bool(s['run_markers']), **design, **design_libraries)
        return dict(counts=self.filtered.counts, denominator=self.filtered.denominator, groups=groups,
                    kwargs=kwargs, pairing=pairing, split=split, split_key=split_key,
                    sample_groups=sample_groups, population=population, several=several,
                    unit_library=unit_library)

    def start_stats(self):
        if self.filtered is None:
            return
        s = self.settings
        try:
            call = self._stats_call()
        except RuntimeError as exc:
            self.stats_status.setText(str(exc))
            return
        counts, denom, groups, kwargs = call['counts'], call['denominator'], call['groups'], call['kwargs']
        split, split_key, sample_groups = call['split'], call['split_key'], call['sample_groups']
        population, several, unit_library = call['population'], call['several'], call['unit_library']
        run_counts = kwargs['run_counts']
        markers_only = bool(s['run_markers']) and not (run_counts or s['run_freq'])
        marker_job = None
        if s['run_markers']:
            marker_split = split if split_key not in ('', None, fa.SPLIT_BY_GROUP) else None
            tag_matrix, spread, spill_note = self._marker_tag_inputs(bool(s['marker_spillover']))
            marker_job = dict(tag_matrix=tag_matrix, spread=spread, spill_note=spill_note,
                              spillover_k=s['marker_spillover_k'] if s['marker_spillover'] else None,
                              transforms=self._marker_transforms(),
                              path=fa.marker_cache_path(self._controller().experiment_dir),
                              signature=(s.get('counts') or {}).get('signature'), channels=self.marker_channels(),
                              reference=s['marker_reference'], controls=list(s.get('control_combos') or []),
                              min_events=int(s['marker_min_events']), groups=sample_groups,
                              group_order=list(s.get('group_names') or []),
                              strata=population if several else None, split_by=marker_split,
                              threshold=float(s['marker_threshold']), clones=bool(s['remove_clones']),
                              unit_library=unit_library, feature_map=self.feature_map())
        pathway_job, pathway_note = None, None
        if s['run_pathways']:
            blocked = self.pathways_blocked()
            if blocked is None and not markers_only:
                pathway_job = dict(evidence=s['pathway_evidence'], overrides=dict(s.get('pathway_names') or {}),
                                   rotations=int(s['pathway_rotations']), min_size=int(s['pathway_min_size']))
            else:
                pathway_note = f"Pathways: not run; they need {blocked or 'the Counts test'}."
        group_job, group_note = None, None
        if s['run_gene_groups']:
            blocked = self.gene_groups_blocked()
            if blocked is None and not markers_only:
                group_job = dict(table=self.gene_group_frame(), overrides=dict(s.get('pathway_names') or {}),
                                 rotations=int(s['pathway_rotations']), min_size=int(s['pathway_min_size']),
                                 library_guides=self.library_guides())
            else:
                group_note = f"Gene groups: not run; they need {blocked or 'the Counts test'}."

        inputs, path = self._stats_inputs(), fst.stats_path(self._controller().experiment_dir)

        def work(_progress):
            if markers_only:
                kwargs.update(run_counts=False, run_freq=False)
            fit_log = [] if pathway_job is not None or group_job is not None else None
            stats = fst.run_statistics(counts, denom, groups, fit_log=fit_log, **kwargs)
            if marker_job is not None:
                stats['markers'] = self._run_markers(stats, counts, marker_job, kwargs)
            if pathway_job is not None:
                stats['pathways'] = self._run_pathways(stats, fit_log, pathway_job)
            elif pathway_note:
                stats['notes'].append(pathway_note)
            if group_job is not None:
                stats['gene_groups'] = self._run_gene_groups(stats, fit_log, group_job)
            elif group_note:
                stats['notes'].append(group_note)
            try:
                fst.save_stats(path, stats, inputs)
            except Exception as exc:
                logger.exception('FlowCode analysis: could not save the statistics to %s', path)
                stats['notes'].append(f'The results could not be saved: {exc}')
            return stats

        if self.plugin._start_task('analysis_stats', work):
            self.run_btn.setEnabled(False)
            self.loo_btn.setEnabled(False)
            self.stats_status.setText('Running statistics…')

    def _marker_tag_inputs(self, spillover: bool) -> tuple:
        """(tag matrix, spillover spread, note) for the marker test, on the
        main thread: the combinations' tags from the design, and with
        *spillover* the spread of each combination's tags into every
        fluorophore (``flowcode_core.spillover_spread``) from the saved
        characterisation and the experiment's noise model. A part that is
        not available is None, and the note says why."""
        import flowcode_core as fcc

        design = self._current_design()
        if design is None:
            return None, None, 'the FlowCode tags are not set, so the tag model is not available.'
        tag_matrix = pd.DataFrame([[1.0 if t in tags else 0.0 for t in design.tags] for tags in design.combo_tags],
                                  index=list(design.combo_names), columns=list(design.tags))
        if not spillover:
            return tag_matrix, None, None
        fc = self.plugin.saved_spectra()
        if fc is None:
            return tag_matrix, None, 'no saved characterisation (section 4), so the tag spillover is not accounted for.'
        try:
            from honeychrome.controller_components import gls_unmix
            c = self._controller()
            noise = gls_unmix.load_noise_state(c.experiment.process, c.experiment_dir, list(fc.detectors))
            if noise is None:
                return tag_matrix, None, ('no noise model fitted (AutoSpectral tab, Noise page), so the '
                                          'tag spillover is not accounted for.')
            fc = fcc.rename_combos(fc, design)
            spread = fcc.spillover_spread(fc, noise.model.g)
        except Exception as exc:
            return tag_matrix, None, f'the tag spillover could not be computed ({exc}).'
        return tag_matrix, pd.DataFrame(spread['values'], index=spread['rows'], columns=spread['columns']), None

    def _marker_transforms(self) -> dict:
        """Copies of the experiment's unmixed transforms (display scale)."""
        from copy import deepcopy
        return deepcopy(dict(getattr(self._controller(), 'unmixed_transformations', None) or {}))

    @staticmethod
    def _run_markers(stats: dict, counts: pd.DataFrame, job: dict, kwargs: dict) -> dict | None:
        """The marker test of a statistics run (worker thread); notes go into
        *stats*."""
        data = fa.load_marker_sums(job['path'], job['signature'])
        if data is None or not job['channels']:
            stats['notes'].append('Markers: no marker intensities with these counts; count again.')
            return None
        removed = stats['removed'] if job['clones'] else None
        if job.get('feature_map') is not None:
            try:
                deltas, meta, used, means, reference_means, spillover = fst.marker_deltas_by_library(
                    data, counts, job['channels'], job['unit_library'], job['feature_map'], job['reference'],
                    job['controls'], job['min_events'], removed=removed, tag_matrix=job['tag_matrix'],
                    spread=job['spread'], transforms=job['transforms'])
            except RuntimeError as exc:
                stats['notes'].append(f'Markers: {exc}')
                return None
        else:
            sums = fa.marker_sums_for(data, list(counts.index), list(counts.columns), job['channels'])
            deltas, meta, used, means, reference_means = fst.marker_deltas(
                sums, counts, job['channels'], job['reference'], job['controls'], job['min_events'],
                removed=removed, return_means=True, tag_matrix=job['tag_matrix'])
            spillover = None
            if job['spread'] is not None:
                spillover = fst.marker_spillover(job['spread'], sums, counts, job['channels'], job['transforms'])
        if job['spill_note']:
            stats['notes'].append(f"Markers: {job['spill_note']}")
        try:
            markers = fst.marker_statistics(deltas, meta, job['groups'], group_order=job['group_order'],
                                            strata=job['strata'], split_by=job['split_by'],
                                            threshold=job['threshold'], pval_threshold=kwargs['pval_threshold'],
                                            fdr_scope=kwargs['fdr_scope'], use_treat=kwargs['use_treat'],
                                            spillover=spillover, spillover_k=job['spillover_k'],
                                            control_combos=job['controls'])
        except RuntimeError as exc:
            stats['notes'].append(f'Markers: {exc}')
            return None
        markers.update(reference=used, channels=list(job['channels']), min_events=job['min_events'],
                       means=means, reference_means=reference_means)
        if markers.get('spillover_k'):
            res = markers['results']
            stats['notes'].append(f"Markers: {int(res['spillover_dominant'].sum())} of {len(res)} features have an "
                                  "uncertainty that is mostly tag spillover (hatched).")
        stats['notes'].extend(f'Markers: {n}' if not n.startswith('Markers') else n for n in markers['notes'])
        return markers

    @staticmethod
    def _run_pathways(stats: dict, fit_log: list, job: dict) -> dict | None:
        """The pathway analysis of a statistics run (worker thread), from the
        fits the run recorded; notes go into *stats*."""
        try:
            result = fpw.run_from_bundle(stats, fit_log, **job)
        except (fpw.PathwayError, OSError, ValueError) as exc:
            stats['notes'].append(f'Pathways: {exc}')
            return None
        stats['notes'].extend(f'Pathways: {n}' for n in result['notes'])
        return result

    def pathway_summary(self) -> str:
        """e.g. 'Pathways: 412 tested in 2 comparison(s); 5 with competitive FDR <= 0.05, 9 self-contained.'"""
        result = (self.stats or {}).get('pathways')
        if not result:
            return ''
        table = result['pathways']
        t = float(self.settings['pval_threshold'])
        return (f" Pathways: {table['set_key'].nunique()} tested in {table['comparison'].nunique()} comparison(s); "
                f"{int((table['competitive_FDR'] <= t).sum())} with competitive FDR <= {t:g}, "
                f"{int((table['FDR'] <= t).sum())} self-contained.")

    @staticmethod
    def _run_gene_groups(stats: dict, fit_log: list, job: dict) -> dict | None:
        """The gene-group analysis of a statistics run (worker thread), from
        the fits the run recorded; notes go into *stats*."""
        try:
            result = fgg.run_gene_groups(stats, fit_log, **job)
        except (fpw.PathwayError, OSError, ValueError) as exc:
            stats['notes'].append(f'Gene groups: {exc}')
            return None
        stats['notes'].extend(f'Gene groups: {n}' for n in result['notes'])
        return result

    def gene_group_summary(self) -> str:
        """e.g. 'Gene groups: 7 tested in 3 comparison(s); 2 with competitive FDR <= 0.05, 4 any change.'"""
        result = (self.stats or {}).get('gene_groups')
        if not result:
            return ''
        table = result['pathways']
        t = float(self.settings['pval_threshold'])
        return (f" Gene groups: {table['set_key'].nunique()} tested in {table['comparison'].nunique()} "
                f"comparison(s); {int((table['competitive_FDR'] <= t).sum())} with competitive FDR <= {t:g}, "
                f"{int((table['mixed_FDR'] <= t).sum())} any change.")

    def on_stats_finished(self, stats):
        self.stats = stats
        self._set_stats_stale(False)
        self._show_marker_k()
        notes = list(self.filtered.notes if self.filtered is not None else []) + list(stats.get('notes') or [])
        key, res = fst.primary_results(stats)
        n_sig = int(res['significant'].sum()) if res is not None else 0
        text = (f"{len(stats['comparisons'])} comparison(s), {len(stats['samples'])} sample(s); "
                f"{n_sig} significant result(s) ({fst.TEST_NAMES.get(key, '')}).")
        markers = stats.get('markers')
        if markers is not None:
            text += (f" Markers: {int(markers['results']['significant'].sum())} significant combination-marker "
                     f"difference(s) in {len(markers['comparisons'])} group(s).")
        text += self.pathway_summary() + self.gene_group_summary()
        self.stats_status.setText(text + ('<br>' + '<br>'.join(notes) if notes else ''))
        self.loo_status.setText('')
        self._update_run_button()
        self._show_results()
        self.section_tabs.setCurrentIndex(3)

    def on_stats_error(self, message):
        self.stats_status.setText(f'Statistics failed: {message}')
        self._update_run_button()

    def start_leave_one_out(self):
        """Run the statistics again without each mouse in turn (each sample
        when unpaired), with the arguments of the current run, on a worker
        thread (``fst.leave_one_out``); the result is attached to the current
        statistics when it finishes."""
        if self.stats is None or self.filtered is None:
            return
        try:
            call = self._stats_call()
        except RuntimeError as exc:
            self.loo_status.setText(str(exc))
            return
        pair_key = self.settings.get(self._design_key('pairing'))
        if call['pairing']:
            levels = dict(call['pairing'])
            by_sample = pair_key == fa.PAIR_BY_SAMPLE
            noun = 'sample' if by_sample else str(pair_key)
        else:
            levels = {u: self.unit_sample(u) for u in call['counts'].index}
            by_sample, noun = True, 'sample'
        labels = {v: self._name(v) for v in set(levels.values()) if v} if by_sample else {}
        counts, denom, groups, kwargs = call['counts'], call['denominator'], call['groups'], call['kwargs']
        inputs = self._stats_inputs()

        def work(progress):
            loo = fst.leave_one_out(counts, denom, groups, levels, kwargs, progress=progress)
            loo.update(noun=noun, labels=labels)
            return loo

        if self.plugin._start_task('analysis_loo', work):
            self._loo_inputs, self._loo_labels = inputs, labels
            self.loo_btn.setEnabled(False)
            self.loo_status.setText(f'Leave one out: running the statistics again without each {noun}.')

    def leave_one_out_text(self) -> str:
        """Status line of the leave-one-out check of the current results;
        '' without one."""
        loo = (self.stats or {}).get('leave_one_out')
        if not loo:
            return ''
        labels = loo.get('labels') or {}
        summary = loo.get('summary')
        text = (f"Leave one out: {len(loo.get('refits') or {})} refit(s), one without each "
                f"{loo.get('noun') or 'mouse'}")
        if summary is None or summary.empty:
            text += '; no significant results to check.'
        else:
            flagged = int((summary['loo_flag'] != '').sum())
            text += (f'; {len(summary) - flagged} of {len(summary)} significant result(s) held in every refit, '
                     f'{flagged} flagged ⚠ (results table).')
        failed = loo.get('failed') or {}
        if failed:
            text += '<br>Could not refit without ' + '; '.join(f'{labels.get(k, k)} ({v})'
                                                               for k, v in failed.items())
        return text

    def on_loo_finished(self, loo):
        if self.stats is None or self._loo_inputs != self._stats_inputs():
            self.loo_status.setText('Leave one out: discarded, as the counts or settings changed while it ran.')
        else:
            self.stats['leave_one_out'] = loo
            self._reapply_thresholds()
            path = fst.stats_path(self._controller().experiment_dir)
            try:
                fst.save_stats(path, self.stats, self._loo_inputs)
            except Exception:
                logger.exception('FlowCode analysis: could not save the leave-one-out check to %s', path)
            self.loo_status.setText(self.leave_one_out_text())
        self._loo_inputs = None
        self._update_loo_button()

    def on_loo_error(self, message):
        self._loo_inputs = None
        self.loo_status.setText(f'Leave one out failed: {message}')
        self._update_loo_button()

    # =====================================================================
    # Results
    # =====================================================================

    def _build_results(self, lay):
        self.figure_tabs = fvs.ViewStack(lambda widget: _figure_section(getattr(widget, 'figure_key', None)))
        row = QHBoxLayout()
        row.addWidget(QLabel('<b>Comparison:</b>'))
        self.cb_comparison = QComboBox()
        self.cb_comparison.currentIndexChanged.connect(lambda _i: self._show_results(redraw_summary=False))
        _no_wheel(self.cb_comparison)
        row.addWidget(self.cb_comparison, stretch=1)
        self.lbl_marker_comparison = QLabel('<b>Markers:</b>')
        self.cb_marker_comparison = QComboBox()
        self.cb_marker_comparison.setToolTip('The group whose marker differences the Marker views show.')
        self.cb_marker_comparison.currentIndexChanged.connect(lambda _i: self._show_marker_results())
        _no_wheel(self.cb_marker_comparison)
        row.addWidget(self.lbl_marker_comparison)
        row.addWidget(self.cb_marker_comparison, stretch=1)
        row.addWidget(QLabel('<b>View:</b>'))
        row.addWidget(self.figure_tabs.selector, stretch=1)
        lay.addLayout(row)
        opts = QHBoxLayout()
        opts.addWidget(QLabel('Comparison heatmap order:'))
        self.cb_heat_order = QComboBox()
        for key, text in HEATMAP_ORDERS.items():
            self.cb_heat_order.addItem(text, key)
        opts.addWidget(self.cb_heat_order)
        opts.addWidget(QLabel('rows:'))
        self.sb_heat_rows = _int_spin(1, 3)
        opts.addWidget(self.sb_heat_rows)
        opts.addWidget(QLabel('PCA loadings:'))
        self.sb_loadings = _int_spin(0, 50)
        opts.addWidget(self.sb_loadings)
        self.figure_defaults_btn = QPushButton('Restore defaults')
        self.figure_defaults_btn.setToolTip(RESTORE_TOOLTIP + ' (heatmap order, heatmap rows, PCA loadings)')
        self.figure_defaults_btn.clicked.connect(self._restore_figure_defaults)
        opts.addWidget(self.figure_defaults_btn)
        opts.addStretch()
        lay.addLayout(opts)
        _no_wheel(self.cb_heat_order)
        self.cb_heat_order.currentIndexChanged.connect(
            lambda _i: self._on_figure_option('heatmap_order', self.cb_heat_order.currentData()))
        self.sb_heat_rows.valueChanged.connect(lambda v: self._on_figure_option('heatmap_rows', int(v)))
        self.sb_loadings.valueChanged.connect(lambda v: self._on_figure_option('n_loadings', int(v)))
        self.results_table = CopyableTableWidget([], [])
        self.results_table.setToolTip('Double-click a row to plot its combination in every sample.')
        self.results_table.cellDoubleClicked.connect(self._on_results_double_clicked)
        self.stale_label = _label(STALE_TEXT, _WARN_STYLE)
        self.stale_label.setVisible(False)
        lay.addWidget(self.stale_label)
        self.figure_tabs.addTab(self.results_table, 'Table')
        lay.addWidget(self.figure_tabs, stretch=1)
        self.figure_msg = _label('')
        lay.addWidget(self.figure_msg)

    def _clear_figure_tabs(self, keep=()):
        """Remove the figure tabs, keeping the results table (tab 0) and the
        figures whose key is in *keep*."""
        for i in range(self.figure_tabs.count() - 1, 0, -1):
            widget = self.figure_tabs.widget(i)
            if getattr(widget, 'figure_key', None) in keep:
                continue
            self.figure_tabs.removeTab(i)
            if widget is not None:
                widget.deleteLater()

    def _figure_name(self, key: str, title: str) -> str:
        """Export Graphic file name: population, comparison (for the
        per-comparison figures) and figure title."""
        parts = [self.counted_population()]
        pair = self.current_pair()
        if _figure_set(key) == 'markers':
            parts.append(f'markers {self.cb_marker_comparison.currentText()}')
        elif pair is not None and key not in SUMMARY_FIGURES + ('pathway_dot_plot', 'guide_allocation',
                                                                'gene_group_spider'):
            parts.append(fst.comparison_label(*pair))
        parts.append(title)
        return ' - '.join(parts)

    def _on_figure_option(self, key, value):
        if self._filling:
            return
        self.settings[key] = value
        self._changed()
        self._show_results()

    def current_pair(self):
        if self.stats is None:
            return None
        i = self.cb_comparison.currentIndex()
        pairs = self.stats['pairs']
        return pairs[i] if 0 <= i < len(pairs) else (pairs[0] if pairs else None)

    def _show_results(self, redraw_summary: bool = True):
        s = self.settings
        self._filling = True
        try:
            self.cb_heat_order.setCurrentIndex(max(self.cb_heat_order.findData(s['heatmap_order']), 0))
            self.sb_heat_rows.setValue(int(s['heatmap_rows']))
            self.sb_loadings.setValue(int(s['n_loadings']))
            if redraw_summary:
                markers = (self.stats or {}).get('markers')
                for combo, labels in ((self.cb_comparison, (self.stats or {}).get('comparisons') or []),
                                      (self.cb_marker_comparison, markers['comparisons'] if markers else [])):
                    current = combo.currentText()
                    combo.blockSignals(True)
                    try:
                        combo.clear()
                        for label in labels:
                            combo.addItem(label)
                        if current:
                            combo.setCurrentIndex(max(combo.findText(current), 0))
                    finally:
                        combo.blockSignals(False)
                shown = markers is not None
                self.lbl_marker_comparison.setVisible(shown)
                self.cb_marker_comparison.setVisible(shown)
        finally:
            self._filling = False
        self._fill_results_table()
        if self.stats is None:
            self._clear_figure_tabs()
            self.figure_msg.setText('Run statistics to see the results.')
            self._show_guide_allocation()
            return
        self._shown_figure = self.figure_tabs.tabText(self.figure_tabs.currentIndex())
        # A new comparison leaves its old views in place until the new ones
        # replace them (on_figures_finished), so the shown view does not fall
        # through to a neighbouring one while the figures are drawn.
        if redraw_summary:
            self._clear_figure_tabs()
        self.start_figures(summary=redraw_summary, markers=redraw_summary)

    def _show_guide_allocation(self):
        """Before any statistics: the guide allocation tab, when a
        gene-group table is loaded."""
        if self.gene_group_frame() is None:
            return
        try:
            alloc = self.gene_group_allocation()
        except Exception as exc:
            self.figure_msg.setText(f'Guide allocation: {exc}')
            return
        items = fgg.figures(None, alloc, self.gene_group_colours(), self.settings.get('gene_group_areas') or {},
                            [], {}, None, None, None, float(self.settings['pval_threshold']), self.plugin.is_dark(),
                            self.counted_population())
        item = items.get('guide_allocation')
        if item is None or item['fig'] is None:
            self.figure_msg.setText(f"Guide allocation: {(item or {}).get('error', '')}")
            return
        self.figures['guide_allocation'] = item
        self._insert_panel('guide_allocation', self._figure_panel('guide_allocation', item), item['title'])

    def _show_marker_results(self):
        """Redraw only the marker tabs (a different marker group chosen)."""
        if self.stats is None or self._filling:
            return
        self._shown_figure = self.figure_tabs.tabText(self.figure_tabs.currentIndex())
        self._clear_figure_tabs(tuple(k for k in self.figures if _figure_set(k) != 'markers'))
        self.start_figures(summary=False, comparison=False, markers=True)

    def _fill_results_table(self):
        table = self.results_table
        pair = self.current_pair()
        if pair is None:
            table.setRowCount(0)
            table.setColumnCount(0)
            return
        label = fst.comparison_label(*pair)
        frames = []
        for key in ('counts', 'freq'):
            res = self.stats.get(key)
            if res is not None:
                view = res[res['comparison'] == label].assign(test=fst.TEST_NAMES[key])
                loo = fst.leave_one_out_columns(self.stats, key)
                if loo is not None:
                    view = view.join(_loo_display(loo), on=['comparison', 'feature'])
                frames.append(view)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for c in LOO_COLUMNS:
            if c in df.columns:
                df[c] = df[c].fillna('')
        cols = [c for c in ('test', 'feature', 'logFC', 'CI.L', 'CI.R', 'P.Value', 'adj.P.Val', 'adj.P.Val.global',
                            'significant') + LOO_COLUMNS if c in df.columns]
        df = df[cols].sort_values(['test', 'P.Value']) if not df.empty else df
        # Sized once after filling: with ResizeToContents active, every
        # setItem re-measures its whole column.
        table.setSortingEnabled(False)
        table.setUpdatesEnabled(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        try:
            table.setColumnCount(len(cols))
            table.setHorizontalHeaderLabels(['combination' if c == 'feature' else c for c in cols])
            table.setRowCount(len(df))
            for r, (_i, row) in enumerate(df.iterrows()):
                for j, c in enumerate(cols):
                    v = row[c]
                    text = (f'{v:.3g}' if isinstance(v, (float, np.floating)) and c != 'logFC'
                            else (f'{v:+.2f}' if c == 'logFC' else str(v)))
                    table.setItem(r, j, _item(text))
            table.resizeColumnsToContents()
        finally:
            table.setSortingEnabled(True)
            table.setUpdatesEnabled(True)

    def _unit_groups(self) -> dict:
        """{counts row: its sample's group}."""
        groups = self.settings.get('sample_groups') or {}
        if self.counts is None:
            return dict(groups)
        return {u: groups.get(self.unit_sample(u), '') for u in self.counts.index}

    def start_figures(self, summary: bool = True, comparison: bool = True, markers: bool = True):
        """Draw figures on a worker: the chosen comparison's, with *summary*
        those over every comparison, and with *markers* the chosen marker
        group's."""
        if self.stats is None:
            return
        wanted = {name for name, on in (('comparison', comparison), ('summary', summary), ('markers', markers))
                  if on}
        if self.plugin.task_running('analysis_figures'):
            self._figures_pending = True
            self._pending_sets = set(getattr(self, '_pending_sets', set())) | wanted
            return
        self._figures_pending = False
        self._pending_sets = set()
        stats = self.stats
        s = dict(self.settings)
        pair = self.current_pair()
        is_dark = self.plugin.is_dark()
        population = self.counted_population()
        labels = self.unit_labels()
        qc = self.qc
        filtered = self.filtered
        groups = self._unit_groups()
        colours = self.group_colours()
        order = list(s.get('group_names') or [])
        combination = self._shown_combination()
        marker_label = self.cb_marker_comparison.currentText()
        marker_ref = fst.MARKER_REFERENCES.get((stats.get('markers') or {}).get('reference'), '')
        marker_feature = self._shown_marker_feature()
        marker_pairing = self._unit_factor(s.get('pairing'), pairing=True) if marker_feature else None
        pathways = stats.get('pathways')
        pathway_label = fst.comparison_label(*pair) if pair is not None else None
        pathway_set = self._shown_pathway(pathway_label)
        group_result = self._group_result()
        group_set = self._shown_gene_group(pathway_label)
        group_alloc = (group_result or {}).get('allocation')
        group_colours = self.gene_group_colours()
        group_areas = dict(s.get('gene_group_areas') or {})
        spider = self.spider_comparisons()
        spider_colours = fgg.comparison_colours(spider, stats.get('pairs'), colours)

        def work(_progress):
            out = {}
            if pathways and ('summary' in wanted or 'comparison' in wanted):
                label = pathway_label if 'comparison' in wanted else None
                out.update(fpw.figures(pathways, stats['comparisons'], label, pathway_set,
                                      self._pathway_ci(stats, label), s['pval_threshold'], is_dark, population,
                                      dot_plot='summary' in wanted))
            if group_result and ('summary' in wanted or 'comparison' in wanted):
                label = pathway_label if 'comparison' in wanted else None
                out.update(fgg.figures(group_result, group_alloc, group_colours, group_areas, spider,
                                       spider_colours, label, group_set if label else None,
                                       self._pathway_ci(stats, label), s['pval_threshold'], is_dark, population,
                                       summary='summary' in wanted,
                                       min_size=group_result['settings'].get('min_size')))
            if 'markers' in wanted and stats.get('markers') is not None and marker_label:
                out.update(fst.marker_figures(stats['markers'], marker_label, s['pval_threshold'],
                                              s['marker_threshold'], s['fdr_scope'], is_dark, population,
                                              marker_ref, s['heatmap_order'], feature=marker_feature,
                                              pairing=marker_pairing, sample_labels=labels, group_colours=colours,
                                              group_order=order))
            if 'comparison' in wanted and pair is not None:
                out.update(fst.comparison_figures(stats, pair, s['pval_threshold'], s['fc_threshold'],
                                                  s['fdr_scope'], is_dark, population, labels,
                                                  group_colours=colours, combination=combination,
                                                  group_order=order))
            if 'summary' in wanted:
                out.update(fst.summary_figures(stats, is_dark, population, s['heatmap_order'],
                                               int(s['heatmap_rows']), int(s['n_loadings']), labels,
                                               group_colours=colours))
                if qc is not None and filtered is not None:
                    make = lambda: fst.qc_figure(qc, filtered.counts, groups, is_dark, population,  # noqa: E731
                                                 group_colours=colours)
                    try:
                        out['qc'] = {'fig': make(), 'title': 'QC', 'error': None, 'make': make}
                    except Exception as exc:
                        out['qc'] = {'fig': None, 'title': 'QC', 'error': str(exc)}
            return {'figures': out, 'sets': wanted}

        if self.plugin._start_task('analysis_figures', work):
            self.figure_msg.setText('Drawing figures…')

    def on_figures_finished(self, result):
        if getattr(self, '_figures_pending', False):
            sets = self._pending_sets
            self.start_figures(summary='summary' in sets, comparison='comparison' in sets, markers='markers' in sets)
            return
        figures, sets = result['figures'], result['sets']
        shown = getattr(self, '_shown_figure', None) or self.figure_tabs.tabText(self.figure_tabs.currentIndex())
        self._shown_figure = None
        kept = {k: v for k, v in self.figures.items() if _figure_set(k) not in sets}
        self.figures = {**figures, **kept}
        self._clear_figure_tabs(keep=tuple(kept))
        errors = []
        for key, item in figures.items():
            if item['fig'] is None:
                errors.append(f"{item['title']}: {item['error']}")
                continue
            self._insert_panel(key, self._figure_panel(key, item), item['title'])
        if 'markers' in sets and (self.stats or {}).get('markers') is not None:
            self._insert_panel('marker_table', self._marker_table(), 'Marker table')
            self.figures['marker_table'] = {'fig': None, 'title': 'Marker table', 'error': None}
        if 'comparison' in sets and (self.stats or {}).get('pathways') and self.current_pair() is not None:
            self._insert_panel('pathway_table', self._pathway_table(), 'Pathway table')
            self.figures['pathway_table'] = {'fig': None, 'title': 'Pathway table', 'error': None}
        if 'comparison' in sets and (self.stats or {}).get('gene_groups') and self.current_pair() is not None:
            self._insert_panel('gene_group_table', self._gene_group_table(), 'Gene group table')
            self.figures['gene_group_table'] = {'fig': None, 'title': 'Gene group table', 'error': None}
        for i in range(self.figure_tabs.count()):
            if self.figure_tabs.tabText(i) == shown:
                self.figure_tabs.setCurrentIndex(i)
                break
        self.figure_msg.setText('; '.join(errors))

    def _insert_panel(self, key: str, widget, title: str):
        """Add a figure tab in its place (``_figure_rank``)."""
        widget.figure_key = key
        rank = _figure_rank(key)
        position = self.figure_tabs.count()
        for i in range(1, self.figure_tabs.count()):
            if _figure_rank(getattr(self.figure_tabs.widget(i), 'figure_key', None)) > rank:
                position = i
                break
        self.figure_tabs.insertTab(position, widget, title)
        return position

    def _marker_table(self):
        """The chosen marker group's results, most significant first."""
        markers = self.stats['markers']
        res = markers['results']
        view = res[res['comparison'] == self.cb_marker_comparison.currentText()]
        cols = [c for c in ('combination', 'channel', 'logFC', 'CI.L', 'CI.R', 'n_samples', 'spillover', 'k',
                            'se_moderated', 'se_total', 'P.Value', 'cluster.adj.P.Val', 'stagewise.adj.P.Val',
                            'significant')
                if c in view.columns]
        view = view.sort_values('stagewise.adj.P.Val' if 'stagewise.adj.P.Val' in view.columns else 'P.Value')
        table = CopyableTableWidget([], [])
        table.setSortingEnabled(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        table.setColumnCount(len(cols))
        table.setHorizontalHeaderLabels(['Δ' if c == 'logFC' else c for c in cols])
        table.setRowCount(len(view))
        for r, (_i, row) in enumerate(view[cols].iterrows()):
            for j, c in enumerate(cols):
                v = row[c]
                text = (f'{v:+.3f}' if c in ('logFC', 'CI.L', 'CI.R') else
                        f'{v:.3g}' if isinstance(v, (float, np.floating)) else str(v))
                table.setItem(r, j, _item(text))
        table.resizeColumnsToContents()
        table.setSortingEnabled(True)
        table.setToolTip('Double-click a row to plot its combination and marker in every sample.')
        table.cellDoubleClicked.connect(lambda row, _col, table=table: self._on_marker_row(table, row))
        return table

    def _on_marker_row(self, table, row: int):
        items = [table.item(row, j) for j in range(2)]
        if all(it is not None for it in items):
            self.show_marker_values(fst.marker_feature(items[0].text(), items[1].text()))

    @staticmethod
    def _pathway_ci(stats: dict, comparison: str | None) -> pd.DataFrame | None:
        """CI.L and CI.R of each combination's Counts result in *comparison*."""
        res = stats.get('counts')
        if res is None or comparison is None:
            return None
        return res[res['comparison'] == comparison].set_index('feature')[['CI.L', 'CI.R']]

    def _shown_pathway(self, comparison: str | None) -> str | None:
        """The pathway of the Pathway members tab in *comparison*: the one
        chosen last when it was tested there, else the first of the table."""
        result = (self.stats or {}).get('pathways')
        if not result or comparison is None:
            return None
        table = result['pathways']
        if self._pathway_set in set(table.loc[table['comparison'] == comparison, 'set_key']):
            return self._pathway_set
        return fpw.top_set(result, comparison)

    def _pathway_table(self):
        """The chosen comparison's pathways, lowest competitive FDR first."""
        result = self.stats['pathways']
        view = fpw.pathway_table(result, fst.comparison_label(*self.current_pair()))
        cols = (('set_name', 'pathway'), ('score', 'score'), ('concordance', 'agreement'), ('members', 'members'),
                ('libraries', 'libraries'), ('competitive_FDR', 'competitive FDR'),
                ('competitive_P.Value', 'competitive p'), ('FDR', 'self-contained FDR'),
                ('P.Value', 'self-contained p'), ('leading', 'leading members'), ('per_library', 'per library'),
                ('go_names', 'GO terms'), ('go_ids', 'GO ids'))
        table = CopyableTableWidget([], [])
        table.setSortingEnabled(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        table.setColumnCount(len(cols))
        table.setHorizontalHeaderLabels([title for _c, title in cols])
        table.setRowCount(len(view))
        for r, (_i, row) in enumerate(view.iterrows()):
            for j, (c, _title) in enumerate(cols):
                v = row[c]
                text = (f'{v:+.2f}' if c == 'score' else f'{v:.0%}' if c == 'concordance' and np.isfinite(v) else
                        f'{v:.3g}' if isinstance(v, (float, np.floating)) else str(v))
                table.setItem(r, j, _item(text, data=row['set_key'] if j == 0 else None))
        table.resizeColumnsToContents()
        table.setColumnWidth(0, min(table.columnWidth(0), 320))
        table.setSortingEnabled(True)
        table.setToolTip('Double-click a row to show its guides in the Pathway members view.')
        table.cellDoubleClicked.connect(lambda row, _col, table=table: self._on_pathway_row(table, row))
        return table

    def _on_pathway_row(self, table, row: int):
        item = table.item(row, 0)
        if item is not None and item.data(Qt.UserRole):
            self.show_pathway(str(item.data(Qt.UserRole)))

    def show_pathway(self, set_key: str):
        """Show *set_key*'s guides in the chosen comparison in the Pathway
        members tab (after the pathway table) and switch to it. Later
        comparison changes keep it when it was tested there."""
        result = (self.stats or {}).get('pathways')
        pair = self.current_pair()
        if not result or pair is None:
            return
        self._pathway_set = set_key
        label = fst.comparison_label(*pair)
        items = fpw.figures(result, self.stats['comparisons'], label, set_key, self._pathway_ci(self.stats, label),
                           float(self.settings['pval_threshold']), self.plugin.is_dark(), self.counted_population(),
                           dot_plot=False)
        item = items.get('pathway_members')
        if item is None or item['fig'] is None:
            self.figure_msg.setText(f"Pathway members: {(item or {}).get('error', 'not tested')}")
            return
        for i in range(self.figure_tabs.count() - 1, 0, -1):
            widget = self.figure_tabs.widget(i)
            if getattr(widget, 'figure_key', None) == 'pathway_members':
                self.figure_tabs.removeTab(i)
                widget.deleteLater()
        self.figures['pathway_members'] = item
        position = self._insert_panel('pathway_members', self._figure_panel('pathway_members', item), item['title'])
        self.figure_tabs.setCurrentIndex(position)

    def _on_dot_click(self, event, fig):
        """Double-click on the dot plot: show that pathway in that comparison."""
        ax = getattr(fig, '_dot_axes', None)
        if not (event.dblclick and event.button == 1) or ax is None or event.inaxes is not ax or event.x is None:
            return
        points = fig._dot_points
        if not len(points):
            return
        px = ax.transData.transform(points)
        d = np.hypot(px[:, 0] - event.x, px[:, 1] - event.y)
        i = int(np.argmin(d))
        if d[i] > 12:
            return
        comparison, set_key = fig._dot_keys[i]
        self._pathway_set = set_key
        pair = self.current_pair()
        if pair is not None and fst.comparison_label(*pair) == comparison:
            self.show_pathway(set_key)
            return
        index = self.cb_comparison.findText(comparison)
        if index >= 0:
            self.cb_comparison.setCurrentIndex(index)
            self._shown_figure = 'Pathway members'

    def _on_member_click(self, event, fig):
        """Double-click on a guide of the members plot: plot it in every sample."""
        ax = getattr(fig, '_member_axes', None)
        if not (event.dblclick and event.button == 1) or ax is None or event.inaxes is not ax or event.ydata is None:
            return
        features = fig._member_features
        i = int(round(event.ydata))
        if 0 <= i < len(features):
            self.show_combination(features[i])

    def _group_result(self) -> dict | None:
        """The gene-group result with the groups in the order chosen in the
        Gene groups dialog, or None."""
        result = (self.stats or {}).get('gene_groups')
        if not result:
            return None
        ran = list(result.get('group_order') or [])
        chosen = [g for g in self.gene_group_order() if g in ran]
        return {**result, 'group_order': chosen + [g for g in ran if g not in chosen]}

    def _shown_gene_group(self, comparison: str | None) -> str | None:
        """The group of the Gene group members tab in *comparison*: the one
        chosen last when it was tested there, else the first in the table
        with either FDR passing, else the first tested."""
        result = self._group_result()
        if not result or comparison is None:
            return None
        table = fgg.group_table(result, comparison)
        if table.empty:
            return None
        if self._group_set in set(table['set_key']):
            return self._group_set
        hits = fgg.export_groups(result, comparison, float(self.settings['pval_threshold']), limit=1)
        return hits[0] if hits else str(table['set_key'].iloc[0])

    def _gene_group_table(self):
        """The chosen comparison's gene groups, in group order."""
        result = self._group_result()
        view = fgg.group_table(result, fst.comparison_label(*self.current_pair()))
        cols = (('set_name', 'group'), ('score', 'score'), ('concordance', 'agreement'), ('members', 'guides'),
                ('libraries', 'libraries'), ('competitive_FDR', 'competitive FDR'),
                ('competitive_P.Value', 'competitive p'), ('mixed_FDR', 'any-change FDR'),
                ('mixed_P.Value', 'any-change p'), ('FDR', 'self-contained FDR'), ('P.Value', 'self-contained p'),
                ('leading', 'leading guides'), ('per_library', 'per library'))
        table = CopyableTableWidget([], [])
        table.setSortingEnabled(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        table.setColumnCount(len(cols))
        table.setHorizontalHeaderLabels([title for _c, title in cols])
        table.setRowCount(len(view))
        for r, (_i, row) in enumerate(view.iterrows()):
            for j, (c, _title) in enumerate(cols):
                v = row[c]
                text = (f'{v:+.2f}' if c == 'score' else f'{v:.0%}' if c == 'concordance' and np.isfinite(v) else
                        f'{v:.3g}' if isinstance(v, (float, np.floating)) else str(v))
                table.setItem(r, j, _item(text, data=row['set_key'] if j == 0 else None))
        table.resizeColumnsToContents()
        table.setSortingEnabled(True)
        table.setToolTip('Double-click a row to show its guides in the Gene group members view.')
        table.cellDoubleClicked.connect(lambda row, _col, table=table: self._on_gene_group_row(table, row))
        return table

    def _on_gene_group_row(self, table, row: int):
        item = table.item(row, 0)
        if item is not None and item.data(Qt.UserRole):
            self.show_gene_group(str(item.data(Qt.UserRole)))

    def show_gene_group(self, set_key: str):
        """Show *set_key*'s guides in the chosen comparison in the Gene group
        members tab and switch to it. Later comparison changes keep it when it
        was tested there."""
        result = self._group_result()
        pair = self.current_pair()
        if not result or pair is None:
            return
        self._group_set = set_key
        label = fst.comparison_label(*pair)
        items = fgg.figures(result, None, self.gene_group_colours(), {}, [], {}, label, set_key,
                            self._pathway_ci(self.stats, label), float(self.settings['pval_threshold']),
                            self.plugin.is_dark(), self.counted_population(), summary=False)
        item = items.get('gene_group_members')
        if item is None or item['fig'] is None:
            self.figure_msg.setText(f"Gene group members: {(item or {}).get('error', 'not tested')}")
            return
        for i in range(self.figure_tabs.count() - 1, 0, -1):
            widget = self.figure_tabs.widget(i)
            if getattr(widget, 'figure_key', None) == 'gene_group_members':
                self.figure_tabs.removeTab(i)
                widget.deleteLater()
        self.figures['gene_group_members'] = item
        position = self._insert_panel('gene_group_members', self._figure_panel('gene_group_members', item),
                                      item['title'])
        self.figure_tabs.setCurrentIndex(position)

    def _on_spider_click(self, event, fig):
        """Double-click on a spider point: show that group in that comparison."""
        ax = getattr(fig, '_spider_axes', None)
        if not (event.dblclick and event.button == 1) or ax is None or event.inaxes is not ax or event.x is None:
            return
        points = fig._spider_points
        if not len(points):
            return
        px = ax.transData.transform(points)
        d = np.hypot(px[:, 0] - event.x, px[:, 1] - event.y)
        i = int(np.argmin(d))
        if d[i] > 12:
            return
        comparison, set_key = fig._spider_keys[i]
        self._group_set = set_key
        pair = self.current_pair()
        if pair is not None and fst.comparison_label(*pair) == comparison:
            self.show_gene_group(set_key)
            return
        index = self.cb_comparison.findText(comparison)
        if index >= 0:
            self.cb_comparison.setCurrentIndex(index)
            self._shown_figure = 'Gene group members'

    def _figure_panel(self, key: str, item: dict) -> fcfp.FigurePanel:
        """A figure tab; on the volcanos and fold-change bars a double-click
        on a combination shows it in the Combination tab (right-click for a
        menu), also in their popped-out windows."""
        panel = fcfp.FigurePanel(self._figure_name(key, item.get('export_title') or item['title']),
                                 self.analysis_root, options_store=self.figure_options_store)
        panel.figure_key = key
        if key in ('counts_volcano', 'freq_volcano', 'fold_change_bars'):
            results = item.get('results')
            panel.connect_canvas = lambda canvas, fig, results=results: self._connect_clicks(canvas, fig, results)
        elif key == 'marker_volcano':
            results = item.get('results')
            panel.connect_canvas = lambda canvas, fig, results=results: self._connect_clicks(
                canvas, fig, results, markers=True)
        elif key in ('pathway_dot_plot', 'pathway_members'):
            handler = self._on_dot_click if key == 'pathway_dot_plot' else self._on_member_click

            def connect(canvas, fig, handler=handler, key=key):
                canvas.mpl_connect('button_press_event', lambda event, fig=fig: handler(event, fig))
                canvas.setToolTip('Double-click a dot to show that pathway\'s guides (Pathway members).'
                                  if key == 'pathway_dot_plot' else
                                  'Double-click a guide to plot it in every sample (Combination view).')
            panel.connect_canvas = connect
        elif key in ('gene_group_spider', 'gene_group_members'):
            handler = self._on_spider_click if key == 'gene_group_spider' else self._on_member_click

            def connect(canvas, fig, handler=handler, key=key):
                canvas.mpl_connect('button_press_event', lambda event, fig=fig: handler(event, fig))
                canvas.setToolTip('Double-click a point to show that group\'s guides (Gene group members).'
                                  if key == 'gene_group_spider' else
                                  'Double-click a guide to plot it in every sample (Combination view).')
            panel.connect_canvas = connect
        fig, make = self._stamped(item['fig'], item.get('make'))
        panel.set_figure(fig, make=make)
        return panel

    def _connect_clicks(self, canvas, fig, results, markers: bool = False):
        if results is None or results.empty:
            return
        canvas.mpl_connect('button_press_event',
                           lambda event, fig=fig, results=results: self._on_figure_click(event, fig, results,
                                                                                          markers))
        canvas.setToolTip('Double-click a point to plot it in every sample (Marker view); right-click for more.'
                          if markers else 'Double-click a combination to plot it in every sample (Combination '
                          'view); right-click for more.')

    def _combination_at(self, event, fig, results) -> str | None:
        """The combination under a mouse event: the bar of the fold-change
        bars, or the volcano point within 12 pixels."""
        ax = event.inaxes
        if ax is None or event.x is None:
            return None
        bars = getattr(fig, '_bar_features', None)
        if bars is not None:
            if ax is not getattr(fig, '_bar_axes', None) or event.xdata is None:
                return None
            i = int(round(event.xdata))
            return bars[i] if 0 <= i < len(bars) else None
        from honeychrome.view_components.differential_plots import significance_column
        col, _label = significance_column(results, self.settings['fdr_scope'])
        xy = np.column_stack([results['logFC'].values.astype(float),
                              -np.log10(np.maximum(results[col].values.astype(float), 1e-300))])
        ok = np.isfinite(xy).all(axis=1)
        if not ok.any():
            return None
        px = ax.transData.transform(xy[ok])
        d = np.hypot(px[:, 0] - event.x, px[:, 1] - event.y)
        i = int(np.argmin(d))
        return None if d[i] > 12 else str(results[ok].iloc[i]['feature'])

    def _on_figure_click(self, event, fig, results, markers: bool = False):
        combination = self._combination_at(event, fig, results)
        if combination is None:
            return
        show = self.show_marker_values if markers else self.show_combination
        window = self._marker_window if markers else self._combination_window
        if event.dblclick and event.button == 1:
            show(combination)
        elif event.button == 3:
            from PySide6.QtGui import QCursor
            from PySide6.QtWidgets import QMenu
            menu = QMenu(self)
            menu.addSection(combination)
            menu.addAction('Show in the Marker view' if markers else 'Show in the Combination view').triggered.connect(
                lambda: show(combination))
            menu.addAction('Open in a window').triggered.connect(lambda: window(combination))
            menu.exec(QCursor.pos())

    def _shown_marker_feature(self) -> str | None:
        """The combination-marker pair of the Marker tab, when the marker test has it."""
        markers = (self.stats or {}).get('markers')
        name = self._marker_feature
        if not name or markers is None or markers.get('means') is None or name not in markers['means'].columns:
            return None
        return name

    def _marker_maker(self, feature: str):
        markers, s = self.stats['markers'], self.settings
        comparison = self.cb_marker_comparison.currentText()
        reference = fst.MARKER_REFERENCES.get(markers.get('reference'), '')
        pairing = self._unit_factor(s.get('pairing'), pairing=True)
        colours, order = self.group_colours(), list(s.get('group_names') or [])
        population, labels, is_dark = self.counted_population(), self.unit_labels(), self.plugin.is_dark()
        fdr_scope = s['fdr_scope']

        def make():
            return fst.marker_value_figure(markers, feature, comparison, fdr_scope, is_dark, population, reference,
                                           pairing=pairing, sample_labels=labels, group_colours=colours,
                                           group_order=order)
        return make

    def show_marker_values(self, feature: str):
        """Plot a combination-marker pair in every sample in the Marker tab
        (after the other marker tabs) and show that tab. It follows later
        marker group changes."""
        if self.stats is None or (self.stats.get('markers') or {}).get('means') is None:
            return
        self._marker_feature = feature
        if self._shown_marker_feature() is None:
            return
        make = self._marker_maker(feature)
        title = f'Marker: {feature}'
        try:
            item = {'fig': make(), 'title': title, 'error': None, 'make': make}
        except Exception as exc:
            self.figure_msg.setText(f'{title}: {exc}')
            return
        for i in range(self.figure_tabs.count() - 1, 0, -1):
            widget = self.figure_tabs.widget(i)
            if getattr(widget, 'figure_key', None) == 'marker_combination':
                self.figure_tabs.removeTab(i)
                widget.deleteLater()
        self.figures['marker_combination'] = item
        position = self._insert_panel('marker_combination', self._figure_panel('marker_combination', item), title)
        self.figure_tabs.setCurrentIndex(position)

    def _marker_window(self, feature: str):
        if self.stats is None or (self.stats.get('markers') or {}).get('means') is None:
            return
        try:
            fig = self._stamped(self._marker_maker(feature)())[0]
        except Exception as exc:
            QMessageBox.warning(self, 'Marker', f'The figure could not be drawn: {exc}')
            return
        fcfp.open_figure_window(self, fig, self._figure_name('marker_combination', f'Marker: {feature}'))

    def _on_results_double_clicked(self, row: int, _col: int):
        for j in range(self.results_table.columnCount()):
            if self.results_table.horizontalHeaderItem(j).text() == 'combination':
                item = self.results_table.item(row, j)
                if item is not None:
                    self.show_combination(item.text())
                return

    def _shown_combination(self) -> str | None:
        """The combination of the Combination tab, when the statistics have it."""
        name = self._combination
        if not name or self.stats is None or name not in self.stats['log2_values'].columns:
            return None
        return name

    def _combination_maker(self, combination: str):
        stats, pair, s = self.stats, self.current_pair(), self.settings
        colours, order = self.group_colours(), list(s.get('group_names') or [])
        population, labels, is_dark = self.counted_population(), self.unit_labels(), self.plugin.is_dark()
        fdr_scope = s['fdr_scope']

        def make():
            return fst.combination_figure(stats, combination, pair, fdr_scope, is_dark, population, labels,
                                          group_colours=colours, group_order=order)
        return make

    def show_combination(self, combination: str):
        """Plot *combination* in every sample in the Combination tab (after
        the comparison's figures) and show that tab. It follows later
        comparison changes."""
        if self.stats is None or self.current_pair() is None:
            return
        self._combination = combination
        if self._shown_combination() is None:
            return
        make = self._combination_maker(combination)
        title = f'Combination: {combination}'
        try:
            item = {'fig': make(), 'title': title, 'error': None, 'make': make}
        except Exception as exc:
            self.figure_msg.setText(f'{title}: {exc}')
            return
        for i in range(self.figure_tabs.count() - 1, 0, -1):
            widget = self.figure_tabs.widget(i)
            if getattr(widget, 'figure_key', None) == 'combination':
                self.figure_tabs.removeTab(i)
                widget.deleteLater()
        self.figures['combination'] = item
        position = self._insert_panel('combination', self._figure_panel('combination', item), title)
        self.figure_tabs.setCurrentIndex(position)

    def _combination_window(self, combination: str):
        if self.stats is None or self.current_pair() is None:
            return
        try:
            fig = self._stamped(self._combination_maker(combination)())[0]
        except Exception as exc:
            QMessageBox.warning(self, 'Combination', f'The figure could not be drawn: {exc}')
            return
        fcfp.open_figure_window(self, fig, self._figure_name('combination', f'Combination: {combination}'))

    def on_figures_error(self, message):
        self.figure_msg.setText(f'Figures failed: {message}')

    # =====================================================================
    # Export
    # =====================================================================

    def _build_export(self, lay):
        blurb = _label('Writes a folder named after the population (and the date and time) with the counts, '
                       'frequencies (also as an R-style pct_matrix.csv), QC, samples, results and settings, a PNG '
                       'of every figure for every comparison, and one PDF report.', '')
        lay.addWidget(blurb)
        row = QHBoxLayout()
        row.addWidget(QLabel('Analysis folder:'))
        self.root_label = _label('', '')
        row.addWidget(self.root_label, stretch=1)
        for text, slot in (('Choose…', self._on_choose_root), ('Default', self._on_default_root)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        lay.addLayout(row)
        self.export_btn = QPushButton('Export analysis')
        self.export_btn.clicked.connect(self.start_export)
        lay.addWidget(self.export_btn)
        self.export_status = _label('')
        lay.addWidget(self.export_status)

    def _on_choose_root(self):
        folder = QFileDialog.getExistingDirectory(self, 'Folder for analysis exports', str(self.analysis_root()))
        if folder:
            self.settings['export_root'] = folder
            self._changed()
            self.root_label.setText(str(self.analysis_root()))

    def _on_default_root(self):
        self.settings['export_root'] = None
        self._changed()
        self.root_label.setText(str(self.analysis_root()))

    def settings_text(self) -> str:
        """Plain-text record of the analysis, for settings.txt and the report."""
        from honeychrome import __version__ as hc_version
        import flowcode_state as fcs

        s = self.settings
        stats = self.stats
        files = self.used_files()
        saved = s.get('counts') or {}
        lines = ([NO_FRET_WARNING, 'The samples were debarcoded without FRET correction (no FRET backbone): '
                  'FRET, residual tag spillover and tandem breakdown were not removed.', '']
                 if self.uncorrected() else []) + [
            f"Exported: {datetime.now().isoformat(timespec='seconds')}",
            f'Honeychrome: {hc_version}; FlowCode plugin: {fcs.PLUGIN_VERSION}',
            f'Experiment: {self._controller().experiment_dir}',
            f'Populations: {self.counted_population()} (each a gate of the unmixed gating'
            + (', or a phenotype scored on the live-unmixed events inside the FlowCode gates: '
               + '; '.join(f'{name}: {summary}' for name, summary in self.counted_phenotypes().items())
               + '; counts are memberships summed' if self.soft_counts() else '') + ')',
            f"Counted: {saved.get('time', '')}",
            (f"Source: live unmix of the raw files ({saved.get('method') or ''})"
             if saved.get('source') == fa.SOURCE_LIVE else f'Exported files: {self.export_folder()}'
             + ''.join(f'; linked from another experiment: {f}' for f in s.get('linked_folders') or [])),
            'Gating: the experiment\'s unmixed gate boundaries (with each sample\'s own overrides) applied to the '
            'FlowCode-unmixed events.'
            + (' Files linked from other experiments are gated with the template boundaries.'
               if saved.get('source') != fa.SOURCE_LIVE and s.get('linked_folders') else ''),
            f"Inclusion: combinations left out by choice: {', '.join(s.get('exclude_combos') or []) or 'none'}; "
            f"min. events per combination {s['min_combo_events']}; min. barcoded events per sample "
            f"{s['min_sample_events']}.",
            '',
            'Samples (file; events; in population; barcoded; group):',
        ]
        pops = self.counted_populations() or [self.population_name()]
        several = len(pops) > 1
        for key, path in files.items():
            for pop in pops:
                unit = fa.unit_id(key, pop, several)
                qc = self.qc.loc[unit] if self.qc is not None and unit in self.qc.index else None
                lines.append(f"  {self._name(key)}{f' [{pop}]' if several else ''}: {Path(path).name}; "
                             + ('; '.join(_number(qc[c], 0) for c in ('events', 'in_population', 'barcoded'))
                                if qc is not None else 'not counted')
                             + f"; {(s.get('sample_groups') or {}).get(key, '')}"
                             + (f"; {self.library_of(key) or 'no library'}" if self.multi_library() else ''))
        if self.multi_library():
            state = self.plugin.state
            lines += ['', 'Libraries (section 2; each sample\'s combinations named by its library):']
            for lib in state.libraries:
                name = str(lib.get('name', ''))
                n = sum(1 for key in files if self.library_of(key) == name)
                label = f" ({lib['label']})" if lib.get('label') else ''
                lines.append(f"  {name}{label}: {len(lib.get('combos') or [])} guides, combinations file "
                             f"{lib.get('csv_path') or '-'}; {n} sample(s) used")
            lines.append('  Guides with the same name in several libraries: '
                         + ('merged' if state.merge_same_names else 'kept apart, named with their library'))
        lines += ['', 'Statistics:']
        tests = [n for k, n in (
            ('run_counts', f"Counts (NB GLM, {fst.COUNT_TEST_NAMES[s['counts_method']]})"),
            ('run_freq', 'Frequencies (moderated LM, log2 % of library size)'),
            ('run_markers', 'Markers (moderated t of combination minus reference, cluster-first FDR)'))
            if (self.run_counts() if k == 'run_counts' else s[k])]
        pair_key, split_key = s.get(self._design_key('pairing')), s.get(self._design_key('split_by'))
        adjust = [c for c in s.get('adjust') or [] if c not in (pair_key, split_key)]
        factor_name = {fa.PAIR_BY_SAMPLE: 'sample', fa.SPLIT_BY_GROUP: 'sample group', fa.SPLIT_BY_LIBRARY: 'library'}
        if self.compare_mode() == 'populations':
            compare = (f"populations (within each sample); reference population: "
                       f"{s.get('reference_population') or '-'}")
        else:
            compare = 'sample groups' + (' (within each population)' if several else '')
            if several and s['population_contrasts'] and s['contrast_mode'] == 'reference':
                compare += (f"; population contrasts against {s.get('reference_population') or '-'} (the "
                            "change against the reference group compared between populations)")
        normalisation = fst.NORMALISATION_NAMES[s['normalisation']]
        if s['normalisation'] == 'control':
            normalisation += f": {', '.join(s.get('control_combos') or []) or 'none chosen'}"
        clone_rule = (f"removed (more than {s['clone_min_count']} events, more than {s['clone_min_pct']} % and "
                      f"above median + SD of the combination's % over the samples of its group; kept when "
                      f"expanded in more than {s['clone_max_levels']} pairing levels)"
                      if s['remove_clones'] else 'not removed')
        lines += [
            f"  Tests: {', '.join(tests)}",
            f"  Library sizes: {normalisation}",
            f"  Expanded clones: {clone_rule}",
            f"  Compare: {compare}",
            f"  Comparisons: {CONTRAST_MODES[s['contrast_mode']]}; reference: {s.get('reference') or '-'}",
            f"  Pairing: {factor_name.get(pair_key, pair_key) or 'none'}; split by: "
            f"{factor_name.get(split_key, split_key) or 'none'}; adjusted for: {', '.join(adjust) or 'none'}",
            f"  FDR <= {s['pval_threshold']} ({FDR_SCOPES[s['fdr_scope']]}); |log2FC| >= {s['fc_threshold']}; "
            f"TREAT: {s['use_treat']}",
        ]
        markers = (stats or {}).get('markers')
        if s['run_markers']:
            used = markers['reference'] if markers else s['marker_reference']
            lines.append(f"  Markers: {', '.join(markers['channels'] if markers else self.marker_channels()) or 'none'}; "
                         f"reference: {fst.MARKER_REFERENCES.get(used, used)}; min. events "
                         f"{s['marker_min_events']}; |Δ| >= {s['marker_threshold']} (transformed units)")
            if not s['marker_spillover'] or (markers and not markers.get('spillover_k')):
                rule = 'not accounted for (moderated standard errors)'
            else:
                rule = ("added to the standard error, SE_total^2 = SE_moderated^2 + (k x spillover)^2, with the "
                        "tags' spillover spread into the marker in transformed units; t, p and CI from SE_total; ")
                if markers:
                    rule += self.marker_k_text()
                elif s['marker_spillover_k'] == 'auto':
                    rule += 'k calibrated on the control combinations'
                else:
                    rule += f"k = {s['marker_spillover_k']:g}"
                if markers and markers.get('spillover_k_auto'):
                    rule += ('; k is the smallest of 0 to 4 (steps of 0.25) at which the control combinations '
                             'have no significant marker hits')
            lines.append(f'  Marker spillover: {rule}')
        if s['run_pathways']:
            lines += fpw.settings_lines((stats or {}).get('pathways'), {
                'evidence': s['pathway_evidence'], 'min_size': s['pathway_min_size'],
                'rotations': s['pathway_rotations'], 'overrides': s.get('pathway_names') or {}})
        if s['run_gene_groups']:
            table = self.gene_group_frame()
            lines += fgg.settings_lines((stats or {}).get('gene_groups'), {
                'file': s.get('gene_group_file'), 'digest': fgg.table_digest(table) if table is not None else '',
                'groups': len(self.gene_group_order()), 'min_size': s['pathway_min_size'],
                'rotations': s['pathway_rotations']})
            if (stats or {}).get('gene_groups'):
                lines.append('  Gene group spider comparisons: ' + ', '.join(self.spider_comparisons()))
        if stats is not None:
            lines.append('  Comparisons run: ' + ', '.join(stats['comparisons']))
            if markers:
                lines.append('  Marker groups tested: ' + ', '.join(markers['comparisons']))
            lines += [f'  Note: {n}' for n in (self.filtered.notes if self.filtered else []) + (stats.get('notes') or [])]
        return '\n'.join(lines) + '\n'

    def start_export(self):
        if self.counts is None or self.filtered is None:
            self.export_status.setText('Count events first (Data).')
            return
        if self.stats is not None and self._stats_stale:
            self.export_status.setText('The results are out of date: run statistics again before exporting.')
            return
        no_fret = self.uncorrected()
        if no_fret and not self.plugin.confirm_no_fret('Export the analysis report of samples debarcoded without '
                                                       'FRET correction?'):
            self.export_status.setText('Export cancelled.')
            return
        population = self.counted_population()
        folder = fst.default_export_folder(self.analysis_root(), population)
        stats = self.stats
        s = dict(self.settings)
        is_dark = False
        labels = self.unit_labels()
        groups = self._unit_groups()
        colours = self.group_colours()
        qc, filtered, counts, meta = self.qc, self.filtered, self.counts, self.meta
        settings_text = self.settings_text() + self.shift_view.settings_text()
        sample_table = self.sample_table_frame()

        figures = []
        if qc is not None:
            figures.append(('QC', lambda: fst.qc_figure(qc, filtered.counts, groups, is_dark, population,
                                                        group_colours=colours)))
        if stats is not None:
            # Each set of figures is drawn once, when its first figure is
            # written; only the set being written is held.
            drawn = {}
            marker_ref = fst.MARKER_REFERENCES.get((stats.get('markers') or {}).get('reference'), '')

            def figure_of(which, key):
                if which not in drawn:
                    drawn.clear()
                    if which == 'summary':
                        drawn[which] = fst.summary_figures(stats, is_dark, population, s['heatmap_order'],
                                                           int(s['heatmap_rows']), int(s['n_loadings']),
                                                           labels, group_colours=colours)
                    elif len(which) == 2:
                        drawn[which] = fst.marker_figures(stats['markers'], which[1], s['pval_threshold'],
                                                          s['marker_threshold'], s['fdr_scope'], is_dark,
                                                          population, marker_ref, s['heatmap_order'])
                    else:
                        drawn[which] = fst.comparison_figures(stats, which, s['pval_threshold'],
                                                              s['fc_threshold'], s['fdr_scope'], is_dark,
                                                              population, labels, group_colours=colours)
                return drawn[which].get(key, {}).get('fig')

            unit_library = stats.get('libraries')
            lib_labels = stats.get('library_labels') or {}
            pca_figures = ([(f'sample_pca {lib}', f'Sample PCA ({lib_labels.get(lib, lib)})')
                            for lib in dict.fromkeys(unit_library.values())] if unit_library
                           else [('sample_pca', 'Sample PCA')])
            heatmaps = [('comparison_heatmap', 'Comparison heatmap')]
            if stats.get('counts') is not None and stats.get('freq') is not None:
                heatmaps.append(('comparison_heatmap_freq', 'Comparison heatmap (frequencies)'))
            for key, name in heatmaps + pca_figures:
                figures.append((name, lambda key=key: figure_of('summary', key)))
            for pair in stats['pairs']:
                label = fst.comparison_label(*pair)
                for key in ('counts_volcano', 'freq_volcano', 'fold_change_bars', 'heatmap_significant',
                            'heatmap_all'):
                    figures.append((f'{label} - {key.replace("_", " ")}',
                                    lambda pair=pair, key=key: figure_of(pair, key)))
            for label in (stats.get('markers') or {}).get('comparisons') or []:
                for key in ('marker_heatmap', 'marker_volcano'):
                    figures.append((f'Markers {label} - {key.replace("marker_", "")}',
                                    lambda label=label, key=key: figure_of(('markers', label), key)))
            pathways = stats.get('pathways')
            threshold = float(s['pval_threshold'])
            if pathways:
                figures.append(('Pathway dot plot', lambda: fpw.dot_plot_figure(
                    pathways, stats['comparisons'], threshold, is_dark, population)))
                for label in stats['comparisons']:
                    ci = self._pathway_ci(stats, label)
                    for key in fpw.export_sets(pathways, label, threshold):
                        figures.append((f'{label} - pathway {fpw.set_name(pathways, key)}',
                                        lambda label=label, key=key, ci=ci: fpw.members_figure(
                                            pathways, label, key, ci, threshold, is_dark, population)))
            groups_result = self._group_result()
            if groups_result:
                group_colours = self.gene_group_colours()
                spider = self.spider_comparisons()
                spider_colours = fgg.comparison_colours(spider, stats.get('pairs'), colours)
                min_size = groups_result['settings'].get('min_size')
                if groups_result.get('allocation') is not None:
                    figures.append(('Guide allocation', lambda: fgg.donut_figure(
                        groups_result['allocation'], group_colours, is_dark, population,
                        s.get('gene_group_areas') or {})))
                figures.append(('Gene group spider', lambda: fgg.spider_figure(
                    groups_result, spider, spider_colours, group_colours, threshold, is_dark, population, min_size)))
                for label in stats['comparisons']:
                    ci = self._pathway_ci(stats, label)
                    for key in fgg.export_groups(groups_result, label, threshold):
                        figures.append((f'{label} - gene group {fpw.set_name(groups_result, key)}',
                                        lambda label=label, key=key, ci=ci: fpw.members_figure(
                                            groups_result, label, key, ci, threshold, is_dark, population,
                                            groups=True)))

        figures += self.shift_view.export_figures(is_dark)
        store = self.figure_options_store()
        if store is not None:
            from honeychrome.view_components import figure_options as fo
            figures = [(name, fo.with_options(make, fo.FigureOptionsRef(
                store, fcfp.OPTIONS_SCOPE, self._report_figure_key(name))))
                for name, make in figures]
        shift_tables = self.shift_view.write_tables

        def work(progress):
            written = fst.write_analysis_folder(folder, population=population, raw_counts=counts, meta=meta,
                                                qc=qc, filtered=filtered, stats=stats, sample_table=sample_table,
                                                settings_text=settings_text, figures=figures, sample_labels=labels,
                                                progress=progress, no_fret=no_fret,
                                                fdr_threshold=float(s['pval_threshold']))
            written = list(written) + list(shift_tables(folder))
            progress(len(figures), len(figures), '')
            return written

        if self.plugin._start_task('analysis_export', work):
            self.export_btn.setEnabled(False)
            self.export_status.setText(f'Writing {folder}…')
            self._export_folder = folder

    def on_export_finished(self, written):
        self.export_btn.setEnabled(True)
        folder = getattr(self, '_export_folder', None)
        self.export_status.setText(f'Wrote {len(written)} file(s) to {folder}')
        if self.plugin.bus is not None:
            self.plugin.bus.statusMessage.emit(f'FlowCode analysis exported to {folder}')

    def on_export_error(self, message):
        self.export_btn.setEnabled(True)
        self.export_status.setText(f'Export failed: {message}')

    # =====================================================================
    # Refresh
    # =====================================================================

    def on_task_finished(self, key, result):
        if key in fshv.TASKS:
            self.shift_view.on_finished(key, result)
            return
        {'analysis_count': self.on_count_finished, 'analysis_stats': self.on_stats_finished,
         'analysis_figures': self.on_figures_finished, 'analysis_export': self.on_export_finished,
         'analysis_loo': self.on_loo_finished, 'analysis_restore': self.on_restore_finished,
         fcgv.TASK: self.gating_view.on_finished}[key](result)

    def on_task_error(self, key, message):
        if key in fshv.TASKS:
            self.shift_view.on_error(key, message)
            return
        {'analysis_count': self.on_count_error, 'analysis_stats': self.on_stats_error,
         'analysis_figures': self.on_figures_error, 'analysis_export': self.on_export_error,
         'analysis_loo': self.on_loo_error, 'analysis_restore': self.on_restore_error,
         fcgv.TASK: self.gating_view.on_error}[key](message)

    def on_task_progress(self, key, n, total, text):
        if key in fshv.TASKS:
            self.shift_view.on_progress(key, n, total, text)
        elif key == fcgv.TASK:
            self.gating_view.on_progress(n, total, text)
        elif key == 'analysis_count' and text:
            self.count_status.setText(f'Counting {text}… ({n + 1}/{total})')
        elif key == 'analysis_export' and text:
            self.export_status.setText(f'Writing {text}… ({n + 1}/{total})')
        elif key == 'analysis_loo' and text:
            name = self._loo_labels.get(text, text)
            self.loo_status.setText(f'Leave one out: refit {n + 1} of {total}, without {name}.')

    def refresh_status(self):
        kind, text = self.status_summary()
        self.set_status(kind, text)

    def refresh(self):
        """Bring the page up to date with the experiment (cheap: reads file
        headers only when the folder, samples or combinations changed)."""
        self._needing = None
        try:
            self._scan_links()
        except Exception as exc:
            logger.exception('FlowCode analysis: linking failed')
            self.links = {'files': {}, 'unlinked': []}
            self.link_note.setText(f'Could not read the export folder: {exc}')
        if (self.settings.get('counts') or {}).get('records'):
            if self.counts is None and not self._left_out and self._library_key == self._library_setup_key():
                self._load_counts()
            elif self._library_key != self._library_setup_key():
                # Library assignments, guide names or the merge toggle changed:
                # name the saved counts again.
                self._load_counts()
                self._invalidate_stats()
        self._fill_gate_tree()
        self._fill_phenotype_list()
        self._fill_data()
        self._check_counts_current()
        self._fill_groups()
        self.maybe_auto_suggest_groupings()
        self._fill_stats_choices()
        self._restore_saved_stats()
        self.gating_view.refresh()
        self.shift_view.refresh()
        self.root_label.setText(str(self.analysis_root()))
        self._update_default_buttons()
        self.refresh_status()
