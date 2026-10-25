"""
flowcode_stats.py
-----------------
Statistics and figures of the FlowCode Analysis page. No Qt.

Each feature is a FlowCode combination (a barcode, i.e. a guide); each
sample contributes its count of barcoded events per combination inside the
analysed population, and the denominator is the sample's barcoded events
over the analysed combinations. The tests are Honeychrome's shared ones
(``controller_components.differential_stats``):

* Counts — negative-binomial GLM per combination with the log library size
  as offset: weights each sample by its cell number, so small samples count
  for less. The default. Tested by a quasi-likelihood F test (default) or
  the Wald test.
* Frequencies — moderated linear model on log2 % of barcoded events, half
  an event added to every count. With a small group the half-event floor
  moves with the library size, so prefer Counts when samples differ much
  in size.

Library sizes (``differential_stats.library_sizes``): *Robust* (default)
rescales each sample's barcoded events by a trimmed ratio of sums, so a few
expanded clones do not push every other combination down; *Total* uses the
barcoded events; *Control* uses the chosen control combinations (e.g. the
non-targeting guides) after their own expanded clones are removed.

Expanded clones (optional, the lab R masterscript's rule): within each
non-reference group (tissue), a combination is a clone in a sample when its
% of barcoded events exceeds the median + SD of that combination's % over
the group's samples in which it was seen, with more than *min count* events
and more than *min %*. It is removed in that pairing level (mouse) across
all its samples, unless the same combination is expanded in more than *max
levels* pairing levels of that group.

Contrasts, pairing (blocking) and adjustment covariates are as in the
Honeycluster and Automated Gating plugins. *Split by* runs the same
comparisons within each level of a covariate (e.g. COPD vs healthy within
each tissue) and labels them ``other vs base [level]``; significance is
then recomputed over the stacked table so the FDR scope covers every level.

Several populations: each row is one population of one sample. Library
sizes and clones are worked out within each population, and group
comparisons run within each population (``[population]`` in the label).
Populations can instead be the groups (compared within each sample).
Population contrasts test whether a group's change against the reference
group differs between two populations (``_population_contrasts``).

Markers: each combination's mean marker intensity minus its reference
cells' in the same sample, tested against zero within each group with a
moderated t and cluster-first FDR over combinations (``marker_deltas``,
``marker_statistics``).
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from honeychrome.controller_components import differential_stats as ds

MIN_SAMPLES_PER_GROUP = 3

DEFAULTS = {
    'run_counts': True,
    'run_freq': False,
    'contrast_mode': 'reference',
    'pval_threshold': 0.05,
    'fc_threshold': 1.0,
    'use_treat': False,
    'fdr_scope': 'per_comparison',
    'counts_method': 'ql',
    'normalisation': 'robust',
}

CLONE_DEFAULTS = {'min_count': 200, 'min_pct': 10.0, 'max_levels': 2}
COUNT_TEST_NAMES = {'ql': 'quasi-likelihood F test', 'wald': 'Wald test'}
NORMALISATION_NAMES = {'robust': 'robust (trimmed ratio of sums)', 'total': 'total barcoded events',
                       'control': 'control combinations'}
_NON_TARGETING = re.compile(r'non[\s_-]*target', re.IGNORECASE)
# Internal labels of the population-contrast design (never shown).
_CELL_SEP = '\x1f'
_BASE = '\x1fbase'


def control_candidates(combinations) -> list:
    """Combinations whose name marks them as non-targeting controls."""
    return [c for c in combinations if _NON_TARGETING.search(str(c))]


def expanded_clones(counts: pd.DataFrame, denominator: pd.Series, samples: list,
                    groups: dict | None = None, *,
                    min_count: int = CLONE_DEFAULTS['min_count'],
                    min_pct: float = CLONE_DEFAULTS['min_pct']) -> pd.DataFrame:
    """
    Expanded clones in *samples*, by the lab R masterscript's rule. Within
    each group (tissue) and for each combination, the % of barcoded events
    is compared across the group's samples in which the combination was
    seen: a sample is a clone when its % exceeds the median + SD of those
    percentages, with more than *min_count* events and more than *min_pct*
    %. A combination seen in fewer than two samples of a group has no SD
    and is never a clone there. Without *groups* all *samples* form one
    group. Returns rows sample, combination, count, pct.
    """
    rows = []
    totals = denominator.reindex(samples).astype(float)
    by_group: dict = {}
    for s in samples:
        if totals.get(s, 0.0) > 0:
            by_group.setdefault((groups or {}).get(s, ''), []).append(s)
    for members in by_group.values():
        sub = counts.loc[members].astype(float)
        pct = 100.0 * sub.div(totals.loc[members], axis=0)
        for combo in sub.columns:
            seen = sub[combo].values > 0
            if seen.sum() < 2:
                continue
            values = pct[combo].values[seen]
            limit = np.median(values) + np.std(values, ddof=1)
            hit = seen & (pct[combo].values > limit) & (sub[combo].values > min_count) & (pct[combo].values > min_pct)
            for i in np.flatnonzero(hit):
                rows.append({'sample': members[i], 'combination': combo,
                             'count': int(sub[combo].values[i]), 'pct': float(pct[combo].values[i])})
    return pd.DataFrame(rows, columns=['sample', 'combination', 'count', 'pct'])


def remove_clones(counts: pd.DataFrame, clones: pd.DataFrame, pairing: dict | None,
                  max_levels: int | None = CLONE_DEFAULTS['max_levels'],
                  groups: dict | None = None) -> tuple:
    """
    *counts* with each expanded clone set to NaN in every sample of its
    pairing level (the mouse's other tissues too), or in its own sample when
    unpaired. A combination expanded in more than *max_levels* levels of the
    same group (tissue) is kept there (None: never kept). Returns (counts,
    table) where the table adds 'group', 'level' and 'action' to *clones*.
    """
    out = counts.astype(float).copy()
    table = clones.copy()
    if table.empty:
        return out, table.assign(group=[], level=[], action=[])
    level_of = {s: str(pairing.get(s, '')) if pairing else s for s in counts.index}
    table['group'] = [(groups or {}).get(s, '') for s in table['sample']]
    table['level'] = [level_of.get(s, s) for s in table['sample']]
    n_levels = table.groupby(['group', 'combination'])['level'].nunique()
    actions = []
    for _i, r in table.iterrows():
        n = int(n_levels[(r['group'], r['combination'])])
        if max_levels is not None and n > max_levels:
            actions.append(f'kept (expanded in {n} levels of {r["group"] or "the samples"})')
            continue
        hit = [s for s in counts.index if level_of.get(s) == r['level']] if pairing else [r['sample']]
        out.loc[hit, r['combination']] = np.nan
        actions.append('removed')
    table['action'] = actions
    return out, table


def comparison_label(base: str, other: str, level: str | None = None) -> str:
    label = f"{other} vs {base}"
    return f"{label} [{level}]" if level is not None else label


def _qualifying(samples: list, groups: dict, order: list) -> list:
    n = {}
    for s in samples:
        g = groups.get(s) or ''
        if g:
            n[g] = n.get(g, 0) + 1
    names = list(order) + sorted(g for g in n if g not in order)
    return [g for g in names if n.get(g, 0) >= MIN_SAMPLES_PER_GROUP]


def run_statistics(counts: pd.DataFrame, denominator: pd.Series, groups: dict, *,
                   group_order: list | None = None, contrast_mode: str = 'reference',
                   reference: str = '', pairing: dict | None = None,
                   covariates: pd.DataFrame | None = None, split_by: dict | None = None,
                   pval_threshold: float = 0.05, fc_threshold: float = 1.0,
                   use_treat: bool = False, fdr_scope: str = 'per_comparison',
                   run_counts: bool = True, run_freq: bool = False,
                   counts_method: str = 'ql', normalisation: str = 'robust',
                   control_combos: list | None = None, clone_filter: dict | None = None,
                   strata: dict | None = None, within_strata: bool = True,
                   clone_groups: dict | None = None, clone_reference: str | None = None,
                   population_contrasts: str | None = None, allow_no_comparisons: bool = False) -> dict:
    """
    Test every combination between sample groups.

    counts      : samples × combinations (``flowcode_analysis.filter_counts``).
                  With several populations each row is one population of
                  one sample (a unit), and *strata* names its population.
    denominator : barcoded events per sample over those combinations.
    groups      : {sample: group}; samples without one are left out.
    pairing     : {sample: pairing level} (e.g. mouse), or None.
    covariates  : adjustment covariates indexed by sample, or None.
    split_by    : {sample: level} to run the comparisons within each level.
    counts_method : 'ql' or 'wald' (``differential_stats.run_glm_counts``).
    normalisation : 'robust', 'total' or 'control' (library sizes).
    control_combos: the control combinations for 'control'.
    clone_filter  : None, or {'min_count', 'min_pct', 'max_levels'} to remove
                  expanded clones before testing (module docstring).
    strata      : {sample: population}, or None for one population. Library
                  sizes and expanded clones are worked out within each
                  population.
    within_strata : with *strata*, run the comparisons within each
                  population (levels ``population`` or ``population · level``).
                  False when the groups are the populations themselves.
    clone_groups, clone_reference : the groups the clone rule compares
                  samples within, and the group it does not examine (None:
                  every group). Default: *groups*, and the reference group
                  in reference mode.
    population_contrasts : the reference population, to also test with
                  *strata* (reference contrasts, groups that are not
                  populations) whether each group's change against the
                  reference group differs between every other population and
                  the reference population; see ``_population_contrasts``.
    allow_no_comparisons : return the prepared counts when no comparison
                  can be made (for the marker test alone) instead of raising.

    Returns {'samples', 'group_vec', 'split', 'pairing', 'comparisons'
    (labels), 'pairs' ([(base, other, level)]), 'counts', 'counts_values',
    'freq', 'freq_values', 'log2_values', 'removed', 'denominator',
    'library_sizes', 'clones', 'strata', 'interaction', 'notes'}; a result
    is None when not requested. 'removed' marks the counts left out as
    expanded clones; 'interaction' describes the population contrasts
    ({label: {'units', 'cell', 'line', 'population'}}).
    Raises RuntimeError when no comparison can be made.
    """
    order = list(group_order or [])
    samples = [s for s in counts.index if groups.get(s)]
    if split_by:
        samples = [s for s in samples if split_by.get(s) not in (None, '')]
    strata = {s: str(strata[s]) for s in samples if strata.get(s) not in (None, '')} if strata else None
    if strata is not None:
        samples = [s for s in samples if s in strata]

    def level_of(s):
        parts = []
        if strata is not None and within_strata:
            parts.append(strata[s])
        if split_by:
            parts.append(str(split_by[s]))
        return ' · '.join(parts) if parts else None

    unit_level = {s: level_of(s) for s in samples}
    levels = list(dict.fromkeys(unit_level.values())) if samples else [None]
    notes, pairs = [], []
    frames = {'counts': [], 'freq': []}
    if clone_groups is None:
        clone_groups = groups
        names_all = _qualifying(samples, groups, order)
        clone_reference = (reference if reference in names_all else (names_all[0] if names_all else '')) \
            if contrast_mode == 'reference' else None
    analysed, denom_used, library, clones = _prepare_counts(
        counts.loc[samples], denominator.loc[samples], pairing, normalisation, control_combos, clone_filter,
        notes, strata, clone_groups, clone_reference)
    log2_all = ds.log2_frequencies(analysed, library.values)

    for level in levels:
        level_samples = [s for s in samples if unit_level[s] == level]
        names = _qualifying(level_samples, groups, order)
        where = '' if level is None else f' in {level}'
        if len(names) < 2:
            notes.append(f"Fewer than two groups with {MIN_SAMPLES_PER_GROUP} or more samples{where}: "
                         "no comparison made.")
            continue
        ref = reference if reference in names else names[0]
        contrasts = ds.build_contrasts(names, contrast_mode, ref)
        used = [s for s in level_samples if groups.get(s) in names]
        group_vec = [groups[s] for s in used]
        pair_vec = [str((pairing or {}).get(s, '')) for s in used] if pairing else None
        cov = None
        if covariates is not None and len(covariates.columns):
            cov = ds.covariate_frame(covariates, used, list(covariates.columns))
        common = dict(pairing_vec=pair_vec, fdr_scope=fdr_scope, covariates=cov, use_treat=use_treat)
        sub = analysed.loc[used]
        size = library.loc[used]
        try:
            if run_counts:
                res = ds.run_glm_counts(sub, group_vec, contrasts, contrast_mode, pval_threshold,
                                        fc_threshold, offset_df=_offsets(sub, size), method=counts_method,
                                        **common)
                frames['counts'].append(_relabel(res, level))
            if run_freq:
                res = ds.run_moderated(log2_all.loc[used], group_vec, contrasts, contrast_mode,
                                       pval_threshold, fc_threshold, **common)
                frames['freq'].append(_relabel(res, level))
        except RuntimeError as exc:
            notes.append(f"Comparisons{where} not made: {exc}")
            continue
        for base, other in contrasts:
            pairs.append((base, other, level))

    interaction = {}
    if population_contrasts and strata is not None and contrast_mode == 'reference':
        interaction = _population_contrasts(
            analysed, log2_all, library, samples, groups, order, reference, pairing, covariates, split_by,
            strata, population_contrasts, run_counts, run_freq, counts_method, frames, pairs, notes,
            dict(pval_threshold=pval_threshold, fc_threshold=fc_threshold, fdr_scope=fdr_scope,
                 use_treat=use_treat))

    if not pairs and not allow_no_comparisons:
        raise RuntimeError(' '.join(notes) or 'No comparison could be made.')

    def stacked(key):
        if not frames[key]:
            return None
        table = pd.concat(frames[key], ignore_index=True)
        return ds.apply_significance(table, pval_threshold, fc_threshold, fdr_scope=fdr_scope,
                                     use_treat=use_treat)

    kept = [s for s in samples if groups.get(s)]
    return {
        'samples': kept,
        'group_vec': [groups[s] for s in kept],
        'split': {s: unit_level[s] for s in kept} if any(v is not None for v in unit_level.values()) else None,
        'pairing': {s: str(pairing[s]) for s in kept if pairing.get(s) not in (None, '')} if pairing else None,
        'comparisons': [comparison_label(b, o, lv) for b, o, lv in pairs],
        'pairs': pairs,
        'counts': stacked('counts'),
        'counts_values': counts.loc[kept] if run_counts else None,
        'freq': stacked('freq'),
        'freq_values': (100.0 * counts.loc[kept].div(denominator.loc[kept], axis=0)),
        'log2_values': ds.log2_frequencies(counts.loc[kept], library.loc[kept].values),
        'removed': analysed.loc[kept].isna(),
        'denominator': denom_used.loc[kept],
        'library_sizes': library.loc[kept],
        'clones': clones,
        'strata': {s: strata[s] for s in kept} if strata is not None else None,
        'interaction': interaction,
        'counts_method': counts_method,
        'normalisation': normalisation,
        'notes': notes,
    }


def _offsets(sub: pd.DataFrame, size: pd.Series) -> pd.DataFrame:
    """The library size of each sample, repeated for every combination."""
    return pd.DataFrame(np.repeat(size.values[:, None], sub.shape[1], axis=1), index=sub.index,
                        columns=sub.columns)


def _population_contrasts(analysed, log2_all, library, samples, groups, order, reference, pairing,
                          covariates, split_by, strata, ref_pop, run_counts, run_freq, counts_method,
                          frames, pairs, notes, thresholds) -> dict:
    """
    Population contrasts: for every population P other than *ref_pop* and
    every group g other than the reference group, does g's change against
    the reference group differ between P and *ref_pop*? The tested effect is
    the interaction (log P g − log P ref) − (log ref_pop g − log ref_pop ref),
    e.g. whether a guide is more enriched in the pancreas (against the
    spleen) among CD8 than among Tconv cells.

    One fit per P (and split level) over the units of P and *ref_pop*. The
    tested coefficients are indicators of P's units in each group g; with
    pairing, each pairing level (mouse) has its own baseline per population
    and its own effect of each group shared by both populations, so every
    mouse is compared with itself (the GLM counterpart of the lab scripts'
    paired t-test of spleen-normalised frequencies); without pairing the
    population and group main effects take their place. Comparisons are
    labelled ``P vs ref_pop [g vs ref]`` and added to *frames* and *pairs*.
    Returns {label: {'units', 'cell', 'line', 'population'}} for the
    figures.
    """
    out = {}
    populations = list(dict.fromkeys(strata[s] for s in samples))
    if ref_pop not in populations:
        notes.append(f'Population contrasts: the reference population {ref_pop} has no samples.')
        return out
    split_levels = list(dict.fromkeys(str(split_by[s]) for s in samples)) if split_by else [None]
    for pop in populations:
        if pop == ref_pop:
            continue
        for lvl in split_levels:
            units = [s for s in samples if strata[s] in (pop, ref_pop)
                     and (lvl is None or str(split_by[s]) == lvl)]
            where = f' ({pop} vs {ref_pop}' + (f', {lvl})' if lvl is not None else ')')
            ref = reference
            cell_n = {}
            for s in units:
                cell_n[(groups[s], strata[s])] = cell_n.get((groups[s], strata[s]), 0) + 1
            names = [g for g in _qualifying(units, groups, order)
                     if cell_n.get((g, pop), 0) >= MIN_SAMPLES_PER_GROUP
                     and cell_n.get((g, ref_pop), 0) >= MIN_SAMPLES_PER_GROUP]
            if ref not in names:
                notes.append(f'Population contrasts{where}: the reference group {ref or "-"} needs '
                             f'{MIN_SAMPLES_PER_GROUP} or more samples in both populations.')
                continue
            tested = [g for g in names if g != ref]
            if not tested:
                continue
            used = [s for s in units if groups[s] in names]
            cells = [f'{groups[s]}{_CELL_SEP}{pop}' if (groups[s] != ref and strata[s] == pop) else _BASE
                     for s in used]
            cov_cols = {}
            if pairing:
                mouse = [str(pairing.get(s, '')) for s in used]
                pair_vec = [f'{m}{_CELL_SEP}{strata[s]}' for m, s in zip(mouse, used)]
                cov_cols['group within pairing level'] = [
                    f'{m}{_CELL_SEP}{groups[s]}' if groups[s] != ref else _BASE for m, s in zip(mouse, used)]
            else:
                pair_vec = None
                cov_cols['population'] = [strata[s] for s in used]
                cov_cols['group'] = [groups[s] for s in used]
            cov = pd.DataFrame(cov_cols, index=used)
            if covariates is not None and len(covariates.columns):
                cov = cov.join(ds.covariate_frame(covariates, used, list(covariates.columns)))
            contrasts = [(_BASE, f'{g}{_CELL_SEP}{pop}') for g in tested]
            labels = {f'{g}{_CELL_SEP}{pop} vs {_BASE}': comparison_label(ref_pop, pop, f'{g} vs {ref}'
                                                                          + (f' · {lvl}' if lvl else ''))
                      for g in tested}
            common = dict(pairing_vec=pair_vec, covariates=cov, **thresholds)
            sub = analysed.loc[used]
            try:
                if run_counts:
                    res = ds.run_glm_counts(sub, cells, contrasts, 'reference',
                                            offset_df=_offsets(sub, library.loc[used]), method=counts_method,
                                            **common)
                    frames['counts'].append(res.assign(comparison=res['comparison'].map(labels)))
                if run_freq:
                    res = ds.run_moderated(log2_all.loc[used], cells, contrasts, 'reference', **common)
                    frames['freq'].append(res.assign(comparison=res['comparison'].map(labels)))
            except RuntimeError as exc:
                notes.append(f'Population contrasts{where} not made: {exc}')
                continue
            for g in tested:
                level = f'{g} vs {ref}' + (f' · {lvl}' if lvl else '')
                pairs.append((ref_pop, pop, level))
                members = [s for s in used if groups[s] in (g, ref)]
                out[comparison_label(ref_pop, pop, level)] = {
                    'units': members,
                    'cell': {s: f'{strata[s]} · {groups[s]}' for s in members},
                    'line': ({s: f'{pairing.get(s, "")} · {strata[s]}' for s in members} if pairing else {}),
                    'population': {s: strata[s] for s in members},
                    'order': [f'{p} · {x}' for p in (ref_pop, pop) for x in (ref, g)],
                    'colour_group': {f'{p} · {x}': x for p in (ref_pop, pop) for x in (ref, g)},
                }
    return out


def _prepare_counts(counts: pd.DataFrame, denominator: pd.Series, pairing: dict | None, normalisation: str,
                    control_combos: list | None, clone_filter: dict | None, notes: list,
                    strata: dict | None, clone_groups: dict, clone_reference: str | None) -> tuple:
    """
    Counts with expanded clones removed (when *clone_filter*), the barcoded
    events left per sample, the library size per sample and the clone table
    (None without a filter). Each population of *strata* is handled on its
    own; clones are looked for in the samples whose clone group is not
    *clone_reference*, compared within their clone group.
    """
    if strata is None:
        return _prepare_stratum(counts, denominator, pairing, normalisation, control_combos, clone_filter,
                                notes, clone_groups, clone_reference, '')
    parts = []
    for pop in dict.fromkeys(strata[s] for s in counts.index):
        rows = [s for s in counts.index if strata[s] == pop]
        parts.append(_prepare_stratum(counts.loc[rows], denominator.loc[rows], pairing, normalisation,
                                      control_combos, clone_filter, notes, clone_groups, clone_reference,
                                      f'{pop}: '))
    analysed = pd.concat([p[0] for p in parts]).loc[counts.index]
    denom_used = pd.concat([p[1] for p in parts]).loc[counts.index]
    library = pd.concat([p[2] for p in parts]).loc[counts.index]
    tables = [p[3].assign(population=pop) for p, pop in zip(parts, dict.fromkeys(strata[s] for s in counts.index))
              if p[3] is not None]
    clones = pd.concat(tables, ignore_index=True) if tables else None
    return analysed, denom_used, library, clones


def _prepare_stratum(counts, denominator, pairing, normalisation, control_combos, clone_filter, notes,
                     clone_groups, clone_reference, prefix) -> tuple:
    examined = [s for s in counts.index
                if clone_groups.get(s) and (clone_reference is None or clone_groups.get(s) != clone_reference)]
    settings = dict(CLONE_DEFAULTS, **(clone_filter or {}))
    found = None
    if clone_filter is not None or normalisation == 'control':
        found = expanded_clones(counts, denominator, examined, clone_groups,
                                min_count=int(settings['min_count']), min_pct=float(settings['min_pct']))

    analysed, clones = counts.astype(float), None
    if clone_filter is not None:
        analysed, clones = remove_clones(counts, found, pairing, int(settings['max_levels']), clone_groups)
        removed = clones[clones['action'] == 'removed']
        kept = clones[clones['action'] != 'removed']
        notes.append(f"{prefix}Expanded clones: {len(removed)} removed, {len(kept)} kept (expanded in more "
                     f"than {int(settings['max_levels'])} pairing levels of one group).")
    removed_events = counts.astype(float).where(analysed.isna(), 0.0).sum(axis=1)
    denom_used = denominator.astype(float) - removed_events

    if normalisation == 'control':
        controls = [c for c in (control_combos or []) if c in counts.columns]
        if not controls:
            raise RuntimeError('Control normalisation: choose at least one control combination.')
        own = found[found['combination'].isin(controls)]
        ctrl_counts, _table = remove_clones(analysed[controls], own, pairing, None)
        sizes = ds.library_sizes(ctrl_counts, denom_used.values, 'control', controls)
        if len(own):
            notes.append(f"{prefix}Control normalisation: {len(own)} expanded control clone(s) left out of "
                         "the control totals.")
    else:
        sizes = ds.library_sizes(analysed, denom_used.values, normalisation)
    library = pd.Series(sizes, index=counts.index)
    factors = library / denom_used.replace(0.0, np.nan)
    if normalisation != 'total' and factors.notna().any():
        notes.append(f"{prefix}Library sizes: {NORMALISATION_NAMES[normalisation]}; factors "
                     f"{factors.min():.2f}–{factors.max():.2f} of the barcoded events.")
    return analysed, denom_used, library, clones


def _relabel(results: pd.DataFrame, level: str | None) -> pd.DataFrame:
    if level is None:
        return results
    out = results.copy()
    out['comparison'] = out['comparison'] + f' [{level}]'
    return out


def reapply_thresholds(stats: dict, pval_threshold: float, fc_threshold: float, fdr_scope: str,
                       use_treat: bool) -> dict:
    """*stats* with p-values, FDR and significance recomputed for new
    thresholds, without refitting."""
    out = dict(stats)
    for key in ('counts', 'freq'):
        if out.get(key) is not None:
            out[key] = ds.apply_significance(out[key], pval_threshold, fc_threshold,
                                             fdr_scope=fdr_scope, use_treat=use_treat)
    return out


def primary_results(stats: dict) -> tuple:
    """(key, results) of the test the summary figures use: counts when run,
    otherwise frequencies."""
    for key in ('counts', 'freq'):
        if stats.get(key) is not None:
            return key, stats[key]
    return None, None


def comparison_samples(stats: dict, pair: tuple) -> list:
    """Samples of one comparison (its two groups, within its split level;
    for a population contrast, both populations in both groups)."""
    contrast = (stats.get('interaction') or {}).get(comparison_label(*pair))
    if contrast is not None:
        return list(contrast['units'])
    base, other, level = pair
    split = stats.get('split') or {}
    return [s for s, g in zip(stats['samples'], stats['group_vec'])
            if g in (base, other) and (level is None or split.get(s) == level)]


def results_table(stats: dict) -> pd.DataFrame | None:
    """Every test's results in one long table, with a 'test' column."""
    frames = []
    counts_name = f"Counts (NB GLM, {COUNT_TEST_NAMES.get(stats.get('counts_method') or 'wald')})"
    for key, name in (('counts', counts_name), ('freq', 'Frequencies (moderated LM)')):
        if stats.get(key) is not None:
            frames.append(stats[key].assign(test=name))
    return pd.concat(frames, ignore_index=True) if frames else None


# ---------------------------------------------------------------------------
# Sample PCA
# ---------------------------------------------------------------------------

def sample_pca(stats: dict, n_loadings: int = 10) -> dict | None:
    """
    Two-component PCA of every sample on log2 % of barcoded events
    (z-scored per combination), with the top loadings by length. Same
    result layout as ``ag_results.sample_pca``, for
    ``ag_report.make_pca_figure``. None with fewer than 2 samples or 2
    varying combinations.
    """
    from sklearn.decomposition import PCA

    values = stats.get('log2_values')
    if values is None or values.shape[0] < 2:
        return None
    sd = values.std(axis=0, ddof=0)
    values = values.loc[:, sd > 1e-12]
    if values.shape[1] < 2:
        return None
    z = (values - values.mean(axis=0)) / values.std(axis=0, ddof=0)
    n_comp = min(2, z.shape[0], z.shape[1])
    pca = PCA(n_components=n_comp)
    pcs = pca.fit_transform(z.values)
    explained = list(pca.explained_variance_ratio_[:n_comp]) + [0.0] * (2 - n_comp)
    if n_comp < 2:
        pcs = np.hstack([pcs, np.zeros((pcs.shape[0], 2 - n_comp))])
    scores = pd.DataFrame(pcs[:, :2], index=z.index, columns=['PC1', 'PC2'])
    load = np.zeros((z.shape[1], 2))
    for c in range(n_comp):
        load[:, c] = pca.components_[c] * np.sqrt(max(pca.explained_variance_[c], 0.0))
    loadings = pd.DataFrame(load, index=z.columns, columns=['PC1', 'PC2'])
    length = np.hypot(loadings['PC1'], loadings['PC2'])
    loadings = loadings.loc[length.sort_values(ascending=False).head(max(1, int(n_loadings))).index]
    group_of = dict(zip(stats['samples'], stats['group_vec']))
    return {'scores': scores, 'loadings': loadings,
            'explained_variance_ratio': (float(explained[0]), float(explained[1])),
            'groups': [group_of.get(s, '') for s in scores.index], 'sources': ['log2 % of barcoded']}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

TEST_NAMES = {'counts': 'Counts', 'freq': 'Frequencies'}
X_LABELS = {'counts': 'log2 fold change (barcode share, NB GLM)',
            'freq': 'log2 fold change (% of barcoded)'}


def comparison_figures(stats: dict, pair: tuple, pval_threshold: float, fc_threshold: float,
                       fdr_scope: str, is_dark: bool, population: str,
                       sample_labels: dict | None = None, group_colours: dict | None = None,
                       combination: str | None = None, group_order: list | None = None) -> dict:
    """
    Figures for one comparison: a volcano per test run, the fold-change bars
    of the primary test, the per-sample heatmap of significant combinations,
    the per-sample heatmap of every combination and, with *combination*, that
    combination's per-sample plot. Returns {key: {'fig', 'title', 'error',
    'results', 'make'}}; 'results' is the table a volcano or the bars were
    drawn from, 'make' draws a fresh copy of the figure (for Pop Out).
    *group_colours* ({group: colour}) colour the groups in every figure.
    """
    from honeychrome.view_components import differential_plots as dp

    base, other, level = pair
    label = comparison_label(base, other, level)
    keep = comparison_samples(stats, pair)
    contrast = (stats.get('interaction') or {}).get(label)
    group_by_sample = (dict(contrast['population']) if contrast is not None
                       else dict(zip(stats['samples'], stats['group_vec'])))
    values = stats['freq_values']
    sample_df = values.loc[[s for s in keep if s in values.index]]
    if sample_labels:
        sample_df = sample_df.rename(index=lambda s: sample_labels.get(s, s))
        group_by_sample = {sample_labels.get(s, s): g for s, g in group_by_sample.items()}
    run_label = f'{population}: {label}'
    colours = dict(group_colours or {})
    out = {}

    def add(key, tab_title, maker, results=None, **kwargs):
        try:
            out[key] = {'fig': maker(**kwargs), 'title': tab_title, 'error': None, 'results': results,
                        'make': lambda: maker(**kwargs)}
        except Exception as exc:
            out[key] = {'fig': None, 'title': tab_title, 'error': str(exc) or type(exc).__name__,
                        'results': None}

    first, first_key = None, None
    for key in ('counts', 'freq'):
        res = stats.get(key)
        if res is None:
            continue
        view = res[res['comparison'] == label].reset_index(drop=True)
        if view.empty:
            continue
        if first is None:
            first, first_key = view, key
        add(f'{key}_volcano', f'{TEST_NAMES[key]} volcano', dp.make_volcano_figure, results=view,
            results_df=view, title=f'{TEST_NAMES[key]} volcano: {label}', pval_threshold=pval_threshold,
            fc_threshold=fc_threshold, is_dark=is_dark, fdr_scope=fdr_scope, x_label=X_LABELS[key],
            run_label=run_label, group_a=base, group_b=other, group_colours=colours)
    if first is not None:
        add('fold_change_bars', 'Fold-change bars', fold_change_bars_figure, results=first,
            results_df=first, title=f'{TEST_NAMES[first_key]}: {label}', group_a=base, group_b=other,
            fc_threshold=fc_threshold, fdr_scope=fdr_scope, is_dark=is_dark, run_label=run_label,
            x_label=X_LABELS[first_key], group_colours=colours)
        add('heatmap_significant', 'Heatmap (significant)', dp.make_heatmap_figure,
            results_df=first, sample_df=sample_df, title=f'Significant combinations: {label}',
            group_by_sample=group_by_sample, group_a=base, group_b=other, is_dark=is_dark,
            run_label=run_label, value_label='% of barcoded', group_colours=colours)
        add('heatmap_all', 'Heatmap (all)', dp.make_heatmap_figure,
            results_df=first.assign(significant=True), sample_df=sample_df,
            title=f'All combinations: {label}', group_by_sample=group_by_sample, group_a=base,
            group_b=other, is_dark=is_dark, run_label=run_label, value_label='% of barcoded',
            group_colours=colours)
    if combination:
        add('combination', f'Combination: {combination}', combination_figure, stats=stats,
            combination=combination, pair=pair, fdr_scope=fdr_scope, is_dark=is_dark, population=population,
            sample_labels=sample_labels, group_colours=colours, group_order=group_order)
    return out


def summary_figures(stats: dict, is_dark: bool, population: str, heatmap_order: str = 'effect',
                    heatmap_rows: int = 1, n_loadings: int = 10,
                    sample_labels: dict | None = None, group_colours: dict | None = None) -> dict:
    """Figures over every comparison: the comparison heatmap of the
    primary test and the sample PCA ({key: {'fig', 'title', 'error',
    'make'}})."""
    from honeychrome.view_components import differential_plots as dp

    _bundled_plugins_on_path()
    import ag_report

    out = {}
    key, res = primary_results(stats)
    if res is not None:
        def make_heatmap():
            return dp.make_comparison_heatmap_figure(
                res, f'{population}: {TEST_NAMES[key].lower()}, all comparisons', is_dark=is_dark,
                order=heatmap_order, n_rows=heatmap_rows, comparison_order=stats['comparisons'])

        try:
            out['comparison_heatmap'] = {'fig': make_heatmap(), 'title': 'Comparison heatmap', 'error': None,
                                         'make': make_heatmap}
        except Exception as exc:
            out['comparison_heatmap'] = {'fig': None, 'title': 'Comparison heatmap',
                                         'error': str(exc) or type(exc).__name__}
    pca = sample_pca(stats, n_loadings=n_loadings)
    if pca is not None:
        def make_pca():
            return ag_report.make_pca_figure(pca, is_dark, sample_labels=sample_labels,
                                             group_colours=group_colours)

        try:
            out['sample_pca'] = {'fig': make_pca(), 'title': 'Sample PCA', 'error': None, 'make': make_pca}
        except Exception as exc:
            out['sample_pca'] = {'fig': None, 'title': 'Sample PCA', 'error': str(exc) or type(exc).__name__}
    return out


NO_GROUP_COLOUR = '#7f7f7f'
P_NAMES = {'stagewise.adj.P.Val': 'stage-wise adj. P', 'adj.P.Val.global': 'adj. P (pooled)',
           'adj.P.Val': 'adj. P', 'P.Value': 'P'}


def _colour_of(group, group_colours: dict | None, index: int) -> str:
    """A group's colour from *group_colours*, else the next tab10 colour;
    samples without a group are grey."""
    import matplotlib

    if not group or group == 'no group':
        return NO_GROUP_COLOUR
    colour = (group_colours or {}).get(group)
    if colour:
        return colour
    return matplotlib.colors.to_hex(matplotlib.colormaps['tab10'](index % 10))


def qc_figure(qc: pd.DataFrame, counts: pd.DataFrame, groups: dict, is_dark: bool, population: str,
              group_colours: dict | None = None):
    """Barcoded events per combination summed over the samples of each
    group (stacked, in the group colours), and the barcoded events per
    sample."""
    from matplotlib.figure import Figure

    from honeychrome.view_components import differential_plots as dp

    fig = Figure(figsize=(max(6.0, 0.22 * counts.shape[1] + 2.0), 6.5), constrained_layout=True)
    ax1, ax2 = fig.subplots(2, 1)
    group_names = list(dict.fromkeys(groups.get(s) or 'no group' for s in counts.index))
    bottom = np.zeros(counts.shape[1])
    colour_of = {g: _colour_of(g, group_colours, i) for i, g in enumerate(group_names)}
    for g in group_names:
        rows = [s for s in counts.index if (groups.get(s) or 'no group') == g]
        vals = counts.loc[rows].sum(axis=0).values.astype(float)
        ax1.bar(range(counts.shape[1]), vals, bottom=bottom, color=colour_of[g], label=g)
        bottom += vals
    ax1.set_xticks(range(counts.shape[1]))
    ax1.set_xticklabels(list(counts.columns), rotation=90, fontsize=6)
    ax1.set_ylabel('Barcoded events')
    ax1.set_title(f'{population}: events per combination', fontsize=10)
    ax1.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False)
    names = [str(qc.loc[s, 'name']) if s in qc.index else str(s) for s in counts.index]
    ax2.bar(range(len(names)), counts.sum(axis=1).values.astype(float),
            color=[colour_of[groups.get(s) or 'no group'] for s in counts.index])
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=90, fontsize=6)
    ax2.set_ylabel('Barcoded events')
    ax2.set_yscale('log')
    ax2.set_title('Barcoded events per sample', fontsize=10)
    dp.style_figure_theme(fig, is_dark)
    return fig


def fold_change_bars(results_df: pd.DataFrame) -> pd.DataFrame:
    """The bars of ``fold_change_bars_figure``: one row per combination with
    a finite log2FC, ordered from most depleted to most enriched."""
    table = results_df[np.isfinite(results_df['logFC'].astype(float))]
    return table.sort_values('logFC', kind='mergesort').reset_index(drop=True)


def fold_change_bars_figure(results_df: pd.DataFrame, title: str, group_a: str, group_b: str,
                            fc_threshold: float, fdr_scope: str, is_dark: bool, run_label: str = '',
                            x_label: str = 'log2 fold change', group_colours: dict | None = None):
    """
    Depletion and enrichment of every combination in one comparison: a bar
    per combination at its log2 fold change (*group_b* over *group_a*),
    ordered from most depleted to most enriched, in the colour of the group
    it is higher in; significant bars are solid, the others faint. The 95%
    CI is drawn as a whisker and the |log2FC| threshold as dashed lines.
    ``fig._bar_features`` lists the combinations left to right, for click
    look-up.
    """
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    from honeychrome.view_components import differential_plots as dp

    bars = fold_change_bars(results_df)
    if bars.empty:
        return dp.message_figure('No combination has a fold change.', title, is_dark, run_label)
    n = len(bars)
    colour_a, colour_b = dp.group_pair_colours(group_a, group_b, group_colours)
    fig = Figure(figsize=(max(5.0, 0.16 * n + 1.6), 4.6), constrained_layout=True)
    ax = fig.add_subplot(111)
    fc = bars['logFC'].values.astype(float)
    sig = bars['significant'].values.astype(bool) if 'significant' in bars.columns else np.zeros(n, bool)
    colours = [colour_b if v > 0 else colour_a for v in fc]
    x = np.arange(n)
    ax.bar(x, fc, color=colours, alpha=None, edgecolor=colours, linewidth=0.6)
    for patch, significant in zip(ax.patches, sig):
        if not significant:
            patch.set_alpha(0.3)
    if {'CI.L', 'CI.R'} <= set(bars.columns):
        lo = bars['CI.L'].values.astype(float)
        hi = bars['CI.R'].values.astype(float)
        ok = np.isfinite(lo) & np.isfinite(hi)
        fg = 'white' if is_dark else 'black'
        ax.errorbar(x[ok], fc[ok], yerr=[fc[ok] - lo[ok], hi[ok] - fc[ok]], fmt='none', ecolor=fg,
                    elinewidth=0.6, capsize=1.5, alpha=0.6)
    for y in (-fc_threshold, fc_threshold):
        if fc_threshold > 0:
            ax.axhline(y, color='#7f7f7f', linestyle='--', linewidth=0.8)
    ax.axhline(0.0, color='#7f7f7f', linewidth=0.8)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(bars['feature'].astype(str).tolist(), rotation=90, fontsize=6)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_ylabel(x_label, fontsize=9)
    p_col, _p_label = dp.significance_column(bars, fdr_scope)
    ax.set_title(f'{title}\nsolid: significant ({P_NAMES.get(p_col, p_col)} and |log2FC| thresholds)',
                 fontsize=10)
    ax.legend(handles=[Patch(color=colour_a, label=f'higher in {group_a}'),
                       Patch(color=colour_b, label=f'higher in {group_b}')], fontsize=7, loc='upper left')
    dp.style_figure_theme(fig, is_dark)
    dp.stamp_run_label(fig, run_label)
    fig._bar_features = bars['feature'].astype(str).tolist()
    fig._bar_axes = ax
    return fig


def combination_values(stats: dict, combination: str, pair: tuple, group_order: list | None = None) -> pd.DataFrame:
    """
    Per-sample values of one combination for ``combination_figure``: one row
    per sample of every group (within the comparison's split level), with
    its group, pairing level, % of library size (half an event added, as the
    Frequencies test uses) and whether the count was left out as an expanded
    clone. Rows follow *group_order*, then the samples' order. For a
    population contrast the rows are its units, grouped by population and
    group, and the pairing level is the mouse within each population.
    """
    _base, _other, level = pair
    split = stats.get('split') or {}
    pairing = stats.get('pairing') or {}
    removed = stats.get('removed')
    log2 = stats['log2_values']
    contrast = (stats.get('interaction') or {}).get(comparison_label(*pair))
    if contrast is not None:
        members = [(s, contrast['cell'][s]) for s in contrast['units']]
        pairing = contrast['line']
        group_order = contrast['order']
    else:
        members = [(s, g) for s, g in zip(stats['samples'], stats['group_vec'])
                   if level is None or split.get(s) == level]
    rows = []
    for s, g in members:
        if s not in log2.index or combination not in log2.columns:
            continue
        rows.append({'sample': s, 'group': g, 'pair': pairing.get(s, ''),
                     'percent': float(2.0 ** log2.loc[s, combination]),
                     'removed': bool(removed is not None and s in removed.index
                                     and combination in removed.columns and removed.loc[s, combination])})
    table = pd.DataFrame(rows, columns=['sample', 'group', 'pair', 'percent', 'removed'])
    order = [g for g in (group_order or []) if g in set(table['group'])]
    order += [g for g in dict.fromkeys(table['group']) if g not in order]
    rank = {g: i for i, g in enumerate(order)}
    return table.sort_values('group', key=lambda col: col.map(rank), kind='mergesort').reset_index(drop=True)


def combination_figure(stats: dict, combination: str, pair: tuple, fdr_scope: str, is_dark: bool,
                       population: str = '', sample_labels: dict | None = None,
                       group_colours: dict | None = None, group_order: list | None = None):
    """
    One combination in every sample: a column per group (within the
    comparison's split level) with a dot per sample in the group's colour
    and the group median, on a log scale of % of library size. Samples of
    the same pairing level (mouse) are joined by lines, darker between the
    two groups of *pair*. Counts left out as expanded clones are hollow.
    The title gives each test's log2FC, CI and adjusted p-value for *pair*.
    """
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from honeychrome.view_components import differential_plots as dp

    base, other, level = pair
    label = comparison_label(base, other, level)
    table = combination_values(stats, combination, pair, group_order)
    if table.empty:
        return dp.message_figure(f'{combination} has no values in {label}.', combination, is_dark)
    contrast = (stats.get('interaction') or {}).get(label)
    if contrast is not None:
        group_colours = {cell: (group_colours or {}).get(g) for cell, g in contrast['colour_group'].items()}
    groups = list(dict.fromkeys(table['group']))
    xpos = {g: i for i, g in enumerate(groups)}
    fg = 'white' if is_dark else 'black'
    fig = Figure(figsize=(max(4.2, 0.75 * len(groups) + 2.2), 4.8), constrained_layout=True)
    ax = fig.add_subplot(111)
    rng = np.random.default_rng(0)
    table = table.assign(x=[xpos[g] + rng.uniform(-0.12, 0.12) for g in table['group']])
    paired = table[table['pair'] != '']
    for _level, rows in paired.groupby('pair', sort=False):
        rows = rows.sort_values('x')
        ax.plot(rows['x'], rows['percent'], color='#9a9a9a', linewidth=0.6, alpha=0.5, zorder=1)
        a, b = rows[rows['group'] == base], rows[rows['group'] == other]
        if len(a) == 1 and len(b) == 1:
            ax.plot([a['x'].iloc[0], b['x'].iloc[0]], [a['percent'].iloc[0], b['percent'].iloc[0]],
                    color=fg, linewidth=0.9, alpha=0.7, zorder=2)
    for i, g in enumerate(groups):
        rows = table[table['group'] == g]
        colour = _colour_of(g, group_colours, i)
        kept = rows[~rows['removed']]
        ax.scatter(kept['x'], kept['percent'], s=26, color=colour, edgecolors=fg, linewidths=0.4, zorder=3)
        gone = rows[rows['removed']]
        if len(gone):
            ax.scatter(gone['x'], gone['percent'], s=26, facecolors='none', edgecolors=colour, linewidths=1.0,
                       zorder=3)
        if len(kept):
            med = float(np.median(kept['percent']))
            ax.hlines(med, i - 0.28, i + 0.28, color=fg, linewidth=1.2, zorder=4)
    ax.set_yscale('log')
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_xticks(range(len(groups)))
    ticks = ax.set_xticklabels([f'{g}\n(n={int((table["group"] == g).sum())})' for g in groups], fontsize=8)
    for tick, g in zip(ticks, groups):
        if g in (base, other):
            tick.set_fontweight('bold')
    ax.tick_params(axis='y', labelsize=8)
    ax.set_ylabel('% of library size (log scale)', fontsize=9)
    lines = [f'{population}: {combination}' if population else combination]
    for key in ('counts', 'freq'):
        res = stats.get(key)
        if res is None:
            continue
        row = res[(res['comparison'] == label) & (res['feature'] == combination)]
        if row.empty:
            continue
        row = row.iloc[0]
        p_col, _p_label = dp.significance_column(res, fdr_scope)
        text = f"{TEST_NAMES[key]}, {label}: log2FC {float(row['logFC']):+.2f}"
        if np.isfinite(float(row.get('CI.L', np.nan))) and np.isfinite(float(row.get('CI.R', np.nan))):
            text += f" [{float(row['CI.L']):+.2f}, {float(row['CI.R']):+.2f}]"
        text += f", {P_NAMES.get(p_col, p_col)} = {float(row[p_col]):.3g}"
        lines.append(text)
    ax.set_title('\n'.join(lines), fontsize=9)
    handles = []
    if len(paired):
        handles.append(Line2D([], [], color='#9a9a9a', linewidth=0.8, label='same pairing level'))
    if table['removed'].any():
        handles.append(Line2D([], [], marker='o', linestyle='', markerfacecolor='none', markeredgecolor=fg,
                              label='left out (expanded clone)'))
    if handles:
        ax.legend(handles=handles, fontsize=7, loc='best')
    names = [(sample_labels or {}).get(s, str(s).replace('\\', '/').rsplit('/', 1)[-1]) for s in table['sample']]
    overlay = ax.scatter(table['x'], table['percent'], s=60, alpha=0, zorder=6)
    fig._hover_handler = dp.make_scatter_hover_handler(
        fig, ax, overlay, [f'{n}: {v:.3g}%' for n, v in zip(names, table['percent'])], is_dark)
    dp.style_figure_theme(fig, is_dark)
    return fig


# ---------------------------------------------------------------------------
# Per-combination marker intensities
# ---------------------------------------------------------------------------

MARKER_DEFAULTS = {'min_events': 10, 'threshold': 0.03, 'spillover_limit': 0.03}
MARKER_REFERENCES = {'tags': 'the tag model (other combinations, same tags)',
                     'controls': 'the control combinations', 'all': 'all other barcoded cells'}
# Huber tuning constant and iterations of the tag model.
TAG_MODEL_HUBER = 1.345
TAG_MODEL_ITERATIONS = 6
MARKER_X_LABEL = 'Δ mean transformed intensity (combination − reference)'


def marker_feature(combination: str, channel: str) -> str:
    return f'{combination}: {channel}'


def marker_deltas(sums: np.ndarray, counts: pd.DataFrame, channels: list, reference: str = 'controls',
                  control_combos: list | None = None, min_events: int = MARKER_DEFAULTS['min_events'],
                  removed: pd.DataFrame | None = None, return_means: bool = False,
                  tag_matrix: pd.DataFrame | None = None) -> tuple:
    """
    Within-sample marker differences of every combination.

    sums    : (samples, combinations, channels) sums of the transformed
              intensities of each combination's events, rows and columns
              as *counts*.
    counts  : samples × combinations, the events those sums are over.
    reference : 'tags' — what the combination's own tags predict from the
              other combinations of the sample (``tag_model_residuals``),
              so a tag fluorophore's spillover into a marker, shared by every
              combination carrying it, is not taken for an effect;
              'controls' — the control combinations (a control is compared
              with the other controls); 'all' — every other barcoded cell of
              the sample. 'tags' without *tag_matrix* falls back to
              'controls', and 'controls' without any control combination to
              'all'.
    tag_matrix: combinations × tags, 1 where the combination carries the
              tag (rows are matched to *counts*' columns).
    min_events: a combination, or its reference, with fewer events in a
              sample has no value there (NaN).
    removed : samples × combinations, True where a count was left out as an
              expanded clone; those cells take part in neither side.

    Returns (deltas, meta, reference used): deltas is samples × features
    (combination-major, ``marker_feature`` names) of the combination's mean
    minus its reference's mean in the same sample; meta lists each
    feature's 'combination', 'channel' and 'cluster_id' (the combination,
    the family for cluster-first testing). With *return_means* the
    combination's and the reference's means (same layout) follow.
    """
    n = counts.values.astype(float).copy()
    s = np.asarray(sums, dtype=float).copy()
    if removed is not None:
        gone = removed.reindex(index=counts.index, columns=counts.columns, fill_value=False).values.astype(bool)
        n[gone] = 0.0
        s[gone] = 0.0
    combos = list(counts.columns)
    controls = [combos.index(c) for c in (control_combos or []) if c in combos]
    if reference == 'tags' and tag_matrix is None:
        reference = 'controls'
    if reference == 'controls' and not controls:
        reference = 'all'
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(n[:, :, None] >= min_events, s / n[:, :, None], np.nan)
        if reference == 'tags':
            tags = tag_matrix.reindex(combos).fillna(0.0).values.astype(float)
            ref_mean = mean - tag_model_residuals(mean, n, tags)
        else:
            if reference == 'controls':
                pool_s, pool_n = s[:, controls].sum(axis=1), n[:, controls].sum(axis=1)
            else:
                pool_s, pool_n = s.sum(axis=1), n.sum(axis=1)
            ref_s = np.repeat(pool_s[:, None, :], len(combos), axis=1)
            ref_n = np.repeat(pool_n[:, None], len(combos), axis=1)
            inside = np.zeros(len(combos), dtype=bool)
            inside[controls if reference == 'controls' else slice(None)] = True
            ref_s[:, inside] -= s[:, inside]
            ref_n[:, inside] -= n[:, inside]
            ref_mean = np.where(ref_n[:, :, None] >= min_events, ref_s / ref_n[:, :, None], np.nan)
    delta = (mean - ref_mean).reshape(len(counts), len(combos) * len(channels))
    names = [marker_feature(c, ch) for c in combos for ch in channels]
    meta = pd.DataFrame({'feature': names, 'combination': [c for c in combos for _ch in channels],
                         'channel': [ch for _c in combos for ch in channels],
                         'cluster_id': [c for c in combos for _ch in channels]})
    out = (pd.DataFrame(delta, index=counts.index, columns=names), meta, reference)
    if return_means:
        shape = (len(counts), len(combos) * len(channels))
        out += (pd.DataFrame(mean.reshape(shape), index=counts.index, columns=names),
                pd.DataFrame(ref_mean.reshape(shape), index=counts.index, columns=names))
    return out


def tag_model_residuals(mean: np.ndarray, n: np.ndarray, tags: np.ndarray) -> np.ndarray:
    """
    Leave-one-out residuals of each combination's mean from the additive tag
    model of its sample.

    mean : (samples, combinations, markers) mean intensities (NaN: too few
           events); n: (samples, combinations) their events; tags:
           (combinations, tags) 0/1.

    For every sample and marker the combinations' means are fitted on their
    tag indicators (one term per tag, no intercept: every combination has
    the same number of tags), weighted by events capped at three times the
    sample's median, with Huber downweighting (``TAG_MODEL_HUBER``) so a few
    combinations with real effects do not move the tag terms. The residual
    is divided by 1 - leverage, so each combination is compared with what
    the other combinations predict for its tags. Needs five more measured
    combinations than tags; NaN otherwise.
    """
    out = np.full(mean.shape, np.nan)
    p = tags.shape[1]
    for i in range(mean.shape[0]):
        counted = n[i] > 0
        if not counted.any():
            continue
        w0 = np.minimum(n[i], 3.0 * np.median(n[i][counted]))
        for k in range(mean.shape[2]):
            y = mean[i, :, k]
            ok = np.isfinite(y) & (w0 > 0)
            if ok.sum() < p + 5:
                continue
            x, yo, base = tags[ok], y[ok], w0[ok]
            if np.linalg.matrix_rank(x) < p:
                keep = np.flatnonzero(x.any(axis=0))
                x = x[:, keep]
            w = base.copy()
            for _it in range(TAG_MODEL_ITERATIONS):
                root = np.sqrt(w)
                q, r = np.linalg.qr(x * root[:, None])
                beta = np.linalg.lstsq(r, q.T @ (yo * root), rcond=None)[0]
                e = yo - x @ beta
                scaled = np.abs(e * root)
                scale = 1.4826 * np.median(scaled)
                if not scale > 0:
                    break
                z = scaled / scale
                w = base * np.where(z <= TAG_MODEL_HUBER, 1.0, TAG_MODEL_HUBER / np.maximum(z, 1e-12))
            root = np.sqrt(w)
            q, _r = np.linalg.qr(x * root[:, None])
            leverage = np.sum(q ** 2, axis=1)
            out[i, np.flatnonzero(ok), k] = e / np.maximum(1.0 - leverage, 1e-6)
    return out


def marker_spillover(spread: pd.DataFrame, sums: np.ndarray, counts: pd.DataFrame, channels: list,
                     transforms: dict) -> pd.DataFrame:
    """
    Expected spillover of each combination's tags into each marker, on the
    display scale, per sample: the transformed distance between the
    sample's mean barcoded-cell level of the marker and that level plus the
    combination's spillover spread (*spread*: combinations × markers, raw
    units, ``flowcode_core.spillover_spread``). Returns samples × features
    (``marker_feature`` names); NaN for a marker without a transform or a
    sample without barcoded cells.
    """
    combos = list(counts.columns)
    n = counts.values.astype(float)
    total = n.sum(axis=1)
    out = np.full((len(counts), len(combos), len(channels)), np.nan)
    raw = spread.reindex(index=combos, columns=channels).values.astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        level = np.asarray(sums, dtype=float).sum(axis=1) / total[:, None]
    for k, ch in enumerate(channels):
        tr = transforms.get(ch)
        xform = getattr(tr, 'xform', None)
        if xform is None or not np.isfinite(raw[:, k]).any():
            continue
        ok = np.isfinite(level[:, k])
        if not ok.any():
            continue
        x0 = xform.inverse(level[ok, k])
        shifted = xform.apply((x0[:, None] + np.nan_to_num(raw[None, :, k])).ravel()).reshape(len(x0), -1)
        out[np.flatnonzero(ok), :, k] = np.abs(shifted - level[ok, k][:, None])
    names = [marker_feature(c, ch) for c in combos for ch in channels]
    return pd.DataFrame(out.reshape(len(counts), -1), index=counts.index, columns=names)


def _marker_significance(results: pd.DataFrame, pval_threshold: float, threshold: float, fdr_scope: str,
                         use_treat: bool) -> pd.DataFrame:
    """Cluster-first significance of the marker results that are tested
    (``tested``); the others keep their estimates, with no adjusted
    p-value and not significant."""
    tested = results['tested'].astype(bool) if 'tested' in results.columns else pd.Series(True, results.index)
    if not tested.any():
        return results.assign(significant=False)
    done = ds.apply_significance(results[tested], pval_threshold, threshold, fdr_scope=fdr_scope,
                                 use_treat=use_treat, hierarchical=True)
    rest = results[~tested].copy()
    for col in done.columns:
        if col not in rest.columns:
            rest[col] = np.nan
    rest['significant'] = False
    return pd.concat([done, rest[done.columns]]).loc[results.index]


def marker_label(group: str, level: str | None = None) -> str:
    return f'{group} [{level}]' if level else group


def marker_statistics(deltas: pd.DataFrame, meta: pd.DataFrame, groups: dict, *,
                      group_order: list | None = None, strata: dict | None = None,
                      split_by: dict | None = None, threshold: float = MARKER_DEFAULTS['threshold'],
                      pval_threshold: float = 0.05, fdr_scope: str = 'per_comparison',
                      use_treat: bool = False, spillover: pd.DataFrame | None = None,
                      spillover_limit: float | None = None) -> dict:
    """
    Does a combination change a marker against its reference cells? For
    every group (within each population and split level), each feature's
    within-sample differences (``marker_deltas``) are tested against zero
    with a moderated t (an intercept-only limma fit: one value per sample,
    so the comparison is paired by sample). Features need two or more
    samples. Significance is cluster-first with the combinations as
    families (``differential_stats.apply_significance(hierarchical=True)``):
    a combination is screened over all its markers, then its markers are
    tested. *threshold* is the |Δ| threshold in transformed units.

    *spillover* (samples × features, ``marker_spillover``) adds each
    feature's mean expected spillover over the group's samples
    ('spillover'). With *spillover_limit*, features above it are not tested
    ('tested' False: they keep their estimate but take no part in the FDR
    and are never significant), because the combination's own tags could
    move the marker by that much.

    Returns {'results' (stacked, one 'comparison' per group and level),
    'comparisons', 'sets' ({comparison: {'group', 'level', 'units'}}),
    'notes'}. Raises RuntimeError when no group can be tested.
    """
    from honeychrome.controller_components import moderated_stats as ms

    order = list(group_order or [])
    units = [u for u in deltas.index if groups.get(u)]
    if split_by:
        units = [u for u in units if split_by.get(u) not in (None, '')]

    def level_of(u):
        parts = ([str(strata[u])] if strata else []) + ([str(split_by[u])] if split_by else [])
        return ' · '.join(parts) or None

    sets: dict = {}
    for u in units:
        sets.setdefault((groups[u], level_of(u)), []).append(u)
    names = order + sorted({g for g, _lv in sets if g not in order})
    frames, notes, labels, tested_sets = [], [], [], {}
    for lv in dict.fromkeys(level for _g, level in sets):
        for g in names:
            members = sets.get((g, lv))
            if not members:
                continue
            label = marker_label(g, lv)
            if len(members) < 2:
                notes.append(f'Markers, {label}: fewer than two samples.')
                continue
            expr = deltas.loc[members].values.T.copy()
            expr[np.isfinite(expr).sum(axis=1) < 2] = np.nan
            try:
                fit = ms.e_bayes(ms.lm_fit(expr, np.ones((len(members), 1))))
            except ValueError as exc:
                notes.append(f'Markers, {label}: {exc}')
                continue
            table = ms.top_table(fit, 0)
            table.insert(0, 'feature', list(deltas.columns))
            table['comparison'] = label
            table['n_samples'] = np.isfinite(expr).sum(axis=1)
            if spillover is not None:
                spill = spillover.reindex(index=members, columns=deltas.columns).mean(axis=0).values
                table['spillover'] = spill
                table['tested'] = ~(np.isfinite(spill) & (spill > spillover_limit)) if spillover_limit is not None \
                    else True
            else:
                table['tested'] = True
            frames.append(table[np.isfinite(table['logFC'].astype(float))])
            labels.append(label)
            tested_sets[label] = {'group': g, 'level': lv, 'units': list(members)}
    if not frames:
        raise RuntimeError(' '.join(notes) or 'No group has marker values to test.')
    results = pd.concat(frames, ignore_index=True).merge(meta, on='feature', how='left', sort=False)
    results = _marker_significance(results, pval_threshold, threshold, fdr_scope, use_treat)
    return {'results': results, 'comparisons': labels, 'sets': tested_sets, 'notes': notes,
            'spillover_limit': spillover_limit if spillover is not None else None}


def reapply_marker_thresholds(markers: dict, pval_threshold: float, threshold: float, fdr_scope: str,
                              use_treat: bool, spillover_limit: float | None = None) -> dict:
    """*markers* with significance recomputed for new thresholds and, when
    the results carry spillover values, a new spillover limit (None: every
    feature tested), without refitting."""
    out = dict(markers)
    results = markers['results']
    if 'spillover' in results.columns:
        spill = results['spillover'].astype(float)
        results = results.assign(tested=~(np.isfinite(spill) & (spill > spillover_limit))
                                 if spillover_limit is not None else True)
        out['spillover_limit'] = spillover_limit
    out['results'] = _marker_significance(results, pval_threshold, threshold, fdr_scope, use_treat)
    return out


def marker_figures(markers: dict, comparison: str, pval_threshold: float, threshold: float, fdr_scope: str,
                   is_dark: bool, population: str, reference_name: str, heatmap_order: str = 'effect',
                   feature: str | None = None, pairing: dict | None = None, sample_labels: dict | None = None,
                   group_colours: dict | None = None, group_order: list | None = None) -> dict:
    """
    Figures of one marker comparison (a group): the combination × marker
    heatmap of significant differences, the volcano of every
    combination-marker pair and, with *feature*, that combination-marker
    pair's per-sample plot (``marker_value_figure``) ({key: {'fig',
    'title', 'error', 'results', 'make'}}).
    """
    from honeychrome.view_components import differential_plots as dp

    view = markers['results'][markers['results']['comparison'] == comparison].reset_index(drop=True)
    run_label = f'{population}: {comparison}, combinations against {reference_name}'
    out = {}

    def add(key, tab_title, maker, results=None, **kwargs):
        try:
            out[key] = {'fig': maker(**kwargs), 'title': tab_title, 'error': None, 'results': results,
                        'make': lambda: maker(**kwargs)}
        except Exception as exc:
            out[key] = {'fig': None, 'title': tab_title, 'error': str(exc) or type(exc).__name__,
                        'results': None}

    grid = view.assign(feature=view['combination'], comparison=view['channel'])
    untested = ([] if 'tested' not in view.columns else
                list(zip(view.loc[~view['tested'].astype(bool), 'combination'],
                         view.loc[~view['tested'].astype(bool), 'channel'])))

    def heatmap():
        fig = dp.make_comparison_heatmap_figure(
            grid, f'{comparison}: marker differences (significant)'
            + ('; hatched: not tested (tag spillover above the limit)' if untested else ''),
            is_dark=is_dark, run_label=run_label, order=heatmap_order, value_label='Δ transformed intensity')
        _hatch_cells(fig, untested)
        return fig

    add('marker_heatmap', 'Marker heatmap', heatmap)
    add('marker_volcano', 'Marker volcano', dp.make_volcano_figure, results=view, results_df=view,
        title=f'Markers: {comparison}', pval_threshold=pval_threshold, fc_threshold=threshold, is_dark=is_dark,
        fdr_scope=fdr_scope, x_label=MARKER_X_LABEL, run_label=run_label, group_a='reference',
        group_b='combination')
    if feature:
        add('marker_combination', f'Marker: {feature}', marker_value_figure, markers=markers, feature=feature,
            comparison=comparison, fdr_scope=fdr_scope, is_dark=is_dark, population=population,
            reference_name=reference_name, pairing=pairing, sample_labels=sample_labels,
            group_colours=group_colours, group_order=group_order)
    return out


def _hatch_cells(fig, cells: list):
    """Hatch the (column label, row label) *cells* of a
    ``make_comparison_heatmap_figure`` grid."""
    from matplotlib.patches import Rectangle

    if not cells:
        return
    wanted = set(cells)
    for ax in fig.axes:
        if not ax.images:
            continue
        columns = [t.get_text() for t in ax.get_xticklabels()]
        rows = [t.get_text() for t in ax.get_yticklabels()]
        for x, col in enumerate(columns):
            for y, row in enumerate(rows):
                if (col, row) in wanted:
                    ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1.0, 1.0, fill=False, hatch='////',
                                           edgecolor='#9a9a9a', linewidth=0.0))


def marker_values(markers: dict, feature: str, comparison: str, pairing: dict | None = None,
                  group_order: list | None = None) -> pd.DataFrame:
    """
    Per-sample means of one combination-marker pair for
    ``marker_value_figure``: one row per sample of every group tested within
    the comparison's population and split level, with its group, pairing
    level, the combination's mean, the reference's mean and their
    difference. Rows follow *group_order*, then the samples' order.
    """
    sets = markers.get('sets') or {}
    level = (sets.get(comparison) or {}).get('level')
    means, refs = markers.get('means'), markers.get('reference_means')
    rows = []
    for label, info in sets.items():
        if info['level'] != level:
            continue
        for u in info['units']:
            if means is None or u not in means.index or feature not in means.columns:
                continue
            m, r = float(means.loc[u, feature]), float(refs.loc[u, feature])
            if not (np.isfinite(m) and np.isfinite(r)):
                continue
            rows.append({'sample': u, 'group': info['group'], 'pair': str((pairing or {}).get(u, '') or ''),
                         'combination': m, 'reference': r, 'delta': m - r})
    table = pd.DataFrame(rows, columns=['sample', 'group', 'pair', 'combination', 'reference', 'delta'])
    order = [g for g in (group_order or []) if g in set(table['group'])]
    order += [g for g in dict.fromkeys(table['group']) if g not in order]
    rank = {g: i for i, g in enumerate(order)}
    return table.sort_values('group', key=lambda col: col.map(rank), kind='mergesort').reset_index(drop=True)


def marker_value_figure(markers: dict, feature: str, comparison: str, fdr_scope: str, is_dark: bool,
                        population: str = '', reference_name: str = '', pairing: dict | None = None,
                        sample_labels: dict | None = None, group_colours: dict | None = None,
                        group_order: list | None = None):
    """
    One combination-marker pair in every sample: a column per group (of the
    comparison's population and split level). In each column the reference
    cells' mean (hollow, left) and the combination's mean (filled, right, in
    the group's colour) of every sample are joined by a line, so the line's
    slope is the tested difference; the group medians are marked. A
    sample's combination means in different groups are joined by faint lines
    within its pairing level (mouse). The title gives the difference, its CI
    and the stage-wise adjusted p-value in *comparison*, whose group is bold.
    """
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from honeychrome.view_components import differential_plots as dp

    table = marker_values(markers, feature, comparison, pairing, group_order)
    if table.empty:
        return dp.message_figure(f'{feature} has no values in {comparison}.', feature, is_dark)
    groups = list(dict.fromkeys(table['group']))
    xpos = {g: i for i, g in enumerate(groups)}
    fg = 'white' if is_dark else 'black'
    fig = Figure(figsize=(max(4.6, 0.95 * len(groups) + 2.2), 4.8), constrained_layout=True)
    ax = fig.add_subplot(111)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.05, 0.05, len(table))
    table = table.assign(x_ref=[xpos[g] - 0.18 for g in table['group']] + jitter,
                         x_comb=[xpos[g] + 0.18 for g in table['group']] + jitter)
    for _level, rows in table[table['pair'] != ''].groupby('pair', sort=False):
        if len(rows) > 1:
            rows = rows.sort_values('x_comb')
            ax.plot(rows['x_comb'], rows['combination'], color='#9a9a9a', linewidth=0.6, alpha=0.35, zorder=1)
    selected = (markers.get('sets') or {}).get(comparison, {}).get('group')
    for i, g in enumerate(groups):
        rows = table[table['group'] == g]
        colour = _colour_of(g, group_colours, i)
        for _i, r in rows.iterrows():
            ax.plot([r['x_ref'], r['x_comb']], [r['reference'], r['combination']], color=colour,
                    linewidth=0.9 if g == selected else 0.6, alpha=0.8, zorder=2)
        ax.scatter(rows['x_ref'], rows['reference'], s=22, facecolors='none', edgecolors=colour, linewidths=0.9,
                   zorder=3)
        ax.scatter(rows['x_comb'], rows['combination'], s=26, color=colour, edgecolors=fg, linewidths=0.4,
                   zorder=3)
        for x, col in ((i - 0.18, 'reference'), (i + 0.18, 'combination')):
            ax.hlines(float(np.median(rows[col])), x - 0.12, x + 0.12, color=fg, linewidth=1.2, zorder=4)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_xticks(range(len(groups)))
    ticks = ax.set_xticklabels([f'{g}\n(n={int((table["group"] == g).sum())})' for g in groups], fontsize=8)
    for tick, g in zip(ticks, groups):
        if g == selected:
            tick.set_fontweight('bold')
    ax.tick_params(axis='y', labelsize=8)
    ax.set_ylabel('Mean transformed intensity', fontsize=9)
    lines = [f'{population}: {feature}' if population else feature]
    res = markers['results']
    row = res[(res['comparison'] == comparison) & (res['feature'] == feature)]
    if not row.empty:
        row = row.iloc[0]
        p_col, _p_label = dp.significance_column(res, fdr_scope)
        text = f"{comparison}: Δ {float(row['logFC']):+.3f}"
        if np.isfinite(float(row.get('CI.L', np.nan))) and np.isfinite(float(row.get('CI.R', np.nan))):
            text += f" [{float(row['CI.L']):+.3f}, {float(row['CI.R']):+.3f}]"
        if 'tested' in row.index and not bool(row['tested']):
            text += f", not tested: tag spillover {float(row['spillover']):.3f}"
        else:
            text += f", {P_NAMES.get(p_col, p_col)} = {float(row[p_col]):.3g}"
            if np.isfinite(float(row.get('spillover', np.nan))):
                text += f", tag spillover {float(row['spillover']):.3f}"
        lines.append(text)
    ax.set_title('\n'.join(lines), fontsize=9)
    handles = [Line2D([], [], marker='o', linestyle='', markerfacecolor='none', markeredgecolor=fg,
                      label=f'reference ({reference_name})' if reference_name else 'reference'),
               Line2D([], [], marker='o', linestyle='', color=fg, label='combination')]
    if (table['pair'] != '').any():
        handles.append(Line2D([], [], color='#9a9a9a', linewidth=0.8, label='same pairing level'))
    ax.legend(handles=handles, fontsize=7, loc='best')
    names = [(sample_labels or {}).get(s, str(s)) for s in table['sample']]
    overlay = ax.scatter(table['x_comb'], table['combination'], s=60, alpha=0, zorder=6)
    fig._hover_handler = dp.make_scatter_hover_handler(
        fig, ax, overlay, [f'{n}: {c:.3f} (reference {r:.3f})'
                           for n, c, r in zip(names, table['combination'], table['reference'])], is_dark)
    dp.style_figure_theme(fig, is_dark)
    return fig


def _bundled_plugins_on_path():
    import honeychrome
    bundled = str(Path(honeychrome.__file__).resolve().parent / 'bundled_plugins')
    if bundled not in sys.path:
        sys.path.insert(0, bundled)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def stamp_invalid(fig, text: str | None = None):
    """Write *text* (default the no-FRET warning) in red across *fig*,
    corner to corner, for results of samples debarcoded without FRET
    correction. Outside the layout, so the figure itself is unchanged; a
    figure is stamped once. Returns *fig*."""
    if fig is None or getattr(fig, '_invalid_stamp', False):
        return fig
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    if text is None:
        from flowcode_state import NO_FRET_WARNING as text
    width, height = fig.get_size_inches()
    per_point = TextPath((0, 0), text, size=1, prop=FontProperties(weight='bold')).get_extents().width
    # 80% of the diagonal, in points
    size = float(np.clip(0.8 * 72.0 * np.hypot(width, height) / max(per_point, 1e-9), 6.0, 60.0))
    angle = float(np.degrees(np.arctan2(height, width)))
    fig.text(0.5, 0.5, text, color='#d62728', alpha=0.4, fontsize=size, weight='bold', rotation=angle,
             rotation_mode='anchor', ha='center', va='center', zorder=1000, in_layout=False)
    fig._invalid_stamp = True
    return fig


def _stamping_pdf_pages(path):
    """A PdfPages whose every page is stamped with ``stamp_invalid``."""
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure

    class StampedPdfPages(PdfPages):
        def savefig(self, figure=None, **kwargs):
            if isinstance(figure, Figure):
                stamp_invalid(figure)
            super().savefig(figure, **kwargs)

    return StampedPdfPages(path)


def default_export_folder(root, population: str, when: datetime | None = None) -> Path:
    """``<root>/<population> <YYYY-MM-DD HHMMSS>``, the population name made
    safe for a folder name."""
    _bundled_plugins_on_path()
    import drc_report

    stamp = (when or datetime.now()).strftime('%Y-%m-%d %H%M%S')
    return Path(root) / f"{drc_report.sanitize_filename(population, 'population')} {stamp}"


def write_analysis_folder(folder, *, population: str, raw_counts: pd.DataFrame, meta: pd.DataFrame,
                          qc: pd.DataFrame, filtered, stats: dict | None, sample_table: pd.DataFrame,
                          settings_text: str, figures: list, sample_labels: dict | None = None,
                          progress=None, no_fret: bool = False) -> list:
    """
    Write one analysis into *folder*:

    * ``counts_all_samples.csv`` — every linked sample, every combination,
      with events in the file and population, barcoded, untransduced and
      unresolved events;
    * ``counts_analysed.csv`` and ``percent_of_barcoded.csv`` — after the
      inclusion rules (``pct_matrix.csv`` is the same percentages with
      sample names first, the layout the lab's R scripts read);
    * ``qc.csv``, ``samples.csv`` (groups, pairing, covariates);
    * ``results.csv`` — every comparison of every test run;
    * ``library_sizes.csv`` — barcoded events and library size per sample;
    * ``clones.csv`` — expanded clones found, and whether each was removed
      (when the clone filter was on);
    * ``markers.csv`` — the marker test of every group (when run);
    * ``settings.txt`` — population, gates, rules, test settings, notes;
    * ``figures/`` — a PNG per figure, and ``<population> report.pdf`` with
      a title page, the settings, the QC table, every figure and the
      results tables.

    *figures* is a list of (name, zero-argument callable returning a
    canvas-less Figure). With *no_fret* (samples debarcoded without FRET
    correction) every figure and report page carries ``stamp_invalid`` and
    the title page leads with the warning. Returns the list of files
    written.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    _bundled_plugins_on_path()
    import drc_report

    folder = Path(folder)
    fig_dir = folder / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def readable(unit):
        return str(unit).replace(_CELL_SEP, ' | ')

    def csv(df, name, index=True):
        path = folder / name
        if index:
            df = df.rename(index=readable)
        df.to_csv(path, index=index)
        written.append(path)

    labels = sample_labels or {}
    csv(raw_counts.join(meta), 'counts_all_samples.csv')
    csv(filtered.counts, 'counts_analysed.csv')
    pct = 100.0 * filtered.counts.div(filtered.denominator.replace(0, np.nan), axis=0)
    csv(pct, 'percent_of_barcoded.csv')
    csv(pct.rename(index=lambda s: labels.get(s, Path(str(s)).stem)).rename_axis('sample').reset_index(),
        'pct_matrix.csv', index=False)
    csv(qc, 'qc.csv')
    csv(sample_table, 'samples.csv')
    results = results_table(stats) if stats else None
    if results is not None:
        csv(results, 'results.csv', index=False)
    if stats and stats.get('library_sizes') is not None:
        sizes = pd.DataFrame({'barcoded': stats['denominator'], 'library_size': stats['library_sizes']})
        sizes['factor'] = sizes['library_size'] / sizes['barcoded'].replace(0, np.nan)
        csv(sizes.rename(index=lambda s: labels.get(s, s)), 'library_sizes.csv')
    if stats and stats.get('clones') is not None:
        clones = stats['clones'].copy()
        clones['sample'] = [labels.get(s, s) for s in clones['sample']]
        csv(clones, 'clones.csv', index=False)
    if stats and stats.get('markers') is not None:
        csv(stats['markers']['results'], 'markers.csv', index=False)
    settings_path = folder / 'settings.txt'
    settings_path.write_text(settings_text, encoding='utf-8')
    written.append(settings_path)

    pdf_path = folder / f"{drc_report.sanitize_filename(population, 'population')} report.pdf"
    from flowcode_state import NO_FRET_WARNING

    with (_stamping_pdf_pages if no_fret else PdfPages)(str(pdf_path)) as pdf:
        drc_report.add_title_page(pdf, 'FlowCode Analysis', ([NO_FRET_WARNING] if no_fret else []) + [
            f'Population: {population}',
            f'Samples analysed: {filtered.counts.shape[0]}; combinations: {filtered.counts.shape[1]}',
            f"Comparisons: {len(stats['comparisons']) if stats else 0}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ])
        _text_pages(pdf, 'Settings', settings_text)
        drc_report.add_dataframe_pages(pdf, qc.rename(index=readable).reset_index(), 'Sample QC')
        for i, (name, make) in enumerate(figures):
            if progress is not None:
                progress(i, len(figures), name)
            try:
                fig = make()
            except Exception:
                fig = None
            if fig is None:
                continue
            if no_fret:
                stamp_invalid(fig)
            png = fig_dir / f"{drc_report.sanitize_filename(name)}.png"
            fig.savefig(png, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
            written.append(png)
            pdf.savefig(fig, bbox_inches='tight')
        if results is not None:
            sig = results[results['significant'].astype(bool)]
            cols = [c for c in ('test', 'comparison', 'feature', 'logFC', 'CI.L', 'CI.R', 'P.Value',
                                'adj.P.Val', 'adj.P.Val.global') if c in sig.columns]
            drc_report.add_dataframe_pages(pdf, sig[cols], 'Significant combinations')
        if stats and stats.get('markers') is not None:
            res = stats['markers']['results']
            sig = res[res['significant'].astype(bool)]
            cols = [c for c in ('comparison', 'combination', 'channel', 'logFC', 'CI.L', 'CI.R', 'n_samples',
                                'stagewise.adj.P.Val') if c in sig.columns]
            drc_report.add_dataframe_pages(pdf, sig[cols].rename(columns={'logFC': 'delta'}),
                                           'Significant marker differences')
    written.append(pdf_path)
    return written


def _text_pages(pdf, title: str, text: str, lines_per_page: int = 55):
    from matplotlib.figure import Figure

    lines = text.splitlines() or ['']
    for start in range(0, len(lines), lines_per_page):
        fig = Figure(figsize=(8.27, 11.69))
        fig.text(0.06, 0.96, title, fontsize=12, weight='bold', va='top')
        fig.text(0.06, 0.93, '\n'.join(lines[start:start + lines_per_page]), fontsize=7,
                 family='monospace', va='top')
        pdf.savefig(fig)
