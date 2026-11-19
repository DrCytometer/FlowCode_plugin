"""
flowcode_synopsis.py
--------------------
The readouts of the FlowCode Analysis page brought together per guide, for
the Synopsis tab (flowcode_synopsis_view.py; figures in
flowcode_synopsis_plots.py). No Qt, and no statistics of its own: every call
is the one its own test made, with the thresholds set on the Statistics and
Phenotype shift tabs.

The readouts fall in two families that answer different questions.

    Accumulation   how many of a guide's cells a group holds, against a
                   baseline group: the counts test and the frequencies test
                   of the Statistics tab. Both use the same counts, so they
                   are two views of one measurement, not two confirmations.
    Phenotype      what a guide's cells look like within one group, against
                   the guide's reference in the same sample: the marker
                   test, the phenotype shift and the immune states. All
                   three read the same cells.

``assemble`` returns, for one population (``scopes``), the long tables of
each readout, one row per guide and group of the calls (``calls``) and one
row per guide (``guides``) with its evidence class and a sentence built from
fixed templates (``guide_sentence``), so the same results always give the
same words. ``overview_text`` is the page's summary; ``write_tables`` the
export.

Evidence classes (``CLASSES``): a guide with a call in both families, in
accumulation only, in phenotype only, or in neither. They say which kinds
of result exist, not how a gene works.

A fold change with no cells on one side is not a measurement: where a guide
has no cell in any sample of a group (or of the baseline), the row is
marked ``unbounded`` and the sentence says "not detected" in place of a
number.

Left out of this version: population contrasts and comparisons between
populations (Compare: Populations) have no baseline group, so they are not
shown; with *Split by*, each split level is a population of its own here and
shares the phenotype readouts of its population.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TESTS = {'counts': 'counts', 'freq': 'frequencies'}
ORDERS = {'group': 'By gene group', 'effect': 'By effect', 'alphabetical': 'Alphabetical'}
SYNOPSIS_DEFAULTS = {'test': '', 'baseline': '', 'order': 'group', 'show_all': False}
BOTH, ACCUMULATION, PHENOTYPE, NONE = ('Accumulation and phenotype', 'Accumulation only', 'Phenotype only',
                                       'No call')
CLASSES = (BOTH, ACCUMULATION, PHENOTYPE, NONE)
# Items named in one clause of a sentence before "and n more".
MAX_LISTED = 6
_LEVEL_SEPARATOR = ' · '

ACCUMULATION_COLUMNS = ('guide', 'group', 'test', 'logFC', 'CI.L', 'CI.R', 'p', 'q', 'significant', 'comparison')
SHIFT_COLUMNS = ('guide', 'group', 'angle_deg', 'p', 'q', 'significant', 'gain_cluster', 'loss_cluster', 'mice',
                 'median_cells')
STATE_COLUMNS = ('guide', 'group', 'state', 'score', 'p', 'q', 'significant', 'depends_on_target', 'members')
MARKER_COLUMNS = ('guide', 'group', 'channel', 'delta', 'q', 'significant', 'spillover_dominant')


def synopsis_settings(stored: dict | None) -> dict:
    """The tab's saved settings with defaults filled in."""
    out = dict(SYNOPSIS_DEFAULTS)
    for key, value in (stored or {}).items():
        if key in out:
            out[key] = value
    if out['test'] not in ('', *TESTS):
        out['test'] = ''
    if out['order'] not in ORDERS:
        out['order'] = SYNOPSIS_DEFAULTS['order']
    out['baseline'] = str(out['baseline'] or '')
    out['show_all'] = bool(out['show_all'])
    return out


# ---------------------------------------------------------------------------
# Scopes, tests and baselines
# ---------------------------------------------------------------------------

def _label_pairs(stats: dict) -> dict:
    """{comparison label: (base, other, level)} of the group comparisons
    (population contrasts left out)."""
    from flowcode_stats import comparison_label

    contrasts = set(stats.get('interaction') or {})
    out = {}
    for pair in stats.get('pairs') or []:
        label = comparison_label(*pair)
        if label not in contrasts:
            out[label] = tuple(pair)
    return out


def _unit_levels(stats: dict) -> dict:
    split = stats.get('split') or {}
    return {s: split.get(s) for s in stats.get('samples') or []}


def _between_populations(stats: dict) -> bool:
    """True when the groups compared are the populations themselves."""
    strata = stats.get('strata')
    if not strata:
        return False
    levels = _unit_levels(stats)
    return not all(level is not None and str(level).split(_LEVEL_SEPARATOR)[0] == str(strata[s])
                   for s, level in levels.items())


def scopes(stats: dict | None, shift: dict | None, population: str = '') -> tuple:
    """
    ([scope], [note]): what the Population picker lists. A scope is
    {'label', 'level', 'population'}: *level* is the statistics level of its
    comparisons (None with one population and no split), *population* the
    gate its phenotype readouts belong to. *population* names the counted
    population when the statistics hold only one.
    """
    notes, out = [], []
    shift_pops = [str(p) for p in (shift or {}).get('populations') or []]
    if stats is not None and _between_populations(stats):
        notes.append('The statistics compare populations within each sample, which has no baseline group: '
                     'accumulation is not shown.')
        stats = None
    if stats is not None:
        strata = stats.get('strata')
        levels = _unit_levels(stats)
        for level in dict.fromkeys(pair[2] for pair in _label_pairs(stats).values()):
            if strata:
                pop = next((str(strata[s]) for s, lv in levels.items() if lv == level), '')
            else:
                pop = shift_pops[0] if len(shift_pops) == 1 else str(population)
            out.append({'label': str(level) if level is not None else (pop or 'All'), 'level': level,
                        'population': pop})
        if stats.get('interaction'):
            notes.append('Population contrasts are not shown.')
    covered = {s['population'] for s in out}
    if not out:
        for pop in shift_pops:
            out.append({'label': pop, 'level': None, 'population': pop})
    elif any(pop not in covered for pop in shift_pops):
        missing = [pop for pop in shift_pops if pop not in covered]
        notes.append(f"Phenotype shift populations without statistics: {', '.join(missing)}.")
    return out, notes


def available_tests(stats: dict | None) -> list:
    return [key for key in TESTS if stats is not None and stats.get(key) is not None]


def resolve_test(stats: dict | None, chosen: str = '') -> str | None:
    """The accumulation test shown: *chosen* when it was run, else counts,
    else frequencies (as the page's summary figures)."""
    tests = available_tests(stats)
    return chosen if chosen in tests else (tests[0] if tests else None)


def _scope_pairs(stats: dict, level) -> dict:
    return {label: pair for label, pair in _label_pairs(stats).items() if pair[2] == level}


def baselines(stats: dict | None, level) -> list:
    """Groups that can be the baseline in *level*: those compared with the
    most other groups (the reference group with reference contrasts, every
    group with pairwise contrasts)."""
    if stats is None:
        return []
    n = {}
    for base, other, _level in _scope_pairs(stats, level).values():
        n[base] = n.get(base, 0) + 1
        n[other] = n.get(other, 0) + 1
    if not n:
        return []
    most = max(n.values())
    return [g for g in n if n[g] == most]


def resolve_baseline(stats: dict | None, level, chosen: str = '') -> str | None:
    """*chosen* when it can be the baseline, else the group that is the
    base of the most comparisons."""
    options = baselines(stats, level)
    if not options:
        return None
    if chosen in options:
        return chosen
    as_base = {}
    for base, _other, _level in _scope_pairs(stats, level).values():
        as_base[base] = as_base.get(base, 0) + 1
    return max(options, key=lambda g: as_base.get(g, 0))


# ---------------------------------------------------------------------------
# Readout tables
# ---------------------------------------------------------------------------

def _q_column(table: pd.DataFrame, fdr_scope: str) -> str:
    if fdr_scope == 'global' and 'adj.P.Val.global' in table.columns:
        return 'adj.P.Val.global'
    return 'adj.P.Val'


def _accumulation(stats: dict, level, baseline: str, fdr_scope: str) -> pd.DataFrame:
    """Every test's rows of the comparisons with *baseline* in *level*,
    signed so that a positive logFC is more cells in 'group' than in the
    baseline."""
    pairs = _scope_pairs(stats, level)
    frames = []
    for test in available_tests(stats):
        table = stats[test]
        q = _q_column(table, fdr_scope)
        for label, (base, other, _level) in pairs.items():
            if baseline not in (base, other):
                continue
            sub = table[table['comparison'] == label]
            if sub.empty:
                continue
            flip = other == baseline
            fc = sub['logFC'].to_numpy(float)
            lo, hi = sub['CI.L'].to_numpy(float), sub['CI.R'].to_numpy(float)
            frames.append(pd.DataFrame({
                'guide': sub['feature'].astype(str).to_numpy(), 'group': base if flip else other, 'test': test,
                'logFC': -fc if flip else fc, 'CI.L': -hi if flip else lo, 'CI.R': -lo if flip else hi,
                'p': sub['P.Value'].to_numpy(float), 'q': sub[q].to_numpy(float),
                'significant': sub['significant'].to_numpy(bool), 'comparison': label}))
    if not frames:
        return pd.DataFrame(columns=list(ACCUMULATION_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def _presence(stats: dict, level) -> tuple:
    """(guide × group cells counted over the group's samples, {group:
    samples}, {group: median barcoded events per sample}) in *level*."""
    values = stats.get('counts_values')
    if values is None:
        values = stats['freq_values'] * stats['denominator'].to_numpy(float)[:, None] / 100.0
    levels = _unit_levels(stats)
    groups = dict(zip(stats['samples'], stats['group_vec']))
    units = [s for s in stats['samples'] if levels[s] == level and s in values.index]
    by_group = {}
    for s in units:
        by_group.setdefault(groups[s], []).append(s)
    cells = pd.DataFrame({g: values.loc[members].sum(axis=0, skipna=True) for g, members in by_group.items()})
    cells.index = cells.index.astype(str)
    denominator = stats['denominator']
    depth = {g: float(np.median(denominator.loc[members].to_numpy(float))) for g, members in by_group.items()}
    return cells, {g: len(members) for g, members in by_group.items()}, depth


def _shift(shift: dict | None, population: str) -> pd.DataFrame:
    if not shift or shift.get('results') is None or shift['results'].empty:
        return pd.DataFrame(columns=list(SHIFT_COLUMNS))
    res = shift['results']
    res = res[res['population'].astype(str) == population]
    return res[list(SHIFT_COLUMNS)].reset_index(drop=True)


def _states(shift: dict | None, population: str) -> pd.DataFrame:
    fsea = (shift or {}).get('fsea')
    if not fsea or fsea.get('results') is None or fsea['results'].empty:
        return pd.DataFrame(columns=list(STATE_COLUMNS))
    res = fsea['results']
    res = res[(res['population'].astype(str) == population) & np.isfinite(res['score'].astype(float))]
    return res[list(STATE_COLUMNS)].reset_index(drop=True)


def _markers(stats: dict | None, level, population: str) -> pd.DataFrame:
    markers = (stats or {}).get('markers')
    if not markers or markers.get('results') is None or markers['results'].empty:
        return pd.DataFrame(columns=list(MARKER_COLUMNS))
    several = any(info.get('population') for info in markers['sets'].values())
    group_of = {label: info['group'] for label, info in markers['sets'].items()
                if (info.get('level') == level if several or level is not None
                    else info.get('population') in ('', population))}
    res = markers['results']
    res = res[res['comparison'].isin(group_of)]
    q = 'stagewise.adj.P.Val' if 'stagewise.adj.P.Val' in res.columns else 'adj.P.Val'
    dominant = res['spillover_dominant'] if 'spillover_dominant' in res.columns else False
    return pd.DataFrame({
        'guide': res['combination'].astype(str).to_numpy(), 'group': res['comparison'].map(group_of).to_numpy(),
        'channel': res['channel'].astype(str).to_numpy(), 'delta': res['logFC'].to_numpy(float),
        'q': res[q].to_numpy(float), 'significant': res['significant'].to_numpy(bool),
        'spillover_dominant': np.asarray(dominant, dtype=bool) if not np.isscalar(dominant) else False})


# ---------------------------------------------------------------------------
# Calls per guide and group
# ---------------------------------------------------------------------------

def _direction(table: pd.DataFrame) -> pd.Series:
    """Signed call per (guide, group): +1, −1 or 0."""
    sign = np.sign(table['logFC'].to_numpy(float)) * table['significant'].to_numpy(bool)
    return pd.Series(sign, index=pd.MultiIndex.from_arrays([table['guide'], table['group']])).astype(int)


def _calls(acc: pd.DataFrame, test: str | None, cells, baseline, shift, states, markers, groups: list) -> pd.DataFrame:
    """One row per guide and group that any readout could test."""
    index = pd.MultiIndex.from_arrays([[], []], names=['guide', 'group'])
    parts = {}
    primary = acc[acc['test'] == test] if test else acc.iloc[0:0]
    other = acc[(acc['test'] != test)] if test else acc.iloc[0:0]
    if not primary.empty:
        key = pd.MultiIndex.from_arrays([primary['guide'], primary['group']])
        parts['logFC'] = pd.Series(primary['logFC'].to_numpy(float), index=key)
        parts['q_accumulation'] = pd.Series(primary['q'].to_numpy(float), index=key)
        parts['accumulation'] = _direction(primary)
    if not other.empty:
        parts['accumulation_other'] = _direction(other)
    if not shift.empty:
        key = pd.MultiIndex.from_arrays([shift['guide'], shift['group']])
        parts['shift_angle'] = pd.Series(shift['angle_deg'].to_numpy(float), index=key)
        parts['q_shift'] = pd.Series(shift['q'].to_numpy(float), index=key)
        parts['shift'] = pd.Series(shift['significant'].to_numpy(bool), index=key)
    for table, value, names in ((states, 'score', ('states_tested', 'states_towards', 'states_away')),
                                (markers, 'delta', ('markers_tested', 'markers_higher', 'markers_lower'))):
        if table.empty:
            continue
        keys = [table['guide'], table['group']]
        called = table['significant'].to_numpy(bool)
        sign = np.sign(table[value].to_numpy(float))
        parts[names[0]] = table.groupby(keys, sort=False).size()
        parts[names[1]] = pd.Series(called & (sign > 0), index=table.index).groupby(keys, sort=False).sum()
        parts[names[2]] = pd.Series(called & (sign < 0), index=table.index).groupby(keys, sort=False).sum()
    if not parts:
        return pd.DataFrame(index=index).reset_index()
    calls = pd.DataFrame(parts)
    calls.index.names = ['guide', 'group']
    calls = calls.reset_index()
    for column in ('logFC', 'q_accumulation', 'shift_angle', 'q_shift'):
        if column not in calls:
            calls[column] = np.nan
    for column in ('accumulation', 'states_tested', 'states_towards', 'states_away', 'markers_tested',
                   'markers_higher', 'markers_lower'):
        calls[column] = calls[column].fillna(0).astype(int) if column in calls else 0
    calls['accumulation_tested'] = np.isfinite(calls['logFC'].to_numpy(float))
    if 'accumulation_other' in calls:
        both = calls['accumulation_other'].notna() & (calls['accumulation'] != 0)
        calls['tests_agree'] = np.where(both, calls['accumulation_other'] == calls['accumulation'], False)
        calls['accumulation_other'] = calls['accumulation_other'].fillna(0).astype(int)
    else:
        calls['accumulation_other'] = 0
        calls['tests_agree'] = False
    calls['shift_tested'] = np.isfinite(calls['shift_angle'].to_numpy(float))
    calls['shift'] = calls['shift'].fillna(False).astype(bool) if 'shift' in calls else False
    calls['phenotype_tested'] = calls['shift_tested'] | (calls['states_tested'] > 0) | (calls['markers_tested'] > 0)
    calls['phenotype_readouts'] = (calls['shift'].astype(int)
                                   + ((calls['states_towards'] + calls['states_away']) > 0).astype(int)
                                   + ((calls['markers_higher'] + calls['markers_lower']) > 0).astype(int))
    n_cells = np.full(len(calls), np.nan)
    unbounded = np.zeros(len(calls), dtype=bool)
    if cells is not None and not cells.empty:
        for i, (guide, group) in enumerate(zip(calls['guide'], calls['group'])):
            if guide in cells.index and group in cells.columns:
                n_cells[i] = float(cells.at[guide, group])
                base = float(cells.at[guide, baseline]) if baseline in cells.columns else np.nan
                unbounded[i] = bool(calls['accumulation_tested'].iat[i]) and (n_cells[i] < 0.5 or base < 0.5)
    calls['cells'] = n_cells
    calls['unbounded'] = unbounded
    rank = {g: i for i, g in enumerate(groups)}
    calls['_g'] = calls['group'].map(rank)
    return calls.sort_values(['guide', '_g'], kind='stable').drop(columns='_g').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sentences
# ---------------------------------------------------------------------------

def _listed(items: list, limit: int = MAX_LISTED) -> str:
    items = [str(i) for i in items]
    if len(items) > limit:
        return ', '.join(items[:limit]) + f' and {len(items) - limit} more'
    return ', '.join(items)


def _plural(n: int, word: str) -> str:
    return f'{n} {word}' if n == 1 else f'{n} {word}s'


def _of_guide(synopsis: dict, table: str, guide: str) -> pd.DataFrame:
    """One guide's rows of a table of the synopsis (split once per table)."""
    cache = synopsis.setdefault('_by_guide', {})
    if table not in cache:
        frame = synopsis[table]
        cache[table] = (frame.iloc[0:0], dict(tuple(frame.groupby('guide', sort=False))))
    empty, rows = cache[table]
    return rows.get(guide, empty)


def guide_class(calls: pd.DataFrame) -> str:
    """The evidence class of one guide's rows of the calls table."""
    accumulation = bool((calls['accumulation'] != 0).any())
    phenotype = bool((calls['phenotype_readouts'] > 0).any())
    return BOTH if accumulation and phenotype else ACCUMULATION if accumulation else PHENOTYPE if phenotype else NONE


def guide_sentence(guide: str, synopsis: dict) -> str:
    """The templated description of one guide: its class, then one clause
    per readout that made a call, strongest group first."""
    mine = _of_guide(synopsis, 'calls', guide)
    baseline, test = synopsis['baseline'], synopsis['test']
    labels = synopsis.get('marker_labels') or {}
    kind = guide_class(mine) if not mine.empty else NONE
    out = [f'{kind}.']
    hits = mine[mine['accumulation'] != 0]
    if not hits.empty:
        clauses = []
        for sign, word in ((1, 'more'), (-1, 'fewer')):
            sub = hits[(hits['accumulation'] == sign) & ~hits['unbounded']]
            sub = sub.reindex(sub['logFC'].abs().sort_values(ascending=False).index)
            if not sub.empty:
                clauses.append(f'{word} in ' + _listed([f'{r.group} ({r.logFC:+.1f})' for r in sub.itertuples()]))
        gone = hits[hits['unbounded'] & (hits['accumulation'] < 0)]
        if not gone.empty:
            clauses.append('not detected in ' + _listed(list(gone['group'])))
        only = hits[hits['unbounded'] & (hits['accumulation'] > 0)]
        if not only.empty:
            clauses.append(f'detected in {_listed(list(only["group"]))} but not in {baseline}')
        text = f"Cells against {baseline} ({TESTS[test]}, log2 fold change): {'; '.join(clauses)}."
        if synopsis['tests'] and len(synopsis['tests']) > 1:
            agree = int(hits['tests_agree'].sum())
            text += f' Counts and frequencies agree in {agree} of {_plural(len(hits), "group")}.'
        out.append(text)
    moved = mine[mine['shift']].sort_values('shift_angle', key=np.abs, ascending=False)
    if not moved.empty:
        out.append('Phenotype shift in '
                   + _listed([f'{r.group} ({abs(r.shift_angle):.0f}°)' for r in moved.itertuples()]) + '.')
    states = _of_guide(synopsis, 'states', guide)
    states = states[states['significant']]
    if not states.empty:
        clauses = []
        for sign, word in ((1, 'towards'), (-1, 'away from')):
            sub = states[np.sign(states['score']) == sign]
            if sub.empty:
                continue
            strength = sub.groupby('state', sort=False)['score'].apply(lambda v: float(np.abs(v).max()))
            names = []
            for state in strength.sort_values(ascending=False).index:
                names.append(f"{state} ({_listed(list(sub.loc[sub['state'] == state, 'group']))})")
            clauses.append(f'{word} ' + _listed(names))
        out.append(f"Immune states: {'; '.join(clauses)}.")
    markers = _of_guide(synopsis, 'markers', guide)
    markers = markers[markers['significant']]
    if not markers.empty:
        top = markers.loc[markers['delta'].abs().idxmax()]
        name = labels.get(top['channel'], top['channel'])
        out.append(f"Markers: {_plural(len(markers), 'change')} in {_plural(markers['group'].nunique(), 'group')}; "
                   f"largest {name} {'higher' if top['delta'] > 0 else 'lower'} in {top['group']} "
                   f"({top['delta']:+.2f}).")
    if not mine.empty:
        n = int(len(mine))
        blind = int((~mine['phenotype_tested']).sum())
        if blind:
            out.append(f'Phenotype could not be tested in {blind} of {_plural(n, "group")} (too few cells).')
    return ' '.join(out)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _guide_order(guides: list, calls: pd.DataFrame, gene_group: dict, group_order: list, order: str) -> list:
    effect = calls.groupby('guide')['logFC'].apply(
        lambda v: float(np.nansum(np.where(calls.loc[v.index, 'accumulation'] != 0, v, 0.0)))) \
        if not calls.empty else pd.Series(dtype=float)

    def by_effect(g):
        return (float(effect.get(g, 0.0)), str(g).lower())
    if order == 'alphabetical':
        return sorted(guides, key=lambda g: str(g).lower())
    if order == 'group' and gene_group:
        rank = {name: i for i, name in enumerate(group_order)}
        return sorted(guides, key=lambda g: (rank.get(gene_group.get(g), len(rank)), *by_effect(g)))
    return sorted(guides, key=by_effect)


def assemble(stats: dict | None, shift: dict | None, scope: dict, *, baseline: str = '', test: str = '',
             order: str = 'group', fdr_scope: str = 'per_comparison', group_order: list | None = None,
             marker_labels: dict | None = None, gene_groups: dict | None = None) -> dict:
    """
    Everything the Synopsis shows for one *scope* (``scopes``).

    stats         : the statistics run (``flowcode_stats.run_statistics``
                    with its marker and gene-group results), or None.
    shift         : the phenotype-shift run with its immune states, or None.
    baseline, test: the chosen baseline group and accumulation test; each
                    falls back to ``resolve_baseline`` / ``resolve_test``.
    order         : guide order, a key of ``ORDERS``.
    group_order   : the sample groups in the Groups tab's order.
    marker_labels : {channel: name shown}.
    gene_groups   : the gene-group result (its 'assignment' and
                    'group_order'), or None.

    Returns {'scope', 'baseline', 'test', 'tests', 'groups', 'guides',
    'gene_group' ({guide: group}), 'gene_group_order', 'accumulation',
    'shift', 'states', 'markers' (long tables), 'calls' (guide × group),
    'guide_table' (one row per guide, with its sentence), 'samples' and
    'depth' ({group: samples, median barcoded events per sample}),
    'marker_labels', 'notes'}.
    """
    level, population = scope.get('level'), str(scope.get('population') or '')
    usable = stats if stats is not None and not _between_populations(stats) else None
    test = resolve_test(usable, test)
    baseline = resolve_baseline(usable, level, baseline) if usable is not None else None
    if usable is not None and baseline is not None:
        acc = _accumulation(usable, level, baseline, fdr_scope)
        cells, samples, depth = _presence(usable, level)
    else:
        acc, cells, samples, depth = pd.DataFrame(columns=list(ACCUMULATION_COLUMNS)), None, {}, {}
    shift_rows, state_rows = _shift(shift, population), _states(shift, population)
    marker_rows = _markers(stats, level, population)

    seen = list(dict.fromkeys([*( [baseline] if baseline else []), *acc['group'], *shift_rows['group'],
                               *state_rows['group'], *marker_rows['group']]))
    listed = [g for g in (group_order or []) if g in seen]
    groups = listed + [g for g in seen if g not in listed]
    calls = _calls(acc, test, cells, baseline, shift_rows, state_rows, marker_rows, groups)

    gene_group, gene_group_order = {}, []
    if gene_groups and gene_groups.get('assignment') is not None:
        table = gene_groups['assignment']
        gene_group = {str(f): str(g) for f, g in zip(table['feature'], table['group']) if isinstance(g, str) and g}
        gene_group_order = [g for g in gene_groups.get('group_order') or [] if g in set(gene_group.values())]
    guides = list(dict.fromkeys([*acc['guide'], *shift_rows['guide'], *state_rows['guide'], *marker_rows['guide']]))
    guides = _guide_order(guides, calls, gene_group, gene_group_order, order)

    notes = []
    blind = [g for g in groups if not calls.loc[calls['group'] == g, 'phenotype_tested'].any()] \
        if not calls.empty else []
    if blind and len(blind) < len(groups):
        notes.append(f"No guide had enough cells for a phenotype readout in {_listed(blind, 30)}.")
    synopsis = {'scope': dict(scope), 'baseline': baseline, 'test': test, 'tests': available_tests(usable),
                'groups': groups, 'guides': guides, 'gene_group': gene_group, 'gene_group_order': gene_group_order,
                'accumulation': acc, 'shift': shift_rows, 'states': state_rows, 'markers': marker_rows,
                'calls': calls, 'samples': samples, 'depth': depth, 'marker_labels': dict(marker_labels or {}),
                'notes': notes}
    synopsis['guide_table'] = _guide_table(synopsis)
    return synopsis


def _guide_table(synopsis: dict) -> pd.DataFrame:
    rows = []
    for guide in synopsis['guides']:
        mine = _of_guide(synopsis, 'calls', guide)
        hits = mine[mine['accumulation'] != 0]
        rows.append({
            'guide': guide, 'gene_group': synopsis['gene_group'].get(guide, ''), 'class': guide_class(mine),
            'groups_more': int((hits['accumulation'] > 0).sum()), 'groups_fewer': int((hits['accumulation'] < 0).sum()),
            'groups_tests_agree': int(hits['tests_agree'].sum()),
            'groups_shift': int(mine['shift'].sum()),
            'state_calls': int((mine['states_towards'] + mine['states_away']).sum()),
            'marker_calls': int((mine['markers_higher'] + mine['markers_lower']).sum()),
            'groups_two_phenotype_readouts': int((mine['phenotype_readouts'] >= 2).sum()),
            'groups_accumulation_tested': int(mine['accumulation_tested'].sum()),
            'groups_phenotype_tested': int(mine['phenotype_tested'].sum()),
            'synopsis': guide_sentence(guide, synopsis)})
    columns = ['guide', 'gene_group', 'class', 'groups_more', 'groups_fewer', 'groups_tests_agree', 'groups_shift',
               'state_calls', 'marker_calls', 'groups_two_phenotype_readouts', 'groups_accumulation_tested',
               'groups_phenotype_tested', 'synopsis']
    return pd.DataFrame(rows, columns=columns)


def overview_text(synopsis: dict) -> str:
    """The summary of one scope in plain lines: what was read, how many
    calls each readout made, the guides of each evidence class and each
    group's calls."""
    table, calls = synopsis['guide_table'], synopsis['calls']
    scope = synopsis['scope']
    lines = [f"{scope['label']}: {_plural(len(synopsis['guides']), 'guide')}, "
             f"{_plural(len(synopsis['groups']), 'group')}."]
    if synopsis['baseline'] and synopsis['test']:
        lines.append(f"Accumulation: cells of each guide in each group against {synopsis['baseline']} "
                     f"({TESTS[synopsis['test']]} test).")
    else:
        lines.append('Accumulation: no statistics to read.')
    if calls.empty:
        return '\n'.join(lines + ['No readout has results for this population.'])
    lines.append(f"Calls (guide × group): accumulation {int((calls['accumulation'] != 0).sum())} of "
                 f"{int(calls['accumulation_tested'].sum())} tested; phenotype shift {int(calls['shift'].sum())} of "
                 f"{int(calls['shift_tested'].sum())}; immune states in "
                 f"{int(((calls['states_towards'] + calls['states_away']) > 0).sum())} of "
                 f"{int((calls['states_tested'] > 0).sum())}; markers in "
                 f"{int(((calls['markers_higher'] + calls['markers_lower']) > 0).sum())} of "
                 f"{int((calls['markers_tested'] > 0).sum())}.")
    lines.append('')
    for kind in CLASSES:
        names = list(table.loc[table['class'] == kind, 'guide'])
        lines.append(f"{kind} ({len(names)}): {_listed(names, 200) if names else 'none'}.")
    lines.append('')
    lines.append('Per group (guides with a call: more cells, fewer cells, phenotype; median barcoded cells per '
                 'sample):')
    for group in synopsis['groups']:
        sub = calls[calls['group'] == group]
        depth = synopsis['depth'].get(group)
        cells = f'{depth:,.0f} cells' if depth is not None and np.isfinite(depth) else 'cells not counted'
        if group == synopsis['baseline']:
            head = f'  {group} (baseline; {cells}):'
            lines.append(f"{head} phenotype {int((sub['phenotype_readouts'] > 0).sum())} of "
                         f"{int(sub['phenotype_tested'].sum())} tested.")
            continue
        phenotype = (f"phenotype {int((sub['phenotype_readouts'] > 0).sum())} of "
                     f"{int(sub['phenotype_tested'].sum())} tested" if sub['phenotype_tested'].any()
                     else 'phenotype not testable')
        lines.append(f"  {group} ({cells}): more {int((sub['accumulation'] > 0).sum())}, fewer "
                     f"{int((sub['accumulation'] < 0).sum())}, {phenotype}.")
    unbounded = int(calls['unbounded'].sum())
    if unbounded:
        lines += ['', f'{_plural(unbounded, "guide × group cell")} had no cell on one side of the comparison: '
                      'shown as not detected, without a fold change.']
    for note in synopsis['notes']:
        lines += ['', note]
    return '\n'.join(lines)


def synopsis_text(synopsis: dict) -> str:
    """The overview followed by every guide's sentence, for the export."""
    lines = [overview_text(synopsis), '', 'Guides', '']
    for row in synopsis['guide_table'].itertuples(index=False):
        group = f' [{row.gene_group}]' if row.gene_group else ''
        lines.append(f'{row.guide}{group}: {row.synopsis}')
    return '\n'.join(lines)


def write_tables(folder, synopses: list) -> list:
    """``synopsis_guides.csv`` (one row per guide), ``synopsis_calls.csv``
    (one row per guide and group) and ``synopsis.txt`` for the scopes in
    *synopses*; returns the paths written."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    synopses = [s for s in synopses if s is not None and not s['calls'].empty]
    if not synopses:
        return []
    written = []
    for name, key in (('synopsis_guides.csv', 'guide_table'), ('synopsis_calls.csv', 'calls')):
        frames = [s[key].assign(population=s['scope']['label'], baseline=s['baseline'] or '',
                                accumulation_test=TESTS.get(s['test'], '')) for s in synopses]
        table = pd.concat(frames, ignore_index=True)
        lead = ['population', 'baseline', 'accumulation_test']
        table = table[lead + [c for c in table.columns if c not in lead]]
        path = folder / name
        table.to_csv(path, index=False)
        written.append(path)
    path = folder / 'synopsis.txt'
    path.write_text('\n\n\n'.join(synopsis_text(s) for s in synopses), encoding='utf-8')
    written.append(path)
    return written
