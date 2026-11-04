"""
flowcode_gene_groups.py
-----------------------
Gene-group effects of FlowCode knockout screens. No Qt.

What it asks
~~~~~~~~~~~~
A library's guides are sorted into gene groups by a table the user
provides (gene → group: chemokine receptors, integrins, AP-1 factors,
sterol metabolism and so on). For each comparison, such as "Gut vs
Spleen", each group gets a score: do this family's knockouts, taken
together, change the readout?

**Reading the score.** Every guide of a group counts the same way (sign
+1; a family has no "enhancer / inhibitor" split), so the score is the
mean of −z over the group's guides, z being each guide's statistic from
the statistics run. A positive score in "Gut vs Spleen" means the group's
knockouts are depleted in the gut: the family is needed for the cells to
accumulate there. A negative score means its knockouts are enriched there.

Three tests per group and comparison:

* **Competitive** (CAMERA, all signs +1): do the family's knockouts move
  one way more than the rest of their library? This is the test of
  direction: it follows the score, and a family whose members move both
  ways does not pass it.
* **Any change** (rotation, mean of z²): is anything happening in the
  family, whatever the direction? Opposite changes add up, so a family
  whose members are needed in different tissues still shows up. One
  strongly changed guide can be enough, so read it with the leading guides.
* **Self-contained** (rotation of the score): its null hypothesis is that
  no member changes at all. Small changes in the same direction add up, so
  it is the most sensitive to a weak, consistent shift; but with strong
  opposite changes it can also pass with a score near 0, so it is not by
  itself a test of direction.

The statistics behind it
~~~~~~~~~~~~~~~~~~~~~~~~
The groups are sets of the signed pathway engine
(``flowcode_pathways.run_pathways``) with every sign +1: the same
percentile-matched rotation test per library, weighted Stouffer
combination across libraries, CAMERA, any-change statistic and BH FDR per
comparison. The one difference from GO pathways is that a group may hold
more than half of a library's tested guides (``max_fraction`` 1).

Matching guides to groups
~~~~~~~~~~~~~~~~~~~~~~~~~
A combination's group is found from, in order: a gene set by hand in the
gene names dialog (looked up in the table by name, then by MGI symbol); its
own name in the table (ignoring case); its MGI symbol (symbol or unique
synonym, ``flowcode_pathways.match_features``) equal to the MGI symbol of
a table gene. Combinations with none are unassigned: counted, not tested.
NON-Targetting combinations are controls, never members.

Built-in tables
~~~~~~~~~~~~~~~
The lab's Homing, Treg transcription factor (OTB01) and Metabolism
(Metab 1–4) tables are installed beside this module (``BUILTIN_TABLES``)
with their gene, group, area and literature reference columns. Picking one
copies its rows into the analysis settings, exactly as loading a CSV does.
When no table is loaded, the built-in table giving a group to the largest
share of the experiment's guides is picked, if that share is at least
``AUTO_PICK_COVERAGE``.

Deliberate v1 simplifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* A guide shared by several libraries after merging identically named
  guides (e.g. a gene with a guide in two Homing sets) is not a group
  member, as in the pathway analysis; it is listed as such in the
  assignment table.
* Each gene belongs to one group only.
"""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

import flowcode_pathways as fpw

GROUP_PREFIX = 'group:'
GROUP_COLUMNS = ('gene', 'group', 'colour', 'area')
REQUIRED_COLUMNS = ('gene', 'group')

MATCH_OVERRIDE = fpw.MATCH_OVERRIDE
MATCH_NAME = 'name'
MATCH_GENE = 'gene symbol'
MATCH_CONTROL = fpw.MATCH_CONTROL
MATCH_UNASSIGNED = 'unassigned'

_COLOUR = re.compile(r'^#[0-9a-fA-F]{6}$')


class GroupTableError(ValueError):
    """The gene-group table cannot be used."""


def read_group_table(path) -> pd.DataFrame:
    """
    A gene-group CSV: columns gene and group (any case, any order), and
    optionally colour (#rrggbb) and area; other columns are ignored. Rows
    with an empty gene or group are skipped; a gene listed twice (ignoring
    case) keeps its first row when both rows give the same group. Returns
    gene, group, colour, area ('' when absent), in file order.

    Raises GroupTableError for a missing column, no usable row, or a gene
    given two different groups.
    """
    with open(Path(path), newline='', encoding='utf-8-sig') as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise GroupTableError('The gene-group table is empty.')
        columns = {str(c).strip().lower(): c for c in reader.fieldnames if c is not None}
        missing = [c for c in REQUIRED_COLUMNS if c not in columns]
        if missing:
            raise GroupTableError('The gene-group table needs the column(s) ' + ', '.join(missing) + '.')
        raw = [{name: str(row.get(columns[name]) or '').strip() if name in columns else ''
                for name in GROUP_COLUMNS} for row in reader]
    kept, first_group, first_name, conflicts = [], {}, {}, {}
    for row in raw:
        if not row['gene'] or not row['group']:
            continue
        key = row['gene'].lower()
        if key in first_group:
            if first_group[key] != row['group']:
                conflicts.setdefault(first_name[key], {first_group[key]}).add(row['group'])
            continue
        first_group[key], first_name[key] = row['group'], row['gene']
        row['colour'] = row['colour'] if _COLOUR.match(row['colour']) else ''
        kept.append(row)
    if conflicts:
        listed = '; '.join(f"{g} ({', '.join(sorted(gs))})" for g, gs in sorted(conflicts.items()))
        raise GroupTableError('Each gene can be in one group only: ' + listed + '.')
    if not kept:
        raise GroupTableError('The gene-group table has no gene with a group.')
    return pd.DataFrame(kept, columns=list(GROUP_COLUMNS))


def group_order(table: pd.DataFrame) -> list:
    """The groups in order of first appearance in the table."""
    return list(dict.fromkeys(table['group']))


def assign_groups(features, table: pd.DataFrame, symbols: dict, synonyms: dict,
                  overrides: dict | None = None) -> pd.DataFrame:
    """
    The group of each feature (combination name). Returns feature, gene
    (the table's gene, '' when none), group ('' when none) and match
    (override, name, gene symbol, control, unassigned).
    """
    group_of = dict(zip(table['gene'], table['group']))
    by_name = {g.lower(): g for g in table['gene']}
    table_symbols = fpw.match_features(list(table['gene']), symbols, synonyms)
    by_symbol: dict[str, str] = {}
    shared: set = set()
    for gene, symbol in zip(table_symbols['feature'], table_symbols['gene']):
        if not symbol:
            continue
        key = symbol.lower()
        if key in by_symbol and by_symbol[key] != gene:
            shared.add(key)
        by_symbol.setdefault(key, gene)
    for key in shared:
        del by_symbol[key]

    def table_gene(name: str) -> str:
        if not name:
            return ''
        if name.lower() in by_name:
            return by_name[name.lower()]
        return by_symbol.get(name.lower(), '')

    rows = []
    for _i, m in fpw.match_features(features, symbols, synonyms, overrides).iterrows():
        feature, symbol, how = m['feature'], m['gene'], m['match']
        if how == fpw.MATCH_CONTROL:
            rows.append((feature, '', '', MATCH_CONTROL))
            continue
        gene, match = '', MATCH_UNASSIGNED
        if how == fpw.MATCH_OVERRIDE:
            gene = table_gene(symbol)
            if not gene:
                gene = by_symbol.get(fpw.match_features([symbol], symbols, synonyms)['gene'].iloc[0].lower(), '')
            match = MATCH_OVERRIDE if gene else MATCH_UNASSIGNED
        elif str(feature).strip().lower() in by_name:
            gene, match = by_name[str(feature).strip().lower()], MATCH_NAME
        elif symbol and symbol.lower() in by_symbol:
            gene, match = by_symbol[symbol.lower()], MATCH_GENE
        rows.append((feature, gene, group_of.get(gene, ''), match))
    return pd.DataFrame(rows, columns=['feature', 'gene', 'group', 'match'])


def group_sets(table: pd.DataFrame) -> pd.DataFrame:
    """The groups as rows of the pathway engine's set table (every sign +1)."""
    return pd.DataFrame({'set_id': GROUP_PREFIX + table['group'], 'set_name': table['group'],
                         'gene': table['gene'], 'sign': 1, 'evidence_class': ''})


# ---------------------------------------------------------------------------
# Built-in tables
# ---------------------------------------------------------------------------

# The lab's libraries, as (name shown, file beside this module). A built-in
# table is read like any other CSV; its rows are copied into the analysis
# settings when picked, so a later version of the file never changes an
# experiment's saved results.
BUILTIN_TABLES = (
    ('Homing (sets 1–3)', 'flowcode_gene_groups_homing.csv'),
    ('Treg transcription factors (OTB01)', 'flowcode_gene_groups_tx_otb01.csv'),
    ('Metabolism (Metab 1–4)', 'flowcode_gene_groups_metab.csv'),
)
# The share of a library's guides a built-in table must give a group before
# it is picked for the user.
AUTO_PICK_COVERAGE = 0.8


def builtin_path(file_name: str) -> Path:
    """The path of a built-in table file."""
    return Path(__file__).with_name(file_name)


def builtin_tables() -> list:
    """(name, file name) of the built-in tables that are installed."""
    return [(name, file) for name, file in BUILTIN_TABLES if builtin_path(file).is_file()]


_BUILTIN_CACHE: dict = {}


def builtin_table(file_name: str) -> pd.DataFrame:
    """A built-in table, read with ``read_group_table`` (kept until the file
    changes)."""
    path = builtin_path(file_name)
    key = (file_name, path.stat().st_mtime_ns)
    if key not in _BUILTIN_CACHE:
        _BUILTIN_CACHE[key] = read_group_table(path)
    return _BUILTIN_CACHE[key].copy()


def builtin_of(file_name: str, table: pd.DataFrame | None) -> str | None:
    """'same' when *table* has the genes and groups of the built-in table
    *file_name* (``table_digest``), 'older' when it was picked from that
    file but the installed file now differs, else None."""
    if table is None or file_name not in {f for _n, f in builtin_tables()}:
        return None
    return 'same' if table_digest(table) == table_digest(builtin_table(file_name)) else 'older'


def coverage(features, table: pd.DataFrame, symbols: dict, synonyms: dict, overrides: dict | None = None) -> float:
    """The share of the non-control *features* given a group by *table*
    (``assign_groups``); 0 when there are none."""
    assigned = assign_groups(features, table, symbols, synonyms, overrides)
    guides = assigned[assigned['match'] != MATCH_CONTROL]
    return float((guides['group'] != '').mean()) if len(guides) else 0.0


def best_builtin(features, symbols: dict, synonyms: dict, overrides: dict | None = None,
                 minimum: float = AUTO_PICK_COVERAGE) -> tuple | None:
    """(name, file name, coverage) of the built-in table giving a group to
    the largest share of *features*, when that share is at least *minimum*;
    else None."""
    best = None
    for name, file in builtin_tables():
        share = coverage(features, builtin_table(file), symbols, synonyms, overrides)
        if share >= minimum and (best is None or share > best[2]):
            best = (name, file, share)
    return best


def run_gene_groups(stats: dict, fit_log: list, table: pd.DataFrame, *, overrides: dict | None = None,
                    rotations: int = fpw.DEFAULT_ROTATIONS, seed: int = fpw.DEFAULT_SEED,
                    min_size: int = fpw.DEFAULT_MIN_SIZE, library_guides: dict | None = None) -> dict:
    """
    Gene-group effects of a statistics run (``flowcode_stats.run_statistics``
    with *fit_log*, counts tested by the QL test), with the bundled MGI names
    for matching. Returns ``flowcode_pathways.run_pathways``'s result (one
    row per group and comparison in 'pathways', set_key ``group:<name>``)
    plus 'assignment' (``assign_groups``), 'group_order', 'table' (the
    groups used) and, with *library_guides* ({library: [combination
    names]}), 'allocation' (``allocation``).
    """
    symbols, synonyms, _meta = fpw.bundled_names()
    features = list(dict.fromkeys(f for r in fit_log for f in r['counts'].columns))
    assignment = assign_groups(features, table, symbols, synonyms, overrides)
    matches = pd.DataFrame({'feature': assignment['feature'], 'gene': assignment['gene'],
                            'match': assignment['match']})
    out = fpw.run_pathways(stats, fit_log, group_sets(table), matches, rotations=rotations, seed=seed,
                           min_size=min_size, max_fraction=1.0)
    unassigned = int((assignment['match'] == MATCH_UNASSIGNED).sum())
    if unassigned:
        out['notes'].append(f'{unassigned} combination(s) have no gene group and are not tested.')
    out['assignment'] = assignment
    out['group_order'] = group_order(table)
    out['table'] = table.reset_index(drop=True)
    if library_guides:
        out['allocation'] = allocation(library_guides, table, symbols, synonyms, overrides, out['group_order'])
    out['settings'].update(groups=len(out['group_order']), unassigned=unassigned,
                           overrides=dict(overrides or {}))
    return out


def table_digest(table: pd.DataFrame) -> str:
    """A short digest of the table's genes and groups (for settings.txt)."""
    text = '\n'.join(f'{g.lower()}\t{grp}' for g, grp in sorted(zip(table['gene'], table['group'])))
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

# ColorBrewer Set1 with its yellow replaced by Dark2's teal: eight colours that
# stay apart for the common kinds of colour blindness. Grey is kept for the
# NON-Targetting guides.
CATEGORICAL = ('#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#1b9e77')
CONTROL_COLOUR = '#999999'
UNASSIGNED_COLOUR = '#dddddd'
CONTROL_SLICE = 'NON-Targetting'
UNASSIGNED_SLICE = 'Unassigned'
# Beyond eight groups, each area of the table gets a ColorBrewer hue family
# (light to dark) and its groups shades within it.
AREA_FAMILIES = (
    ('#fee391', '#fec44f', '#fe9929', '#ec7014', '#cc4c02', '#8c2d04'),     # YlOrBr
    ('#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#54278f', '#3f007d'),     # Purples
    ('#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b'),     # Greens
    ('#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'),     # Blues
    ('#fa9fb5', '#f768a1', '#dd3497', '#ae017e', '#7a0177', '#49006a'),     # RdPu
    ('#c7eae5', '#80cdc1', '#35978f', '#01665e', '#003c30', '#00261e'),     # BrBG, teal side
)
_SHADES = {1: (2,), 2: (1, 3), 3: (0, 2, 4), 4: (0, 1, 3, 5), 5: (0, 1, 2, 3, 5), 6: (0, 1, 2, 3, 4, 5)}
# More than eight groups and no areas: ColorBrewer Paired, then Set3.
_MANY = ('#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6',
         '#6a3d9a', '#ffff99', '#b15928', '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462',
         '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f')


def group_areas(table: pd.DataFrame) -> dict:
    """{group: area} from the table's area column ('' where none)."""
    out: dict = {}
    for group, area in zip(table['group'], table['area']):
        out.setdefault(group, area or out.get(group, ''))
    return out


def group_colours(order: list, areas: dict | None = None, chosen: dict | None = None) -> dict:
    """
    {group: colour} for the groups in *order*: up to eight from
    ``CATEGORICAL`` in order; more than eight, when every group has an
    area, a hue family per area (in order of first appearance) with shades
    within it, else ColorBrewer Paired then Set3. Colours in *chosen*
    (from the table or set by hand) win.
    """
    areas = areas or {}
    if len(order) <= len(CATEGORICAL):
        out = {g: CATEGORICAL[i] for i, g in enumerate(order)}
    elif all(areas.get(g) for g in order):
        by_area: dict[str, list] = {}
        for g in order:
            by_area.setdefault(areas[g], []).append(g)
        out = {}
        for a, (_area, members) in enumerate(by_area.items()):
            family = AREA_FAMILIES[a % len(AREA_FAMILIES)]
            picks = _SHADES.get(len(members), tuple(i % len(family) for i in range(len(members))))
            out.update({g: family[picks[i]] for i, g in enumerate(members)})
    else:
        out = {g: _MANY[i % len(_MANY)] for i, g in enumerate(order)}
    out.update({g: c for g, c in (chosen or {}).items() if g in out and _COLOUR.match(str(c))})
    return out


def table_colours(table: pd.DataFrame) -> dict:
    """{group: colour} given in the table's colour column (first per group)."""
    out: dict = {}
    for group, colour in zip(table['group'], table['colour']):
        if colour and group not in out:
            out[group] = colour
    return out


# ---------------------------------------------------------------------------
# Guide allocation
# ---------------------------------------------------------------------------

def allocation(library_guides: dict, table: pd.DataFrame, symbols: dict, synonyms: dict,
               overrides: dict | None = None, order: list | None = None) -> pd.DataFrame:
    """
    Guides per group, counted per library and pooled: *library_guides* is
    {library: [combination names]} (one entry for a single library). A
    guide in two libraries counts twice. Returns group (the groups in
    *order*, then NON-Targetting and Unassigned when present), guides, and
    one column per library.
    """
    order = list(order or group_order(table))
    libs = list(library_guides)
    counts = {g: dict.fromkeys(libs, 0) for g in order + [CONTROL_SLICE, UNASSIGNED_SLICE]}
    for lib, guides in library_guides.items():
        names = list(dict.fromkeys(str(g) for g in guides))
        assigned = assign_groups(names, table, symbols, synonyms, overrides).set_index('feature')
        for name in guides:
            row = assigned.loc[str(name)]
            if row['match'] == MATCH_CONTROL:
                key = CONTROL_SLICE
            elif row['group'] in counts:
                key = row['group']
            else:
                key = UNASSIGNED_SLICE
            counts[key][lib] += 1
    rows = [(g, sum(v.values()), *[v[lib] for lib in libs]) for g, v in counts.items()
            if g in order or sum(v.values())]
    return pd.DataFrame(rows, columns=['group', 'guides', *libs])


def donut_figure(alloc: pd.DataFrame, colours: dict, is_dark: bool = False, population: str = '',
                 areas: dict | None = None, title: str = 'Guide allocation'):
    """
    The guides of every library, pooled, as one ring: a slice per group
    (its count printed when the slice is at least 2.8% of the ring), the
    total in the middle, a legend with counts and shares. With *areas*
    (more than eight groups) the legend has area headings.
    """
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch
    from honeychrome.view_components import differential_plots as dp

    view = alloc[alloc['guides'] > 0]
    if view.empty:
        return dp.message_figure('No guide is in the libraries.', title, is_dark)
    total = int(view['guides'].sum())
    ink = '#eeeeee' if is_dark else '#222222'
    muted = '#aaaaaa' if is_dark else '#666666'
    surface = '#2b2b2b' if is_dark else 'white'
    colour_of = dict(colours)
    colour_of.update({CONTROL_SLICE: CONTROL_COLOUR, UNASSIGNED_SLICE: UNASSIGNED_COLOUR})
    names, sizes = list(view['group']), [int(n) for n in view['guides']]
    fills = [colour_of.get(g, UNASSIGNED_COLOUR) for g in names]
    fig = Figure(figsize=(10.0, 5.6), layout='constrained')
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])
    ax = fig.add_subplot(grid[0, 0])
    wedges, _texts = ax.pie(sizes, colors=fills, startangle=90, counterclock=False,
                            wedgeprops=dict(width=0.38, edgecolor=surface, linewidth=2))
    for wedge, n, fill in zip(wedges, sizes, fills):
        if n / total < 0.028:
            continue
        angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2.0)
        r, g, b = (int(fill[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
        text = '#222222' if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55 else 'white'
        ax.text(0.81 * np.cos(angle), 0.81 * np.sin(angle), str(n), ha='center', va='center', fontsize=9,
                color=text, fontweight='bold')
    libraries = [c for c in alloc.columns if c not in ('group', 'guides')]
    ax.text(0, 0.07, str(total), ha='center', va='center', fontsize=22, color=ink, fontweight='bold')
    ax.text(0, -0.15, 'guides' + (f'\n{len(libraries)} libraries' if len(libraries) > 1 else ''), ha='center',
            va='center', fontsize=10, color=muted, linespacing=1.4)
    ax.set_aspect('equal')
    legend_ax = fig.add_subplot(grid[0, 1])
    legend_ax.axis('off')

    def entry(g, n):
        return Patch(facecolor=colour_of.get(g, UNASSIGNED_COLOUR), edgecolor='none',
                     label=f'{g}  {n} ({100.0 * n / total:.0f}%)')

    count_of = dict(zip(names, sizes))
    headings = set()
    handles = []
    if areas and len(names) > len(CATEGORICAL):
        by_area: dict[str, list] = {}
        for g in names:
            if g not in (CONTROL_SLICE, UNASSIGNED_SLICE):
                by_area.setdefault(areas.get(g) or 'Other', []).append(g)
        for area, members in by_area.items():
            handles.append(Patch(facecolor='none', edgecolor='none', label=area))
            headings.add(area)
            handles += [entry(g, count_of[g]) for g in members]
    else:
        handles += [entry(g, count_of[g]) for g in names if g not in (CONTROL_SLICE, UNASSIGNED_SLICE)]
    handles += [entry(g, count_of[g]) for g in (CONTROL_SLICE, UNASSIGNED_SLICE) if g in count_of]
    leg = legend_ax.legend(handles=handles, loc='center left', frameon=False, handlelength=1.2, handleheight=1.0,
                           fontsize=8.5 if len(handles) > 12 else 10, labelspacing=0.45 if len(handles) > 12 else 0.7)
    for text in leg.get_texts():
        text.set_color(muted if text.get_text() in headings else ink)
        if text.get_text() in headings:
            text.set_fontweight('bold')
    ax.set_title(f'{population}: {title.lower()}' if population else title, fontsize=11, color=ink)
    dp.style_figure_theme(fig, is_dark)
    for text in leg.get_texts():
        text.set_color(muted if text.get_text() in headings else ink)
    return fig


# ---------------------------------------------------------------------------
# Spider plot, table and export
# ---------------------------------------------------------------------------

SPIDER_COMPARISONS = 6
COMPARISON_FALLBACK = ('#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#999999')


def significant(row, fdr_threshold: float) -> tuple:
    """(competitive, any-change) FDR at or below *fdr_threshold*."""
    return (bool(float(row['competitive_FDR']) <= fdr_threshold),
            bool(float(row['mixed_FDR']) <= fdr_threshold))


def group_table(result: dict, comparison: str | None = None) -> pd.DataFrame:
    """The group results (of one comparison, or all) with each library's
    summary, in the table's group order."""
    table = fpw.pathway_table(result, comparison)
    order = {GROUP_PREFIX + g: i for i, g in enumerate(result.get('group_order') or [])}
    return table.assign(_order=table['set_key'].map(order)).sort_values(
        ['comparison', '_order'], kind='stable', ignore_index=True).drop(columns='_order')


def spider_comparisons(comparisons: list, chosen: list | None, pairs: list | None = None,
                       reference: str | None = None, limit: int = SPIDER_COMPARISONS) -> list:
    """The comparisons drawn on the spider: those in *chosen* that were run
    (at most *limit*); without a choice, those against the *reference*
    group (``pairs``: (base, other, level)), else the first *limit*."""
    if chosen:
        return [c for c in comparisons if c in set(chosen)][:limit]
    if pairs and reference:
        from flowcode_stats import comparison_label
        against = [comparison_label(*p) for p in pairs if p[0] == reference]
        against = [c for c in comparisons if c in set(against)]
        if against:
            return against[:limit]
    return list(comparisons)[:limit]


def comparison_colours(comparisons: list, pairs: list | None, group_colours_by_name: dict | None) -> dict:
    """{comparison: colour}: the Groups table colour of the comparison's
    first-named group, else ColorBrewer Set1 in order."""
    from flowcode_stats import comparison_label
    first = {comparison_label(*p): p[1] for p in pairs or []}
    out, spare = {}, iter(COMPARISON_FALLBACK * 4)
    used = set()
    for c in comparisons:
        colour = (group_colours_by_name or {}).get(first.get(c, ''))
        if not colour or colour in used:
            colour = next(c2 for c2 in spare if c2 not in used)
        out[c] = colour
        used.add(colour)
    return out


def _nice_limit(value: float) -> float:
    """The smallest of 1, 2, 2.5, 5 times a power of ten at or above *value*."""
    power = 10.0 ** np.floor(np.log10(value))
    return float(next(m * power for m in (1.0, 2.0, 2.5, 5.0, 10.0) if m * power >= value * (1.0 - 1e-9)))


SPOKE_LABEL_GAP = 1.06
SPOKE_LABEL_WIDTH = 18
SPOKE_LABEL_PAD = 0.12
SPIDER_DIAMETER = 5.6
SPIDER_MARGIN = 0.15


def _spoke_label(ax, angle: float, radius: float, text: str, colour: str):
    """A group's name just outside the web at *angle*, wrapped to
    ``SPOKE_LABEL_WIDTH`` characters a line and aligned away from the web
    (left-aligned on the right, right-aligned on the left, centred at the
    top and bottom), so long names neither cross the plot nor get cut.
    The label is left out of the layout; ``_fit_spider`` makes room
    for it."""
    import textwrap

    across, up = np.sin(angle), np.cos(angle)
    ha = 'left' if across > 0.2 else 'right' if across < -0.2 else 'center'
    va = 'bottom' if up > 0.2 else 'top' if up < -0.2 else 'center'
    label = ax.text(angle, radius, textwrap.fill(text, SPOKE_LABEL_WIDTH, break_on_hyphens=False), ha=ha, va=va,
                    ma=ha if ha != 'center' else 'center', fontsize=8, color=colour, clip_on=False)
    label.set_in_layout(False)
    return label


def _fit_spider(fig, ax, labels: list, legend, title):
    """
    Lay the spider out by hand: a web ``SPIDER_DIAMETER`` inches across,
    centred, with room on both left and right for the widest side label and
    above and below for the tallest top or bottom label, the title above
    and the legend below; the figure is sized to fit. The web is a circle
    centred in its space, so room made for a long label on one side only
    would push a label on the other side off the figure.
    """
    from matplotlib.backends.backend_agg import RendererAgg

    renderer = RendererAgg(1, 1, fig.dpi)
    side, end = 0.0, 0.0
    for label in labels:
        box = label.get_window_extent(renderer=renderer)
        if label.get_ha() != 'center':
            side = max(side, box.width / fig.dpi)
        if label.get_va() != 'center':
            end = max(end, box.height / fig.dpi)
    offset = (SPOKE_LABEL_GAP - 1.0) * SPIDER_DIAMETER / 2.0 + SPOKE_LABEL_PAD
    side += offset
    end += offset
    key = legend.get_window_extent(renderer=renderer)
    heading = title.get_window_extent(renderer=renderer).height / fig.dpi
    width = max(SPIDER_DIAMETER + 2.0 * side, key.width / fig.dpi) + 2.0 * SPIDER_MARGIN
    height = (2.0 * SPIDER_MARGIN + heading + SPIDER_MARGIN + 2.0 * end + SPIDER_DIAMETER + SPIDER_MARGIN
              + key.height / fig.dpi)
    fig.set_size_inches(width, height)
    bottom = SPIDER_MARGIN + key.height / fig.dpi + SPIDER_MARGIN + end
    ax.set_position([(width - SPIDER_DIAMETER) / 2.0 / width, bottom / height,
                     SPIDER_DIAMETER / width, SPIDER_DIAMETER / height])
    legend.set_bbox_to_anchor((0.5, SPIDER_MARGIN / height), transform=fig.transFigure)
    title.set_y(1.0 - SPIDER_MARGIN / height)


def spider_figure(result: dict, comparisons: list, colours: dict, label_colours: dict, fdr_threshold: float,
                  is_dark: bool = False, population: str = '', min_size: int | None = None):
    """
    Gene groups as spokes, *comparisons* as polygons (*colours*:
    {comparison: colour}). The radius is the group score on a symmetric
    scale with a dashed zero ring: outside the ring the group's knockouts
    are depleted in the comparison's first group (the family is needed
    there), inside they are enriched. Filled point: competitive FDR at or
    below the threshold; ring: any-change FDR at or below it. Groups not
    tested (fewer guides than the minimum) are labelled and have no point.
    Hover and double-click need ``fig._spider_points`` (data theta, r),
    ``_spider_keys`` ((comparison, set_key)) and ``_spider_axes``.
    """
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from honeychrome.view_components import differential_plots as dp

    table = result['pathways']
    order = list(result.get('group_order') or [])
    shown = [c for c in comparisons if c in set(table['comparison'])]
    if not order or not shown:
        return dp.message_figure('No gene group could be tested.', 'Gene groups', is_dark)
    view = table[table['comparison'].isin(shown)]
    lim = float(np.nanmax(np.abs(view['score'].astype(float)))) if not view.empty else 1.0
    lim = _nice_limit(max(lim, 1.0))
    fg = '#eeeeee' if is_dark else '#222222'
    face = '#2b2b2b' if is_dark else 'white'
    angles = np.linspace(0.0, 2.0 * np.pi, len(order), endpoint=False)
    fig = Figure(figsize=(8.6, 8.4))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 2.08 * lim)
    ring = np.linspace(0.0, 2.0 * np.pi, 361)
    ax.plot(ring, np.full(ring.size, lim), ls='--', lw=1.0, color='grey', zorder=1)
    ticks = [t for t in (-lim, -lim / 2.0, 0.0, lim / 2.0, lim)]
    ax.set_yticks([t + lim for t in ticks])
    ax.set_yticklabels([f'{t:+g}' if t else '0' for t in ticks], fontsize=7, color='grey')
    ax.set_rlabel_position(180.0 / len(order))
    tested_keys = set(view['set_key'])
    ax.set_xticks(angles)
    ax.set_xticklabels([])
    points, keys, texts, fills, edges, rings = [], [], [], [], [], []
    for comparison in shown:
        rows = view[view['comparison'] == comparison].set_index('set_key')
        colour = colours.get(comparison, fg)
        radii = []
        for angle, g in zip(angles, order):
            key = GROUP_PREFIX + g
            if key not in rows.index:
                radii.append(np.nan)
                continue
            r = rows.loc[key]
            radius = float(np.clip(float(r['score']), -lim, lim)) + lim
            radii.append(radius)
            comp, mixed = significant(r, fdr_threshold)
            points.append((angle, radius))
            keys.append((comparison, key))
            fills.append(colour if comp else face)
            edges.append(colour)
            rings.append(mixed)
            texts.append(f"{g} — {comparison}\nscore {r['score']:+.2f}, {int(r['members'])} guide(s), agreement "
                         f"{r['concordance']:.0%}\ncompetitive p {r['competitive_P.Value']:.3g}, FDR "
                         f"{r['competitive_FDR']:.3g}\nany change p {r['mixed_P.Value']:.3g}, FDR "
                         f"{r['mixed_FDR']:.3g}\nself-contained p {r['P.Value']:.3g}, FDR {r['FDR']:.3g}\n"
                         f"leading: {r['leading']}")
        closed = np.append(radii, radii[0])
        ax.plot(np.append(angles, angles[0]), closed, color=colour, lw=1.8, zorder=2)
    pts = np.array(points) if points else np.zeros((0, 2))
    if len(pts):
        ring_pick = np.array(rings, dtype=bool)
        if ring_pick.any():
            ax.scatter(pts[ring_pick, 0], pts[ring_pick, 1], s=190, facecolors='none', edgecolors=fg,
                       linewidths=1.3, zorder=3)
    sc = ax.scatter(pts[:, 0], pts[:, 1], s=55, c=fills or 'none', edgecolors=edges or 'none', linewidths=1.6,
                    zorder=4)
    lines = [Line2D([], [], color=colours.get(c, fg), lw=1.8, label=c) for c in shown]
    key = [Line2D([], [], ls='none', marker='o', ms=7, mfc='grey', mec='grey',
                  label=f'filled: competitive FDR ≤ {fdr_threshold:g}'),
           Line2D([], [], ls='none', marker='o', ms=7, mfc=face, mec='grey', label='open: not significant'),
           Line2D([], [], ls='none', marker='o', ms=12, mfc='none', mec=fg,
                  label=f'ringed: any-change FDR ≤ {fdr_threshold:g}'),
           Line2D([], [], ls='--', color='grey', label='score 0 (outside: knockouts depleted)')]
    line_columns = 1 if len(lines) <= len(key) else 2
    rows = max(len(key), -(-len(lines) // line_columns))
    blank = Line2D([], [], color='none', label='')
    handles = lines + [blank] * (rows * line_columns - len(lines)) + key + [blank] * (rows - len(key))
    legend = fig.legend(handles=handles, loc='lower center', ncol=line_columns + 1, fontsize=8,
                        frameon=False, columnspacing=2.5)
    title = 'Gene groups: score (mean of −z of the group\'s guides)'
    heading = fig.suptitle(f'{population}: {title}' if population else title, fontsize=10, va='top')
    dp.style_figure_theme(fig, is_dark)
    untested = f'(n < {min_size or fpw.DEFAULT_MIN_SIZE}, not tested)'
    spoke_labels = []
    for angle, g in zip(angles, order):
        tested = GROUP_PREFIX + g in tested_keys
        spoke_labels.append(_spoke_label(ax, angle, 2.08 * lim * SPOKE_LABEL_GAP,
                                         g if tested else f'{g} {untested}',
                                         label_colours.get(g, fg) if tested else 'grey'))
    _fit_spider(fig, ax, spoke_labels, legend, heading)
    fig._hover_handler = dp.make_scatter_hover_handler(fig, ax, sc, texts, is_dark)
    fig._spider_points = pts
    fig._spider_keys = keys
    fig._spider_axes = ax
    return fig


def export_groups(result: dict, comparison: str, fdr_threshold: float,
                  limit: int = fpw.MEMBER_FIGURES_PER_COMPARISON) -> list:
    """The groups of *comparison* whose guides are drawn on export: the
    competitive or the any-change FDR at or below the threshold, in table
    order, at most *limit*."""
    table = group_table(result, comparison)
    hits = table[(table['competitive_FDR'] <= fdr_threshold) | (table['mixed_FDR'] <= fdr_threshold)]
    return list(hits['set_key'][:limit])


def figures(result: dict | None, alloc: pd.DataFrame | None, colours: dict, areas: dict, comparisons: list,
            comparison_colour: dict, comparison: str | None, set_key: str | None, ci: pd.DataFrame | None,
            fdr_threshold: float, is_dark: bool, population: str, summary: bool = True,
            min_size: int | None = None) -> dict:
    """{key: {'fig', 'title', 'error', 'make', 'export_title'}} for the
    page: with *summary* the guide allocation (from *alloc*) and the spider
    (over *comparisons*); the chosen group's guides in *comparison*."""
    makers = []
    if summary and alloc is not None:
        makers.append(('guide_allocation', 'Guide allocation', 'Guide allocation',
                       lambda: donut_figure(alloc, colours, is_dark, population, areas)))
    if summary and result:
        makers.append(('gene_group_spider', 'Gene group spider', 'Gene group spider',
                       lambda: spider_figure(result, comparisons, comparison_colour, colours, fdr_threshold,
                                             is_dark, population, min_size)))
    if result and comparison is not None and set_key is not None:
        makers.append(('gene_group_members', 'Gene group members',
                       f'Gene group members - {fpw.set_name(result, set_key)}',
                       lambda: fpw.members_figure(result, comparison, set_key, ci, fdr_threshold, is_dark,
                                                  population, groups=True)))
    out = {}
    for key, title, export_title, make in makers:
        try:
            out[key] = {'fig': make(), 'title': title, 'error': None, 'make': make, 'export_title': export_title}
        except Exception as exc:
            out[key] = {'fig': None, 'title': title, 'error': str(exc) or type(exc).__name__}
    return out


def settings_lines(result: dict | None, settings: dict) -> list:
    """settings.txt lines of a gene-group run (or of the chosen settings
    when there is no result)."""
    run = (result or {}).get('settings') or {}
    lines = [
        f"  Gene groups: table {settings.get('file') or '-'} (digest {settings.get('digest') or '-'}; "
        f"{run.get('groups', settings.get('groups', '-'))} groups); min. "
        f"{run.get('min_size', settings.get('min_size'))} guides; "
        f"{run.get('rotations', settings.get('rotations'))} rotations, seed {run.get('seed', fpw.DEFAULT_SEED)}",
        '  Gene group tests: every guide +1; score = mean of -z (positive: the group\'s knockouts are depleted in the '
        'first group); competitive test: CAMERA against the rest of the library (filled points); any-change test: '
        'rotation of the mean of z^2 (ringed points); self-contained rotation test of the score (table only); '
        'libraries combined by weighted Stouffer; BH FDR within each comparison.',
    ]
    if run.get('unassigned'):
        lines.append(f"  Combinations with no gene group (not tested): {run['unassigned']}")
    assignment = (result or {}).get('assignment')
    if assignment is not None and not assignment.empty:
        by_symbol = assignment[assignment['match'] == MATCH_GENE]
        if not by_symbol.empty:
            lines.append('  Gene groups matched by MGI symbol: '
                         + ', '.join(f'{f} = {g}' for f, g in zip(by_symbol['feature'], by_symbol['gene'])))
    return lines
