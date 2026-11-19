"""
flowcode_synopsis_plots.py
--------------------------
Figures of the Synopsis tab (flowcode_synopsis.py), one population at a
time. No Qt.

    atlas_figure    groups (rows) × guides (columns). Colour: the log2 fold
                    change of the guide's cells in the group against the
                    baseline, where the test calls it. Dot: a phenotype call
                    in that group, larger with each readout that makes one
                    (phenotype shift, immune states, markers). Grey cell:
                    no phenotype readout could be tested there, so a
                    missing dot says nothing. Cross: a call with no cell on
                    one side of the comparison (no fold change to show).
    dossier_figure  one guide across the groups, one panel per readout on a
                    shared group axis: fold change with its interval, cells
                    counted, shift angle, immune-state scores and the
                    markers with a call.

Colours follow the page's other figures: RdBu_r for signed effects (red
up), grey for untested. Figures that can be double-clicked carry
``fig._points`` (x, y, key) and ``fig._point_axes``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import flowcode_synopsis as fsy

CELL = 0.27                       # inches per atlas cell
_UNTESTED = '#8c8c8c'
_NEUTRAL = '#7f7f7f'
_UP, _DOWN = '#b2182b', '#2166ac'   # RdBu_r ends
_DOT_SIZES = {1: 9, 2: 26, 3: 52}
MAX_MARKER_ROWS = 14
MAX_FOLD = 6.0


def _message(text: str, title: str, is_dark: bool, run_label: str):
    from honeychrome.view_components import differential_plots as dp
    return dp.message_figure(text, title, is_dark, run_label)


def _grid(calls: pd.DataFrame, column: str, groups: list, guides: list, fill=np.nan) -> np.ndarray:
    out = np.full((len(groups), len(guides)), fill, dtype=float)
    if calls.empty:
        return out
    row = {g: i for i, g in enumerate(groups)}
    col = {g: i for i, g in enumerate(guides)}
    for guide, group, value in zip(calls['guide'], calls['group'], calls[column]):
        if guide in col and group in row:
            out[row[group], col[guide]] = float(value)
    return out


def _cell_label(r, baseline: str, test: str) -> str:
    lines = [f'{r.guide}, {r.group}']
    if r.accumulation_tested:
        call = 'not called' if r.accumulation == 0 else 'called'
        lines.append('no cell on one side of the comparison' if r.unbounded else
                     f'cells against {baseline}: log2FC {r.logFC:+.2f}, q {r.q_accumulation:.3g} ({call}, '
                     f'{fsy.TESTS.get(test, "")})')
    if r.shift_tested:
        lines.append(f"phenotype shift {abs(r.shift_angle):.0f}°, q {r.q_shift:.3g}"
                     f"{' (called)' if r.shift else ''}")
    if r.states_tested:
        lines.append(f'immune states: {r.states_towards} towards, {r.states_away} away of {r.states_tested} tested')
    if r.markers_tested:
        lines.append(f'markers: {r.markers_higher} higher, {r.markers_lower} lower of {r.markers_tested} tested')
    if not r.phenotype_tested:
        lines.append('phenotype not testable (too few cells)')
    if np.isfinite(r.cells):
        lines.append(f'{r.cells:,.0f} cells counted in the group')
    return '\n'.join(lines)


def atlas_figure(synopsis: dict, group_colours: dict | None = None, is_dark: bool = False, run_label: str = '',
                 n_rows: int = 1, show_all: bool = False):
    """
    The guide × group atlas of one population (module docstring). Guides are
    in the synopsis' order, with a strip of gene-group colours
    (*group_colours*, {gene group: colour}) above them when the guides have
    gene groups. *n_rows* splits the guides over stacked panels (1–3);
    *show_all* colours every tested cell by its fold change and outlines
    the calls.
    """
    import matplotlib
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from honeychrome.view_components import differential_plots as dp

    scope, baseline, test = synopsis['scope'], synopsis['baseline'], synopsis['test']
    title = f"Synopsis atlas: {scope['label']}"
    calls, guides = synopsis['calls'], list(synopsis['guides'])
    groups = [g for g in synopsis['groups'] if g != baseline] + ([baseline] if baseline else [])
    if calls.empty or not guides or not groups:
        return _message('Run statistics or the phenotype shift to see the atlas.', title, is_dark, run_label)

    fold = _grid(calls, 'logFC', groups, guides)
    called = _grid(calls, 'accumulation', groups, guides, 0.0) != 0
    unbounded = _grid(calls, 'unbounded', groups, guides, 0.0) > 0
    readouts = _grid(calls, 'phenotype_readouts', groups, guides, 0.0)
    testable = _grid(calls, 'phenotype_tested', groups, guides, 0.0) > 0
    shown = (called | bool(show_all)) & np.isfinite(fold)
    measured = shown & ~unbounded
    vmax = float(np.abs(fold[measured]).max()) if measured.any() else 1.0
    vmax = min(max(vmax, 1.0), MAX_FOLD)
    fold = np.clip(fold, -vmax, vmax)

    n_rows = int(min(max(1, n_rows), 3, len(guides)))
    per_row = int(np.ceil(len(guides) / n_rows))
    chunks = [guides[i * per_row:(i + 1) * per_row] for i in range(n_rows)]
    gene_group = synopsis['gene_group'] if group_colours else {}
    depth = synopsis['depth']
    row_names = [f'{g} (baseline)' if g == baseline else str(g) for g in groups]
    depth_names = [f'{depth[g]:,.0f}' if g in depth and np.isfinite(depth[g]) else '' for g in groups]
    label_w = max(dp.text_width_inches(n, 7) for n in row_names) + 0.3
    band = max(dp.text_width_inches(str(g), 7) for g in guides) + 0.1
    cells_h = len(groups) * CELL + (0.14 if gene_group else 0.0)
    fig_w = min(max(6.5, per_row * CELL + label_w + 1.9), 28.0)
    fig = Figure(figsize=(fig_w, max(3.0, (cells_h + band) * n_rows + 1.7)), constrained_layout=True)
    axes = fig.subplots(n_rows, 1, squeeze=False)[:, 0]
    cmap = matplotlib.colormaps['RdBu_r'].copy()
    cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    fg = 'white' if is_dark else 'black'
    bg = dp.figure_background(is_dark)
    index = {g: i for i, g in enumerate(guides)}
    points, labels, image = [], [], None
    info = {(r.guide, r.group): r for r in calls.itertuples(index=False)}
    for ax, chunk in zip(axes, chunks):
        cols = [index[g] for g in chunk]
        blind = np.ma.masked_array(np.ones((len(groups), len(cols))), mask=testable[:, cols])
        ax.imshow(blind, aspect='auto', cmap=matplotlib.colors.ListedColormap([_UNTESTED]), alpha=0.28,
                  interpolation='nearest')
        image = ax.imshow(np.ma.masked_array(fold[:, cols], mask=~shown[:, cols]), aspect='auto', cmap=cmap,
                          norm=norm, interpolation='nearest')
        ax.grid(False)
        dp.cell_grid_lines(ax, len(cols), len(groups), '#898989', 0.4)
        ys, xs = np.nonzero(readouts[:, cols] > 0)
        if len(xs):
            ax.scatter(xs, ys, s=[_DOT_SIZES[int(readouts[y, cols[x]])] for x, y in zip(xs, ys)], c=fg,
                       edgecolors=bg, linewidths=0.6, zorder=3)
        ys, xs = np.nonzero((unbounded & shown)[:, cols])
        if len(xs):
            # These cells sit at the dark ends of the colour scale.
            ax.scatter(xs, ys, s=30, marker='x', c='white', linewidths=1.1, zorder=4)
        if show_all:
            ys, xs = np.nonzero(called[:, cols])
            for x, y in zip(xs, ys):
                ax.add_patch(matplotlib.patches.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor=fg,
                                                          linewidth=1.1, zorder=2))
        hx, hy, hl = [], [], []
        for y, group in enumerate(groups):
            for x, guide in enumerate(chunk):
                r = info.get((guide, group))
                if r is not None:
                    hx.append(x)
                    hy.append(y)
                    hl.append(_cell_label(r, baseline, test))
        hover = ax.scatter(hx, hy, s=60, alpha=0.0)
        points.append((ax, [(x, y, chunk[x]) for x, y in zip(hx, hy)]))
        labels.append((ax, hover, hl))
        ax.set_xticks(range(len(chunk)))
        ax.set_xticklabels(chunk, rotation=90, fontsize=7)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(row_names, fontsize=7)
        ax.tick_params(length=0)
        top = -0.5
        if gene_group:
            top = -0.5 - 0.5
            for x, guide in enumerate(chunk):
                colour = group_colours.get(gene_group.get(guide))
                if colour:
                    ax.add_patch(matplotlib.patches.Rectangle((x - 0.5, top), 1, 0.4, color=colour, clip_on=False,
                                                              linewidth=0))
        ax.set_xlim(-0.5, per_row - 0.5)
        ax.set_ylim(len(groups) - 0.5, top)
        side = ax.secondary_yaxis('right')
        side.set_yticks(range(len(groups)))
        side.set_yticklabels(depth_names, fontsize=6)
        side.tick_params(length=0, colors=fg)
        for spine in side.spines.values():
            spine.set_visible(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cb = fig.colorbar(image, ax=list(axes), shrink=0.7, pad=0.04)
    limit = f' (colours end at ±{vmax:g})' if vmax >= MAX_FOLD else ''
    cb.set_label(dp.wrap_to_width(f"log2 fold change against {baseline}"
                                  f"{'' if show_all else ', where called'}{limit}", 7,
                                  max(0.8, 0.7 * n_rows * cells_h)), fontsize=7)
    cb.ax.tick_params(labelsize=7)
    handles = [Line2D([], [], linestyle='', marker='o', color=fg, markeredgecolor=bg,
                      markersize=np.sqrt(_DOT_SIZES[n]), label=f'phenotype call by {n} readout{"s" if n > 1 else ""}')
               for n in (1, 2, 3)]
    handles.append(Patch(facecolor=_UNTESTED, alpha=0.28, label='phenotype not testable'))
    if (unbounded & shown).any():
        handles.append(Line2D([], [], linestyle='', marker='x', color='white', markersize=5,
                              markerfacecolor=_DOWN, markeredgewidth=1.1,
                              label='no cell on one side: no fold change'))
    if show_all:
        handles.append(Patch(facecolor='none', edgecolor=fg, label='accumulation called'))
    for name in synopsis['gene_group_order'] if gene_group else []:
        if name in group_colours:
            handles.append(Patch(facecolor=group_colours[name], label=dp.wrap_label(name, 34)))
    legend = fig.legend(handles=handles, loc='outside lower center', ncols=min(4, len(handles)), fontsize=7,
                        frameon=False)
    for text in legend.get_texts():
        text.set_color(fg)
    what = f"cells against {baseline} ({fsy.TESTS.get(test, '')} test)" if baseline and test else 'phenotype only'
    axes[0].set_title(dp.wrap_to_width(f'{title}: {what}; right: median barcoded cells per sample', 10,
                                       fig_w - 0.4), fontsize=10)
    dp.style_figure_theme(fig, is_dark)
    cb.ax.yaxis.label.set_color(fg)
    cb.ax.tick_params(colors=fg)
    dp.stamp_run_label(fig, run_label)
    handlers = [dp.make_scatter_hover_handler(fig, ax, hover, hl, is_dark) for ax, hover, hl in labels]

    def on_hover(event):
        for handler in handlers:
            handler(event)
    fig._hover_handler = on_hover
    fig._atlas_points = points
    return fig


def _clean(ax, fg: str):
    ax.grid(False)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=7, colors=fg)


def _heat(ax, grid: np.ndarray, called: np.ndarray, names: list, fg: str, limit: float | None = None):
    import matplotlib
    from matplotlib.colors import TwoSlopeNorm

    cmap = matplotlib.colormaps['RdBu_r'].copy()
    cmap.set_bad(_UNTESTED, 0.28)
    finite = np.abs(grid[np.isfinite(grid)])
    lim = float(limit or (finite.max() if len(finite) else 1.0)) or 1.0
    image = ax.imshow(np.ma.masked_invalid(grid), aspect='auto', cmap=cmap,
                      norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim), interpolation='nearest')
    ys, xs = np.nonzero(called)
    ax.scatter(xs, ys, s=46, facecolors='none', edgecolors=fg, linewidths=1.0)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def dossier_figure(synopsis: dict, guide: str, is_dark: bool = False, run_label: str = ''):
    """
    One guide across the groups (module docstring). Panels without results
    are left out; a ringed heatmap cell and a filled point or bar are
    calls; grey heatmap cells were not tested.
    """
    from matplotlib.figure import Figure
    from honeychrome.view_components import differential_plots as dp

    scope, baseline, test = synopsis['scope'], synopsis['baseline'], synopsis['test']
    groups = list(synopsis['groups'])
    calls = synopsis['calls']
    mine = calls[calls['guide'] == guide].set_index('group')
    group_name = synopsis['gene_group'].get(guide, '')
    title = f"{guide}{f' [{group_name}]' if group_name else ''}: {scope['label']}"
    if mine.empty or not groups:
        return _message('This guide has no result in this population.', title, is_dark, run_label)
    x = np.arange(len(groups))
    pos = {g: i for i, g in enumerate(groups)}
    acc = synopsis['accumulation']
    acc = acc[acc['guide'] == guide]
    states = synopsis['states']
    states = states[states['guide'] == guide]
    markers = synopsis['markers']
    markers = markers[markers['guide'] == guide]
    state_names = list(dict.fromkeys(states['state']))
    hit = markers[markers['significant']]
    channels = list(hit.reindex(hit['delta'].abs().sort_values(ascending=False).index)['channel'].drop_duplicates())
    channels = channels[:MAX_MARKER_ROWS]
    marker_names = synopsis.get('marker_labels') or {}

    panels = []
    if not acc.empty:
        panels += [('fold', 2.0), ('cells', 0.9)]
    if mine['shift_tested'].any():
        panels.append(('shift', 1.2))
    if state_names:
        panels.append(('states', 0.24 * len(state_names) + 0.25))
    if channels:
        panels.append(('markers', 0.24 * len(channels) + 0.25))
    fg = 'white' if is_dark else 'black'
    label_w = max([dp.text_width_inches(dp.wrap_label(s, 34), 7) for s in state_names]
                  + [dp.text_width_inches(str(marker_names.get(c, c)), 7) for c in channels] + [1.0]) + 0.3
    fig_w = min(max(7.5, len(groups) * 0.42 + label_w + 1.4), 20.0)
    sentence = dp.wrap_to_width(fsy.guide_sentence(guide, synopsis), 8, fig_w - 0.5)
    head = (sentence.count('\n') + 2) * 8 * 1.3 / 72.0 + 0.25
    band = max(dp.text_width_inches(str(g), 7) for g in groups) + 0.2
    fig = Figure(figsize=(fig_w, sum(h for _k, h in panels) + head + band + 0.35 * len(panels)),
                 layout='constrained')
    axes = fig.subplots(len(panels), 1, sharex=True, squeeze=False,
                        gridspec_kw={'height_ratios': [h for _k, h in panels]})[:, 0]
    for ax, (kind, _h) in zip(axes, panels):
        if kind == 'fold':
            limit = MAX_FOLD
            bounded = mine[mine['accumulation_tested'] & ~mine['unbounded']]
            if not bounded.empty:
                limit = min(MAX_FOLD, max(2.0, float(bounded['logFC'].abs().max()) * 1.4))
            ax.axhline(0.0, color=_NEUTRAL, linewidth=0.8)
            for offset, key, marker in ((-0.12, test, 'o'), (0.12, next((t for t in synopsis['tests'] if t != test),
                                                                         None), 's')):
                sub = acc[acc['test'] == key] if key else acc.iloc[0:0]
                for group, fc, low, high, is_called in zip(sub['group'], sub['logFC'], sub['CI.L'], sub['CI.R'],
                                                           sub['significant']):
                    if group not in pos:
                        continue
                    xi = pos[group] + (offset if len(synopsis['tests']) > 1 else 0.0)
                    colour = (_UP if fc > 0 else _DOWN) if is_called else _NEUTRAL
                    if bool(mine['unbounded'].get(group, False)):
                        ax.scatter([xi], [np.sign(fc) * limit * 0.94], marker='x', s=28, c=colour,
                                   linewidths=1.0, zorder=3)
                        continue
                    ax.plot([xi, xi], [low, high], color=colour, linewidth=0.9, zorder=2)
                    ax.scatter([xi], [fc], marker=marker, s=26, zorder=3, edgecolors=colour,
                               facecolors=colour if is_called else 'none', linewidths=1.0)
            if baseline in pos:
                ax.axvspan(pos[baseline] - 0.5, pos[baseline] + 0.5, color=_UNTESTED, alpha=0.18, linewidth=0)
                ax.text(pos[baseline], 0.0, 'baseline', rotation=90, ha='center', va='center', fontsize=6,
                        color=fg)
            ax.set_ylim(-limit, limit)
            names = [fsy.TESTS[t] for t in ([test] + [t for t in synopsis['tests'] if t != test])]
            marks = ' (circles), '.join(names[:1]) + (f' (circles), {names[1]} (squares)' if len(names) > 1 else '')
            ax.set_ylabel(f'log2 fold change\nagainst {baseline}', fontsize=7)
            ax.set_title(f'Cells: {marks}; filled: called; bars: 95% interval; cross: no cell on one side',
                         fontsize=7, loc='left', color=fg)
            _clean(ax, fg)
        elif kind == 'cells':
            values = np.array([float(mine['cells'].get(g, np.nan)) for g in groups])
            ax.bar(x, np.where(np.isfinite(values), np.maximum(values, 0.5), np.nan), color=_NEUTRAL, width=0.6)
            ax.set_yscale('log')
            ax.set_ylim(0.5, max(10.0, float(np.nanmax(values)) * 2.0) if np.isfinite(values).any() else 10.0)
            ax.set_ylabel('cells', fontsize=7)
            ax.set_title("The guide's cells counted in the group's samples", fontsize=7, loc='left', color=fg)
            _clean(ax, fg)
        elif kind == 'shift':
            angle = np.array([abs(float(mine['shift_angle'].get(g, np.nan))) for g in groups])
            is_called = np.array([bool(mine['shift'].get(g, False)) for g in groups])
            ax.bar(x[is_called], angle[is_called], color=fg, width=0.6)
            ax.bar(x[~is_called], angle[~is_called], facecolor='none', edgecolor=_NEUTRAL, width=0.6, linewidth=0.9)
            for xi in x[~np.isfinite(angle)]:
                ax.axvspan(xi - 0.5, xi + 0.5, color=_UNTESTED, alpha=0.18, linewidth=0)
            ax.set_ylim(0, max(20.0, float(np.nanmax(angle)) * 1.15) if np.isfinite(angle).any() else 20.0)
            ax.set_ylabel('shift angle (°)', fontsize=7)
            ax.set_title('Phenotype shift: filled where called; grey: not testable', fontsize=7, loc='left',
                         color=fg)
            _clean(ax, fg)
        elif kind == 'states':
            grid = np.full((len(state_names), len(groups)), np.nan)
            ring = np.zeros(grid.shape, dtype=bool)
            for r in states.itertuples(index=False):
                if r.group in pos:
                    grid[state_names.index(r.state), pos[r.group]] = r.score
                    ring[state_names.index(r.state), pos[r.group]] = bool(r.significant)
            image = _heat(ax, grid, ring, [dp.wrap_label(s, 34, 2) for s in state_names], fg)
            cb = fig.colorbar(image, ax=ax, pad=0.01, aspect=8)
            cb.set_label('state score', fontsize=6)
            cb.ax.tick_params(labelsize=6, colors=fg)
            cb.ax.yaxis.label.set_color(fg)
            ax.set_title('Immune states (red: towards the state; ringed: called)', fontsize=7, loc='left',
                         color=fg)
        else:
            grid = np.full((len(channels), len(groups)), np.nan)
            ring = np.zeros(grid.shape, dtype=bool)
            for r in markers[markers['channel'].isin(channels)].itertuples(index=False):
                if r.group in pos:
                    grid[channels.index(r.channel), pos[r.group]] = r.delta
                    ring[channels.index(r.channel), pos[r.group]] = bool(r.significant)
            image = _heat(ax, grid, ring, [str(marker_names.get(c, c)) for c in channels], fg)
            cb = fig.colorbar(image, ax=ax, pad=0.01, aspect=8)
            cb.set_label('difference', fontsize=6)
            cb.ax.tick_params(labelsize=6, colors=fg)
            cb.ax.yaxis.label.set_color(fg)
            more = int(hit['channel'].nunique()) - len(channels)
            ax.set_title('Markers with a call (difference from the reference, display scale; ringed: called)'
                         + (f'; {more} more not shown' if more > 0 else ''), fontsize=7, loc='left', color=fg)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(groups, rotation=90, fontsize=7)
    axes[-1].set_xlim(-0.5, len(groups) - 0.5)
    fig.suptitle(f'{title}\n{sentence}', fontsize=8, x=0.01, ha='left')
    fig.align_ylabels(list(axes))
    dp.style_figure_theme(fig, is_dark)
    dp.stamp_run_label(fig, run_label)
    return fig
