"""
flowcode_fsea_plots.py
----------------------
Figures of the per-guide immune states (flowcode_fsea.py), one population
and sample group at a time. No Qt.

    state_heatmap_figure  states × guides, coloured by the state score
                          (red: the guide moves its cells along the state,
                          blue: away from it), ringed where significant
    state_volcano_figure  the guides of one state: score against −log10 FDR
    guide_profile_figure  one guide: its state scores, and each state's
                          members as sign-aligned t (positive: moved in the
                          direction the state lists)
    member_figure         one guide's value of one member in every mouse
                          beside its reference

Figures that can be double-clicked carry ``fig._points`` (x, y, key) and
``fig._point_axes``; the guide profile ``fig._member_features`` and
``fig._member_axes``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_GUIDES = 60
MAX_LABELS = 25
_SIGN_COLOURS = {1: '#d95f02', -1: '#1b9e77'}       # RColorBrewer Dark2: towards / away from the state


def _message(text: str, title: str, is_dark: bool, run_label: str):
    from honeychrome.view_components import differential_plots as dp
    return dp.message_figure(text, title, is_dark, run_label)


def _subset(table: pd.DataFrame, population: str, group: str) -> pd.DataFrame:
    if table is None or table.empty:
        return pd.DataFrame(columns=getattr(table, 'columns', []))
    return table[(table['population'] == population) & (table['group'] == group)]


def _finish(fig, is_dark: bool, run_label: str):
    from honeychrome.view_components import differential_plots as dp
    dp.style_figure_theme(fig, is_dark)
    dp.stamp_run_label(fig, run_label)
    return fig


def _short(text: str, width: int = 42) -> str:
    text = str(text)
    return text if len(text) <= width else text[:width - 1].rstrip() + '…'


def _tested(sub: pd.DataFrame) -> pd.DataFrame:
    return sub[np.isfinite(sub['score'].astype(float))]


def state_heatmap_figure(results: pd.DataFrame, population: str, group: str, fdr: float = 0.1,
                         is_dark: bool = False, run_label: str = '', max_guides: int = MAX_GUIDES):
    """States (rows) × guides (columns), coloured by the main score, ringed
    where q is below *fdr*; a dot marks a call that holds only with the
    guide's own target rows. Guides with a call come first (strongest
    first); at most *max_guides* columns."""
    from matplotlib.figure import Figure
    from honeychrome.view_components import differential_plots as dp

    title = f'Immune states: {population}, {group}'
    sub = _subset(results, population, group)
    tested = _tested(sub)
    if tested.empty and not sub['depends_on_target'].any():
        return _message('No guide × state could be tested in this group.', title, is_dark, run_label)
    states = list(dict.fromkeys(sub['state']))
    strength = sub.assign(_s=np.where(sub['significant'], sub['score'].abs(), 0.0),
                          _a=sub['score'].abs().fillna(0.0))
    order = strength.groupby('guide')[['_s', '_a']].max().sort_values(['_s', '_a'], ascending=False)
    guides = list(order.index[:max_guides])
    grid = np.full((len(states), len(guides)), np.nan)
    for r in sub.itertuples(index=False):
        if r.guide in guides:
            grid[states.index(r.state), guides.index(r.guide)] = r.score
    lim = max(1.0, float(np.nanmax(np.abs(grid))) if np.isfinite(grid).any() else 1.0)
    fig = Figure(figsize=(max(6.0, 0.24 * len(guides) + 4.5), max(2.8, 0.42 * len(states) + 2.2)),
                 layout='constrained')
    ax = fig.add_subplot(111)
    im = ax.imshow(np.ma.masked_invalid(grid), cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto',
                   interpolation='nearest')
    ring = 'white' if is_dark else 'black'
    xs, ys, keys, labels = [], [], [], []
    for r in sub.itertuples(index=False):
        if r.guide not in guides:
            continue
        x, y = guides.index(r.guide), states.index(r.state)
        if r.significant:
            ax.scatter([x], [y], s=70, facecolors='none', edgecolors=ring, linewidths=1.2)
        if r.depends_on_target:
            ax.scatter([x], [y], s=8, c=ring)
        xs.append(x)
        ys.append(y)
        keys.append((r.guide, r.state))
        main = (f'score {r.score:+.2f}, q {r.q:.3g} ({int(r.members)} members)' if np.isfinite(r.score)
                else f'fewer than the minimum members without its target rows ({int(r.members)})')
        labels.append(f'{r.guide} — {r.state}\n{main}\nwith all rows: score {r.score_all:+.2f}, q {r.q_all:.3g} '
                      f'({int(r.members_all)} members)')
    hover = ax.scatter(xs, ys, s=60, alpha=0.0)
    ax.set_xticks(range(len(guides)))
    ax.set_xticklabels(guides, rotation=90, fontsize=7)
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels([_short(s) for s in states], fontsize=8)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label('State score (red: towards, blue: away)', fontsize=8)
    n_sig = int(sub['significant'].sum())
    shown = '' if len(order) <= max_guides else f' (strongest {max_guides} of {len(order)})'
    ax.set_title(f'{title}{shown}\nringed: q < {fdr:g} without the guide\'s own target rows ({n_sig}); '
                 'dot: called only with them; blank: not tested', fontsize=9)
    fg = dp.style_figure_theme(fig, is_dark)
    cb.ax.yaxis.label.set_color(fg)
    cb.ax.tick_params(colors=fg)
    dp.stamp_run_label(fig, run_label)
    fig._hover_handler = dp.make_scatter_hover_handler(fig, ax, hover, labels, is_dark)
    fig._points = [(x, y, k) for x, y, k in zip(xs, ys, keys)]
    fig._point_axes = ax
    return fig


def state_volcano_figure(results: pd.DataFrame, population: str, group: str, state: str, fdr: float = 0.1,
                         is_dark: bool = False, run_label: str = ''):
    """The guides of one state: score (x) against −log10 FDR (y); guides with
    q below *fdr* outlined and labelled (strongest first)."""
    from matplotlib.figure import Figure
    from honeychrome.view_components import differential_plots as dp

    title = f'{_short(state, 60)}: {population}, {group}'
    sub = _subset(results, population, group)
    sub = _tested(sub[sub['state'] == state])
    if sub.empty:
        return _message('No guide could be tested for this state in this group.', title, is_dark, run_label)
    x = sub['score'].to_numpy(float)
    y = -np.log10(np.maximum(sub['q'].to_numpy(float), 1e-300))
    sig = sub['significant'].to_numpy(bool)
    fg = 'white' if is_dark else 'black'
    fig = Figure(figsize=(7.0, 5.0), layout='constrained')
    ax = fig.add_subplot(111)
    colours = [_SIGN_COLOURS[1] if v > 0 else _SIGN_COLOURS[-1] for v in x]
    sc = ax.scatter(x, y, s=30, c=colours, edgecolors=np.where(sig, fg, 'none'), linewidths=0.9, alpha=0.9)
    ax.axvline(0.0, color='#7f7f7f', linewidth=0.8)
    ax.axhline(-np.log10(fdr), color='#7f7f7f', linewidth=0.6, linestyle=':')
    for i in [i for i in np.argsort(-y) if sig[i]][:MAX_LABELS]:
        ax.annotate(str(sub['guide'].iloc[i]), (x[i], y[i]), xytext=(3, 3), textcoords='offset points',
                    fontsize=7, color=fg)
    ax.set_xlabel('State score (mean signed z of the members; positive: towards the state)', fontsize=9)
    ax.set_ylabel('−log10 FDR', fontsize=9)
    ax.grid(False)
    ax.set_title(f'{title}\n{len(sub)} guides tested, {int(sig.sum())} with q < {fdr:g} (outlined)', fontsize=10)
    labels = [f"{r.guide}: score {r.score:+.2f}, p {r.p:.3g}, q {r.q:.3g}\n{int(r.members)} members, "
              f"{int(r.mice)} mice; any direction q {r.q_mixed:.3g}" for r in sub.itertuples(index=False)]
    _finish(fig, is_dark, run_label)
    fig._hover_handler = dp.make_scatter_hover_handler(fig, ax, sc, labels, is_dark)
    fig._points = list(zip(x, y, sub['guide'].astype(str)))
    fig._point_axes = ax
    return fig


def guide_profile_figure(results: pd.DataFrame, members: pd.DataFrame, rows: dict, population: str, group: str,
                         guide: str, fdr: float = 0.1, is_dark: bool = False, run_label: str = ''):
    """
    One guide in one group. Left: each tested state's score (starred where
    q is below *fdr*). Right: each state's members, as the member's t times
    the sign the state lists, so a bar to the right moved the way the state
    says; members on the guide's own target are hatched (left out of the
    main score). *rows* is {state: the state's scored rows}.
    """
    from matplotlib.figure import Figure

    title = f'{guide}: immune states ({population}, {group})'
    res = _subset(results, population, group)
    res = res[res['guide'] == guide]
    mem = _subset(members, population, group)
    mem = mem[mem['guide'] == guide].set_index('feature')
    if res.empty or mem.empty:
        return _message('This guide has no tested state in this group.', title, is_dark, run_label)
    states = list(res['state'])
    lines = []
    for state in states:
        r = rows.get(state)
        if r is None:
            continue
        for row in r.itertuples(index=False):
            if row.feature in mem.index:
                lines.append((state, row.feature, row.label, int(row.sign),
                              str(row.gene).lower() == str(guide).lower()))
    fig = Figure(figsize=(10.0, max(3.2, 0.26 * len(lines) + 1.8)), layout='constrained')
    left, right = fig.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1.6]})
    fg = 'white' if is_dark else 'black'
    y = np.arange(len(states))[::-1]
    scores = res['score'].to_numpy(float)
    shown = np.where(np.isfinite(scores), scores, res['score_all'].to_numpy(float))
    left.barh(y, shown, color=[_SIGN_COLOURS[1] if v > 0 else _SIGN_COLOURS[-1] for v in shown],
              hatch=None, alpha=0.9)
    for yi, r, v in zip(y, res.itertuples(index=False), shown):
        mark = '*' if r.significant else ('(*)' if r.depends_on_target else '')
        if mark:
            left.annotate(mark, (v, yi), xytext=(4 if v >= 0 else -4, -3), textcoords='offset points',
                          ha='left' if v >= 0 else 'right', fontsize=9, color=fg)
    left.axvline(0.0, color='#7f7f7f', linewidth=0.8)
    left.margins(x=0.15)
    left.set_yticks(y)
    left.set_yticklabels([_short(s, 34) for s in states], fontsize=8)
    left.set_xlabel('State score', fontsize=8)
    left.set_title(f'* q < {fdr:g}; (*) only with the target rows', fontsize=8)
    left.grid(False)
    ym = np.arange(len(lines))[::-1]
    values = np.array([sign * float(mem.loc[f, 't']) for _s, f, _l, sign, _t in lines])
    bars = right.barh(ym, values, color=[_SIGN_COLOURS[1] if v > 0 else _SIGN_COLOURS[-1] for v in values],
                      alpha=0.9)
    for bar, (_s, _f, _l, _sign, target) in zip(bars, lines):
        if target:
            bar.set_hatch('///')
            bar.set_alpha(0.5)
    right.axvline(0.0, color='#7f7f7f', linewidth=0.8)
    right.set_yticks(ym)
    right.set_yticklabels([f"{'↑' if sign > 0 else '↓'} {_short(label, 40)}" for _s, _f, label, sign, _t in lines],
                          fontsize=7)
    previous = None
    for yi, (state, _f, _l, _s, _t) in zip(ym, lines):
        if previous is not None and state != previous:
            right.axhline(yi + 0.5, color='#7f7f7f', linewidth=0.6, linestyle=':')
        previous = state
    right.set_xlabel('Member t × listed direction (right: as the state lists)', fontsize=8)
    right.set_title('Members by state (top to bottom as on the left; hatched: own target)', fontsize=8)
    right.grid(False)
    fig.suptitle(title, fontsize=10)
    fig._member_features = [f for _s, f, _l, _sign, _t in lines][::-1]
    fig._member_axes = right
    return _finish(fig, is_dark, run_label)


def member_figure(values: pd.DataFrame, guide: str, label: str, population: str, group: str,
                  sample_labels: dict | None = None, is_dark: bool = False, run_label: str = ''):
    """One guide's share of one member in every mouse (filled) beside its
    reference's (open), joined per mouse."""
    from matplotlib.figure import Figure

    title = f'{guide}: {label} ({population}, {group})'
    if values is None or values.empty:
        return _message('This guide has no mouse with a value for this member.', title, is_dark, run_label)
    sample_labels = sample_labels or {}
    fg = 'white' if is_dark else 'black'
    fig = Figure(figsize=(max(5.0, 0.7 * len(values) + 2.5), 4.4), layout='constrained')
    ax = fig.add_subplot(111)
    x = np.arange(len(values))
    g = values['guide_share'].to_numpy(float)
    r = values['reference_share'].to_numpy(float)
    for xi, a, b in zip(x, g, r):
        ax.plot([xi, xi], [b, a], color='#7f7f7f', linewidth=0.8)
    ax.scatter(x, r, s=36, facecolors='none', edgecolors=fg, linewidths=1.0, label='reference')
    ax.scatter(x, g, s=36, c=_SIGN_COLOURS[1], label=guide)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{sample_labels.get(s, s)}\n({int(c)} cells)" for s, c in zip(values['sample'],
                                                                                        values['cells'])],
                       rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Share', fontsize=9)
    ax.set_ylim(0, min(1.0, max(float(np.nanmax(np.concatenate([g, r]))) * 1.15, 0.05)))
    ax.grid(False)
    ax.legend(fontsize=8, frameon=False, loc='best')
    mean = float(values['difference'].mean())
    ax.set_title(f'{title}\nmean logit difference from the reference {mean:+.2f}', fontsize=10)
    fig._points = None
    return _finish(fig, is_dark, run_label)
