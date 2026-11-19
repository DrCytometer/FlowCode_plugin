"""
test_flowcode_plots.py
----------------------
Figures of the FlowCode plugin's Debarcode & FRET section
(flowcode_plots.py): the spillover heatmap, the panel expression grid and
the combo frequency bars. Built off screen with matplotlib only.

The plugin lives outside the package; its folder defaults to
user_plugins/flowcode in this repository and can be set with
FLOWCODE_PLUGIN_DIR.

Usage:
    pytest tests/test_flowcode_plots.py -m numpy_only
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_FLOWCODE_DIR = Path(os.environ.get('FLOWCODE_PLUGIN_DIR', _REPO / 'user_plugins' / 'flowcode'))
if str(_FLOWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(_FLOWCODE_DIR))

import flowcode_plots as fp  # noqa: E402

pytestmark = pytest.mark.numpy_only


def _matrix(n_rows=3, n_cols=5, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.normal(0, 1, (n_rows, n_cols))
    on = np.zeros((n_rows, n_cols), dtype=bool)
    for r in range(n_rows):
        on[r, [r % n_cols, (r + 1) % n_cols]] = True
    return {'rows': [f'C{r}' for r in range(n_rows)], 'combo_index': list(range(n_rows)),
            'columns': [f'F{c}' for c in range(n_cols)], 'values': values, 'on': on,
            'pairwise_combos': ['G1']}


def _texts(ax):
    return [t.get_text() for t in ax.texts]


def test_symmetric_limit_is_the_99th_percentile_of_absolute_values():
    values = np.concatenate([np.linspace(-5, 5, 1001), [np.nan, np.inf]])
    assert fp.symmetric_limit(values) == pytest.approx(np.quantile(np.abs(np.linspace(-5, 5, 1001)), 0.99))
    assert fp.symmetric_limit(np.zeros(4)) == pytest.approx(1e-6)
    assert fp.symmetric_limit(np.array([np.nan])) == 1.0


def test_heatmap_is_centred_on_zero_and_outlines_the_on_tag_cells():
    from matplotlib.patches import Rectangle

    m = _matrix()
    fig = fp.spillover_heatmap_figure(m, {'F0': 'HA - F0'}, 'unmixed')
    ax = fig.axes[0]
    image = ax.images[0]
    lim = fp.symmetric_limit(m['values'])
    assert image.get_clim() == pytest.approx((-lim, lim))
    np.testing.assert_array_equal(image.get_array(), m['values'])
    outlines = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert len(outlines) == int(m['on'].sum())
    corners = sorted((p.get_x() + 0.5, p.get_y() + 0.5) for p in outlines)
    assert corners == sorted((float(c), float(r)) for r, c in zip(*np.nonzero(m['on'])))
    assert [t.get_text() for t in ax.get_xticklabels()][0] == 'HA - F0'
    assert len(_texts(ax)) == m['values'].size
    assert '-0.0' not in _texts(ax)
    assert 'G1' in ax.get_xlabel()


def test_large_heatmaps_drop_the_cell_values():
    fig = fp.spillover_heatmap_figure(_matrix(n_rows=30, n_cols=20), space='detector')
    assert _texts(fig.axes[0]) == []


def test_heatmap_without_characterised_combos_explains_why():
    m = dict(_matrix(), rows=[], values=np.zeros((0, 5)), on=np.zeros((0, 5), dtype=bool))
    fig = fp.spillover_heatmap_figure(m)
    assert 'pairwise model' in ' '.join(_texts(fig.axes[0]))


def _expression(n_markers=4, bins=60, seed=1):
    rng = np.random.default_rng(seed)
    centres = np.linspace(0.0, 1.0, bins)

    def trace(mu):
        t = np.exp(-0.5 * ((centres - mu) / 0.08) ** 2)
        return t / t.max()

    groups = ('untransduced', 'before', 'after')
    return {
        'markers': [f'M{j}' for j in range(n_markers)],
        'is_tag': [j < 2 for j in range(n_markers)],
        'centres': [centres] * n_markers,
        'limits': [(0.0, 1.0)] * n_markers,
        'ticks': [[(0.0, '0'), (0.5, '10³')]] * n_markers,
        'traces': {g: [trace(rng.uniform(0.2, 0.8)) for _ in range(n_markers)] for g in groups},
        'medians': {g: [0.3] * n_markers for g in groups},
        'n': {'untransduced': 120, 'before': 2500, 'after': 2500},
    }


def test_panel_expression_has_a_row_per_marker_and_a_column_per_group():
    data = _expression()
    fig = fp.panel_expression_figure(data)
    axes = np.array(fig.axes).reshape(4, 3)
    assert [ax.get_title().split('\n')[0] for ax in axes[0]] == [fp.GROUP_TITLES[g] for g in
                                                                ('untransduced', 'before', 'after')]
    assert axes[0, 1].get_title().endswith('2,500 events')
    assert [ax.get_ylabel() for ax in axes[:, 0]] == data['markers']
    from matplotlib.colors import to_hex
    assert to_hex(axes[0, 0].yaxis.label.get_color()) == fp.ON_COLOUR.lower()
    assert to_hex(axes[3, 0].yaxis.label.get_color()) != fp.ON_COLOUR.lower()
    for r in range(4):
        for c, group in enumerate(('untransduced', 'before', 'after')):
            ax = axes[r, c]
            assert ax.get_xlim() == (0.0, 1.0)
            assert [t.get_text() for t in ax.get_xticklabels()] == ['0', '10³']
            line = ax.lines[0]
            np.testing.assert_allclose(line.get_ydata(), data['traces'][group][r])
            assert to_hex(line.get_color()) == fp.GROUP_COLOURS[group].lower()
            assert ax.lines[1].get_xdata()[0] == pytest.approx(0.3)


def test_panel_expression_skips_empty_groups_and_handles_no_markers():
    data = _expression(n_markers=2)
    data['traces']['untransduced'] = [None, None]
    data['medians']['untransduced'] = [float('nan')] * 2
    fig = fp.panel_expression_figure(data)
    axes = np.array(fig.axes).reshape(2, 3)
    assert len(axes[0, 0].lines) == 0 and len(axes[0, 1].lines) == 1
    empty = fp.panel_expression_figure(dict(data, markers=[]))
    assert 'No events' in ' '.join(_texts(empty.axes[0]))


def test_frequency_bars_are_percentages_of_barcoded_events():
    counts = [300, 100, 600]
    totals = {'total_events': 1500, 'untransduced': 400, 'unresolved': 100, 'rescued': 25}
    fig = fp.combo_frequency_figure(['A', 'B', 'C'], counts, totals)
    ax = fig.axes[0]
    heights = [p.get_height() for p in ax.patches]
    np.testing.assert_allclose(heights, [30.0, 10.0, 60.0])
    assert _texts(ax) == ['300', '100', '600']
    footer = fig._supxlabel.get_text()
    assert 'Untransduced 400 (26.7%)' in footer and 'unresolved 100 (6.7%)' in footer and 'rescued 25' in footer


def test_frequency_bars_turn_horizontal_above_twenty_combos():
    names = [f'C{i}' for i in range(21)]
    fig = fp.combo_frequency_figure(names, np.arange(1, 22), {'total_events': 300})
    ax = fig.axes[0]
    widths = [p.get_width() for p in ax.patches]
    np.testing.assert_allclose(sum(widths), 100.0)
    assert [t.get_text() for t in ax.get_yticklabels()] == names
