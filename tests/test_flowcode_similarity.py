"""
test_flowcode_similarity.py
---------------------------
The comparison of guides with each other in cell numbers
(flowcode_similarity.py) on a small paired experiment: five groups of six
mice, twelve guides, two guides planted with the same gains and losses and
one with the opposite.

    pytest tests/test_flowcode_similarity.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PLUGIN_DIR = Path(os.environ.get('FLOWCODE_PLUGIN_DIR',
                                 Path(__file__).resolve().parents[1] / 'user_plugins' / 'flowcode'))
sys.path.insert(0, str(PLUGIN_DIR))

import flowcode_similarity as fsim  # noqa: E402

GROUPS = ('A', 'B', 'C', 'D', 'E')
MICE = [f'M{i}' for i in range(6)]
GUIDES = [f'G{i}' for i in range(12)]
SAMPLES = [f'Raw/{g}/{m}.fcs' for g in GROUPS for m in MICE]
GROUP_OF = {s: s.split('/')[1] for s in SAMPLES}
MOUSE_OF = {s: s.split('/')[2][:-4] for s in SAMPLES}
FIRST, SECOND, OPPOSITE = 'G0', 'G1', 'G2'
PATTERN = {'B': 4.0, 'C': 1.0, 'D': 0.25, 'E': 2.0}
KWARGS = dict(group_order=list(GROUPS), contrast_mode='reference', reference='A', pairing=MOUSE_OF)


def _counts(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mean = np.full((len(SAMPLES), len(GUIDES)), 300.0)
    for row, sample in enumerate(SAMPLES):
        factor = PATTERN.get(GROUP_OF[sample], 1.0)
        mean[row, GUIDES.index(FIRST)] *= factor
        mean[row, GUIDES.index(SECOND)] *= factor
        mean[row, GUIDES.index(OPPOSITE)] /= factor
    return pd.DataFrame(rng.poisson(mean).astype(float), index=SAMPLES, columns=GUIDES)


@pytest.fixture(scope='module')
def result():
    counts = _counts()
    return fsim.run(counts, counts.sum(axis=1), GROUP_OF, MOUSE_OF, KWARGS, 'freq', n_splits=6, seed=1)


def test_guides_with_the_same_pattern_are_each_others_partner(result):
    mine = fsim.partners(result, None, FIRST)
    assert list(mine['rank']) == [1, 2, 3, 4, 5] and mine['partner'].iloc[0] == SECOND
    assert mine['similarity'].iloc[0] > 0.9
    assert mine['halves'].iloc[0] == 12 and mine['halves_in_top'].iloc[0] == 12
    assert OPPOSITE not in set(mine['partner'])
    back = fsim.partners(result, None, SECOND)
    assert back['partner'].iloc[0] == FIRST
    assert fsim.partners(result, None, 'no such guide').empty


def test_summary_says_what_the_list_rests_on(result):
    row = fsim.level_summary(result, None)
    assert row['guides'] == len(GUIDES) and row['comparisons'] == 4
    assert row['chance'] == pytest.approx(5 / 11) and np.isnan(row['tag_share'])
    assert -1.0 <= row['agreement_min'] <= row['agreement'] <= row['agreement_max'] <= 1.0
    text = fsim.summary_text(result, None, 'Mouse')
    assert 'frequencies test' in text and 'split by Mouse' in text and 'over 6 splits' in text
    assert 'tags are not set' in text
    assert fsim.level_summary(result, 'another level') is None


def test_too_few_comparisons_gives_a_note_and_no_table():
    counts = _counts()
    keep = [s for s in SAMPLES if GROUP_OF[s] in ('A', 'B', 'C')]
    kwargs = dict(KWARGS, group_order=['A', 'B', 'C'])
    out = fsim.run(counts.loc[keep], counts.loc[keep].sum(axis=1), GROUP_OF, MOUSE_OF, kwargs, 'freq', n_splits=2)
    assert out['table'].empty and out['summary'].empty
    assert out['notes'] and 'at least 4 group comparisons' in out['notes'][0]


def test_signature_is_centred_and_similarity_uses_shared_comparisons():
    table = pd.DataFrame({'feature': ['a', 'b', 'c'] * 2, 'comparison': ['x'] * 3 + ['y'] * 3,
                          't': [3.0, 1.0, -1.0, 0.0, 2.0, 4.0], 'df.total': [np.inf] * 6})
    block = fsim.signature(table, ['x', 'y'])
    assert list(block.columns) == ['x', 'y'] and np.allclose(block.median(axis=0), 0.0)
    assert block.at['a', 'x'] == pytest.approx(2.0) and block.at['c', 'y'] == pytest.approx(2.0)
    values = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, np.nan],
                           [5.0, 4.0, np.nan, np.nan, np.nan]], index=list('abc'))
    sim = fsim.similarity(values, min_shared=4)
    assert sim.at['a', 'b'] == pytest.approx(1.0) and np.isnan(sim.at['a', 'c']) and np.isnan(sim.at['a', 'a'])


def test_similarity_from_shared_tags_is_taken_out():
    rng = np.random.default_rng(5)
    names = [f'g{i}' for i in range(40)]
    tag_names = ['T1', 'T2', 'T3', 'T4', 'T5']
    tags = pd.DataFrame(0.0, index=names, columns=tag_names)
    for row, name in enumerate(names):
        tags.loc[name, tag_names[row % 5]] = 1.0
        tags.loc[name, tag_names[(row // 5) % 5]] = 1.0
    noise = rng.normal(0.0, 0.1, (40, 40))
    noise = (noise + noise.T) / 2.0
    both_t1 = np.outer(tags['T1'], tags['T1'])
    sim = pd.DataFrame(noise + 0.5 * both_t1, index=names, columns=names)
    cleaned, share = fsim.remove_tag_effect(sim, tags)
    i, k = np.triu_indices(40, 1)
    shared = both_t1[i, k] > 0
    before = sim.to_numpy()[i, k]
    after = cleaned.to_numpy()[i, k]
    assert before[shared].mean() - before[~shared].mean() > 0.4
    assert abs(after[shared].mean() - after[~shared].mean()) < 0.05
    assert share > 0.3 and after.mean() == pytest.approx(before.mean(), abs=1e-9)
    assert np.allclose(cleaned.to_numpy(), cleaned.to_numpy().T)
    same, none = fsim.remove_tag_effect(sim, None)
    assert same is sim and none is None
    few, none = fsim.remove_tag_effect(sim.iloc[:4, :4], tags)
    assert none is None and few.equals(sim.iloc[:4, :4])
    assert fsim.shared_tags(tags, names).at['g0', 'g0'] == 1.0
    assert fsim.shared_tags(tags, names + ['unknown']) is None


def test_halves_keep_every_group_when_samples_are_not_paired():
    rng = np.random.default_rng(2)
    own = {s: s for s in SAMPLES}
    half = fsim.split_levels(own, GROUP_OF, rng)
    assert len(half) == 15
    assert {g: sum(GROUP_OF[s] == g for s in half) for g in GROUPS} == {g: 3 for g in GROUPS}
    mice = fsim.split_levels(MOUSE_OF, GROUP_OF, rng)
    assert len(mice) == 3 and mice <= set(MICE)


def test_table_is_written_with_population_names(result, tmp_path):
    paths = fsim.write_table(tmp_path, result, {'': 'T cells'})
    assert [p.name for p in paths] == ['synopsis_similar_guides.csv']
    table = pd.read_csv(paths[0])
    assert set(table['population']) == {'T cells'} and set(table['test']) == {'freq'}
    assert len(table) == 5 * len(GUIDES) and 'agreement_of_population' in table.columns
    assert fsim.write_table(tmp_path, None) == []


def test_similar_guides_are_found_shown_and_kept_in_the_dossier(qapp, qtbot, tmp_path):
    import flowcode_stats as fst
    import flowcode_synopsis_view as fsyv
    import flowcode_synthetic as syn
    import flowcode_unmix_tab as module
    import test_flowcode_tab as tft

    module._ensure_qt_imports()
    syn.write_experiment(tmp_path)
    controller = tft._with_unmixed_gate(tft.FlowCodeController(tmp_path))
    plugin, _bus = tft._plugin(module, qtbot, controller)
    plugin.tabs.setCurrentIndex(5)
    page = plugin.pages[6]
    view = page.synopsis_view
    tft._settle(plugin, qtbot)
    page.section_tabs.setCurrentWidget(view)
    assert not view.similar_btn.isEnabled() and view.similar_status.text() == fsyv.SIMILAR_NEEDS

    counts = _counts()
    page.stats = fst.run_statistics(counts, counts.sum(axis=1), GROUP_OF, run_counts=True, run_freq=True, **KWARGS)
    view._can_compare = lambda: page.stats is not None
    view._similar_call = lambda: dict(counts=counts, denominator=counts.sum(axis=1), groups=GROUP_OF,
                                      levels=MOUSE_OF, kwargs=dict(KWARGS, run_counts=True, run_freq=True),
                                      tags=None, split_by='Mouse')
    view.refresh()
    qtbot.waitUntil(lambda: view.synopsis is not None, timeout=tft.LONG)
    tft._settle(plugin, qtbot)
    view.show_dossier(FIRST)
    assert view.similar_btn.isEnabled() and view.similar_btn.text() == 'Find similar guides'
    assert view.similar_table.rowCount() == 0 and view.similar_status.text() == fsyv.SIMILAR_HINT

    view.start_similar()
    assert plugin.task_running(fsyv.TASK_SIMILAR) and not view.similar_btn.isEnabled()
    page.on_task_progress(fsyv.TASK_SIMILAR, 2, 20, '')
    assert view.similar_status.text() == f'{fsyv.SIMILAR_RUNNING} Split 3 of 20.'
    page.on_task_progress(fsyv.TASK_SIMILAR, 20, 20, '')
    assert view.similar_status.text() == f'{fsyv.SIMILAR_RUNNING} Split 3 of 20.'
    qtbot.waitUntil(lambda: 'similar_guides' in page.stats and not plugin.task_running(fsyv.TASK_SIMILAR),
                    timeout=tft.LONG)
    tft._settle(plugin, qtbot)
    assert page.stats['similar_guides']['test'] == 'counts' and page.stats['similar_guides']['split_by'] == 'Mouse'
    assert view.similar_table.rowCount() == 5 and view.similar_table.item(0, 0).text() == SECOND
    assert view.similar_table.horizontalHeaderItem(2).text() == 'among the 5 most similar (halves)'
    assert view.similar_table.item(0, 2).text() == '40 of 40'
    assert 'split by Mouse' in view.similar_status.text() and view.similar_btn.text() == 'Find similar guides again'
    saved, _inputs, _when = fst.load_stats(fst.stats_path(tmp_path))
    assert len(saved['similar_guides']['table']) == 5 * len(GUIDES)

    view.show_dossier(SECOND)
    assert view.similar_table.item(0, 0).text() == FIRST
    _figures, write_tables = view.export_plan(False)
    assert 'synopsis_similar_guides.csv' in [p.name for p in write_tables(tmp_path / 'out')]
    assert 'similar guides: cell numbers (counts test)' in view.settings_text()
    page.reset_session()
    assert view.similar_table.rowCount() == 0
