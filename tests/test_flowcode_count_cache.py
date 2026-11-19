"""
test_flowcode_count_cache.py
----------------------------
Count events from the compact per-sample cache (flowcode_count_cache.py)
against counting the whole sample, on the synthetic FlowCode-unmixed events
of ``flowcode_synthetic.shift_sample``: Stable (Time, a per-sample axis) →
Viable (viability dye) → T (CD3) → CD8, three marker-only channels.

    pytest tests/test_flowcode_count_cache.py
"""

import os
import sys
from pathlib import Path

import numpy as np

PLUGIN_DIR = Path(os.environ.get('FLOWCODE_PLUGIN_DIR',
                                 Path(__file__).resolve().parents[1] / 'user_plugins' / 'flowcode'))
sys.path.insert(0, str(PLUGIN_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import flowcode_analysis as fa  # noqa: E402
import flowcode_count_cache as fcnt  # noqa: E402
import flowcode_synthetic as syn  # noqa: E402

MARKERS = list(syn.SHIFT_MARKER_FLUORS)
FLUORS = list(syn.SHIFT_FLUORS)
VIABILITY = [syn.SHIFT_VIABILITY]


def _same_records(a: dict, b: dict):
    assert set(a) == set(b)
    for pop in a:
        assert {k: v for k, v in a[pop].items() if k != 'marker_sums'} == \
            {k: v for k, v in b[pop].items() if k != 'marker_sums'}
        np.testing.assert_array_equal(a[pop]['marker_sums'], b[pop]['marker_sums'])


def test_counts_from_kept_events_equal_those_of_the_whole_sample(tmp_path):
    snapshot, pnn = syn.shift_gating(time_gate=True), syn.shift_pnn()
    for seed in (1, 2):
        key = f'Raw/Spleen/M{seed}.fcs'
        events = syn.shift_sample(seed, untransduced=20000)
        reads = []

        def load(events=events):
            reads.append(1)
            return events, pnn
        for gates in (['T', 'CD8'], ['CD8']):
            whole = fa.count_populations(events, pnn, key, snapshot, gates, MARKERS)
            first, read = fcnt.count_sample(key, load, snapshot, gates, MARKERS, tmp_path, 'unmix', FLUORS, VIABILITY)
            again, read_again = fcnt.count_sample(key, load, snapshot, gates, MARKERS, tmp_path, 'unmix', FLUORS,
                                                  VIABILITY)
            _same_records(first, whole)
            _same_records(again, whole)
            assert not read_again
        assert len(reads) == 1
    assert fcnt.storing_gate(snapshot, ['T', 'CD8'], FLUORS, VIABILITY) == 'Viable'
    files = list(tmp_path.glob('*.npz'))
    assert len(files) == 2
    with np.load(files[0]) as z:
        assert set(z['markers']) == {'FITC', 'PerCP', 'APC-Cy7'}
        assert len(z['marker_values']) < len(z['values']) < len(events)
        assert z['values'].shape[1] < len(pnn)


def test_a_new_storing_gate_or_unmixing_reads_the_sample_again(tmp_path):
    snapshot, pnn = syn.shift_gating(time_gate=True), syn.shift_pnn()
    events = syn.shift_sample(3)
    reads = []

    def load():
        reads.append(1)
        return events, pnn
    for gates, unmixing in ((['T'], 'a'), (['T'], 'a'), (['Stable'], 'a'), (['T'], 'b')):
        fcnt.count_sample('M3', load, snapshot, gates, MARKERS, tmp_path, unmixing, FLUORS, VIABILITY)
    assert len(reads) == 3
    assert len(list(tmp_path.glob('*.npz'))) == 1
