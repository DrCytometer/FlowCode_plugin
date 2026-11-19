"""
test_flowcode_method.py
-----------------------
The FlowCode plugin's use of the experiment's unmixing method
(user_plugins/flowcode/flowcode_method.py): with nothing set on the
AutoSpectral tab the final and debarcoding unmixes are the per-cell AF the
plugin always used; with the noise model the final unmix is
unmix_method.unmix_fluorescence of the sample (its spectra set and particle
class), and the debarcoding space follows it only when asked.
"""

import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_FLOWCODE_DIR = Path(os.environ.get('FLOWCODE_PLUGIN_DIR', _REPO / 'user_plugins' / 'flowcode'))
if str(_FLOWCODE_DIR) not in sys.path:
    sys.path.insert(0, str(_FLOWCODE_DIR))

import flowcode_core as fcc  # noqa: E402
import flowcode_method as fcm  # noqa: E402
from honeychrome.controller_components import unmix_method as um  # noqa: E402
from honeychrome.controller_components.autospectral_functions import build_af_precomputed  # noqa: E402
from test_unmix_method import A, B, _corrections, _raw, fitted, make_controller  # noqa: E402,F401


def _plugin_controller(fitted, tmp_path, unmixing=None):
    ctrl, env = make_controller(fitted, tmp_path, unmixing)
    _panel, snap, _fit, af, _entry, _events = fitted
    spectra = snap['spectra']
    ctrl._build_fluor_spectra = lambda: spectra
    ctrl.precomputed_for_profile = lambda name: ctrl.af_precomputed_cache.get(name)
    ctrl.af_precomputed_cache = {'P': build_af_precomputed(spectra, af, env)}
    return ctrl, env


def test_without_a_method_both_unmixes_are_per_cell_af(fitted, tmp_path):
    ctrl, env = _plugin_controller(fitted, tmp_path)
    plan = fcm.build_plan(ctrl, profile_lists=[['P']])
    assert plan.state is None and plan.token == '' and not plan.needs_time_probe()
    af = plan.af_for(['P'])
    assert af.profile_names == ('P',) and af.n_spectra == 3
    d = plan.spectra.shape[1]
    raw = _raw(fitted, A)[:, :d]
    expected = fcc.af_only_unmix(raw, plan.spectra, list(plan.fluor_names), af.spectra, envelope=env)
    for out in (plan.final(raw, af, A), plan.debarcode(raw, af, A)):
        np.testing.assert_array_equal(out['unmixed'], expected['unmixed'])
        np.testing.assert_array_equal(out['af_idx'], expected['af_idx'])
    assert plan.af_for(['missing']) is None


def test_noise_model_final_unmix_and_debarcoding_choice(fitted, tmp_path):
    ctrl, _env = _plugin_controller(fitted, tmp_path, {'noise_model': True, 'corrections': True})
    ctrl.experiment.process[um.CORRECTIONS_KEY] = _corrections(fitted, [B])[0]
    plan = fcm.build_plan(ctrl, profile_lists=[['P']])
    assert plan.state.method == um.M2PG and plan.token
    assert 'M2pg' in plan.label
    af = plan.af_for(['P'])
    d = plan.spectra.shape[1]
    for key in (A, B):
        raw = _raw(fitted, key)[:, :d]
        expected = um.unmix_fluorescence(raw, plan.state, af, key)
        out = plan.final(raw, af, key)
        np.testing.assert_array_equal(out['unmixed'], expected['unmixed'])
        np.testing.assert_array_equal(out['af_idx'], expected['af_idx'] + 1)
        r1 = plan.debarcode(raw, af, key)
        np.testing.assert_array_equal(
            r1['unmixed'], fcc.af_only_unmix(raw, plan.spectra, list(plan.fluor_names), af.spectra,
                                             envelope=af.precomputed.get('envelope'))['unmixed'])
    assert plan.set_for(B) == 'Day1' and plan.set_for(A) == ''
    assert not np.array_equal(plan.spectra_for(B), plan.spectra) and plan.spectra_for(A) is plan.spectra
    assert 'spectral corrections Day1' in plan.sample_label(af, B)

    experiment = fcm.build_plan(ctrl, fcm.DEBARCODE_EXPERIMENT, [['P']])
    raw = _raw(fitted, A)[:, :d]
    np.testing.assert_array_equal(experiment.debarcode(raw, af, A)['unmixed'], experiment.final(raw, af, A)['unmixed'])


def test_method_summary(fitted, tmp_path):
    ctrl, _env = _plugin_controller(fitted, tmp_path)
    summary = fcm.method_summary(ctrl)
    assert summary['noise'] == 'fitted' and summary['method'] == um.R1 and summary['corrections'] == {}
    ctrl.experiment.process['unmixing'] = {'noise_model': True, 'corrections': True}
    ctrl.experiment.process[um.CORRECTIONS_KEY] = _corrections(fitted, [B])[0]
    summary = fcm.method_summary(ctrl)
    names = fitted[1]['fluor_names']
    assert summary['noise'] == 'in use' and summary['method'] == um.M2PG
    assert summary['corrections'] == {'Day1': [names[1]]}
    assert fcm.method_api_available()
