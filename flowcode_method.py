"""
flowcode_method.py
------------------
The experiment's unmixing method as FlowCode uses it. No Qt.

``build_plan(controller, ...)`` (main thread) reads the experiment's method
(``unmix_method.method_state``: per-cell AF, the noise model (M2pg), the
clamped-data weights or spectral variants (R2), and the Fix My Unmix
spectral corrections per sample group) and the AF libraries FlowCode needs,
and returns an ``UnmixPlan`` the workers call:

``plan.final(raw_fl, af, sample_key)``
    The final unmix of FRET-corrected raw fluorescence: the sample's method
    with its group's corrected spectra. Without a method in effect (nothing
    ticked on the AutoSpectral tab) it is Honeychrome's Python per-cell AF
    on the base spectra.

``plan.debarcode(raw_fl, af, sample_key)``
    The debarcoding space. ``'r1'`` (default): per-cell AF on the base
    spectra, the space the thresholds have always been set in.
    ``'experiment'``: the final unmix with the debarcoding AF library.

Both return ``unmixed`` (n x F, fluorophores in ``fluor_names`` order,
uncompensated), ``af_scale`` and ``af_idx`` (1-based into ``af.spectra``,
0 for none).

v1 simplifications
    * The pooled backbone (section 3 space, section 4 figures and Assess) is
      unmixed with ``sample_key=None``: base spectra, as cells.
    * The debarcoding AF library and each sample's export library are
      combined once per distinct profile list.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import flowcode_core as fcc

DEBARCODE_R1 = 'r1'
DEBARCODE_EXPERIMENT = 'experiment'
DEBARCODE_METHODS = (DEBARCODE_R1, DEBARCODE_EXPERIMENT)


def method_api_available() -> bool:
    """True when this Honeychrome resolves the experiment's unmixing method."""
    import importlib.util
    try:
        if importlib.util.find_spec('honeychrome.controller_components.unmix_method') is None:
            return False
        from honeychrome.controller_components import sample_loader
    except ImportError:
        return False
    return hasattr(sample_loader, 'AFState')


def af_state_for(controller, profile_names):
    """``sample_loader.AFState`` of the named AF profiles (stored, with a
    cached precomputed library), or None. Main thread only."""
    from honeychrome.controller_components.autospectral_functions import (
        af_index_lookup, combine_af_precomputed, precompute_joint_cov_extras,
    )
    from honeychrome.controller_components.sample_loader import AFState

    af_profiles = controller.experiment.process.get('af_profiles') or {}
    names = [str(n) for n in (profile_names or []) if str(n) in af_profiles]
    if not names:
        return None
    precomputed = [controller.precomputed_for_profile(n) for n in names]
    if any(p is None for p in precomputed):
        return None
    spectra = np.vstack([np.asarray(af_profiles[n]['spectra'], dtype=np.float64) for n in names])
    if len(precomputed) == 1:
        combined = precomputed[0]
    else:
        combined = dict(combine_af_precomputed(precomputed))
        combined.update(precompute_joint_cov_extras(combined, spectra))
    return AFState(profile_names=tuple(names), precomputed=combined, spectra=spectra,
                   index_map=af_index_lookup(af_profiles, names))


@dataclass(frozen=True, eq=False)
class UnmixPlan:
    """Everything the FlowCode workers need to unmix, read on the main thread."""
    state: object                    # unmix_method.MethodState or None (per-cell AF, base spectra)
    spectra: np.ndarray              # base reference spectra, fluorophores x detectors
    fluor_names: tuple
    debarcode_method: str = DEBARCODE_R1
    af_libraries: dict = field(default_factory=dict)    # profile-name tuple -> AFState or None
    label: str = ''
    token: str = ''

    # -- AF libraries -----------------------------------------------------------

    def af_for(self, profile_names):
        return self.af_libraries.get(tuple(str(n) for n in (profile_names or [])))

    # -- methods ----------------------------------------------------------------

    def effective_method(self, af, sample_key=None) -> str:
        from honeychrome.controller_components import unmix_method as um
        return um.R1 if self.state is None else self.state.effective_method(af, sample_key)

    def set_for(self, sample_key) -> str:
        return '' if self.state is None else self.state.set_for(sample_key)

    def spectra_for(self, sample_key) -> np.ndarray:
        """The spectra a sample's final unmix uses (its group's corrected set, else the base)."""
        set_id = self.set_for(sample_key)
        return self.state.spectra_for(set_id) if set_id else self.spectra

    def sample_label(self, af, sample_key=None) -> str:
        """Method of one sample, for the run record and the file keyword."""
        from honeychrome.controller_components import unmix_method as um
        method = self.effective_method(af, sample_key)
        text = um.method_label(method, None if self.state is None else self.state.prior_lambda)
        set_id = self.set_for(sample_key)
        return text + (f', spectral corrections {set_id}' if set_id else '')

    def needs_time_probe(self) -> bool:
        from honeychrome.controller_components import unmix_method as um
        return um.needs_time_probe(self.state)

    # -- unmixing (worker-safe) -------------------------------------------------

    def _r1_base(self, raw_fl, af) -> dict:
        out = fcc.af_only_unmix(raw_fl, self.spectra, list(self.fluor_names), af.spectra,
                                envelope=(af.precomputed or {}).get('envelope'))
        return {'unmixed': out['unmixed'], 'af_scale': out['af_scale'], 'af_idx': out['af_idx']}

    def final(self, raw_fl: np.ndarray, af, sample_key=None) -> dict:
        raw_fl = np.ascontiguousarray(raw_fl, dtype=np.float64)
        if self.state is None:
            return self._r1_base(raw_fl, af)
        from honeychrome.controller_components import unmix_method as um
        out = um.unmix_fluorescence(raw_fl, self.state, af, sample_key)
        idx = np.asarray(out['af_idx'], dtype=np.int64)
        return {'unmixed': out['unmixed'], 'af_scale': np.asarray(out['af_scale'], dtype=np.float64),
                'af_idx': np.where(idx >= 0, idx + 1, 0)}

    def debarcode(self, raw_fl: np.ndarray, af, sample_key=None) -> dict:
        if self.debarcode_method == DEBARCODE_EXPERIMENT:
            return self.final(raw_fl, af, sample_key)
        return self._r1_base(np.ascontiguousarray(raw_fl, dtype=np.float64), af)


def build_plan(controller, debarcode_method: str = DEBARCODE_R1, profile_lists=()) -> UnmixPlan:
    """
    The experiment's method and the AF libraries of *profile_lists* (each a
    list of AF profile names: the debarcoding library, each sample's export
    library). Main thread only.
    """
    from honeychrome.controller_components import unmix_method as um

    names = list(fcc_fluor_names(controller))
    spectra = np.asarray(controller._build_fluor_spectra(), dtype=np.float64)
    state = um.method_state(controller)
    libraries = {}
    for profile_names in profile_lists:
        key = tuple(str(n) for n in (profile_names or []))
        if key not in libraries:
            libraries[key] = af_state_for(controller, key)
    if debarcode_method not in DEBARCODE_METHODS:
        debarcode_method = DEBARCODE_R1
    method = um.R1 if state is None else state.method
    label = um.method_label(method, None if state is None else state.prior_lambda)
    return UnmixPlan(state=state, spectra=spectra, fluor_names=tuple(names), debarcode_method=debarcode_method,
                     af_libraries=libraries, label=label, token=um.method_token(state))


def fcc_fluor_names(controller) -> list:
    """Fluorophore labels with a reference spectrum, in spectral-model order."""
    process = controller.experiment.process
    profiles = process.get('profiles') or {}
    return [c['label'] for c in process.get('spectral_model') or [] if c['label'] in profiles]


def method_summary(controller, state=False) -> dict:
    """What the AutoSpectral tab has set, for the prerequisites panel and the
    section 5 method line: ``label``, ``token``, ``noise`` ('in use',
    'fitted', 'not fitted'), ``corrections`` ({group: [corrected
    fluorophores]} in effect). *state* is the experiment's
    ``unmix_method.method_state`` when already read. Main thread only."""
    from honeychrome.controller_components import gls_unmix as gu
    from honeychrome.controller_components import unmix_method as um

    process = controller.experiment.process
    settings = um.unmixing_settings(process)
    fitted = bool(process.get(gu.PROCESS_KEY))
    if state is False:
        state = um.method_state(controller)
    noise = 'in use' if fitted and settings['noise_model'] else ('fitted' if fitted else 'not fitted')
    corrections = {}
    if state is not None:
        for group, spectra in state.corrected_sets.items():
            changed = [n for i, n in enumerate(state.fluor_names)
                       if not np.array_equal(spectra[i], state.spectra[i])]
            corrections[group] = changed
    method = um.R1 if state is None else state.method
    return {'label': um.method_label(method, None if state is None else state.prior_lambda),
            'token': um.method_token(state), 'noise': noise, 'corrections': corrections,
            'method': method, 'lambda': None if state is None else state.prior_lambda}
