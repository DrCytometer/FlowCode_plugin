"""
flowcode_similarity.py
----------------------
Which guides behave alike in cell numbers across the groups, for the
Synopsis tab of the FlowCode Analysis page (no Qt).

In plain terms. Every guide has a pattern of gains and losses over the
group comparisons of the statistics: more cells here, fewer there. Two
guides whose patterns rise and fall together are similar, which is what
knockouts of genes acting in one pathway are expected to show. For each
guide this module lists the guides most like it, and says how often each
pairing is found again when only half the mice are used.

Signature. For one population (a statistics level), a guide's signature
is its z-score in every group comparison: the z with the same two-sided
p-value as the test's t, so an uncertain fold change sits near zero. Each
comparison is centred on the median guide, so that a shift shared by the
whole library (an error in the baseline group, for example) does not make
every pair look alike.

Similarity. The Pearson correlation of two signatures over the comparisons
both guides were tested in; undefined with fewer than ``MIN_SHARED``.

Tags. Guides that share FlowCode tags share whatever the tags themselves
do to detection, so they can look alike for a technical reason. The
similarity of every pair is regressed on which single tags the two guides
share and which pairs of tags they share, and that fitted part is taken
out (the mean similarity is kept). ``tag_share`` is the fraction of the
variance of the similarity that the tags accounted for.

Stability. The pairing levels (the mice with paired samples, otherwise the
samples of each group) are split in two at random ``n_splits`` times. The
statistics are run again on each half, with every other argument
unchanged, and the similarity is worked out in each. For a guide's
partner, ``halves_in_top`` counts the half-data maps in which the partner
is again among the guide's ``top`` most similar guides, of ``halves``
maps in which both were tested. By chance that is about
``top / (guides - 1)`` of them. ``agreement`` is the Spearman correlation,
over all pairs of guides, between the two halves' similarities: near zero
when nothing repeats.

Simplifications. Only cell numbers are used: marker and phenotype-shift
similarity are left out until they have been checked on a library with
control guides. Population contrasts are left out, as in the Synopsis.
With several libraries a guide carried by more than one takes the tags of
the first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TOP = 5
N_SPLITS = 20
MIN_SHARED = 4
MIN_GUIDES = 8
# Pairs of guides needed per term of the tag regression.
PAIRS_PER_TAG_TERM = 5
TABLE_COLUMNS = ('level', 'guide', 'partner', 'rank', 'similarity', 'halves_in_top', 'halves', 'shared_tags')
SUMMARY_COLUMNS = ('level', 'guides', 'comparisons', 'agreement', 'agreement_min', 'agreement_max', 'tag_share',
                   'chance')


def level_key(level) -> str:
    """The level of a statistics comparison as stored in the tables."""
    return '' if level is None else str(level)


def signature(table: pd.DataFrame, labels: list) -> pd.DataFrame:
    """Guides × comparisons: the z-score of every guide in the comparisons
    *labels* of a results table, each comparison centred on its median
    guide (module docstring)."""
    from scipy import stats as st

    sub = table[table['comparison'].isin(labels)]
    t = sub['t'].to_numpy(float)
    df = sub['df.total'].to_numpy(float)
    df = np.where(np.isfinite(df) & (df > 0), df, 1e9)
    p = 2.0 * st.t.sf(np.abs(t), df)
    z = np.sign(t) * st.norm.isf(np.clip(p, 1e-300, 1.0) / 2.0)
    wide = sub.assign(z=z).pivot_table(index='feature', columns='comparison', values='z', aggfunc='first')
    wide = wide.reindex(columns=[c for c in labels if c in wide.columns])
    wide.index = wide.index.astype(str)
    return wide - wide.median(axis=0, skipna=True)


def similarity(block: pd.DataFrame, min_shared: int = MIN_SHARED) -> pd.DataFrame:
    """Guides × guides Pearson correlation of the rows of *block* over the
    columns both rows have; NaN with fewer than *min_shared* in common."""
    values = block.to_numpy(float)
    valid = np.isfinite(values).astype(float)
    filled = np.where(valid > 0, values, 0.0)
    n = valid @ valid.T
    with np.errstate(divide='ignore', invalid='ignore'):
        sum_a, sum_b = filled @ valid.T, valid @ filled.T
        cov = filled @ filled.T - sum_a * sum_b / n
        var_a = (filled ** 2) @ valid.T - sum_a ** 2 / n
        var_b = valid @ (filled ** 2).T - sum_b ** 2 / n
        out = cov / np.sqrt(var_a * var_b)
    out[(n < min_shared) | ~np.isfinite(out)] = np.nan
    np.fill_diagonal(out, np.nan)
    return pd.DataFrame(out, index=block.index, columns=block.index)


def shared_tags(tags: pd.DataFrame | None, guides: list) -> pd.DataFrame | None:
    """Guides × guides number of tags two guides share, or None without
    the tags of every guide."""
    if tags is None or not set(guides) <= set(tags.index):
        return None
    has = tags.loc[guides].to_numpy(float)
    return pd.DataFrame(has @ has.T, index=guides, columns=guides)


def remove_tag_effect(sim: pd.DataFrame, tags: pd.DataFrame | None) -> tuple:
    """
    (*sim* with the part explained by shared tags taken out, the fraction
    of its variance that part made up). The part is fitted over all pairs
    of guides by least squares on one term per tag both guides carry and
    one per pair of tags both carry. Returns (*sim*, None) when the tags
    are not known or the pairs are too few for the number of terms.
    """
    guides = list(sim.index)
    if tags is None or not set(guides) <= set(tags.index):
        return sim, None
    has = tags.loc[guides].to_numpy(float)
    has = has[:, has.sum(axis=0) > 0]
    i, k = np.triu_indices(len(guides), 1)
    single = has[i] * has[k]
    n_tags = single.shape[1]
    pairs = [single[:, a] * single[:, b] for a in range(n_tags) for b in range(a + 1, n_tags)]
    terms = np.column_stack([single] + pairs) if pairs else single
    terms = terms[:, terms.std(axis=0) > 0]
    y = sim.to_numpy(float)[i, k]
    ok = np.isfinite(y)
    if terms.shape[1] == 0 or ok.sum() < PAIRS_PER_TAG_TERM * (terms.shape[1] + 1) or not y[ok].var() > 0:
        return sim, None
    centred = terms - terms[ok].mean(axis=0)
    design = np.column_stack([np.ones(len(y)), centred])
    beta, *_ = np.linalg.lstsq(design[ok], y[ok], rcond=None)
    fitted = centred @ beta[1:]
    out = sim.to_numpy(float).copy()
    out[i, k] = y - fitted
    out[k, i] = y - fitted
    share = float(1.0 - (y[ok] - design[ok] @ beta).var() / y[ok].var())
    return pd.DataFrame(out, index=sim.index, columns=sim.columns), max(share, 0.0)


def _top(sim: pd.DataFrame, top: int) -> dict:
    """{guide: [its *top* most similar guides, most similar first]}."""
    values = sim.to_numpy(float)
    names = list(sim.index)
    out = {}
    for row, guide in enumerate(names):
        order = [j for j in np.argsort(-np.nan_to_num(values[row], nan=-np.inf)) if np.isfinite(values[row, j])]
        out[guide] = [names[j] for j in order[:top]]
    return out


def _levels_of(stats: dict) -> dict:
    """{level: [comparison label]} of the group comparisons of a run."""
    import flowcode_synopsis as fsy

    out = {}
    for label, (_base, _other, level) in fsy._label_pairs(stats).items():
        out.setdefault(level, []).append(label)
    return out


def _maps(stats: dict, test: str, tags: pd.DataFrame | None) -> dict:
    """{level: (similarity with the tag effect removed, tag share, number
    of comparisons)} of a statistics run, for the levels that can be read."""
    table = stats.get(test)
    out = {}
    if table is None:
        return out
    for level, labels in _levels_of(stats).items():
        block = signature(table, labels)
        block = block.dropna(how='all')
        if block.shape[1] < MIN_SHARED or block.shape[0] < MIN_GUIDES:
            continue
        sim, share = remove_tag_effect(similarity(block), tags)
        out[level] = (sim, share, block.shape[1])
    return out


def split_levels(levels: dict, groups: dict, rng) -> set:
    """
    A random half of the pairing levels. When every level lies in one
    group (unpaired samples), half of each group's levels, so that both
    halves keep every group.
    """
    group_of = {}
    for sample, level in levels.items():
        if level in (None, '') or not groups.get(sample):
            continue
        group_of.setdefault(level, set()).add(groups[sample])
    names = sorted(group_of, key=str)
    if names and all(len(g) == 1 for g in group_of.values()):
        chosen = set()
        by_group = {}
        for level in names:
            by_group.setdefault(next(iter(group_of[level])), []).append(level)
        for members in by_group.values():
            chosen |= {members[j] for j in rng.permutation(len(members))[:len(members) // 2]}
        return chosen
    return {names[j] for j in rng.permutation(len(names))[:len(names) // 2]}


def run(counts: pd.DataFrame, denominator: pd.Series, groups: dict, levels: dict, kwargs: dict, test: str, *,
        tags: pd.DataFrame | None = None, n_splits: int = N_SPLITS, top: int = TOP, seed: int = 0,
        progress=None) -> dict:
    """
    The guides most like each guide in cell numbers, and how often each
    pairing holds in half the data (module docstring).

    counts, denominator, groups, kwargs : the arguments of the statistics
              run (``flowcode_stats.run_statistics``).
    levels  : {sample: pairing level} (the mouse; the sample itself when
              samples are not paired).
    test    : 'counts' or 'freq', the test the z-scores are taken from.
    tags    : guides × tags (1 where the guide carries the tag), or None.
    progress: (n, total, text) is called before each split.

    Returns {'test', 'top', 'n_splits', 'seed', 'table' (``TABLE_COLUMNS``:
    each guide's *top* partners per level), 'summary' (``SUMMARY_COLUMNS``:
    one row per level), 'notes'}.
    """
    import flowcode_stats as fst
    from scipy import stats as st

    kw = dict(kwargs, allow_no_comparisons=True, run_counts=test == 'counts', run_freq=test == 'freq')
    kw.pop('fit_log', None)
    notes = []
    full = _maps(fst.run_statistics(counts, denominator, groups, **kw), test, tags)
    if not full:
        notes.append(f'Similar guides need at least {MIN_SHARED} group comparisons and {MIN_GUIDES} guides in a '
                     'population.')
    rng = np.random.default_rng(seed)
    held = {level: {} for level in full}
    present = {level: {} for level in full}
    agreement = {level: [] for level in full}
    wanted = {level: _top(sim, top) for level, (sim, _share, _n) in full.items()}
    failed = 0
    for split in range(n_splits if full else 0):
        if progress is not None:
            progress(split, n_splits, '')
        half = split_levels(levels, groups, rng)
        maps = []
        for inside in (True, False):
            kept = [s for s in counts.index if (levels.get(s) in half) == inside and levels.get(s) not in (None, '')]
            try:
                maps.append(_maps(fst.run_statistics(counts.loc[kept], denominator.loc[kept], groups, **kw), test,
                                  tags))
            except RuntimeError:
                maps.append({})
        if not all(maps):
            failed += 1
        for level in full:
            for part in maps:
                if level not in part:
                    continue
                sim = part[level][0]
                best = _top(sim, top)
                for guide, partners in wanted[level].items():
                    if guide not in best:
                        continue
                    for partner in partners:
                        if partner not in sim.index or not np.isfinite(sim.at[guide, partner]):
                            continue
                        key = (guide, partner)
                        present[level][key] = present[level].get(key, 0) + 1
                        held[level][key] = held[level].get(key, 0) + int(partner in best[guide])
            if all(level in part for part in maps):
                a, b = maps[0][level][0], maps[1][level][0]
                common = [g for g in a.index if g in b.index]
                i, k = np.triu_indices(len(common), 1)
                ua, ub = a.loc[common, common].to_numpy(float)[i, k], b.loc[common, common].to_numpy(float)[i, k]
                ok = np.isfinite(ua) & np.isfinite(ub)
                if ok.sum() > 2:
                    agreement[level].append(float(st.spearmanr(ua[ok], ub[ok])[0]))
    if progress is not None:
        progress(n_splits, n_splits, '')
    if failed:
        notes.append(f'{failed} of {n_splits} splits left a half in which no comparison could be made.')
    rows, summary = [], []
    for level, (sim, share, n_comparisons) in full.items():
        common = shared_tags(tags, list(sim.index))
        for guide, partners in wanted[level].items():
            for rank, partner in enumerate(partners, start=1):
                rows.append((level_key(level), guide, partner, rank, float(sim.at[guide, partner]),
                             held[level].get((guide, partner), 0), present[level].get((guide, partner), 0),
                             float(common.at[guide, partner]) if common is not None else np.nan))
        values = agreement[level]
        summary.append((level_key(level), len(sim.index), n_comparisons,
                        float(np.mean(values)) if values else np.nan, float(np.min(values)) if values else np.nan,
                        float(np.max(values)) if values else np.nan, np.nan if share is None else share,
                        min(1.0, top / max(len(sim.index) - 1, 1))))
    if full and tags is None:
        notes.append('The FlowCode tags are not set, so similarity from shared tags was not removed.')
    elif full and any(share is None for _sim, share, _n in full.values()):
        notes.append('Too few guides to estimate the effect of shared tags, so it was not removed.')
    return {'test': test, 'top': int(top), 'n_splits': int(n_splits), 'seed': int(seed),
            'table': pd.DataFrame(rows, columns=list(TABLE_COLUMNS)),
            'summary': pd.DataFrame(summary, columns=list(SUMMARY_COLUMNS)), 'notes': notes}


def partners(result: dict | None, level, guide: str) -> pd.DataFrame:
    """The rows of *guide* in *level* from a ``run`` result, most similar first."""
    if not result or result.get('table') is None:
        return pd.DataFrame(columns=list(TABLE_COLUMNS))
    table = result['table']
    mine = table[(table['level'].fillna('').astype(str) == level_key(level))
                 & (table['guide'].astype(str) == str(guide))]
    return mine.sort_values('rank').reset_index(drop=True)


def level_summary(result: dict | None, level) -> dict | None:
    """The summary row of *level* as a dict, or None."""
    if not result or result.get('summary') is None:
        return None
    summary = result['summary']
    row = summary[summary['level'].fillna('').astype(str) == level_key(level)]
    return row.iloc[0].to_dict() if len(row) else None


def summary_text(result: dict | None, level, split_by: str = 'sample') -> str:
    """One line on what the list of similar guides rests on in *level*;
    *split_by* names what the halves are halves of (the pairing factor)."""
    import flowcode_synopsis as fsy

    row = level_summary(result, level)
    if row is None:
        return ' '.join((result or {}).get('notes') or [])
    parts = [f"Cell numbers ({fsy.TESTS.get(result['test'], result['test'])} test) over "
             f"{int(row['comparisons'])} comparisons, {int(row['guides'])} guides."]
    if np.isfinite(row['agreement']):
        parts.append(f"Agreement between two halves of the data, split by {split_by}: {row['agreement']:.2f} "
                     f"({row['agreement_min']:.2f} to {row['agreement_max']:.2f} over {int(result['n_splits'])} "
                     'splits; 0 is none, 1 is identical).')
    if np.isfinite(row['tag_share']):
        parts.append(f"Shared tags explained {100.0 * row['tag_share']:.0f}% of the similarity and were removed.")
    parts.append(f"By chance a partner is among the {int(result['top'])} most similar in about "
                 f"{100.0 * row['chance']:.0f}% of the halves.")
    return ' '.join(parts + list(result.get('notes') or []))


def write_table(folder, result: dict | None, level_labels: dict | None = None) -> list:
    """``synopsis_similar_guides.csv`` in *folder* from a ``run`` result
    (*level_labels*: {stored level: population name}); the paths written."""
    if not result or result.get('table') is None or result['table'].empty:
        return []
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    table = result['table'].copy()
    keys = table['level'].fillna('').astype(str)
    table.insert(0, 'population', [str((level_labels or {}).get(k, k)) for k in keys])
    table = table.drop(columns='level')
    table.insert(1, 'test', result['test'])
    summary = result['summary'].copy()
    summary['level'] = summary['level'].fillna('').astype(str)
    for column in ('agreement', 'tag_share'):
        table[f'{column}_of_population'] = keys.map(summary.set_index('level')[column]).to_numpy()
    path = folder / 'synopsis_similar_guides.csv'
    table.to_csv(path, index=False)
    return [path]
