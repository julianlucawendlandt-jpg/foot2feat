import importlib

import numpy as np

from analysis.recoverability.features import FeatureTable
from analysis.recoverability.tests.test_likelihood import toy_panel

N = importlib.import_module("analysis.recoverability.04_shuffle_nulls")


def test_permutations():
    panel = toy_panel(200, seed=5)
    ft = FeatureTable(panel)
    rng = np.random.default_rng(0)
    strata = N.position_strata(ft)

    p = N.perm_sequence(ft, rng)
    assert sorted(p) == list(range(len(ft.seg))), "not a permutation"
    assert np.array_equal(strata[p], strata), "sequence null left its stratum"
    assert (p != np.arange(len(p))).mean() > 0.5, "sequence null barely moved"

    p = N.perm_structure(ft, rng)
    q = np.quantile(ft.gc, [0.2, 0.4, 0.6, 0.8])
    gcq = np.searchsorted(q, ft.gc)
    assert np.array_equal(strata[p], strata) and np.array_equal(gcq[p], gcq)

    p = N.perm_start_units(ft, rng)
    c = np.nonzero(ft.start_center)[0]
    main = c[ft.dAUG[c] == 0]
    assert np.all(ft.dAUG[p[main]] == 0)
    inside = ft.cat["start"][:, 0] >= 0
    assert np.array_equal(p[~inside], np.nonzero(~inside)[0])
    assert np.all(ft.cat["start"][p[inside], 0] >= 0)
    k = ft.cat["kozak"][main, 0]
    assert sorted(k) == sorted(ft.cat["kozak"][p[main], 0])
    print("ok  permutations preserve strata / GC / start-site matching")


if __name__ == "__main__":
    test_permutations()
