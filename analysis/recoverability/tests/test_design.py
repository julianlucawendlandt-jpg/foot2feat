import importlib
import os
import subprocess
import sys

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from analysis.recoverability import config as C

KEY, FAM = "gse139131_unsel", "SW"
need = pytest.mark.skipif(not os.path.exists(C.results_path("design", "cache", "info_{}_{}_10.npz".format(KEY, FAM))),
                          reason="needs design/cache from a completed run")


@pytest.fixture(scope="module")
def mods():
    DE = importlib.import_module("analysis.recoverability.12_experiment_design")
    return DE, DE.E


@pytest.fixture(scope="module")
def pilot_info(mods):
    DE, E = mods
    return DE.Pilot(KEY), E.Info(KEY, FAM)


@need
def test_r2_reproduces_validated_predictions(mods, pilot_info):
    DE, E = mods
    P, _ = pilot_info
    truth = {"SW": "B", "ST": "C", "AUG": "D"}
    v2 = {}
    for r in C.read_tsv(C.results_path("theory", "theory_validation.tsv")):
        if r["dataset"] == KEY and r["direction"] == "rep1->rep2" and r["truth"] in ("B", "C", "D"):
            v2[(r["family"], int(float(r["size_index"])), int(float(r["depth"])))] = r
    eng, ref, sim, old = [], [], [], []
    for fam in ("SW", "ST", "AUG"):
        info = E.Info(KEY, fam)
        sizes = DE.SP.SIZE_K if fam == "AUG" else DE.SP.SIZE_S
        for si in (0, 1, 2):
            for d in (5, 10, 20, 50, 100, 200, 500):
                r = E.loo_design(info, d, 2, P.kappa, size=DE.size_units(fam, sizes[si]), cv2=P.cv2[fam],
                                 shared_depth=True)
                o = DE.design(P, DE.planted_gain(P, fam, sizes[si]), fam, d, 2)
                eng.append(r["power_one_direction"])
                ref.append(float(v2[(fam, si, d)]["pred_power"]))
                sim.append(float(v2[(fam, si, d)]["sim_power"]))
                old.append(o["power_one_direction"] if o else 0.0)
    eng, ref, sim, old = map(np.array, (eng, ref, sim, old))
    assert np.mean(np.abs(eng - ref)) < 0.05
    assert np.mean(np.abs(eng - sim)) <= np.mean(np.abs(old - sim)) + 0.02


@need
def test_r2_joint_probability(mods, pilot_info):
    DE, E = mods
    for m, v, rho, t in ((0.002, 1e-6, 0.5, 0.001), (0.0, 4e-6, 0.2, 0.001), (0.003, 1e-6, 0.0, 0.002)):
        q = E._p_all_exceed(m, v, rho, t, 2)
        cov = v * np.array([[1, rho], [rho, 1]])
        direct = stats.multivariate_normal(mean=[-m, -m], cov=cov).cdf([-t, -t])
        assert abs(q - direct) < 2e-3
        if rho == 0:
            p = 1 - stats.norm.cdf((t - m) / np.sqrt(v))
            assert abs(q - p * p) < 1e-6
    q4 = E._p_all_exceed(0.002, 1e-6, 0.3, 0.001, 4)
    cov = 1e-6 * (0.7 * np.eye(4) + 0.3)
    z = np.random.default_rng(1).multivariate_normal(np.full(4, 0.002), cov, size=400000)
    assert abs(q4 - np.mean(np.all(z > 0.001, axis=1))) < 5e-3


@need
@pytest.mark.parametrize("R", [2, 3, 4])
def test_loo_covariance_symmetric_psd(mods, pilot_info, R):
    DE, E = mods
    P, info = pilot_info
    res = E.loo_design(info, 50, R, P.kappa, size=1.0, return_parts=True)
    S = E.unit_covariance(res)
    assert np.allclose(S, S.T, rtol=1e-8, atol=1e-12 * np.abs(S).max())
    ev = np.linalg.eigvalsh(0.5 * (S + S.T))
    assert ev.min() > -1e-8 * ev.max()


def test_pooled_statistic_is_read_weighted(mods):
    DE, E = mods
    rng = np.random.default_rng(0)
    reads = rng.integers(100, 10000, size=12).astype(float)
    gains = rng.normal(0.01, 0.005, size=12) * reads
    assert np.isclose(E.pooled_dll(gains, reads), gains.sum() / reads.sum())
    assert not np.isclose(E.pooled_dll(gains, reads), np.mean(gains / reads))


@need
def test_identical_libraries_add_information(mods, pilot_info):
    DE, E = mods
    P, info = pilot_info
    r2 = E.loo_design(info, 200, 2, P.kappa)
    r4 = E.loo_design(info, 200, 4, P.kappa)
    assert np.isclose(r4["beta_information_total"] / r2["beta_information_total"], 2.0, rtol=0.05)
    assert np.isclose(r2["beta_information_total"], 2 * r2["beta_information_one_library"], rtol=0.05)


@need
def test_fewer_reads_fewer_eligible(mods, pilot_info):
    DE, E = mods
    P, info = pilot_info
    el = [E.loo_design(info, d, 2, P.kappa)["eligible_transcripts"] for d in (400, 100, 30, 10, 4)]
    assert all(a > b for a, b in zip(el, el[1:]))


@need
def test_planner_runs_without_costs():
    out = subprocess.run([sys.executable, "-m", "analysis.recoverability.plan_experiment", "--pilot", KEY,
                          "--effect", FAM, "--effect-size", "0.5", "--replicates", "2,3"],
                         capture_output=True, text=True, env=dict(os.environ))
    assert out.returncode == 0, out.stderr[-2000:]
    assert "DESIGNS AT" in out.stdout and "ILLUSTRATIVE" not in out.stdout


@need
def test_costs_move_optimum_not_frontier(mods, pilot_info):
    DE, E = mods
    P, info = pilot_info
    size = DE.size_units(FAM, 0.5)
    f1 = DE.frontier(P, info, size, (2, 4), (1e8, 1e9), P.cv2[FAM])
    need = {2: 5e8, 4: 2e8, 6: 1.5e8}
    _, cheap_lib = DE.cost_overlay(need, c_lib=10, c_mill=3)
    _, dear_lib = DE.cost_overlay(need, c_lib=5000, c_mill=3)
    f2 = DE.frontier(P, info, size, (2, 4), (1e8, 1e9), P.cv2[FAM])
    assert [r["power_pooled"] for r in f1] == [r["power_pooled"] for r in f2]
    assert cheap_lib["R"] != dear_lib["R"]


@need
def test_status_labels(mods, pilot_info):
    DE, E = mods
    P, info = pilot_info
    assert DE.design_row(P, info, 1.0, 2, 1e8, 1.0)["validation_status"] == "validated"
    assert DE.design_row(P, info, 1.0, 3, 1e8, 1.0)["validation_status"] == "extrapolated"
    rows, _ = DE.cost_overlay({2: 1e8, 3: 1e8})
    assert all(r["validation_status"] == "illustrative" for r in rows)
    assert DE.bin_status(10).startswith("simulation-validated")
    for b in (1, 5, 20):
        assert "not simulation-validated" in DE.bin_status(b)
