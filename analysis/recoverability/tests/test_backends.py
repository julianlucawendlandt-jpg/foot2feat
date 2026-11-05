import os

import numpy as np

os.environ.setdefault("UTR_CHUNK_MB", "0.05")

from analysis.recoverability import likelihood as L
from analysis.recoverability.features import FeatureTable
from analysis.recoverability.tests.test_likelihood import base_eta, simulate, toy_panel


def test_agreement():
    panel = toy_panel(300, seed=11)
    ft = FeatureTable(panel)
    rng = np.random.default_rng(2)
    eta = (base_eta(ft) - 0.6 * ft.cont["paired"][:, 1]
           + np.where(ft.cat["kozak"][:, 0] == 0, 1.0, 0.0))
    simulate(ft, panel, eta, 150, rng)
    y = ft.counts(panel, "rep1")
    tr, te = np.arange(0, 220), np.arange(220, 300)
    worst = 0.0
    for b in (1, 5, 20):
        for name in ("P", "SW", "ST", "AUG", "S5", "SW+ST+AUG"):
            pt = L.Problem(ft, tr, b, L.MODELS[name])
            yb = pt.bin_counts(y)
            fs = L.fit(pt, yb, backend="scipy")
            ftc = L.fit(pt, yb, backend="torch")
            dobj = abs(fs.info["objective_per_read"] - ftc.info["objective_per_read"])
            dbeta = float(np.abs(fs.beta - ftc.beta).max())
            pe = L.Problem(ft, te, b, L.MODELS[name])
            a, n = L.loglik(fs, pe, pe.bin_counts(y), per_transcript=True)
            c, _ = L.loglik(ftc, pe, pe.bin_counts(y), per_transcript=True)
            dll = float(abs(a.sum() - c.sum()) / n.sum())
            assert dobj < 1e-7, (b, name, dobj)
            assert dll < 1e-6, (b, name, dll)
            worst = max(worst, dbeta)
            print("ok  b={:<2} {:<10} d_obj {:.1e}  max|d_beta| {:.1e}  heldout/read {:.1e}  "
                  "({} chunks, iters {}/{})".format(
                      b, name, dobj, dbeta, dll, len(pt._torch.chunks),
                      fs.info["iterations"], ftc.info["iterations"]))
    print("all backend checks passed (worst coefficient difference {:.1e})".format(worst))


if __name__ == "__main__":
    test_agreement()
