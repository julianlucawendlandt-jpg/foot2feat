import numpy as np

from analysis.recoverability import likelihood as L
from analysis.recoverability.features import FeatureTable


def toy_panel(n=400, seed=0, length=(60, 160)):
    rng = np.random.default_rng(seed)
    panel = []
    for i in range(n):
        m = int(rng.integers(*length))
        s = "".join(rng.choice(list("ACGT"), size=m))
        for _ in range(int(rng.integers(0, 3))):
            p = int(rng.integers(5, m - 10))
            s = s[:p] + "ATG" + s[p + 3:]
        seq = s + "ATG" + "".join(rng.choice(list("ACGT"), size=30))
        paired = (rng.random(len(seq)) < 0.4).astype(float)
        dg = rng.normal(-8, 3, size=len(seq))
        panel.append({"transcript_id": "t%d" % i, "chrom": "c%d" % (i % 5),
                      "seq": seq, "msi": m, "paired": paired, "local_dG": dg,
                      "counts": {}})
    return panel


def simulate(ft, panel, eta, depth, rng, rep="rep1"):
    for t in range(len(panel)):
        a, b = ft.tx_start[t], ft.tx_start[t + 1]
        e = eta[a:b]
        p = np.exp(e - e.max())
        p /= p.sum()
        y = rng.multinomial(depth, p)
        full = np.zeros(len(panel[t]["seq"]), dtype=np.int64)
        full[:len(y)] = y
        panel[t]["counts"][rep] = full


def base_eta(ft):
    return -0.01 * np.clip(ft.dAUG, 0, 300) + 2.0 * (np.abs(ft.dAUG) <= 2)


def test_normalisation_and_binning():
    panel = toy_panel(50)
    ft = FeatureTable(panel)
    rng = np.random.default_rng(1)
    eta = base_eta(ft) + rng.normal(0, 0.5, len(ft.seg))
    simulate(ft, panel, eta, 200, rng)
    y = ft.counts(panel, "rep1")
    for b in (1, 3, 5, 10, 20):
        prob = L.Problem(ft, np.arange(50), b, ["sw"])
        beta = rng.normal(0, 0.3, prob.K)
        lq = L.bin_logq(prob, L.nucleotide_logp(prob, beta))
        tot = np.bincount(prob.bin_tx, weights=np.exp(lq), minlength=prob.T)
        assert np.allclose(tot, 1.0, atol=1e-10), b
        e = prob.X @ beta
        ll_hand = 0.0
        yb = prob.bin_counts(y)
        for t in range(prob.T):
            m = prob.seg == t
            pn = np.exp(e[m] - e[m].max())
            pn /= pn.sum()
            bins = prob.bin[m]
            pb = np.bincount(bins - bins.min(), weights=pn)
            ybt = yb[bins.min():bins.max() + 1]
            ll_hand += float((ybt[ybt > 0] * np.log(pb[ybt > 0])).sum())
        assert abs(L.loglik(beta, prob, yb) - ll_hand) < 1e-6, b
        for t in range(3):
            pos = np.arange(ft.tx_start[t], ft.tx_start[t + 1])
            sel = np.isin(ft.dAUG[pos], (0, -1, -2))
            ids = prob.bin[prob.seg == t][sel]
            if b >= 3:
                assert len(set(ids)) == 1, (b, ids)
    print("ok  normalisation, exact bin aggregation, AUG anchoring")


def test_subset_invariant_indices():
    panel = toy_panel(40)
    ft = FeatureTable(panel)
    a = L.Problem(ft, np.arange(0, 20), 5, [])
    b = L.Problem(ft, np.arange(20, 40), 5, [])
    assert a.slices["pos_aug"] == b.slices["pos_aug"]
    for prob, txs in ((a, range(0, 20)), (b, range(20, 40))):
        pos = ft.positions(list(txs))
        idx = prob.X[:, prob.slices["pos_aug"]].tocsr().indices
        m = {}
        for d, i in zip(ft.dAUG[pos], idx):
            assert m.setdefault(d, i) == i
    ia = dict(zip(ft.dAUG[ft.positions(range(0, 20))], a.X[:, a.slices["pos_aug"]].tocsr().indices))
    ib = dict(zip(ft.dAUG[ft.positions(range(20, 40))], b.X[:, b.slices["pos_aug"]].tocsr().indices))
    for d in set(ia) & set(ib):
        assert ia[d] == ib[d]
    print("ok  positional indices identical across transcript subsets")


def planted(block_eta, blocks, name, depth=300, n=600, seed=3, b=1):
    panel = toy_panel(n, seed=seed)
    ft = FeatureTable(panel)
    rng = np.random.default_rng(seed)
    simulate(ft, panel, base_eta(ft) + block_eta(ft), depth, rng)
    y = ft.counts(panel, "rep1")
    tr, te = np.arange(0, n * 3 // 4), np.arange(n * 3 // 4, n)
    p0 = L.Problem(ft, tr, b, [])
    m0 = L.fit(p0, p0.bin_counts(y))
    p1 = L.Problem(ft, tr, b, blocks)
    m1 = L.fit(p1, p1.bin_counts(y), init=m0.as_dict())
    info = m1.info
    q0, q1 = L.Problem(ft, te, b, []), L.Problem(ft, te, b, blocks)
    a = L.loglik(m0, q0, q0.bin_counts(y), per_transcript=True)
    c = L.loglik(m1, q1, q1.bin_counts(y), per_transcript=True)
    rw, et = L.delta_ll(a[0], a[1], c[0], c[1])
    return m1, rw, et, info


def test_planted_structure():
    m, rw, et, info = planted(lambda ft: -0.8 * ft.cont["paired"][:, 1], ["paired"], "ST")
    w = m.coef("paired")
    assert w[1] < -0.5 and abs(w[0]) < 0.2, w
    assert rw > 0.005, rw
    print("ok  planted structure: coef %s, held-out dLL/read %.4f (eq-tx %.4f)"
          % (np.round(w, 3), rw, et))


def test_planted_kozak():
    def eta(ft):
        k = ft.cat["kozak"][:, 0]
        return np.where(k == 0, 1.5, np.where(k == 2, -1.0, 0.0))
    m, rw, et, _ = planted(eta, ["kozak"], "AUG")
    w = m.coef("kozak")
    assert w[0] > w[1] > w[2], w
    assert rw > 0, rw
    print("ok  planted Kozak: strong>adequate>weak %s, dLL/read %.4f" % (np.round(w, 3), rw))


def test_planted_window():
    def eta(ft):
        sw = ft.cat["sw"]
        oi = 10 + 4
        return 0.7 * (sw[:, oi] == oi * 4 + 2)
    m, rw, _, _ = planted(eta, ["sw"], "SW")
    w = m.coef("sw").reshape(21, 4)
    assert w[14].argmax() == 2 and w[14, 2] - w[14].mean() > 0.4, w[14]
    assert rw > 0.005, rw
    print("ok  planted window effect at +4G recovered, dLL/read %.4f" % rw)


def test_null_feature():
    m, rw, et, _ = planted(lambda ft: 0.0 * ft.d5, ["paired", "dG"], "ST")
    assert abs(rw) < 2e-3, rw
    print("ok  null feature: held-out dLL/read %.5f (eq-tx %.5f)" % (rw, et))


if __name__ == "__main__":
    test_normalisation_and_binning()
    test_subset_invariant_indices()
    test_planted_structure()
    test_planted_kozak()
    test_planted_window()
    test_null_feature()
    print("all likelihood tests passed")
