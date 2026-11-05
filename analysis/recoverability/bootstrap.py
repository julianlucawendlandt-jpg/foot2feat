import numpy as np

from analysis.recoverability import config as C


def delta_ll_ci(d, n, n_boot=C.N_BOOT, seed=0, alpha=0.05):
    d, n = np.asarray(d, float), np.asarray(n, float)
    keep = n > 0
    d, n = d[keep], n[keep]
    rw = d.sum() / n.sum()
    et = (d / n).mean()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    brw = d[idx].sum(1) / n[idx].sum(1)
    bet = (d[idx] / n[idx]).mean(1)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {"read_weighted": float(rw), "equal_transcript": float(et),
            "rw_lo": float(np.percentile(brw, q[0])), "rw_hi": float(np.percentile(brw, q[1])),
            "et_lo": float(np.percentile(bet, q[0])), "et_hi": float(np.percentile(bet, q[1])),
            "p_rw_le_0": float((brw <= 0).mean()), "n_transcripts": int(len(d)),
            "n_reads": float(n.sum())}


def mean_ci(values, n_boot=C.N_BOOT, seed=0, alpha=0.05, weights=None):
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    v = v[ok]
    w = None if weights is None else np.asarray(weights, float)[ok]
    if len(v) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    if w is None:
        m, bm = v.mean(), v[idx].mean(1)
    else:
        m, bm = (v * w).sum() / w.sum(), (v[idx] * w[idx]).sum(1) / w[idx].sum(1)
    lo, hi = np.percentile(bm, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(m), "lo": float(lo), "hi": float(hi), "n": int(len(v))}
