import numpy as np
import scipy.linalg
import scipy.sparse as sp

from analysis.recoverability.features import CAT_LEVELS, CONT_DIMS

POS_MAX = 300

MODELS = {
    "P": [],
    "S1": ["nt1"],
    "S3": ["k3"],
    "S5": ["k5"],
    "SW": ["sw"],
    "ST_paired": ["paired"],
    "ST_dG": ["dG"],
    "ST": ["paired", "dG"],
    "AUG": ["start", "m3", "p4", "kozak"],
    "UORF": ["start", "m3", "p4", "kozak", "nuaug", "in_uorf", "uorf_geom"],
    "SW+ST": ["sw", "paired", "dG"],
    "SW+AUG": ["sw", "start", "m3", "p4", "kozak"],
    "ST+AUG": ["paired", "dG", "start", "m3", "p4", "kozak"],
    "SW+ST+AUG": ["sw", "paired", "dG", "start", "m3", "p4", "kozak"],
}

_ANCHOR = 2
_BIG = 60000


def position_bins(d5, dAUG, b, pos_max=POS_MAX):
    lo = (-pos_max + _ANCHOR + _BIG) // b
    hi = (pos_max + _ANCHOR + _BIG) // b
    da = np.clip(dAUG, -pos_max, pos_max)
    aug = np.floor_divide(da + _ANCHOR + _BIG, b) - lo
    five = np.floor_divide(np.clip(d5, 0, pos_max), b)
    return aug, five, int(hi - lo + 1), int(pos_max // b + 1)


def resolution_bins(seg, dAUG, b):
    key = seg.astype(np.int64) * 1000000 + np.floor_divide(dAUG + _ANCHOR + _BIG, b)
    _, inv = np.unique(key, return_inverse=True)
    return inv.astype(np.int64), int(inv.max()) + 1 if len(inv) else 0


def _onehot_multi(idx, n_levels):
    P, m = idx.shape
    rows = np.repeat(np.arange(P), m)
    cols = idx.reshape(-1)
    keep = cols >= 0
    return sp.csr_matrix((np.ones(keep.sum()), (rows[keep], cols[keep])),
                         shape=(P, n_levels))


class Problem(object):
    def __init__(self, ft, tx_idx, b, blocks, permute=None):
        pos = ft.positions(tx_idx)
        self.pos = pos
        _, seg = np.unique(ft.seg[pos], return_inverse=True)
        self.seg = seg
        self.T = int(seg.max()) + 1 if len(seg) else 0
        self.bin, self.n_bins = resolution_bins(seg, ft.dAUG[pos], b)
        self.bin_tx = np.zeros(self.n_bins, dtype=np.int64)
        self.bin_tx[self.bin] = seg
        self.b = b

        aug, five, na, nf = position_bins(ft.d5[pos], ft.dAUG[pos], b)
        mats, self.slices, k0 = [], {}, 0
        permute = permute or {}
        spec = [("pos_aug", aug[:, None], na), ("pos_five", five[:, None], nf)]
        for k in blocks:
            src = pos if k not in permute else permute[k][pos]
            if k in CAT_LEVELS:
                spec.append((k, ft.cat[k][src], CAT_LEVELS[k]))
            elif k in CONT_DIMS:
                spec.append((k, ft.cont[k][src], None))
            else:
                raise KeyError(k)
        for name, v, n in spec:
            if n is None:
                m = sp.csr_matrix(v)
                n = v.shape[1]
            else:
                m = _onehot_multi(v, n)
            mats.append(m)
            self.slices[name] = slice(k0, k0 + n)
            k0 += n
        self.X = sp.hstack(mats, format="csr")
        self.K = k0
        self.blocks = list(blocks)
        self._pat = None

    def bin_counts(self, y_flat):
        y = np.asarray(y_flat, dtype=np.float64)[self.pos]
        return np.bincount(self.bin, weights=y, minlength=self.n_bins)

    def init_beta(self, init=None):
        beta = np.zeros(self.K)
        if init:
            for k, s in self.slices.items():
                if k in init and len(init[k]) == s.stop - s.start:
                    beta[s] = init[k]
        return beta


def _seg_logsumexp(x, seg, T):
    m = np.full(T, -np.inf)
    np.maximum.at(m, seg, x)
    m[~np.isfinite(m)] = 0.0
    return m + np.log(np.bincount(seg, weights=np.exp(x - m[seg]), minlength=T))


def nucleotide_logp(prob, beta):
    eta = prob.X @ beta
    return eta - _seg_logsumexp(eta, prob.seg, prob.T)[prob.seg]


def bin_logq(prob, logp):
    return _seg_logsumexp(logp, prob.bin, prob.n_bins)


class Fit(object):
    def __init__(self, prob, beta, info):
        self.beta, self.info = beta, info
        self.slices = dict(prob.slices)

    def coef(self, block):
        return self.beta[self.slices[block]]

    def as_dict(self):
        return {k: self.beta[s].copy() for k, s in self.slices.items()}


def loglik(fit_or_beta, prob, ybin, per_transcript=False):
    if isinstance(fit_or_beta, Fit):
        assert set(fit_or_beta.slices) == set(prob.slices), \
            "fit and problem have different feature blocks"
        beta = prob.init_beta(fit_or_beta.as_dict())
    else:
        beta = np.asarray(fit_or_beta, dtype=np.float64)
    lq = bin_logq(prob, nucleotide_logp(prob, beta))
    contrib = np.where(ybin > 0, ybin * lq, 0.0)
    if not per_transcript:
        return float(contrib.sum())
    ll = np.bincount(prob.bin_tx, weights=contrib, minlength=prob.T)
    n = np.bincount(prob.bin_tx, weights=ybin, minlength=prob.T)
    return ll, n


def _fisher_pattern(prob):
    if getattr(prob, "_pat", None) is not None:
        return prob._pat
    Xc = prob.X.tocoo()
    pat = {"row": Xc.row, "val": Xc.data}
    for name, owner, n in (("A", prob.bin, prob.n_bins), ("Xbar", prob.seg, prob.T)):
        key = owner[Xc.row].astype(np.int64) * prob.K + Xc.col
        uk, inv = np.unique(key, return_inverse=True)
        pat[name] = (inv, uk // prob.K, uk % prob.K, n)
    prob._pat = pat
    return pat


def fisher_information(prob, p, q, N_t, lam):
    pat = _fisher_pattern(prob)
    v = pat["val"] * p[pat["row"]]
    out = []
    for name, rw in (("A", N_t[prob.bin_tx] / np.maximum(q, 1e-300)), ("Xbar", N_t)):
        inv, rows, cols, n = pat[name]
        M = sp.csr_matrix((np.bincount(inv, weights=v), (rows, cols)), shape=(n, prob.K))
        Ms = sp.diags(np.sqrt(rw)) @ M
        out.append((Ms.T @ Ms).toarray())
    I = out[0] - out[1]
    I[np.diag_indices_from(I)] += lam
    return I


def fit(prob, ybin, prior_sd=1.0, init=None, max_iter=200, tol=1e-8, refresh=4,
        backend=None):
    from analysis.recoverability.config import backend as _default_backend
    backend = backend or _default_backend()
    if backend == "torch":
        from analysis.recoverability.gpu import fit_torch
        return fit_torch(prob, ybin, prior_sd=prior_sd, init=init, max_iter=max_iter,
                         tol=tol, refresh=refresh)
    f = _fit_scipy(prob, ybin, prior_sd=prior_sd, init=init, max_iter=max_iter,
                   tol=tol, refresh=refresh)
    f.info["backend"] = "scipy"
    return f


def _fit_scipy(prob, ybin, prior_sd=1.0, init=None, max_iter=200, tol=1e-8, refresh=4):
    ybin = np.asarray(ybin, dtype=np.float64)
    N_t = np.bincount(prob.bin_tx, weights=ybin, minlength=prob.T)
    N = float(N_t.sum())
    assert N > 0, "no reads in the fitting set"
    lam = 1.0 / prior_sd ** 2
    beta = prob.init_beta(init)
    Xt = prob.X.T.tocsr()

    def objective(bt):
        lq = bin_logq(prob, nucleotide_logp(prob, bt))
        ll = float(np.where(ybin > 0, ybin * lq, 0.0).sum())
        return (-ll + 0.5 * lam * bt @ bt) / N, ll

    obj, ll = objective(beta)
    it, converged, fac, age, n_refresh = 0, False, None, refresh, 0
    grad = np.zeros(prob.K)
    while it < max_iter:
        it += 1
        logp = nucleotide_logp(prob, beta)
        p = np.exp(logp)
        q = np.exp(bin_logq(prob, logp))
        ratio = np.where(q[prob.bin] > 0, ybin[prob.bin] / q[prob.bin], 0.0)
        grad = Xt @ (ratio * p - N_t[prob.seg] * p) - lam * beta
        fresh = age >= refresh or fac is None
        if fresh:
            I = fisher_information(prob, p, q, N_t, lam)
            try:
                fac = scipy.linalg.cho_factor(I)
            except np.linalg.LinAlgError:
                I[np.diag_indices_from(I)] += 1e-6 * np.abs(np.diag(I)).max()
                fac = scipy.linalg.cho_factor(I)
            age, n_refresh = 0, n_refresh + 1
        step = scipy.linalg.cho_solve(fac, grad)
        t = 1.0
        while True:
            new_obj, new_ll = objective(beta + t * step)
            if new_obj <= obj + 1e-14 or t < 1e-6:
                break
            t *= 0.5
        if not new_obj <= obj:
            if fresh:
                converged = True
                break
            age = refresh
            continue
        beta = beta + t * step
        change = obj - new_obj
        obj, ll = new_obj, new_ll
        age += 1
        if change < tol:
            converged = True
            break
    info = {"objective_per_read": obj, "train_ll": ll, "n_reads": N,
            "n_transcripts": prob.T, "n_params": prob.K, "iterations": it,
            "fisher_refreshes": n_refresh, "converged": converged,
            "max_abs_grad_per_read": float(np.abs(grad).max() / N)}
    return Fit(prob, beta, info)


def delta_ll(ll_a, n_a, ll_b, n_b):
    assert np.allclose(n_a, n_b)
    d = ll_b - ll_a
    keep = n_a > 0
    return float(d[keep].sum() / n_a[keep].sum()), float((d[keep] / n_a[keep]).mean())
