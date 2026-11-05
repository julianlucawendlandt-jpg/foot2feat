import os
import random
import time

import numpy as np

CENTRE_SLICE_ELEMS = 16 * 2 ** 20
OOM_RETRIES = 6
Q_FLOOR = 1e-30


def _torch():
    import warnings
    import torch
    warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
    warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled")
    return torch


class _TorchProblem(object):
    def __init__(self, prob, device):
        torch = _torch()
        self.device = device
        X = prob.X.tocsr()
        Xt = prob.X.T.tocsr()
        f64 = torch.float64

        def csr(m):
            return torch.sparse_csr_tensor(
                torch.as_tensor(m.indptr, dtype=torch.int64),
                torch.as_tensor(m.indices, dtype=torch.int64),
                torch.as_tensor(m.data, dtype=f64), size=m.shape).to(device)

        self.X, self.Xt = csr(X), csr(Xt)
        self.seg = torch.as_tensor(prob.seg, dtype=torch.int64, device=device)
        self.bin = torch.as_tensor(prob.bin, dtype=torch.int64, device=device)
        self.bin_tx = torch.as_tensor(prob.bin_tx, dtype=torch.int64, device=device)
        self.T, self.n_bins, self.K = prob.T, prob.n_bins, prob.K
        Xc = X.tocoo()
        order = np.lexsort((Xc.col, Xc.row))
        row, col, val = Xc.row[order], Xc.col[order], Xc.data[order]
        self.row = torch.as_tensor(row, dtype=torch.int64, device=device)
        self.col = torch.as_tensor(col, dtype=torch.int64, device=device)
        self.val = torch.as_tensor(val, dtype=torch.float32, device=device)
        assert np.all(np.diff(prob.seg) >= 0), "positions must be grouped by transcript"
        tx_pos_start = np.searchsorted(prob.seg, np.arange(prob.T + 1))
        self.tx_nnz_start = np.searchsorted(row, tx_pos_start)
        self.tx_bin_lo = np.array([prob.bin[tx_pos_start[t]:tx_pos_start[t + 1]].min()
                                   if tx_pos_start[t + 1] > tx_pos_start[t] else 0
                                   for t in range(prob.T)] + [prob.n_bins])
        self.tx_bin_hi = np.array([prob.bin[tx_pos_start[t]:tx_pos_start[t + 1]].max() + 1
                                   if tx_pos_start[t + 1] > tx_pos_start[t] else 0
                                   for t in range(prob.T)])
        mb = float(os.environ.get("UTR_CHUNK_MB", 512 if str(device).startswith("cuda") else 128))
        self.chunk_elems = max(int(mb * 2 ** 20 / 4), self.K)
        self.chunks = self._plan_chunks()

    def _plan_chunks(self):
        chunks, t0 = [], 0
        while t0 < self.T:
            lo, hi, t1 = self.tx_bin_lo[t0], self.tx_bin_hi[t0], t0 + 1
            while t1 < self.T:
                nlo, nhi = min(lo, self.tx_bin_lo[t1]), max(hi, self.tx_bin_hi[t1])
                if (nhi - nlo) * self.K > self.chunk_elems:
                    break
                lo, hi, t1 = nlo, nhi, t1 + 1
            chunks.append((t0, t1))
            t0 = t1
        return chunks


def _seg_lse(torch, x, seg, n):
    m = torch.full((n,), -torch.inf, dtype=x.dtype, device=x.device)
    m = m.scatter_reduce(0, seg, x, "amax", include_self=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    s = torch.zeros(n, dtype=x.dtype, device=x.device).index_add_(0, seg, torch.exp(x - m[seg]))
    return m + torch.log(s)


def _spmv(torch, A, v):
    return torch.sparse.mm(A, v.unsqueeze(1)).squeeze(1)


def fit_torch(prob, ybin, prior_sd=1.0, init=None, max_iter=200, tol=1e-8, refresh=4,
              device=None):
    torch = _torch()
    for attempt in range(OOM_RETRIES + 1):
        try:
            return _fit_torch(prob, ybin, prior_sd, init, max_iter, tol, refresh, device)
        except torch.OutOfMemoryError:
            prob._torch = None
            torch.cuda.empty_cache()
            if attempt == OOM_RETRIES:
                break
            time.sleep(random.uniform(5, 30) * (attempt + 1))
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    from analysis.recoverability.likelihood import _fit_scipy
    f = _fit_scipy(prob, ybin, prior_sd=prior_sd, init=init, max_iter=max_iter,
                   tol=tol, refresh=refresh)
    f.info["backend"] = "scipy (cuda OOM fallback)"
    return f


def _fit_torch(prob, ybin, prior_sd, init, max_iter, tol, refresh, device):
    torch = _torch()
    from analysis.recoverability.likelihood import Fit
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tp = getattr(prob, "_torch", None)
    if tp is None or tp.device != device:
        tp = _TorchProblem(prob, device)
        prob._torch = tp
    f64 = torch.float64
    y = torch.as_tensor(np.asarray(ybin, dtype=np.float64), device=device)
    N_t = torch.zeros(tp.T, dtype=f64, device=device).index_add_(0, tp.bin_tx, y)
    N = float(N_t.sum())
    assert N > 0, "no reads in the fitting set"
    lam = 1.0 / prior_sd ** 2
    beta = torch.as_tensor(prob.init_beta(init), dtype=f64, device=device)
    pos_y = y > 0

    def logp_of(bt):
        eta = _spmv(torch, tp.X, bt)
        return eta - _seg_lse(torch, eta, tp.seg, tp.T)[tp.seg]

    def objective(bt):
        lq = _seg_lse(torch, logp_of(bt), tp.bin, tp.n_bins)
        ll = float((y[pos_y] * lq[pos_y]).sum())
        return (-ll + 0.5 * lam * float(bt @ bt)) / N, ll

    def information(p, q):
        K = tp.K
        w_bin = N_t[tp.bin_tx] / torch.clamp(q, min=Q_FLOOR)
        I = torch.zeros(K, K, dtype=f64, device=device)
        pv = (tp.val * p[tp.row].to(torch.float32))
        Xbar = torch.zeros(tp.T * K, dtype=f64, device=device)
        Xbar.index_add_(0, tp.seg[tp.row] * K + tp.col, pv.to(f64))
        Xbar = Xbar.view(tp.T, K)
        for t0, t1 in tp.chunks:
            n0, n1 = int(tp.tx_nnz_start[t0]), int(tp.tx_nnz_start[t1])
            if n1 <= n0:
                continue
            blo = int(tp.tx_bin_lo[t0:t1].min())
            bhi = int(tp.tx_bin_hi[t0:t1].max())
            A = torch.zeros((bhi - blo) * K, dtype=torch.float32, device=device)
            idx = (tp.bin[tp.row[n0:n1]] - blo) * K + tp.col[n0:n1]
            A.index_add_(0, idx, pv[n0:n1])
            A = A.view(bhi - blo, K)
            step_rows = max(1, CENTRE_SLICE_ELEMS // K)
            for r0 in range(0, bhi - blo, step_rows):
                r1 = min(r0 + step_rows, bhi - blo)
                A[r0:r1] -= (q[blo + r0:blo + r1].unsqueeze(1)
                             * Xbar[tp.bin_tx[blo + r0:blo + r1]]).to(torch.float32)
            A.mul_(torch.sqrt(w_bin[blo:bhi]).to(torch.float32).unsqueeze(1))
            I += (A.T @ A).to(f64)
            del A, idx
        I += lam * torch.eye(K, dtype=f64, device=device)
        return I

    def factor(I):
        L_, info_ = torch.linalg.cholesky_ex(I)
        scale = torch.diag(I).abs().max()
        eye = torch.eye(tp.K, dtype=f64, device=device)
        for jitter in (1e-6, 1e-4, 1e-2, 1.0):
            if int(info_) == 0:
                return L_
            L_, info_ = torch.linalg.cholesky_ex(I + jitter * scale * eye)
        if int(info_) == 0:
            return L_
        return torch.diag(torch.sqrt(torch.clamp(torch.diag(I), min=1e-12)))

    obj, ll = objective(beta)
    it, converged, L_, age, n_refresh = 0, False, None, refresh, 0
    grad = torch.zeros(tp.K, dtype=f64, device=device)
    while it < max_iter:
        it += 1
        logp = logp_of(beta)
        p = torch.exp(logp)
        q = torch.exp(_seg_lse(torch, logp, tp.bin, tp.n_bins))
        qb = q[tp.bin]
        ratio = torch.where(qb > 0, y[tp.bin] / qb, torch.zeros_like(qb))
        grad = _spmv(torch, tp.Xt, ratio * p - N_t[tp.seg] * p) - lam * beta
        fresh = age >= refresh or L_ is None
        if fresh:
            L_ = factor(information(p, q))
            age, n_refresh = 0, n_refresh + 1
        step = torch.cholesky_solve(grad.unsqueeze(1), L_).squeeze(1)
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
            "max_abs_grad_per_read": float(grad.abs().max()) / N,
            "backend": "torch:" + str(device)}
    return Fit(prob, beta.cpu().numpy(), info)
