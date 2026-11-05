import argparse
import importlib
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.recoverability import config as C
from analysis.recoverability import likelihood as L
from analysis.recoverability import splits
from analysis.recoverability.bootstrap import delta_ll_ci
from analysis.recoverability.chainfast import log_visits
from model.rate_net import RateNet

EFF = importlib.import_module("analysis.recoverability.03_nested_effects")
torch.set_default_dtype(torch.float64)
DEVICE = "cuda" if C.cuda_available() and torch.cuda.is_available() else "cpu"
OBS_BINS = 6
SEP = 10


class Batch(object):
    def __init__(self, ft, tx, b):
        self.prob = L.Problem(ft, tx, b, [])
        pos = self.prob.pos
        self.pos = pos
        self.seg = torch.as_tensor(self.prob.seg)
        last = np.zeros(len(pos), bool)
        ends = np.r_[np.nonzero(np.diff(self.prob.seg))[0], len(pos) - 1]
        last[ends] = True
        self.last = torch.as_tensor(last)
        aug = (ft.cat["start"][pos, 0] == 0) & ft.start_center[pos]
        self.aug = torch.as_tensor(aug)
        kz = ft.cat["kozak"][pos, 0]
        self.kozak = torch.as_tensor(np.where(aug & (kz >= 0), kz, 1))
        self.pa = torch.as_tensor(ft.cont["paired"][pos, 1])
        self.dga = torch.as_tensor(ft.cont["dG"][pos, 1])
        self.obs = torch.as_tensor(np.minimum(ft.d5[pos] // 10, OBS_BINS))
        self.bin = torch.as_tensor(self.prob.bin)
        self.n_bins = self.prob.n_bins
        self.bin_tx = torch.as_tensor(self.prob.bin_tx)
        self.T = self.prob.T
        x = np.zeros((len(pos) + SEP * (self.T + 1), 6))
        gather = np.zeros(len(pos), dtype=np.int64)
        o, prev = SEP, -1
        for i, (p_, s) in enumerate(zip(pos, self.prob.seg)):
            if s != prev and prev != -1:
                o += SEP
            prev = s
            nt = ft.cat["nt1"][p_, 0]
            if nt >= 0:
                x[o, nt] = 1.0
            x[o, 4] = ft.cont["paired"][p_, 0]
            x[o, 5] = ft.cont["dG"][p_, 0]
            gather[i] = o
            o += 1
        self.x = torch.as_tensor(x[:o + SEP])
        self.gather = torch.as_tensor(gather)
        for name in ("seg", "last", "aug", "kozak", "pa", "dga", "obs", "bin", "bin_tx",
                     "x", "gather"):
            setattr(self, name, getattr(self, name).to(DEVICE))

    def ybin(self, y_flat):
        return torch.as_tensor(self.prob.bin_counts(y_flat), device=DEVICE)


def seg_lse(x, seg, n):
    m = torch.full((n,), -torch.inf, device=x.device).scatter_reduce(0, seg, x.detach(), "amax")
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    return m + torch.log(torch.zeros(n, device=x.device).index_add(0, seg, torch.exp(x - m[seg])))


def bin_ll(eta, bt, yb, per_transcript=False):
    logp = eta - seg_lse(eta, bt.seg, bt.T)[bt.seg]
    lq = seg_lse(logp, bt.bin, bt.n_bins)
    contrib = torch.where(yb > 0, yb * lq, torch.zeros_like(lq))
    if not per_transcript:
        return contrib.sum()
    z = torch.zeros(bt.T, device=contrib.device)
    return (z.index_add(0, bt.bin_tx, contrib).detach().cpu().numpy(),
            z.index_add(0, bt.bin_tx, yb).detach().cpu().numpy())


class Constrained(torch.nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        j = lambda *s: 0.1 * torch.randn(*s, generator=g)
        self.w0 = torch.nn.Parameter(6.0 + j(1))
        self.a = torch.nn.Parameter(j(2))
        self.d0 = torch.nn.Parameter(-1.0 + j(1))
        self.c = torch.nn.Parameter(j(2))
        self.dA = torch.nn.Parameter(1.0 + j(1))
        self.kd = torch.nn.Parameter(j(3))
        self.pA = torch.nn.Parameter(1.0 + j(1))
        self.kp = torch.nn.Parameter(j(3))
        self.g = torch.nn.Parameter(torch.zeros(OBS_BINS + 1))

    def rates(self, bt):
        w = self.w0 + self.a[0] * bt.pa + self.a[1] * bt.dga
        d = self.d0 + self.c[0] * bt.pa + self.c[1] * bt.dga
        d = torch.where(bt.aug, self.dA + self.kd[bt.kozak], d)
        p = self.pA + self.kp[bt.kozak]
        return w, p, d

    def eta(self, bt):
        w, p, d = self.rates(bt)
        g = torch.cat([self.g[:OBS_BINS], self.g.new_zeros(1)])
        return log_visits(w, p, d, bt.aug, bt.last, bt.seg, drop_at_starts=False) + g[bt.obs]

    def describe(self):
        return {"w0": float(self.w0), "step_pairing": float(self.a[0]), "step_dG": float(self.a[1]),
                "d0": float(self.d0), "dwell_pairing": float(self.c[0]), "dwell_dG": float(self.c[1]),
                "dwell_AUG": float(self.dA), "dwell_kozak_strong": float(self.kd[0] - self.kd[2]),
                "commit_AUG": float(self.pA), "commit_kozak_strong": float(self.kp[0] - self.kp[2])}


class Flexible(torch.nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = RateNet(in_channels=6, hidden=32, window=10).double()
        self.g = torch.nn.Parameter(torch.zeros(OBS_BINS + 1))

    def rates(self, bt):
        h = self.net.net(bt.x.t().unsqueeze(0)).squeeze(0)
        return h[0][bt.gather], h[1][bt.gather], h[2][bt.gather]

    def eta(self, bt):
        w, p, d = self.rates(bt)
        g = torch.cat([self.g[:OBS_BINS], self.g.new_zeros(1)])
        return log_visits(w, p, d, bt.aug, bt.last, bt.seg, drop_at_starts=False) + g[bt.obs]


def train(model, bt, yb, kind, epochs, val=None):
    N = float(yb.sum())
    info = {}
    if kind == "B":
        opt = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=300,
                                line_search_fn="strong_wolfe", tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            loss = -bin_ll(model.eta(bt), bt, yb) / N + 1e-4 * sum((q ** 2).sum() for q in model.parameters())
            loss.backward()
            return loss
        opt.step(closure)
        return model, info
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    best, best_state, best_ep, since = -np.inf, None, 0, 0
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        loss = -bin_ll(model.eta(bt), bt, yb) / N
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if val is not None and ep % 10 == 0:
            bv, yv = val
            with torch.no_grad():
                v = float(bin_ll(model.eta(bv), bv, yv)) / float(yv.sum())
            if v > best + 1e-7:
                best, best_ep, since = v, ep, 0
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
            else:
                since += 10
                if since >= 50:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    info["best_epoch"] = best_ep
    return model, info


def run(key, bins, seeds, epochs, synthetic=False, train_reps=None):
    d = C.DATASETS[key]
    recs, meta, reps, reads, ft, y = EFF.prepare(key)
    EFF.ensure_folds(d["species"], recs, reps, key)
    tag = key
    if synthetic:
        y, truth = plant_truth(ft, recs, reps, reads)
        tag = "synthetic:" + key
    all_ok = np.min([reads[r] for r in reps], axis=0) >= EFF.MIN_READS
    rows, prow, t0 = [], [], time.time()
    store = {}
    nf = splits.load(d["species"])["n_folds"]
    for b in bins:
        for k in range(nf):
            tr_mask, te_mask = splits.train_test(ft.chrom, d["species"], k)
            te = np.nonzero(te_mask & all_ok)[0]
            bte = Batch(ft, te, b)
            for A in (train_reps or reps):
                tr_all = np.nonzero(tr_mask & (reads[A] >= EFF.MIN_READS))[0]
                vr = np.random.default_rng(100 + k).random(len(tr_all)) < 0.15
                tr, va = tr_all[~vr], tr_all[vr]
                btr = Batch(ft, tr, b)
                ytr = btr.ybin(y[A])
                bva = Batch(ft, va, b)
                yva = bva.ybin(y[A])
                pt = L.Problem(ft, tr, b, [])
                fP = L.fit(pt, pt.bin_counts(y[A]), prior_sd=EFF.PRIOR_SD)
                pe = L.Problem(ft, te, b, [])
                base = {B: L.loglik(fP, pe, pe.bin_counts(y[B]), per_transcript=True) for B in reps}
                trainP = fP.info["train_ll"]
                for kind, cls in (("B", Constrained), ("C", Flexible)):
                    fields = []
                    for seed in seeds:
                        m, tinfo = train(cls(seed).to(DEVICE), btr, ytr, kind, epochs, val=(bva, yva))
                        with torch.no_grad():
                            tr_ll = float(bin_ll(m.eta(btr), btr, ytr))
                            rows.append({"dataset": tag, "replicate": A, "split": "fold{}_train".format(k),
                                         "seed": seed, "bin_width": b, "model": kind,
                                         "metric": "train_dll_per_read",
                                         "value": (tr_ll - trainP) / float(ytr.sum()),
                                         "n_reads": float(ytr.sum()), "n_transcripts": btr.T,
                                         "best_epoch": tinfo.get("best_epoch", "")})
                            for B in reps:
                                ll, n = bin_ll(m.eta(bte), bte, bte.ybin(y[B]), per_transcript=True)
                                store.setdefault((kind, A, B, b, seed), []).append(
                                    (te, ll - base[B][0], n))
                            w, p, dd = m.rates(bte)
                            fields.append(torch.stack([w, dd]).cpu().numpy())
                            if kind == "B":
                                for nm, v in m.describe().items():
                                    prow.append({"dataset": tag, "replicate": A, "split": "fold{}".format(k),
                                                 "seed": seed, "bin_width": b, "model": "B",
                                                 "metric": nm, "value": v})
                            if synthetic:
                                eta_m = m.eta(bte).cpu().numpy()
                                tv = truth[bte.pos]
                                ok = np.isfinite(tv)
                                cm = eta_m - np.bincount(bte.prob.seg, eta_m)[bte.prob.seg] / np.bincount(bte.prob.seg)[bte.prob.seg]
                                ct = tv - np.bincount(bte.prob.seg, tv)[bte.prob.seg] / np.bincount(bte.prob.seg)[bte.prob.seg]
                                rows.append({"dataset": tag, "replicate": A, "split": "fold{}_test".format(k),
                                             "seed": seed, "bin_width": b, "model": kind,
                                             "metric": "truth_corr_centered_log_occupancy",
                                             "value": float(np.corrcoef(cm[ok], ct[ok])[0, 1])})
                    if len(fields) > 1:
                        cs = [np.corrcoef(fields[i].ravel(), fields[j].ravel())[0, 1]
                              for i in range(len(fields)) for j in range(i + 1, len(fields))]
                        rows.append({"dataset": tag, "replicate": A, "split": "fold{}_test".format(k),
                                     "seed": "", "bin_width": b, "model": kind,
                                     "metric": "rate_field_corr_across_seeds", "value": float(np.mean(cs))})
                print("[chain] {} b={} fold {} rep {} ({:.0f}s)".format(tag, b, k, A, time.time() - t0))
    for (kind, A, B, b, seed), parts in store.items():
        dlt = np.concatenate([p[1] for p in parts])
        n = np.concatenate([p[2] for p in parts])
        ci = delta_ll_ci(dlt, n, n_boot=500, seed=0)
        rows.append({"dataset": tag, "replicate": "{}->{}".format(A, B), "split": "chrom_heldout",
                     "seed": seed, "bin_width": b, "model": kind, "metric": "heldout_dll_vs_P",
                     "value": ci["read_weighted"], "ci_lo": ci["rw_lo"], "ci_hi": ci["rw_hi"],
                     "n_transcripts": ci["n_transcripts"], "n_reads": ci["n_reads"],
                     "cross_replicate": A != B})
    return rows, prow


def plant_truth(ft, recs, reps, reads):
    from data.synthetic import draw_counts
    tx = np.arange(len(recs))
    bt = Batch(ft, tx, 1)
    m = Constrained(0).to(DEVICE)
    with torch.no_grad():
        m.w0.fill_(6.0); m.a.copy_(torch.tensor([-0.4, 0.0]))
        m.d0.fill_(-1.0); m.c.copy_(torch.tensor([0.5, 0.0]))
        m.dA.fill_(1.5); m.kd.copy_(torch.tensor([0.6, 0.0, -0.6]))
        m.pA.fill_(1.0); m.kp.copy_(torch.tensor([0.8, 0.0, -0.8]))
        eta = m.eta(bt).cpu().numpy()
    truth = np.full(len(ft.seg), np.nan)
    truth[bt.pos] = eta
    rng = np.random.default_rng(7)
    y = {}
    for rep in reps:
        v = np.zeros(len(ft.seg))
        for t in range(len(recs)):
            a, e = ft.tx_start[t], ft.tx_start[t + 1]
            v[a:e] = draw_counts(truth[a:e], reads[reps[0]][t], rng)
        y[rep] = v
    return y, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gse174329_eif3b", choices=sorted(C.DATASETS))
    ap.add_argument("--bins", default="10")
    ap.add_argument("--train-reps", default="rep1",
                    help="replicates to TRAIN on (all are scored); 'all' for every one")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=400, help="maximum; C stops early")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    C.ensure_layout()
    tr_reps = None if args.train_reps == "all" else args.train_reps.split(",")
    rows, prow = run(args.dataset, [int(x) for x in args.bins.split(",")],
                     [int(x) for x in args.seeds.split(",")], args.epochs, args.synthetic,
                     train_reps=tr_reps)
    tag = ("synthetic:" if args.synthetic else "") + args.dataset
    for name, data in (("chain_ablation.tsv", rows), ("chain_params.tsv", prow)):
        C.merge_tsv(C.results_path("chain", name), data, lambda r: r["dataset"] != tag)
    C.update_manifest("chain_ablation_" + tag, {"bins": args.bins, "seeds": args.seeds,
                                                "max_epochs": args.epochs, "obs_bins": OBS_BINS,
                                                "train_reps": args.train_reps,
                                                "early_stopping": "15% of training transcripts, "
                                                "every 10 epochs, patience 50"})
    print("[chain] done")


if __name__ == "__main__":
    main()
