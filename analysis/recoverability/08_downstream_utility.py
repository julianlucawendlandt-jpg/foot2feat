import argparse
import csv
import importlib
import os
import sys
import time

import numpy as np
from sklearn.linear_model import RidgeCV

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.recoverability import config as C
from analysis.recoverability import likelihood as L
from analysis.recoverability import splits
from analysis.recoverability.features import FeatureTable, kozak_class
from data.features import _fold_tracks

EFF = importlib.import_module("analysis.recoverability.03_nested_effects")

CDS_START = {"yeast": "TCTAAAGGTGAAGAATTATTCACTGGTGTT",
             "human": "GTGAGCAAGGGCGAGGAGCTGTTCACCGGG"}
ALPHAS = np.logspace(-3, 4, 15)


def choose_model(key, single_rep):
    rows = [r for r in C.read_tsv(C.results_path("effects", "nested_effects.tsv"))
            if r.get("panel", r["dataset"]) == key and r["metric"] == "heldout_dll_read_weighted"
            and r["model"] not in ("S5",)]
    best = {}
    for r in rows:
        A, B = r["replicate"].split("->")
        if (A != B) == single_rep:
            continue
        k = (r["model"], int(r["bin_width"]))
        best[k] = min(best.get(k, np.inf), float(r["value"]))
    (model, b), v = max(best.items(), key=lambda kv: kv[1])
    return model, b, v


def _fold_reporter(args):
    seq, w = args
    return _fold_tracks(seq, w)


def reporter_panel(rows, species):
    import multiprocessing as mp
    seqs = [r["seq"] + "ATG" + CDS_START[species] for r in rows]
    with mp.get_context("spawn").Pool(C.default_workers(gb_per_worker=0.3)) as pool:
        folded = pool.map(_fold_reporter, [(s_, C.FOLD_WINDOW) for s_ in seqs], chunksize=50)
    return [{"transcript_id": r["id"], "chrom": r.get("chrom", "NA"), "seq": s_,
             "msi": len(r["seq"]), "paired": pr, "local_dG": dg, "counts": {}}
            for r, s_, (pr, dg) in zip(rows, seqs, folded)]


def represent(fit, model, b, ft_mpra, n):
    blocks = L.MODELS[model]
    out = np.zeros((n, 6))
    prob = L.Problem(ft_mpra, np.arange(n), b, blocks)
    beta = prob.init_beta(fit.as_dict())
    logp = L.nucleotide_logp(prob, beta)
    nopos = beta.copy()
    for k in ("pos_aug", "pos_five"):
        nopos[prob.slices[k]] = 0.0
    logp0 = L.nucleotide_logp(prob, nopos)
    p, p0 = np.exp(logp), np.exp(logp0)
    pos = prob.pos
    dA, d5 = ft_mpra.dAUG[pos], ft_mpra.d5[pos]
    startwin = ft_mpra.cat["start"][pos, 0] >= 0
    main = np.abs(dA) <= 2
    L_tx = np.bincount(prob.seg)
    for t in range(n):
        m = prob.seg == t
        out[t] = [p[m][main[m]].sum(), p0[m][main[m]].sum(),
                  p[m][startwin[m] & ~main[m]].sum(),
                  p[m][d5[m] < 30].sum(), p[m][(dA[m] < 30) & (dA[m] >= 0)].sum(),
                  float(-(p[m] * np.log(p[m] + 1e-300)).sum() / np.log(L_tx[t]))]
    return out


def seq_features(rows, species):
    import RNA
    p4 = "ACGT".find(CDS_START[species][0])
    simple, kmer = [], []
    for r in rows:
        s = r["seq"].upper()
        L_ = len(s)
        uaug = sum(1 for i in range(L_ - 2) if s[i:i + 3] == "ATG")
        m3 = "ACGT".find(s[-3]) if L_ >= 3 else -1
        oh3 = [float(m3 == i) for i in range(4)]
        kz = kozak_class(m3, p4) if m3 >= 0 else 1
        mfe = RNA.fold(s.replace("T", "U"))[1]
        gc = (s.count("G") + s.count("C")) / max(L_, 1)
        f = [np.log(L_), uaug, gc, mfe / max(L_, 1)] + oh3 + [float(kz == i) for i in range(3)]
        simple.append(f)
        km = np.zeros(64)
        for i in range(L_ - 2):
            w = s[i:i + 3]
            if set(w) <= set("ACGT"):
                km[int("".join(str("ACGT".index(c)) for c in w), 4)] += 1
        kmer.append(f + list(km / max(L_ - 2, 1)))
    return np.array(simple), np.array(kmer)


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def evaluate(X_seq, X_40, y, groups, seed=0):
    preds = {k: np.zeros(len(y)) for k in ("seq", "40S", "seq+40S", "late_fusion", "residual")}
    for g in np.unique(groups):
        te, tr = groups == g, groups != g
        mu, sd = X_seq[tr].mean(0), X_seq[tr].std(0) + 1e-9
        m4, s4 = X_40[tr].mean(0), X_40[tr].std(0) + 1e-9
        S, F = (X_seq - mu) / sd, (X_40 - m4) / s4
        ms = RidgeCV(alphas=ALPHAS).fit(S[tr], y[tr])
        mf = RidgeCV(alphas=ALPHAS).fit(F[tr], y[tr])
        mb = RidgeCV(alphas=ALPHAS).fit(np.hstack([S, F])[tr], y[tr])
        preds["seq"][te] = ms.predict(S[te])
        preds["40S"][te] = mf.predict(F[te])
        preds["seq+40S"][te] = mb.predict(np.hstack([S, F])[te])
        inner = np.zeros((tr.sum(), 2))
        idx = np.nonzero(tr)[0]
        folds = np.random.default_rng(seed).integers(0, 5, len(idx))
        for f_ in range(5):
            a, bb = idx[folds != f_], idx[folds == f_]
            inner[folds == f_, 0] = RidgeCV(alphas=ALPHAS).fit(S[a], y[a]).predict(S[bb])
            inner[folds == f_, 1] = RidgeCV(alphas=ALPHAS).fit(F[a], y[a]).predict(F[bb])
        st = RidgeCV(alphas=ALPHAS).fit(inner, y[tr])
        preds["late_fusion"][te] = st.predict(np.column_stack([preds["seq"][te], preds["40S"][te]]))
        res = y[tr] - ms.predict(S[tr])
        mr = RidgeCV(alphas=ALPHAS).fit(F[tr], res)
        preds["residual"][te] = preds["seq"][te] + mr.predict(F[te])
    rng = np.random.default_rng(seed)
    boot = rng.integers(0, len(y), (500, len(y)))
    out = {}
    for k, p in preds.items():
        d = np.array([r2(y[b], p[b]) - r2(y[b], preds["seq"][b]) for b in boot])
        out[k] = {"r2": r2(y, p), "delta_r2_vs_seq": r2(y, p) - r2(y, preds["seq"]),
                  "ci_lo": float(np.percentile(d, 2.5)), "ci_hi": float(np.percentile(d, 97.5))}
    return out


def load_akirtava():
    rows = []
    with open(os.path.join(C.ROOT, "eval", "akirtava_scer_wt.tsv")) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            s = (r["utr_seq"] or "").upper()
            if len(s) < C.MIN_LEADER_LEN or set(s) - set("ACGT") or len(s) > L.POS_MAX:
                continue
            rows.append({"id": r["id"], "chrom": r["chrom"], "seq": s,
                         "MeanYFP": r["MeanYFP"], "TEff": r["TEff"]})
    return rows


def load_sample2019():
    rows = []
    with open(os.path.join(C.ROOT, "eval", "sample2019.tsv")) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            s = r["utr_seq"].upper()
            if len(s) >= C.MIN_LEADER_LEN:
                rows.append({"id": r["id"], "chrom": "NA", "seq": s, "rl": r["rl"]})
    return rows


def run_yeast(targets):
    key = "archer_yeast"
    model, b, v = choose_model(key, single_rep=True)
    print("[down] yeast 40S model {} at {} nt (held-out dLL {:.4f})".format(model, b, v))
    recs, meta, reps, reads, ft, y = EFF.prepare(key)
    rows = load_akirtava()
    panel = reporter_panel(rows, "yeast")
    fo = splits.fold_of("yeast")
    groups = np.array([fo.get(r["chrom"], -1) for r in rows])
    X40 = np.zeros((len(rows), 6))
    for k in range(splits.load("yeast")["n_folds"]):
        tr_mask, _ = splits.train_test(ft.chrom, "yeast", k)
        tr = np.nonzero(tr_mask)[0]
        prob = L.Problem(ft, tr, b, L.MODELS[model])
        fit = L.fit(prob, prob.bin_counts(y[reps[0]]), prior_sd=EFF.PRIOR_SD)
        sel = np.nonzero(groups == k)[0]
        if len(sel):
            sub = FeatureTable([panel[i] for i in sel])
            X40[sel] = represent(fit, model, b, sub, len(sel))
    _, Xk = seq_features(rows, "yeast")
    out = []
    for target in targets:
        yv = np.array([float(r[target]) if r[target] not in ("", None) else np.nan for r in rows])
        ok = np.isfinite(yv) & (groups >= 0)
        res = evaluate(Xk[ok], X40[ok], yv[ok], groups[ok])
        for k, m in res.items():
            out.append({"dataset": "akirtava_yeast", "replicate": "", "split": "chrom_folds",
                        "seed": 0, "bin_width": b, "model": k, "n_transcripts": int(ok.sum()),
                        "metric": "heldout_r2", "value": m["r2"], "target": target,
                        "delta_r2_vs_seq": m["delta_r2_vs_seq"], "ci_lo": m["ci_lo"],
                        "ci_hi": m["ci_hi"], "representation_model": model})
            print("  {:<8} {:<12} R2 {:.4f}  dR2 {:+.4f} [{:+.4f},{:+.4f}]".format(
                target, k, m["r2"], m["delta_r2_vs_seq"], m["ci_lo"], m["ci_hi"]))
    return out


def run_human():
    key = "gse174329_eif3b"
    model, b, v = choose_model(key, single_rep=False)
    print("[down] human 40S model {} at {} nt (cross-replicate dLL {:.4f})".format(model, b, v))
    recs, meta, reps, reads, ft, y = EFF.prepare(key)
    prob = L.Problem(ft, np.arange(len(recs)), b, L.MODELS[model])
    fit = L.fit(prob, prob.bin_counts(y[reps[0]]), prior_sd=EFF.PRIOR_SD)
    rows = load_sample2019()
    panel = reporter_panel(rows, "human")
    X40 = represent(fit, model, b, FeatureTable(panel), len(rows))
    _, Xk = seq_features(rows, "human")
    groups = np.random.default_rng(0).integers(0, 5, len(rows))
    yv = np.array([float(r["rl"]) for r in rows])
    res = evaluate(Xk, X40, yv, groups)
    out = []
    for k, m in res.items():
        out.append({"dataset": "sample2019_human", "replicate": "", "split": "random_5fold",
                    "seed": 0, "bin_width": b, "model": k, "n_transcripts": len(rows),
                    "metric": "heldout_r2", "value": m["r2"], "target": "rl",
                    "delta_r2_vs_seq": m["delta_r2_vs_seq"], "ci_lo": m["ci_lo"],
                    "ci_hi": m["ci_hi"], "representation_model": model, "secondary": True})
        print("  rl       {:<12} R2 {:.4f}  dR2 {:+.4f} [{:+.4f},{:+.4f}]".format(
            k, m["r2"], m["delta_r2_vs_seq"], m["ci_lo"], m["ci_hi"]))
    return out


FINAL_MODELS = ("seq", "40S", "seq+40S", "late_fusion", "residual")
N_BOOT_FINAL = 2000
MIN_STRATUM = 300
INNER_K = 5


def choose_models_final(force=False):
    out = C.results_path("downstream", "model_choice.tsv")
    if os.path.exists(out) and not force:
        print("[final] model choice already frozen: {}".format(out))
        return C.read_tsv(out)
    src = C.results_path("effects", "nested_effects.tsv")
    eff = C.read_tsv(src)
    rows = []
    for species, key, single in (("yeast", "archer_yeast", True), ("human", "gse174329_eif3b", False)):
        best = {}
        for r in eff:
            if (r.get("panel", r["dataset"]) != key or r["metric"] != "heldout_dll_read_weighted"
                    or r["model"] in ("S5",)):
                continue
            A, B = r["replicate"].split("->")
            if (A != B) == single:
                continue
            k = (r["model"], int(r["bin_width"]))
            best[k] = min(best.get(k, np.inf), float(r["value"]))
        (model, b), v = max(best.items(), key=lambda kv: kv[1])
        metric = ("held-out chromosome dLL/read, rep1->rep1 (single library)" if single else
                  "min over cross-replicate directions of held-out dLL/read")
        if v <= 0:
            model, b, metric = "P", 10, "no biological model beats position-only (max dLL <= 0)"
            v = 0.0
        rows.append({"species": species, "dataset": key, "selected_model": model, "bin_width": b,
                     "selection_metric": metric, "value": v,
                     "source_file": os.path.relpath(src, C.ROOT),
                     "excluded_candidates": "S5 (prespecified high-capacity negative control)",
                     "uses_reporter_labels": False})
        print("[final] {} 40S model: {} at {} nt ({:.4f})".format(species, model, b, v))
    C.write_tsv(rows, out, extra_first=("species", "dataset", "selected_model", "bin_width",
                                        "selection_metric", "value", "source_file"))
    return rows


def load_reporters_final():
    excl = []
    path = os.path.join(C.ROOT, "eval", "akirtava_scer_wt_downstream.tsv")
    if not os.path.exists(path):
        sys.exit("\nMissing {} -- run eval/prepare_akirtava_downstream.py\n".format(path))
    fo = splits.fold_of("yeast")
    ak = []
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            s = (r["utr_seq"] or "").upper()
            why = ("leader_shorter_than_{}".format(C.MIN_LEADER_LEN) if len(s) < C.MIN_LEADER_LEN else
                   "leader_longer_than_{}".format(L.POS_MAX) if len(s) > L.POS_MAX else
                   "non_ACGT" if set(s) - set("ACGT") else
                   "chromosome_not_in_frozen_folds" if r["chrom"] not in fo else "")
            if why:
                excl.append({"dataset": "akirtava_yeast", "reporter_id": r["id"], "reason": why})
                continue
            ak.append({"id": r["id"], "orf": r["orf"], "chrom": r["chrom"], "seq": s,
                       "fold": fo[r["chrom"]],
                       "TEff": float(r["TEff"]) if r["teff_primary"] == "1" else None,
                       "MeanYFP": float(r["MeanYFP"]) if r["meanyfp_valid"] == "1" else None,
                       "combined_depth": float(r["combined_depth"]) if r["combined_depth"] else None})
    sm = []
    with open(os.path.join(C.ROOT, "eval", "sample2019.tsv")) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            s = r["utr_seq"].upper()
            why = ("not_native_human_set" if r.get("set", "human") != "human" else
                   "leader_shorter_than_{}".format(C.MIN_LEADER_LEN) if len(s) < C.MIN_LEADER_LEN else
                   "reads_below_100" if float(r.get("total_reads") or 0) < 100 else
                   "non_ACGT" if set(s) - set("ACGT") else "")
            if why:
                excl.append({"dataset": "sample2019_human", "reporter_id": r["id"], "reason": why})
                continue
            sm.append({"id": r["id"], "chrom": "NA", "seq": s, "rl": float(r["rl"])})
    for rr in ak + sm:
        rr["uAUG"] = sum(1 for i in range(len(rr["seq"]) - 2) if rr["seq"][i:i + 3] == "ATG")
    return ak, sm, excl


def sample_folds(rows, k=5, seed=0):
    n = len(rows)
    order = np.random.default_rng(seed).permutation(n)
    fold = np.empty(n, dtype=int)
    fold[order] = np.arange(n) % k
    seqs = [r["seq"] for r in rows]
    uniq = sorted(set(seqs))
    cid = {s: i for i, s in enumerate(uniq)}
    cl = np.array([cid[s] for s in seqs])
    size = np.bincount(cl, minlength=len(uniq))
    cfold_of = np.empty(len(uniq), dtype=int)
    load = np.zeros(k)
    for c in np.random.default_rng(seed + 1).permutation(len(uniq)):
        f = int(np.argmin(load))
        cfold_of[c] = f
        load[f] += size[c]
    cfold = cfold_of[cl]
    path = C.results_path("downstream", "sample2019_folds.tsv")
    table = [{"id": r["id"], "fold": int(f), "sequence_cluster": int(c), "cluster_fold": int(g)}
             for r, f, c, g in zip(rows, fold, cl, cfold)]
    if os.path.exists(path):
        old = C.read_tsv(path)
        same = (len(old) == len(table) and all(
            o["id"] == t["id"] and int(o["fold"]) == t["fold"] and int(o["cluster_fold"]) == t["cluster_fold"]
            for o, t in zip(old, table)))
        assert same, "sample2019_folds.tsv exists and differs -- folds must not change"
    else:
        C.write_tsv(table, path, extra_first=("id", "fold", "sequence_cluster", "cluster_fold"))
    return fold, cl, cfold


def restandardise(ft_rep, ft_tcp):
    for k, (mu_t, sd_t) in ft_tcp.cont_stats.items():
        mu_r, sd_r = ft_rep.cont_stats[k]
        raw = ft_rep.cont[k] * sd_r + mu_r
        ft_rep.cont[k] = (raw - mu_t) / sd_t
        ft_rep.cont_stats[k] = (mu_t, sd_t)
    return ft_rep


def represent_rows(fit, model, b, ft_rep, idx):
    idx = np.asarray(idx)
    out = np.zeros((len(idx), 6))
    prob = L.Problem(ft_rep, idx, b, L.MODELS[model])
    beta = prob.init_beta(fit.as_dict())
    logp = L.nucleotide_logp(prob, beta)
    nopos = beta.copy()
    for k in ("pos_aug", "pos_five"):
        nopos[prob.slices[k]] = 0.0
    p, p0 = np.exp(logp), np.exp(L.nucleotide_logp(prob, nopos))
    pos = prob.pos
    dA, d5 = ft_rep.dAUG[pos], ft_rep.d5[pos]
    startwin = ft_rep.cat["start"][pos, 0] >= 0
    main_ = np.abs(dA) <= 2
    seg = prob.seg
    order = np.argsort(idx)
    L_tx = np.bincount(seg)
    for t in range(len(idx)):
        m = seg == t
        out[order[t]] = [p[m][main_[m]].sum(), p0[m][main_[m]].sum(),
                         p[m][startwin[m] & ~main_[m]].sum(), p[m][d5[m] < 30].sum(),
                         p[m][(dA[m] < 30) & (dA[m] >= 0)].sum(),
                         float(-(p[m] * np.log(p[m] + 1e-300)).sum() / np.log(L_tx[t]))]
    return out


def _fit_pred(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    m = RidgeCV(alphas=ALPHAS).fit((Xtr - mu) / sd, ytr)
    return m.predict((Xte - mu) / sd)


def _inner_folds(groups, seed):
    ug = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(len(ug))
    f = np.empty(len(ug), dtype=int)
    f[perm] = np.arange(len(ug)) % INNER_K
    return f[np.searchsorted(ug, groups)]


def oof_final(Xs, Xf, y, outer, inner_groups, seed=0, only_fold=None):
    n = len(y)
    P = {m: np.full(n, np.nan) for m in FINAL_MODELS}
    Xb = np.hstack([Xs, Xf])
    for k in np.unique(outer):
        if only_fold is not None and k != only_fold:
            continue
        itr, ite = np.nonzero(outer != k)[0], np.nonzero(outer == k)[0]
        inner = _inner_folds(inner_groups[itr], seed * 1000 + int(k))
        oof_s, oof_f = np.zeros(len(itr)), np.zeros(len(itr))
        for j in range(INNER_K):
            a, bb = itr[inner != j], itr[inner == j]
            if len(bb) == 0:
                continue
            oof_s[inner == j] = _fit_pred(Xs[a], y[a], Xs[bb])
            oof_f[inner == j] = _fit_pred(Xf[a], y[a], Xf[bb])
        s_te = _fit_pred(Xs[itr], y[itr], Xs[ite])
        f_te = _fit_pred(Xf[itr], y[itr], Xf[ite])
        P["seq"][ite] = s_te
        P["40S"][ite] = f_te
        P["seq+40S"][ite] = _fit_pred(Xb[itr], y[itr], Xb[ite])
        stack = RidgeCV(alphas=ALPHAS).fit(np.column_stack([oof_s, oof_f]), y[itr])
        P["late_fusion"][ite] = stack.predict(np.column_stack([s_te, f_te]))
        P["residual"][ite] = s_te + _fit_pred(Xf[itr], y[itr] - oof_s, Xf[ite])
    return P


def canaries(Xs, Xf, y, outer, inner_groups, P, seed=0):
    k = np.unique(outer)[0]
    te = np.nonzero(outer == k)[0]
    y2 = y.copy()
    y2[te] = np.random.default_rng(99).permutation(y[te]) + 1e3
    P2 = oof_final(Xs, Xf, y2, outer, inner_groups, seed, only_fold=k)
    lab = all(np.array_equal(P[m][te], P2[m][te]) for m in FINAL_MODELS)
    i0, others = te[0], te[1:]
    Xs3, Xf3 = Xs.copy(), Xf.copy()
    Xs3[others] = Xs3[others] * 3.0 + 7.0
    Xf3[others] = Xf3[others] * 3.0 + 7.0
    P3 = oof_final(Xs3, Xf3, y, outer, inner_groups, seed, only_fold=k)
    loc = all(np.isclose(P[m][i0], P3[m][i0], rtol=0, atol=1e-10) for m in FINAL_MODELS)
    return lab, loc


def _metrics(y, p):
    from scipy.stats import pearsonr, spearmanr
    ss = ((y - y.mean()) ** 2).sum()
    return {"r2": float(1 - ((y - p) ** 2).sum() / ss),
            "pearson": float(pearsonr(y, p)[0]), "spearman": float(spearmanr(y, p)[0]),
            "rmse": float(np.sqrt(((y - p) ** 2).mean()))}


def _r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def summarise(y, P, units, n_boot, seed, base):
    ug, inv = np.unique(units, return_inverse=True)
    rows_of = [np.nonzero(inv == g)[0] for g in range(len(ug))]
    rng = np.random.default_rng(seed)
    draws = [np.concatenate([rows_of[g] for g in rng.integers(0, len(ug), len(ug))])
             for _ in range(n_boot)]
    out = []
    seq_boot = np.array([_r2(y[d], P["seq"][d]) for d in draws])
    for m in FINAL_MODELS:
        met = _metrics(y, P[m])
        row = dict(base, model=m, n=len(y), n_units=len(ug), n_boot=n_boot, **met)
        if m != "seq":
            bd = np.array([_r2(y[d], P[m][d]) for d in draws]) - seq_boot
            row.update(delta_r2=met["r2"] - _r2(y, P["seq"]),
                       delta_ci_lo=float(np.percentile(bd, 2.5)),
                       delta_ci_hi=float(np.percentile(bd, 97.5)),
                       p_delta_le_0=float((bd <= 0).mean()),
                       primary_test=(m == "seq+40S"))
        out.append(row)
    return out


def run_final(n_boot=N_BOOT_FINAL):
    t0 = time.time()
    choice = {r["species"]: r for r in C.read_tsv(C.results_path("downstream", "model_choice.tsv"))}
    checks, summary, preds_out = [], [], []

    def check(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        print("[check] {:<5} {} {}".format("PASS" if ok else "FAIL", name, detail))

    check("no reporter label used in 40S model selection",
          all(r["uses_reporter_labels"] in ("False", False) and
              r["source_file"].endswith("nested_effects.tsv") for r in choice.values()),
          "model_choice.tsv written from effects/nested_effects.tsv before reporters are loaded")
    ak, sm, excl = load_reporters_final()
    C.write_tsv(excl, C.results_path("downstream", "reporter_exclusions.tsv"),
                extra_first=("dataset", "reporter_id", "reason"))

    ych = choice["yeast"]
    ymodel, yb = ych["selected_model"], int(ych["bin_width"])
    recs, meta, reps, reads, ft, ytcp = EFF.prepare("archer_yeast")
    panel_ak = reporter_panel(ak, "yeast")
    ft_ak = restandardise(FeatureTable(panel_ak), ft)
    folds_y = splits.load("yeast")
    X40_ak = np.full((len(ak), 6), np.nan)
    fold_ak = np.array([r["fold"] for r in ak])
    excl_ok = True
    for k in range(folds_y["n_folds"]):
        tr_mask, _ = splits.train_test(ft.chrom, "yeast", k)
        tr = np.nonzero(tr_mask & (reads[reps[0]] >= EFF.MIN_READS))[0]
        excl_ok &= not (set(ft.chrom[tr]) & set(folds_y["folds"][k]))
        prob = L.Problem(ft, tr, yb, L.MODELS[ymodel])
        fit = L.fit(prob, prob.bin_counts(ytcp[reps[0]]), prior_sd=EFF.PRIOR_SD)
        sel = np.nonzero(fold_ak == k)[0]
        if len(sel):
            X40_ak[sel] = represent_rows(fit, ymodel, yb, ft_ak, sel)
    check("yeast 40S fit excludes the reporter's chromosome fold", excl_ok,
          "training chromosomes never intersect fold k; eligibility reads[rep1] >= {}".format(EFF.MIN_READS))
    Xs_ak = seq_features(ak, "yeast")[1]
    orf = np.array([r["orf"] for r in ak])
    uaug_ak = np.array([r["uAUG"] for r in ak])
    for target in ("TEff", "MeanYFP"):
        yv = np.array([np.nan if r[target] is None else r[target] for r in ak])
        ok = np.isfinite(yv)
        if target == "TEff":
            dep = np.array([r["combined_depth"] or 0 for r in ak])[ok]
            check("Akirtava TEff primary rows satisfy combined PoLib depth >= 5000",
                  bool((dep >= 5000).all()), "n = {}".format(int(ok.sum())))
        idx = np.nonzero(ok)[0]
        P = oof_final(Xs_ak[idx], X40_ak[idx], yv[idx], fold_ak[idx], orf[idx])
        lab, loc = canaries(Xs_ak[idx], X40_ak[idx], yv[idx], fold_ak[idx], orf[idx], P)
        check("no outer-test label used (yeast {})".format(target), lab, "label-scramble canary")
        check("scaling uses training rows only (yeast {})".format(target), loc, "feature-locality canary")
        _collect("akirtava_yeast", target, "rep1", "chrom_folds", "gene", yv[idx], P, orf[idx],
                 uaug_ak[idx], [ak[i] for i in idx], fold_ak[idx], n_boot, ymodel, yb, summary,
                 preds_out, min_stratum=1)
    check("bootstrap unit for Akirtava is the gene/ORF", True, "units = ORF systematic names")

    hch = choice["human"]
    hmodel, hb = hch["selected_model"], int(hch["bin_width"])
    recs, meta, reps, reads, ft, ytcp = EFF.prepare("gse174329_eif3b")
    panel_sm = reporter_panel(sm, "human")
    ft_sm = restandardise(FeatureTable(panel_sm), ft)
    X40 = {}
    ntr = {}
    for rep in reps:
        tr = np.nonzero(reads[rep] >= EFF.MIN_READS)[0]
        prob = L.Problem(ft, tr, hb, L.MODELS[hmodel])
        fit = L.fit(prob, prob.bin_counts(ytcp[rep]), prior_sd=EFF.PRIOR_SD)
        X40[rep] = represent_rows(fit, hmodel, hb, ft_sm, np.arange(len(sm)))
        ntr[rep] = len(tr)
    check("human 40S fitted separately on rep1 and rep2",
          len(X40) == 2 and not np.array_equal(X40[reps[0]], X40[reps[1]]),
          "training transcripts {}".format(ntr))
    X40["mean"] = (X40[reps[0]] + X40[reps[1]]) / 2
    Xs_sm = seq_features(sm, "human")[1]
    yv = np.array([r["rl"] for r in sm])
    fold, cl, cfold = sample_folds(sm)
    fold2, _, cfold2 = sample_folds(sm)
    dup_same = all(len(set(cfold[cl == c])) == 1 for c in np.unique(cl))
    check("identical rows receive deterministic fold assignments",
          np.array_equal(fold, fold2) and np.array_equal(cfold, cfold2) and dup_same,
          "re-derived folds identical; identical sequences share a cluster fold")
    rid = np.arange(len(sm))
    uaug_sm = np.array([r["uAUG"] for r in sm])
    for src in (reps[0], reps[1], "mean"):
        for split, folds_, inner_g, unit, unit_name in (
                ("random_balanced", fold, rid, rid, "leader"),
                ("random_balanced", fold, rid, cl, "sequence_cluster"),
                ("sequence_cluster", cfold, cl, cl, "sequence_cluster")):
            if unit_name == "sequence_cluster" and split == "random_balanced" and src == "mean":
                continue
            P = oof_final(Xs_sm, X40[src], yv, folds_, inner_g)
            if unit_name == "leader":
                lab, loc = canaries(Xs_sm, X40[src], yv, folds_, inner_g, P)
                check("no outer-test label used (human, 40S {})".format(src), lab, "label-scramble canary")
                check("scaling uses training rows only (human, 40S {})".format(src), loc,
                      "feature-locality canary")
            _collect("sample2019_human", "rl", src, split, unit_name, yv, P, unit, uaug_sm, sm,
                     folds_, n_boot, hmodel, hb, summary, preds_out, min_stratum=MIN_STRATUM)
    check("residual training target uses inner out-of-fold sequence predictions", True,
          "oof_final: residual fitted to y_train - inner OOF seq prediction")
    check("late-fusion stacker uses inner out-of-fold predictions", True,
          "oof_final: stacker fitted on inner OOF [seq, 40S] predictions")

    import gzip
    pred_path = C.results_path("downstream", "downstream_predictions.tsv.gz")
    cols = ["dataset", "reporter_id", "gene_or_orf", "chromosome", "outer_fold", "split_mode",
            "target", "observed", "sequence_prediction", "fortyS_prediction",
            "seq_plus_40S_prediction", "late_fusion_prediction", "residual_prediction",
            "fortyS_source_rep", "uAUG_count", "uAUG_stratum"]
    with gzip.open(pred_path, "wt", newline="") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in preds_out:
            fh.write("\t".join("{:.10g}".format(r[c]) if isinstance(r[c], float) else str(r[c])
                               for c in cols) + "\n")
    back = {}
    with gzip.open(pred_path, "rt") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            back.setdefault((r["dataset"], r["target"], r["fortyS_source_rep"], r["split_mode"]),
                            []).append(r)
    col_of = {"seq": "sequence_prediction", "40S": "fortyS_prediction",
              "seq+40S": "seq_plus_40S_prediction", "late_fusion": "late_fusion_prediction",
              "residual": "residual_prediction"}
    worst = 0.0
    for s in summary:
        rr = back[(s["dataset"], s["target"], s["fortyS_source"], s["split_mode"])]
        if s["stratum"] != "all":
            rr = [r for r in rr if r["uAUG_stratum"] == s["stratum"]]
        y_ = np.array([float(r["observed"]) for r in rr])
        p_ = np.array([float(r[col_of[s["model"]]]) for r in rr])
        worst = max(worst, abs(_r2(y_, p_) - s["r2"]))
    check("all prediction rows reproduce the summary R2", worst < 1e-6,
          "max |R2(recomputed) - R2(summary)| = {:.2e} (predictions stored to 10 significant digits)".format(worst))
    C.write_tsv(summary, C.results_path("downstream", "downstream_utility_final.tsv"),
                extra_first=("dataset", "target", "fortyS_source", "split_mode", "bootstrap_unit",
                             "stratum", "model", "n", "r2"))
    C.write_tsv(checks, C.results_path("downstream", "validation_checks.tsv"),
                extra_first=("check", "passed", "detail"))
    C.update_manifest("downstream_final", {
        "model_choice": {k: {"model": v["selected_model"], "bin": v["bin_width"]} for k, v in choice.items()},
        "n_boot": n_boot, "inner_k": INNER_K, "alphas": [float(a) for a in ALPHAS],
        "min_stratum_human": MIN_STRATUM, "teff_min_combined_depth": 5000,
        "human_folds": "downstream/sample2019_folds.tsv (seed 0)",
        "yeast_folds": "frozen yeast chromosome folds", "seconds": round(time.time() - t0)})
    failed = [c["check"] for c in checks if not c["passed"]]
    print("[final] done in {:.0f}s; {} checks, {} failed".format(time.time() - t0, len(checks), len(failed)))
    if failed:
        sys.exit("validation checks failed: {}".format(failed))


def _collect(dataset, target, src, split, unit_name, y, P, units, uaug, rows, folds_, n_boot,
             model, b, summary, preds_out, min_stratum):
    strat = np.where(uaug > 0, ">=1_uAUG", "no_uAUG")
    base = {"dataset": dataset, "target": target, "fortyS_source": src, "split_mode": split,
            "bootstrap_unit": unit_name, "fortyS_model": model, "bin_width": b}
    for name, mask in (("all", np.ones(len(y), bool)), ("no_uAUG", strat == "no_uAUG"),
                       (">=1_uAUG", strat == ">=1_uAUG")):
        if name != "all" and mask.sum() < min_stratum:
            continue
        sub = {m: P[m][mask] for m in FINAL_MODELS}
        summary += summarise(y[mask], sub, np.asarray(units)[mask], n_boot,
                             seed=0, base=dict(base, stratum=name))
    if unit_name == "sequence_cluster" and split == "random_balanced":
        return
    for i, r in enumerate(rows):
        preds_out.append({"dataset": dataset, "reporter_id": r["id"], "gene_or_orf": r.get("orf", ""),
                          "chromosome": r.get("chrom", ""), "outer_fold": int(folds_[i]),
                          "split_mode": split, "target": target, "observed": float(y[i]),
                          "sequence_prediction": float(P["seq"][i]), "fortyS_prediction": float(P["40S"][i]),
                          "seq_plus_40S_prediction": float(P["seq+40S"][i]),
                          "late_fusion_prediction": float(P["late_fusion"][i]),
                          "residual_prediction": float(P["residual"][i]), "fortyS_source_rep": src,
                          "uAUG_count": int(uaug[i]), "uAUG_stratum": strat[i]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("legacy", "choose", "run"), default="legacy",
                    help="legacy = the original exploratory analysis (downstream_utility.tsv); "
                         "choose = freeze the 40S model per species; run = final protocol")
    ap.add_argument("--targets", default="MeanYFP,TEff")
    ap.add_argument("--skip-human", action="store_true")
    ap.add_argument("--n-boot", type=int, default=N_BOOT_FINAL)
    ap.add_argument("--force", action="store_true", help="choose: overwrite model_choice.tsv")
    args = ap.parse_args()
    C.ensure_layout()
    if args.stage == "choose":
        choose_models_final(force=args.force)
        return
    if args.stage == "run":
        run_final(args.n_boot)
        return
    rows = run_yeast(args.targets.split(","))
    if not args.skip_human:
        rows += run_human()
    C.write_tsv(rows, C.results_path("downstream", "downstream_utility.tsv"))
    print("[down] done")


if __name__ == "__main__":
    main()
