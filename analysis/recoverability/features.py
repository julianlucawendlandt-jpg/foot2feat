import numpy as np

from data.features import NEAR_COGNATES, STOP_CODONS

BASE = {"A": 0, "C": 1, "G": 2, "T": 3}
SW_HALF = 10
START_HALF = 2
AHEAD = (5, 20)
DG_AHEAD = 12
GC_HALF = 20
START_CODONS = ("ATG",) + tuple(c for c in NEAR_COGNATES if c != "ATG")

CAT_LEVELS = {"nt1": 4, "k3": 64, "k5": 1024, "sw": (2 * SW_HALF + 1) * 4,
              "start": len(START_CODONS), "m3": 4, "p4": 4, "kozak": 3,
              "nuaug": 4, "in_uorf": 1}
CONT_DIMS = {"paired": 2, "dG": 2, "uorf_geom": 2}


def encode(seq):
    return np.array([BASE.get(c, -1) for c in seq.upper()], dtype=np.int64)


def kmer_codes(x, k):
    n, h = len(x), k // 2
    out = np.full(n, -1, dtype=np.int64)
    for j in range(h, n - h):
        w = x[j - h:j + h + 1]
        if (w < 0).any():
            continue
        c = 0
        for b in w:
            c = c * 4 + int(b)
        out[j] = c
    return out


def kozak_class(m3, p4):
    s = int(m3 in (0, 2)) + int(p4 == 2)
    return {2: 0, 1: 1, 0: 2}[s]


def uorfs(seq, msi):
    out = []
    for j in range(0, msi):
        if seq[j:j + 3] != "ATG":
            continue
        stop, k = None, j + 3
        while k + 3 <= len(seq):
            if seq[k:k + 3] in STOP_CODONS:
                stop = k
                break
            k += 3
        out.append((j, stop))
    return out


def transcript_features(seq, msi, paired, local_dG, cds=False):
    seq = seq.upper()
    n_eval = len(seq) if cds else msi + 3
    x = encode(seq)
    j = np.arange(n_eval)
    f = {"j": j, "d5": j.copy(), "dAUG": msi - j}

    f["nt1"] = x[:n_eval, None].copy()
    f["k3"] = kmer_codes(x, 3)[:n_eval, None]
    f["k5"] = kmer_codes(x, 5)[:n_eval, None]

    sw = np.full((n_eval, 2 * SW_HALF + 1), -1, dtype=np.int64)
    for oi, o in enumerate(range(-SW_HALF, SW_HALF + 1)):
        src = j + o
        ok = (src >= 0) & (src < len(x))
        b = np.full(n_eval, -1, dtype=np.int64)
        b[ok] = x[src[ok]]
        sw[:, oi] = np.where(b >= 0, oi * 4 + b, -1)
    f["sw"] = sw

    start = np.full((n_eval, 1), -1, dtype=np.int64)
    m3 = np.full((n_eval, 1), -1, dtype=np.int64)
    p4 = np.full((n_eval, 1), -1, dtype=np.int64)
    koz = np.full((n_eval, 1), -1, dtype=np.int64)
    geom = np.zeros((n_eval, 2), dtype=np.float64)
    center = np.zeros(n_eval, dtype=bool)
    for s in range(0, min(msi + 1, len(seq) - 2)):
        cod = seq[s:s + 3]
        if cod not in START_CODONS:
            continue
        lo, hi = max(0, s - START_HALF), min(n_eval, s + START_HALF + 1)
        if lo >= hi:
            continue
        start[lo:hi, 0] = START_CODONS.index(cod)
        if s < n_eval:
            center[s] = True
        if cod == "ATG":
            a3 = int(x[s - 3]) if s >= 3 else -1
            a4 = int(x[s + 3]) if s + 3 < len(x) else -1
            if a3 >= 0:
                m3[lo:hi, 0] = a3
            if a4 >= 0:
                p4[lo:hi, 0] = a4
            if a3 >= 0 and a4 >= 0:
                koz[lo:hi, 0] = kozak_class(a3, a4)
    f.update(start=start, m3=m3, p4=p4, kozak=koz, start_center=center)

    nu = np.zeros(n_eval, dtype=np.int64)
    inside = np.full((n_eval, 1), -1, dtype=np.int64)
    for s, stop in uorfs(seq, msi):
        if s + 3 < n_eval:
            nu[s + 3:] += 1
        end = stop + 3 if stop is not None else n_eval
        inside[s + 3:min(end, n_eval), 0] = 0
        lo, hi = max(0, s - START_HALF), min(n_eval, s + START_HALF + 1)
        length = ((stop - s) // 3) if stop is not None else (len(seq) - s) // 3
        gap = (msi - (stop + 3)) if stop is not None and stop + 3 <= msi else 0
        geom[lo:hi, 0] = np.log1p(length)
        geom[lo:hi, 1] = np.log1p(max(gap, 0))
    f["nuaug"] = np.minimum(nu, 3)[:, None]
    f["in_uorf"] = inside
    f["uorf_geom"] = geom

    paired = np.asarray(paired, dtype=np.float64)
    dg = np.asarray(local_dG, dtype=np.float64)
    ahead = np.zeros(n_eval)
    for i in range(n_eval):
        w = paired[min(i + AHEAD[0], len(paired)):min(i + AHEAD[1] + 1, len(paired))]
        ahead[i] = w.mean() if len(w) else np.nan
    dga = np.array([dg[min(i + DG_AHEAD, len(dg) - 1)] for i in range(n_eval)])
    f["paired"] = np.stack([paired[:n_eval], ahead], axis=1)
    f["dG"] = np.stack([dg[:n_eval], dga], axis=1)

    gc = np.isin(x, (1, 2)).astype(np.float64)
    gcl = np.array([gc[max(0, i - GC_HALF):i + GC_HALF + 1].mean() for i in range(n_eval)])
    f["gc"] = gcl
    return f


class FeatureTable(object):
    def __init__(self, panel, cds=False):
        per = [transcript_features(t["seq"], t["msi"], t["paired"], t["local_dG"],
                                   cds=cds) for t in panel]
        self.ids = [t["transcript_id"] for t in panel]
        self.chrom = np.array([t["chrom"] for t in panel])
        lens = np.array([len(p["j"]) for p in per])
        self.tx_start = np.concatenate([[0], np.cumsum(lens)])
        self.seg = np.repeat(np.arange(len(per)), lens)
        cat = lambda k: np.concatenate([p[k] for p in per])
        self.d5 = cat("d5")
        self.dAUG = cat("dAUG")
        self.gc = cat("gc")
        self.start_center = cat("start_center")
        self.cat = {k: cat(k) for k in CAT_LEVELS}
        self.cont = {}
        self.cont_stats = {}
        for k in CONT_DIMS:
            v = cat(k).astype(np.float64)
            if k == "uorf_geom":
                sd = v[v != 0].std() if (v != 0).any() else 1.0
                sd = sd if sd > 0 else 1.0
                self.cont[k] = v / sd
                self.cont_stats[k] = (0.0, sd)
                continue
            mu = np.nanmean(v, axis=0)
            v = np.where(np.isnan(v), mu, v)
            sd = v.std(axis=0)
            sd = np.where(sd > 0, sd, 1.0)
            self.cont[k] = (v - mu) / sd
            self.cont_stats[k] = (mu, sd)
        self.n_eval = lens
        self.cds = cds

    def counts(self, panel, rep):
        return np.concatenate([np.asarray(t["counts"][rep][:n], dtype=np.float64)
                               for t, n in zip(panel, self.n_eval)])

    def positions(self, tx_idx):
        tx_idx = np.asarray(tx_idx)
        return np.concatenate([np.arange(self.tx_start[t], self.tx_start[t + 1])
                               for t in tx_idx]) if len(tx_idx) else np.zeros(0, int)
