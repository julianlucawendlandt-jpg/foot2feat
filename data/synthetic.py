import numpy as np
import torch

from data.features import featurize
from model.chain import run_chain

BASE_STEP_LOGIT = 6.5
GC_STEP_PENALTY = 3.0
GC_WINDOW = 5
BASE_COMMIT_LOGIT = 3.0
KOZAK_MINUS3_A = 3.0
KOZAK_MINUS1_A = 1.5
DWELL_OVER_COMMIT = 2.0
BASE_DWELL_LOGIT = -2.0
GC_DWELL_BONUS = 2.0
TRUE_RHO_LOAD = 2.0
CDS_LENGTH = 100

HAIRPIN_STEM = "GCGCGGCGCC"
HAIRPIN_LOOP = "TTTT"


def _rc(s):
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def ground_truth_rates(seq):
    seq = seq.upper().replace("U", "T")
    n = len(seq)
    gc = np.array([1.0 if c in "GC" else 0.0 for c in seq], dtype=np.float64)

    gc_frac = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - GC_WINDOW), min(n, i + GC_WINDOW + 1)
        gc_frac[i] = gc[lo:hi].mean()

    w = BASE_STEP_LOGIT - GC_STEP_PENALTY * gc_frac

    p = np.full(n, BASE_COMMIT_LOGIT)
    for i in range(n):
        if i >= 3 and seq[i - 3] == "A":
            p[i] += KOZAK_MINUS3_A
        if i >= 1 and seq[i - 1] == "A":
            p[i] += KOZAK_MINUS1_A
    d = BASE_DWELL_LOGIT + GC_DWELL_BONUS * gc_frac
    at_start = np.zeros(n, dtype=bool)
    for i in range(n - 2):
        if seq[i:i + 3] == "ATG":
            at_start[i] = True
    d = np.where(at_start, p + DWELL_OVER_COMMIT, d)
    return w, p, d


def _random_seq(n, rng, gc=0.38):
    probs = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
    return "".join(rng.choice(list("ACGT"), size=n, p=probs))


def make_utr(rng, length=120, n_uorfs=1, hairpin=True):
    seq = list(_random_seq(length, rng))

    s = "".join(seq)
    for i in range(len(s) - 2):
        if s[i:i + 3] == "ATG":
            seq[i + 1] = "A"
        s = "".join(seq)

    if hairpin:
        stem = HAIRPIN_STEM + HAIRPIN_LOOP + _rc(HAIRPIN_STEM)
        pos = rng.integers(10, max(11, length - len(stem) - 30))
        seq[pos:pos + len(stem)] = list(stem)

    for _ in range(n_uorfs):
        strong = bool(rng.integers(0, 2))
        context = "AAAA" if strong else "TCTC"
        pos = int(rng.integers(length // 2, length - 12))
        seq[pos:pos + 7] = list(context + "ATG")

    seq = "".join(seq[:length]) + "ATG"
    return seq, len(seq) - 3


def make_example(rng, length=120, total_counts=3000, near_cognate=False,
                 structure=False, transcript_id="synthetic", flank=10):
    seq, main_start_index = make_utr(rng, length=length)
    left = _random_seq(flank, rng)
    right = _random_seq(flank, rng)
    context = left + seq + right
    feats = featurize(seq, main_start_index, near_cognate=near_cognate,
                      structure=structure, context_seq=context,
                      context_offset=flank)

    w, p, d = (a[flank:flank + len(seq)] for a in ground_truth_rates(context))
    with torch.no_grad():
        profile, fates = run_chain(
            torch.tensor(w, dtype=torch.float32),
            torch.tensor(p, dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
            torch.tensor(feats["start_mask"]),
            torch.tensor(feats["is_main_start"]),
        )
    profile = profile.numpy()

    rate = profile / profile.sum() * total_counts
    ssu_counts = rng.poisson(rate).astype(np.int64)
    input_counts = rng.poisson(np.full(len(seq), total_counts / len(seq))).astype(np.int64)

    abundance = float(rng.lognormal(mean=np.log(0.5), sigma=0.6))
    input_cds_counts = rng.poisson(np.full(CDS_LENGTH, abundance)).astype(np.int64)
    p_commit_main = float(fates[0])
    rs_cds_counts = rng.poisson(
        np.full(CDS_LENGTH, TRUE_RHO_LOAD * p_commit_main * abundance)).astype(np.int64)

    return {
        "transcript_id": transcript_id,
        "utr_seq": seq,
        "main_start_index": main_start_index,
        "utr_context_seq": context,
        "context_offset": flank,
        "ssu_counts": ssu_counts,
        "input_counts": input_counts,
        "rs_cds_counts": rs_cds_counts,
        "input_cds_counts": input_cds_counts,
        "true_w": w,
        "true_p": p,
        "true_d": d,
        "true_fates": fates.numpy(),
    }


def make_dataset(n=64, seed=0, length=120, total_counts=3000, flank=10, **kwargs):
    rng = np.random.default_rng(seed)
    return [
        make_example(rng, length=length, total_counts=total_counts, flank=flank,
                     transcript_id=f"synthetic_{i:04d}", **kwargs)
        for i in range(n)
    ]


def _softmax(e):
    p = np.exp(e - np.max(e))
    return p / p.sum()


def draw_counts(eta, depth, rng, concentration=None):
    p = _softmax(np.asarray(eta, dtype=np.float64))
    depth = int(depth)
    if depth <= 0:
        return np.zeros(len(p), dtype=np.int64)
    if concentration is not None and np.isfinite(concentration):
        p = rng.dirichlet(np.maximum(p * concentration, 1e-8))
    return rng.multinomial(depth, p).astype(np.int64)


def estimate_concentration(pairs, min_reads=20):
    stats = []
    for y1, y2 in pairs:
        y1, y2 = np.asarray(y1, float), np.asarray(y2, float)
        n1, n2 = y1.sum(), y2.sum()
        if n1 < min_reads or n2 < min_reads:
            continue
        p = (y1 + y2) / (n1 + n2)
        nz = p > 0
        x = float((((y1 / n1 - y2 / n2) ** 2)[nz] / p[nz]).sum())
        stats.append((x, int(nz.sum()), n1, n2))
    if not stats:
        return float("inf"), float("nan")
    obs = sum(s[0] for s in stats)
    mult = sum((k - 1) * (1 / n1 + 1 / n2) for _x, k, n1, n2 in stats)
    ratio = obs / mult

    def expected(alpha):
        return sum((k - 1) * ((n1 + alpha) / (1 + alpha) / n1 + (n2 + alpha) / (1 + alpha) / n2)
                   for _x, k, n1, n2 in stats)

    if obs <= mult:
        return float("inf"), ratio
    lo, hi = 1e-3, 1e9
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if expected(mid) > obs:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi)), ratio


def calibrate_concentration(pairs, rng, min_reads=20, pseudo=0.5, n_iter=25):
    pairs = [(np.asarray(a, float), np.asarray(b, float)) for a, b in pairs
             if np.sum(a) >= min_reads and np.sum(b) >= min_reads]
    _, obs = estimate_concentration(pairs, min_reads)
    if not pairs or not np.isfinite(obs):
        return float("inf"), obs, float("nan")
    etas = [np.log(a + b + pseudo) for a, b in pairs]

    def sim_ratio(alpha, seed):
        r = np.random.default_rng(seed)
        sim = [(draw_counts(e, a.sum(), r, alpha), draw_counts(e, b.sum(), r, alpha))
               for e, (a, b) in zip(etas, pairs)]
        return estimate_concentration(sim, min_reads)[1]

    base = sim_ratio(None, 0)
    if obs <= base:
        return float("inf"), obs, base
    lo, hi = 1e-2, 1e6
    for i in range(n_iter):
        mid = np.sqrt(lo * hi)
        if sim_ratio(mid, i) > obs:
            lo = mid
        else:
            hi = mid
    a = float(np.sqrt(lo * hi))
    return a, obs, sim_ratio(a, 999)
