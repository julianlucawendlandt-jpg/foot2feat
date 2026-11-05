import json
import os

import numpy as np

from analysis.recoverability import config as C

FOLD_FILES = {"human": "human_chromosome_folds.json",
              "yeast": "yeast_chromosome_folds.json"}
EXCLUDE = {"chrM", "Mito", "chrY"}


def assign(depth_by_chrom, n_folds):
    folds = [[] for _ in range(n_folds)]
    load = np.zeros(n_folds)
    for chrom, d in sorted(depth_by_chrom.items(), key=lambda kv: (-kv[1], kv[0])):
        f = int(np.argmin(load))
        folds[f].append(chrom)
        load[f] += d
    return folds, load


def freeze(species, depth_by_chrom, source, overwrite=False):
    path = C.results_path(FOLD_FILES[species])
    if os.path.exists(path) and not overwrite:
        return load(species)
    depth = {c: float(d) for c, d in depth_by_chrom.items() if c not in EXCLUDE}
    folds, ld = assign(depth, C.N_FOLDS[species])
    out = {"species": species, "n_folds": len(folds),
           "balanced_on": source, "excluded": sorted(EXCLUDE),
           "folds": [sorted(f, key=_chrom_key) for f in folds],
           "fold_depth": [float(x) for x in ld]}
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    return out


def load(species):
    with open(C.results_path(FOLD_FILES[species])) as fh:
        return json.load(fh)


def fold_of(species):
    return {c: i for i, f in enumerate(load(species)["folds"]) for c in f}


def train_test(chroms, species, k):
    fo = fold_of(species)
    f = np.array([fo.get(c, -1) for c in chroms])
    test = f == k
    train = (f >= 0) & ~test
    assert not set(np.asarray(chroms)[train]) & set(np.asarray(chroms)[test]), \
        "split leakage: a chromosome is in both partitions"
    return train, test


def _chrom_key(c):
    s = c[3:] if c.startswith("chr") else c
    roman = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
             "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13,
             "XIV": 14, "XV": 15, "XVI": 16}
    if s in roman:
        return (0, roman[s])
    return (0, int(s)) if s.isdigit() else (1, s)
