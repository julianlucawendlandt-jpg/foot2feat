import warnings

import numpy as np

NUCS = "ACGT"
NUC_INDEX = {c: i for i, c in enumerate(NUCS)}

NEAR_COGNATES = ("ATG", "TTG", "CTG", "GTG", "ACG", "AAG", "AGG", "ATA", "ATC", "ATT")

STOP_CODONS = ("TAA", "TAG", "TGA")

N_UORF_GEOM = 2


def one_hot(seq):
    seq = seq.upper().replace("U", "T")
    x = np.zeros((len(seq), 4), dtype=np.float32)
    for i, c in enumerate(seq):
        j = NUC_INDEX.get(c)
        if j is not None:
            x[i, j] = 1.0
    return x


def start_positions(seq, near_cognate=False):
    seq = seq.upper().replace("U", "T")
    codons = NEAR_COGNATES if near_cognate else ("ATG",)
    mask = np.zeros(len(seq), dtype=bool)
    for i in range(len(seq) - 2):
        if seq[i:i + 3] in codons:
            mask[i] = True
    return mask


def main_start_track(n, main_start_index):
    mask = np.zeros(n, dtype=bool)
    if main_start_index is not None and 0 <= main_start_index < n:
        mask[main_start_index] = True
    return mask


def uorf_returns(seq, start_mask, main_start_index=None):
    seq = seq.upper().replace("U", "T")
    n = len(seq)
    return_idx = np.full(n, -1, dtype=np.int32)
    uorf_len = np.zeros(n, dtype=np.float32)
    reinit_gap = np.zeros(n, dtype=np.float32)

    limit = (n - 1) if main_start_index is None else int(main_start_index)

    for j in np.nonzero(np.asarray(start_mask))[0]:
        j = int(j)
        if main_start_index is not None and j == int(main_start_index):
            continue
        stop = None
        k = j + 3
        while k + 3 <= n:
            if seq[k:k + 3] in STOP_CODONS:
                stop = k
                break
            k += 3
        if stop is None:
            continue
        ret = stop + 3
        if ret > limit or ret >= n:
            continue
        return_idx[j] = ret
        uorf_len[j] = (stop - j) // 3
        reinit_gap[j] = limit - ret

    return return_idx, uorf_len, reinit_gap


def _fold_tracks(seq, window):
    n = len(seq)
    try:
        import RNA
    except ImportError:
        warnings.warn(
            "ViennaRNA (import RNA) not installed -- structure tracks are all "
            "zero.  pip install ViennaRNA to switch them on.",
            RuntimeWarning, stacklevel=3,
        )
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)

    rna = seq.upper().replace("T", "U")

    structure, _ = RNA.fold(rna)
    paired = np.array([0.0 if c == "." else 1.0 for c in structure], dtype=np.float32)

    local_dG = np.zeros(n, dtype=np.float32)
    if n >= 1:
        half = window // 2
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            local_dG[i] = RNA.fold(rna[lo:hi])[1]
    return paired, local_dG


def featurize(seq, main_start_index=None, near_cognate=False,
              structure=True, fold_window=40, context_seq=None,
              context_offset=0, paired=None, local_dG=None):
    n = len(seq)
    ctx = seq if context_seq is None else context_seq
    if context_seq is not None:
        assert ctx[context_offset:context_offset + n] == seq, \
            "context_seq[context_offset:context_offset+N] must equal seq"

    m = len(ctx)
    if structure and paired is not None and local_dG is not None:
        paired = np.asarray(paired, dtype=np.float32)
        local_dG = np.asarray(local_dG, dtype=np.float32)
        assert len(paired) == len(local_dG) == m, (
            "precomputed structure tracks are {}/{} long but the context is "
            "{}".format(len(paired), len(local_dG), m))
    elif structure:
        paired, local_dG = _fold_tracks(ctx, fold_window)
    else:
        paired = np.zeros(m, dtype=np.float32)
        local_dG = np.zeros(m, dtype=np.float32)

    start_mask = start_positions(seq, near_cognate=near_cognate)
    ret_idx, uorf_len, reinit_gap = uorf_returns(seq, start_mask, main_start_index)

    feats = {
        "one_hot": one_hot(seq),
        "start_mask": start_mask,
        "is_main_start": main_start_track(n, main_start_index),
        "uorf_return": ret_idx,
        "uorf_geom": (np.log1p(np.stack([uorf_len, reinit_gap], axis=1)) / 5.0
                      ).astype(np.float32),
        "paired": paired,
        "local_dG": local_dG / 10.0,
        "context_offset": int(context_offset),
        "n_states": n,
    }
    ctx_oh = one_hot(ctx)
    feats["net_input"] = np.concatenate(
        [ctx_oh, paired[:, None], feats["local_dG"][:, None]], axis=1
    ).astype(np.float32)
    return feats


NET_INPUT_CHANNELS = 6
