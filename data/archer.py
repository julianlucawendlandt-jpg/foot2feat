import os

import numpy as np

DEFAULT_CACHE = os.path.join("cache", "archer_examples.npz")

REQUIRED_KEYS = ("transcript_id", "utr_seq", "main_start_index", "ssu_counts")


def save_examples(examples, path=DEFAULT_CACHE):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, examples=np.array(examples, dtype=object))
    return path


def load_examples(path=DEFAULT_CACHE, min_total_counts=1, min_length=10):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "No cached examples at {}.  Run the pipeline first:\n"
            "  python pipeline/02_align_count.py --stream\n"
            "  python pipeline/03_build_examples.py\n"
            "(see README.md, Installation, for the required binaries)".format(path)
        )

    with np.load(path, allow_pickle=True) as z:
        raw = list(z["examples"])

    kept, dropped = [], 0
    for ex in raw:
        try:
            assert all(k in ex for k in REQUIRED_KEYS)
            n = len(ex["utr_seq"])
            if n < min_length:
                dropped += 1
                continue
            if len(ex["ssu_counts"]) != n:
                dropped += 1
                continue
            if "input_counts" in ex and len(ex["input_counts"]) != n:
                dropped += 1
                continue
            if int(np.sum(ex["ssu_counts"])) < min_total_counts:
                dropped += 1
                continue
            i = ex["main_start_index"]
            if not (0 <= i < n) or ex["utr_seq"][i:i + 3].upper().replace("U", "T") != "ATG":
                dropped += 1
                continue
            kept.append(ex)
        except Exception:
            dropped += 1

    print("[archer] loaded {} examples from {} ({} dropped)".format(len(kept), path, dropped))
    return kept
