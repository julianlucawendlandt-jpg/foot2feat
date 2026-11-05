import re
import array
from collections import defaultdict

import numpy as np

CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=X])")

REF_CONSUMING = frozenset("MDN=X")

FLAG_UNMAPPED = 0x4
FLAG_REVERSE = 0x10
FLAG_SECONDARY = 0x100
FLAG_SUPPLEMENTARY = 0x800


def cigar_reference_span(cigar):
    if cigar == "*":
        return 0
    return sum(int(n) for n, op in CIGAR_RE.findall(cigar) if op in REF_CONSUMING)


def five_prime_position(flag, pos1, cigar, offset=0):
    if flag & FLAG_REVERSE:
        pos0 = (pos1 - 1) + cigar_reference_span(cigar) - 1
        return pos0 - offset, "-"
    return (pos1 - 1) + offset, "+"


def parse_sam_line(line, min_mapq=10):
    if not line or line.startswith("@"):
        return None
    f = line.rstrip("\n").split("\t")
    if len(f) < 11:
        return None
    flag = int(f[1])
    if flag & (FLAG_UNMAPPED | FLAG_SECONDARY | FLAG_SUPPLEMENTARY):
        return None
    if int(f[4]) < min_mapq:
        return None
    rname = f[2]
    if rname == "*":
        return None
    return rname, flag, int(f[3]), f[5]


class CountTracks(object):
    __slots__ = ("data",)

    def __init__(self, data=None):
        self.data = dict(data or {})

    def __contains__(self, key):
        return key in self.data

    def keys(self):
        return self.data.keys()

    def total(self):
        return int(sum(int(v.sum()) for _p, v in self.data.values()))

    def window(self, chrom, strand, start, end):
        n = max(0, end - start)
        out = np.zeros(n, dtype=np.int32)
        hit = self.data.get((chrom, strand))
        if hit is None or n == 0:
            return out
        pos, val = hit
        lo = np.searchsorted(pos, start, "left")
        hi = np.searchsorted(pos, end, "left")
        if hi > lo:
            out[pos[lo:hi] - start] = val[lo:hi]
        return out

    def merge(self, other):
        for key, (p2, v2) in other.data.items():
            cur = self.data.get(key)
            if cur is None:
                self.data[key] = (p2.copy(), v2.copy())
                continue
            p1, v1 = cur
            pos = np.concatenate([p1, p2])
            val = np.concatenate([v1, v2])
            order = np.argsort(pos, kind="stable")
            pos, val = pos[order], val[order]
            uniq, first = np.unique(pos, return_index=True)
            self.data[key] = (uniq, np.add.reduceat(val, first).astype(np.int32))
        return self

    def to_npz(self):
        out = {}
        for (chrom, strand), (pos, val) in self.data.items():
            out["{}|{}|pos".format(chrom, strand)] = pos
            out["{}|{}|val".format(chrom, strand)] = val
        return out

    @classmethod
    def from_npz(cls, z):
        data = {}
        keys = [k for k in z.files if not k.startswith("_")]
        if any(k.endswith("|pos") for k in keys):
            for k in keys:
                if not k.endswith("|pos"):
                    continue
                chrom, strand, _ = k.rsplit("|", 2)
                data[(chrom, strand)] = (
                    z[k], z["{}|{}|val".format(chrom, strand)])
        else:
            for k in keys:
                chrom, strand = k.rsplit("|", 1)
                arr = z[k]
                nz = np.nonzero(arr)[0]
                data[(chrom, strand)] = (nz.astype(np.int64),
                                         arr[nz].astype(np.int32))
        return cls(data)


def accumulate_tracks(lines, chrom_lengths, offset=0, min_mapq=10,
                      coverage=False, keep_reads=False):
    raw = defaultdict(lambda: array.array("l"))
    spans = defaultdict(lambda: array.array("H")) if keep_reads else None
    diff = {} if coverage else None
    stats = defaultdict(int)

    for line in lines:
        stats["total"] += 1
        rec = parse_sam_line(line, min_mapq=min_mapq)
        if rec is None:
            stats["filtered"] += 1
            continue
        rname, flag, pos1, cigar = rec
        if rname not in chrom_lengths:
            stats["unknown_chrom"] += 1
            continue

        pos0, strand = five_prime_position(flag, pos1, cigar, offset=offset)
        if 0 <= pos0 < chrom_lengths[rname]:
            raw[(rname, strand)].append(pos0)
            if spans is not None:
                spans[(rname, strand)].append(min(cigar_reference_span(cigar), 65535))
            stats["assigned"] += 1
        else:
            stats["out_of_bounds"] += 1

        if diff is not None:
            d = diff.get((rname, strand))
            if d is None:
                d = diff[(rname, strand)] = np.zeros(
                    chrom_lengths[rname] + 1, dtype=np.int32)
            lo = pos1 - 1
            hi = min(lo + cigar_reference_span(cigar), len(d) - 1)
            if 0 <= lo < hi:
                d[lo] += 1
                d[hi] -= 1
                stats["covered"] += 1

    counts = CountTracks()
    for key, buf in raw.items():
        pos = np.frombuffer(buf, dtype=np.int64 if buf.itemsize == 8 else np.int32)
        pos = np.sort(pos.astype(np.int64))
        uniq, cnt = np.unique(pos, return_counts=True)
        counts.data[key] = (uniq, cnt.astype(np.int32))

    cov = None
    if diff is not None:
        cov = CountTracks()
        for k, v in diff.items():
            dense = np.cumsum(v[:-1], dtype=np.int32)
            nz = np.nonzero(dense)[0]
            cov.data[k] = (nz.astype(np.int64), dense[nz])
    if spans is not None:
        reads = {}
        for key, buf in raw.items():
            pos = np.frombuffer(buf, dtype=np.int64 if buf.itemsize == 8 else np.int32).astype(np.int64)
            sp_ = np.frombuffer(spans[key], dtype=np.uint16)
            order = np.argsort(pos, kind="stable")
            reads[key] = (pos[order], sp_[order].copy())
        return counts, cov, dict(stats), reads
    return counts, cov, dict(stats)


def accumulate_five_prime_ends(lines, chrom_lengths, offset=0, min_mapq=10):
    counts, _, stats = accumulate_tracks(
        lines, chrom_lengths, offset=offset, min_mapq=min_mapq, coverage=False)
    return counts, stats


def utr_interval(cds_start, cds_end, strand, utr_length, chrom_length,
                 include_start_codon=True):
    pad = 3 if include_start_codon else 0
    if strand == "+":
        start = cds_start - utr_length
        end = cds_start + pad
    else:
        start = cds_end - pad
        end = cds_end + utr_length

    start = max(0, start)
    end = min(chrom_length, end)
    if end - start < pad + 1:
        return None
    return start, end


def first_cds_blocks(cds_blocks, strand, length):
    blocks = sorted(cds_blocks, reverse=(strand == "-"))
    out, need = [], length
    for start, end in blocks:
        if need <= 0:
            break
        span = end - start
        if span <= 0:
            continue
        take = min(span, need)
        if strand == "+":
            out.append((start, start + take))
        else:
            out.append((end - take, end))
        need -= take
    return out


def project_onto_blocks(counts, chrom, strand, blocks, offset=0):
    parts = [project_onto_utr(counts, chrom, strand, s, e, offset=offset)
             for s, e in blocks]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)


def project_onto_utr(counts, chrom, strand, start, end, offset=0):
    shift = -offset if strand == "+" else offset
    vec = counts.window(chrom, strand, start + shift, end + shift)
    if strand == "-":
        vec = vec[::-1]
    return np.ascontiguousarray(vec)


COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(seq):
    return seq.translate(COMPLEMENT)[::-1]


def utr_sequence(genome, chrom, strand, start, end):
    seq = genome[chrom][start:end].upper()
    return revcomp(seq) if strand == "-" else seq


assert cigar_reference_span("30M") == 30
assert cigar_reference_span("10M5I10M") == 20
assert cigar_reference_span("10M5D10M") == 25
assert cigar_reference_span("3S27M2S") == 27

assert five_prime_position(0, 101, "30M") == (100, "+")
assert five_prime_position(FLAG_REVERSE, 101, "30M") == (129, "-")
assert five_prime_position(0, 101, "30M", offset=12) == (112, "+")
assert five_prime_position(FLAG_REVERSE, 101, "30M", offset=12) == (117, "-")

assert utr_interval(1000, 2000, "+", 50, 10000) == (950, 1003)
assert utr_interval(1000, 2000, "-", 50, 10000) == (1997, 2050)

assert first_cds_blocks([(1000, 2000)], "+", 100) == [(1000, 1100)]
assert first_cds_blocks([(1000, 2000)], "-", 100) == [(1900, 2000)]
assert first_cds_blocks([(1000, 1030)], "+", 100) == [(1000, 1030)]
assert first_cds_blocks([(1000, 1020), (1300, 2000)], "+", 100) == [(1000, 1020), (1300, 1380)]
assert first_cds_blocks([(1000, 1700), (1980, 2000)], "-", 100) == [(1980, 2000), (1620, 1700)]
assert utr_interval(1000, 2000, "+", 50, 10000)[1] == first_cds_blocks([(1000, 2000)], "+", 100)[0][0] + 3
