import torch

from analysis.recoverability.chainfast import log_visits
from model.chain import run_chain

torch.set_default_dtype(torch.float64)


def test_against_exact():
    g = torch.Generator().manual_seed(0)
    lens = [12, 40, 7, 25]
    ws, ps, ds, ss, mains, lasts, segs, exact = [], [], [], [], [], [], [], []
    for t, n in enumerate(lens):
        w = 4 + torch.randn(n, generator=g)
        p = torch.randn(n, generator=g) * 2
        d = torch.randn(n, generator=g) * 2
        start = torch.rand(n, generator=g) < 0.2
        main = torch.zeros(n, dtype=torch.bool)
        main[n - 3] = True
        start[n - 3] = True
        start[n - 2:] = False
        for das in (True, False):
            prof, _ = run_chain(w, p, d, start, main, drop_at_starts=das)
            exact.append((t, das, prof))
        ws.append(w); ps.append(p); ds.append(d); ss.append(start)
        last = torch.zeros(n, dtype=torch.bool)
        last[-1] = True
        lasts.append(last)
        segs.append(torch.full((n,), t, dtype=torch.int64))
    W, Pp, D, S = map(torch.cat, (ws, ps, ds, ss))
    LST, SEG = torch.cat(lasts), torch.cat(segs)
    offs = [0]
    for n in lens:
        offs.append(offs[-1] + n)
    for das in (True, False):
        lv = log_visits(W, Pp, D, S, LST, SEG, drop_at_starts=das)
        for t, d_, prof in exact:
            if d_ != das:
                continue
            got = torch.exp(lv[offs[t]:offs[t + 1]])
            assert torch.allclose(got, prof, rtol=1e-9, atol=1e-12), (t, das, got, prof)
    print("ok  closed-form visits match the exact chain solve (both drop modes)")


if __name__ == "__main__":
    test_against_exact()
