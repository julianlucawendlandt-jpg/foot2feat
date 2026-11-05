import torch

NEG = -1e30


def log_visits(w, p, d, start, last, seg, drop_at_starts=True):
    neg = torch.full_like(w, NEG)
    step_l = torch.where(last, neg, w)
    commit_l = torch.where(start, p, neg)
    drop_l = torch.zeros_like(w)
    if not drop_at_starts:
        drop_l = torch.where(start, neg, drop_l)
    logits = torch.stack([step_l, commit_l, d, drop_l], dim=1)
    logp = torch.log_softmax(logits, dim=1)
    log_step, log_stay = logp[:, 0], logp[:, 2]
    log_leave = torch.log1p(-torch.exp(log_stay).clamp(max=1 - 1e-12))
    inc = torch.where(last, torch.zeros_like(w), log_step - log_leave)
    cs = torch.cumsum(inc, 0)
    first = torch.ones_like(seg, dtype=torch.bool)
    first[1:] = seg[1:] != seg[:-1]
    start_idx = torch.cummax(torch.where(first, torch.arange(len(seg), device=seg.device), torch.zeros_like(seg)), 0)[0]
    excl = cs - inc - (cs[start_idx] - inc[start_idx])
    return excl - log_leave
