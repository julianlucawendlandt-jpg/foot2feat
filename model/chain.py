import torch

COMMIT_MAIN = 0
COMMIT_UORF = 1
DROP_OFF = 2
READ_THROUGH = 3
N_ABSORBING = 4

FATE_NAMES = ("commit_main", "commit_uORF", "drop_off", "read_through")


def build_QR(w, p, d, start_mask, is_main_start, r=None, uorf_return=None,
             drop_at_starts=True):
    N = w.shape[0]
    assert w.shape == p.shape == d.shape == (N,), "w, p, d must be per-position scalars"
    assert start_mask.shape == (N,) and is_main_start.shape == (N,)
    assert not bool((is_main_start & ~start_mask).any()), \
        "the main start must also be flagged in start_mask"

    w, p, d = w.double(), p.double(), d.double()
    dtype, device = w.dtype, w.device

    can_step = torch.ones(N, dtype=torch.bool, device=device)
    can_step[N - 1] = False

    after_main = (torch.cumsum(is_main_start.to(torch.int64), dim=0) > 0) & ~is_main_start

    can_commit = start_mask & ~after_main

    neg_inf = torch.finfo(dtype).min
    step_logit = torch.where(can_step, w, torch.full_like(w, neg_inf))
    commit_logit = torch.where(can_commit, p, torch.full_like(p, neg_inf))
    stay_logit = d
    drop_logit = torch.zeros(N, dtype=dtype, device=device)
    if not drop_at_starts:
        drop_logit = torch.where(can_commit, torch.full_like(drop_logit, neg_inf),
                                 drop_logit)

    probs = torch.softmax(
        torch.stack([step_logit, commit_logit, stay_logit, drop_logit], dim=1), dim=1)
    P_step, P_commit, P_stay, P_drop = probs[:, 0], probs[:, 1], probs[:, 2], probs[:, 3]

    idx = torch.arange(N, device=device)
    Q = torch.zeros(N, N, dtype=dtype, device=device)
    Q[idx, idx] = P_stay
    if N > 1:
        Q[idx[:-1], idx[1:]] = P_step[:-1]

    after = after_main.to(dtype)
    main = is_main_start.to(dtype)

    P_init_uorf = P_commit * (1.0 - main)
    if r is not None and uorf_return is not None:
        has_return = uorf_return >= 0
        reinit_frac = torch.sigmoid(r.double()) * has_return.to(dtype)
    else:
        has_return = torch.zeros(N, dtype=torch.bool, device=device)
        reinit_frac = torch.zeros(N, dtype=dtype, device=device)
    P_reinit = P_init_uorf * reinit_frac

    R = torch.zeros(N, N_ABSORBING, dtype=dtype, device=device)
    R[:, COMMIT_MAIN] = P_commit * main
    R[:, COMMIT_UORF] = P_init_uorf - P_reinit
    R[:, DROP_OFF] = P_drop * (1.0 - after)
    R[:, READ_THROUGH] = P_drop * after

    src = torch.nonzero(has_return).flatten()
    if src.numel():
        dst = uorf_return[src].long()
        assert bool((dst > src).all()), \
            "a uORF return must land downstream of its own start codon"
        assert bool((dst < N).all()), "uORF return index outside the state space"
        Q.index_put_((src, dst), P_reinit[src], accumulate=True)

    row_sums = Q.sum(dim=1) + R.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
        f"transient rows must sum to 1, got range [{row_sums.min():.6f}, {row_sums.max():.6f}]"
    assert bool((P_stay < 1.0).all()), "a self-loop with probability 1 would trap the particle"
    assert bool((torch.triu(Q) == Q).all()), \
        "Q must stay upper triangular; a backward return edge would allow cycles"

    return Q, R, {"p_init_uorf": P_init_uorf, "p_reinit": P_reinit}


def solve_chain(Q, R, start_state=0):
    n = Q.shape[0]
    I = torch.eye(n, dtype=Q.dtype, device=Q.device)

    fundamental = torch.linalg.solve(I - Q, I)

    profile = fundamental[start_state]
    fates = (fundamental @ R)[start_state]

    assert torch.all(profile >= -1e-6), "expected visit counts cannot be negative"
    assert torch.allclose(fates.sum(), torch.ones_like(fates.sum()), atol=1e-4), \
        f"fates must sum to 1, got {fates.detach().sum().item():.6f}"

    return profile, fates


def run_chain(w, p, d, start_mask, is_main_start, r=None, uorf_return=None,
              return_aux=False, drop_at_starts=True):
    Q, R, aux = build_QR(w, p, d, start_mask, is_main_start,
                         r=r, uorf_return=uorf_return,
                         drop_at_starts=drop_at_starts)
    profile, fates = solve_chain(Q, R)
    if not return_aux:
        return profile, fates
    return profile, fates, {
        "uorf_init_flux": (profile * aux["p_init_uorf"]).sum(),
        "reinit_flux": (profile * aux["p_reinit"]).sum(),
    }
