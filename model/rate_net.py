import torch
import torch.nn as nn

from data.features import NET_INPUT_CHANNELS, N_UORF_GEOM


class RateNet(nn.Module):
    def __init__(self, in_channels=NET_INPUT_CHANNELS, hidden=32, window=10):
        super().__init__()
        self.window = window
        k = 2 * window + 1

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=k, padding=window),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, 3, kernel_size=1),
        )

        self.reinit_head = nn.Sequential(
            nn.Linear(N_UORF_GEOM, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )

        self.struct_head = nn.Conv1d(hidden, 2, kernel_size=1)

        final = self.net[-1]
        nn.init.zeros_(final.weight)
        with torch.no_grad():
            final.bias.copy_(torch.tensor([6.0, 2.0, 2.0]))

        nn.init.zeros_(self.reinit_head[-1].weight)
        nn.init.zeros_(self.reinit_head[-1].bias)

    def forward(self, x):
        assert x.dim() == 2, "RateNet takes one UTR at a time, shaped (N, C)"
        z = x.t().unsqueeze(0)
        h = self.net[:-1](z)
        out = self.net[-1](h).squeeze(0)
        struct = self.struct_head(h).squeeze(0)
        return out[0], out[1], out[2], struct

    def reinit_logit(self, geom):
        assert geom.dim() == 2 and geom.shape[1] == N_UORF_GEOM, \
            "geom must be (N, {}), got {}".format(N_UORF_GEOM, tuple(geom.shape))
        return self.reinit_head(geom).reshape(-1)
