"""Modules for Experiment 1. Implements PROTOCOL.md §2, §3, §9.

All five conditions share ONE architecture, ONE parameter count, and ONE
initialization. A condition changes only what is written into the delta slots.
Constructing any condition under the same torch seed yields identical weights.

Nothing here implements E1, replay, active inference, actions, or goal priors.
"""

import math

import torch
import torch.nn as nn

CONDITIONS = ("B0", "B1", "B2", "B3", "A")

# condition -> (aux self-prediction loss, delta_w slot live, delta_s slot mode)
_SPEC = {
    "B0": (False, False, "zero"),
    "B1": (True, False, "zero"),
    "B2": (True, True, "zero"),
    "B3": (True, True, "roll"),  # batch-rolled delta_s, same timestep
    "A": (True, True, "self"),
}


def m_dim_for(h_dim: int) -> int:
    return max(4, h_dim // 2)


class Agent(nn.Module):
    """f_phi, p_w, p_s, g_theta.

    Step order at t (h_t has consumed x_0..x_t):
        m_t       = f_phi(h_t)
        logit_t   = p_w(h_t, m_t)          -> scored against x_{t+1}
        h_hat_t+1 = p_s(m_t)               -> from m ONLY (never reads h_t)
        h_{t+1}   = g_theta(h_t, [x_{t+1}, m_t, dw^{t-1}, ds^{t-1}])
        dw^t      = x_{t+1} - sigmoid(logit_t)
        ds^t      = h_{t+1} - h_hat_t+1

    Both deltas enter g at lag 1. delta_w is available at lag 0, but feeding it
    earlier than delta_s would confound "type of error" with "delay" in the
    A-vs-B2 comparison. delta_s cannot be lag 0 -- it depends on h_{t+1}, which
    is what g is computing. Errors land one step late by construction.
    """

    def __init__(self, h_dim: int, condition: str):
        super().__init__()
        assert condition in CONDITIONS
        self.h_dim = h_dim
        self.condition = condition
        self.use_aux, self.use_dw, self.ds_mode = _SPEC[condition]

        m_dim = m_dim_for(h_dim)
        self.m_dim = m_dim

        # f_phi: h -> m. One hidden layer: a purely linear self-model would make a
        # null result attributable to f's weakness rather than to delta_s.
        self.f_phi = nn.Sequential(nn.Linear(h_dim, h_dim), nn.Tanh(), nn.Linear(h_dim, m_dim))
        # p_s: m -> h_hat. Reads m only (PROTOCOL/spec §2 constraint).
        self.p_s = nn.Sequential(nn.Linear(m_dim, h_dim), nn.Tanh(), nn.Linear(h_dim, h_dim))
        # p_w: (h, m) -> logit
        self.p_w = nn.Linear(h_dim + m_dim, 1)

        # delta_s slot is LayerNorm(ds) (+) log||ds||. affine=False so LN(0)=0
        # exactly -- an affine LN would hand the zero-slot conditions a learnable
        # bias and break the "inert slot" guarantee.
        self.ds_norm = nn.LayerNorm(h_dim, elementwise_affine=False)

        in_dim = 1 + m_dim + 1 + (h_dim + 1)  # x, m, dw slot, ds slot
        self.g = nn.GRUCell(in_dim, h_dim)

    def init_state(self, batch: int, device):
        z = lambda *s: torch.zeros(*s, device=device)
        return z(batch, self.h_dim), z(batch, 1), z(batch, self.h_dim)

    def _ds_slot(self, ds: torch.Tensor) -> torch.Tensor:
        """Build the delta_s input slot. Zero conditions bypass LayerNorm entirely."""
        if self.ds_mode == "zero":
            return torch.zeros(ds.shape[0], self.h_dim + 1, device=ds.device)
        if self.ds_mode == "roll":
            # batch element i receives element (i-1)'s delta_s at the SAME timestep:
            # matched dim, delay, normalization path, and marginal variance;
            # conditional relationship to this sequence destroyed.
            ds = torch.roll(ds, shifts=1, dims=0)
        mag = torch.log(ds.norm(dim=-1, keepdim=True) + 1e-6)
        return torch.cat([self.ds_norm(ds), mag], dim=-1)

    def forward_window(self, x_win: torch.Tensor, state):
        """x_win: (B, W+1) float in {0,1}. Predicts x_win[:, k+1] for k in 0..W-1.

        Returns (loss_w, loss_s, new_state, stats). Losses are means over the window.
        """
        h, dw, ds = state
        W = x_win.shape[1] - 1

        loss_w = x_win.new_zeros(())
        loss_s = x_win.new_zeros(())
        sum_dw = sum_ds = sum_dh = 0.0

        for k in range(W):
            m = self.f_phi(h)
            logit = self.p_w(torch.cat([h, m], dim=-1)).squeeze(-1)
            h_hat = self.p_s(m)

            x_next = x_win[:, k + 1]
            loss_w = loss_w + nn.functional.binary_cross_entropy_with_logits(logit, x_next)

            # Deltas enter DETACHED: they are features, not gradient paths. Without
            # this, g can be trained to shrink delta_s (failure mode 3.2).
            dw_slot = dw if self.use_dw else torch.zeros_like(dw)
            g_in = torch.cat([x_next.unsqueeze(-1), m, dw_slot, self._ds_slot(ds)], dim=-1)
            h_new = self.g(g_in, h)

            if self.use_aux:
                # Stop-grad on the TARGET: gradient reaches p_s and f_phi only.
                # Otherwise the cheapest fix is for g to move h toward h_hat -- 3.2
                # again, from the other side.
                loss_s = loss_s + ((h_new.detach() - h_hat) ** 2).mean()

            with torch.no_grad():
                sum_dh += (h_new - h).norm(dim=-1).mean().item()
                sum_dw += dw.abs().mean().item()
                sum_ds += ds.norm(dim=-1).mean().item()

            dw = (x_next - torch.sigmoid(logit)).detach().unsqueeze(-1)
            ds = (h_new - h_hat).detach()
            h = h_new

        stats = {"dh": sum_dh / W, "dw": sum_dw / W, "ds": sum_ds / W}
        return loss_w / W, loss_s / W, (h, dw, ds), stats

    @torch.no_grad()
    def predict_window(self, x_win: torch.Tensor, state):
        """Eval path. Returns (B, W) probabilities and the carried state."""
        h, dw, ds = state
        W = x_win.shape[1] - 1
        out = []
        for k in range(W):
            m = self.f_phi(h)
            logit = self.p_w(torch.cat([h, m], dim=-1)).squeeze(-1)
            h_hat = self.p_s(m)
            out.append(torch.sigmoid(logit))

            x_next = x_win[:, k + 1]
            dw_slot = dw if self.use_dw else torch.zeros_like(dw)
            g_in = torch.cat([x_next.unsqueeze(-1), m, dw_slot, self._ds_slot(ds)], dim=-1)
            h_new = self.g(g_in, h)

            dw = (x_next - torch.sigmoid(logit)).unsqueeze(-1)
            ds = h_new - h_hat
            h = h_new
        return torch.stack(out, dim=1), (h, dw, ds)


def lr_at(step: int, peak=3e-3, floor=3e-4, warmup=500, total=30_000) -> float:
    """Linear warmup then cosine decay. PROTOCOL.md §4."""
    if step < warmup:
        return peak * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * prog))


def build(h_dim: int, condition: str, seed: int) -> Agent:
    """Identical weights across conditions for a given (h_dim, seed)."""
    torch.manual_seed(seed)
    return Agent(h_dim, condition)
