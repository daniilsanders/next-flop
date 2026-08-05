"""Stage 2 architecture: K-chain agent with a horizon self-predictor and an auditable FIFO.

All conditions share ONE architecture, ONE parameter count, ONE initialization. A condition
changes only what is written into the delta_s slot.

    B1  aux loss, delta_w slot zeros, delta_s slot zeros
    B2  aux loss, delta_w live,       delta_s slot zeros
    B3  aux loss, delta_w live,       delta_s batch-rolled at consumption (matched control)
    A   aux loss, delta_w live,       delta_s live

TIMING. Two independent parameters:

    horizon k   p_s(m_t) predicts h_{t+k}; the target matures k steps later
    delay   d   steps from PREDICTION to CONSUMPTION

At step t the predictor emits h_hat_{t+k}. At step t+k the realized h_{t+k} exists, so the
detached delta_s^(k) is computed then. It is held and routed into the recurrence at step
t+d, i.e. (d-k) steps after maturity.

  - d = k+1 reproduces "route it in on the following step": delay 2 at k=1, 9 at k=8.
  - d = 9 held constant across horizons instead isolates the horizon, because otherwise the
    k=1 and k=8 arms differ in BOTH what delta_s represents AND when it arrives, and the
    interaction test cannot separate them.

Which of those Protocol v2 freezes is NOT decided here. Both are supported; the caller
passes `delay` explicitly and it is recorded in every run.
"""

import math
from collections import deque

import torch
import torch.nn as nn

CONDITIONS = ("B1", "B2", "B3", "A")
_SPEC = {  # (delta_w slot live, delta_s slot mode)
    "B1": (False, "zero"),
    "B2": (True, "zero"),
    "B3": (True, "roll"),
    "A": (True, "self"),
}


def m_dim_for(h_dim):
    return max(4, h_dim // 2)


class AgentK2(nn.Module):
    def __init__(self, h_dim, K, condition, horizon=1, delay=None, normalise_aux=True):
        super().__init__()
        assert condition in CONDITIONS
        assert horizon >= 1
        self.h_dim, self.K, self.condition = h_dim, K, condition
        self.horizon = horizon
        self.delay = horizon + 1 if delay is None else delay
        assert self.delay >= horizon + 1, "consumption cannot precede maturity + 1"
        # A prediction emitted at iteration j targets h_{j+k}, which is h_new at iteration
        # j+k-1 -- maturity is k-1 steps after emission, not k. Consumption must then land
        # release_lag steps after maturity so that total emission->consumption == delay.
        self.release_lag = self.delay - horizon + 1
        self.normalise_aux = normalise_aux
        self.use_dw, self.ds_mode = _SPEC[condition]

        m_dim = m_dim_for(h_dim)
        self.m_dim = m_dim
        self.f_phi = nn.Sequential(nn.Linear(h_dim, h_dim), nn.Tanh(), nn.Linear(h_dim, m_dim))
        self.p_s = nn.Sequential(nn.Linear(m_dim, h_dim), nn.Tanh(), nn.Linear(h_dim, h_dim))

        # Content-addressed read/write (see modelk.py: a Linear over [h, m, onehot(k)] is
        # additive in k and cannot select which part of h to read, making the state
        # unreachable at any capacity).
        self.read = nn.Embedding(K, h_dim + m_dim)
        self.read_bias = nn.Embedding(K, 1)
        nn.init.normal_(self.read.weight, std=1.0 / math.sqrt(h_dim + m_dim))
        nn.init.zeros_(self.read_bias.weight)
        self.write_dim = min(16, max(4, h_dim))
        self.write = nn.Embedding(K, self.write_dim)
        nn.init.normal_(self.write.weight, std=1.0 / math.sqrt(self.write_dim))

        self.ds_norm = nn.LayerNorm(h_dim, elementwise_affine=False)
        in_dim = self.write_dim + m_dim + 1 + (h_dim + 1)
        self.g = nn.GRUCell(in_dim, h_dim)

    # ---------------------------------------------------------------- state

    def init_state(self, batch, device):
        z = lambda *s: torch.zeros(*s, device=device)
        return {"h": z(batch, self.h_dim), "dw": z(batch, 1),
                "pend": deque(maxlen=self.horizon),  # predictions awaiting maturity
                "rel": deque(maxlen=max(1, self.release_lag))}  # matured, awaiting release

    def detach_state(self, s):
        return {"h": s["h"].detach(), "dw": s["dw"].detach(),
                "pend": deque((p.detach() for p in s["pend"]), maxlen=s["pend"].maxlen),
                "rel": deque((r.detach() for r in s["rel"]), maxlen=s["rel"].maxlen)}

    # ---------------------------------------------------------------- slots

    def _ds_slot(self, batch, device, released):
        """released: the matured delta_s due for consumption now, or None."""
        if self.ds_mode == "zero" or released is None:
            return torch.zeros(batch, self.h_dim + 1, device=device)
        ds = torch.roll(released, shifts=1, dims=0) if self.ds_mode == "roll" else released
        mag = torch.log(ds.norm(dim=-1, keepdim=True) + 1e-6)
        return torch.cat([self.ds_norm(ds), mag], dim=-1)

    def _read_head(self, h, m, k):
        return (torch.cat([h, m], -1) * self.read(k)).sum(-1) + self.read_bias(k).squeeze(-1)

    # ---------------------------------------------------------------- rollout

    def forward_window(self, x_win, k_win, state, collect=False):
        h, dw = state["h"], state["dw"]
        pend, rel = state["pend"], state["rel"]
        W = x_win.shape[1] - 1
        B = x_win.shape[0]
        dev = x_win.device

        loss_w = x_win.new_zeros(())
        loss_s = x_win.new_zeros(())
        n_aux = 0
        log = {"dh": 0.0, "ds_norm": 0.0, "n_ds_consumed": 0}
        traj = {"dw": [], "ds": []} if collect else None

        for j in range(W):
            m = self.f_phi(h)
            kn = k_win[:, j + 1].long()
            logit = self._read_head(h, m, kn)
            x_next = x_win[:, j + 1]
            loss_w = loss_w + nn.functional.binary_cross_entropy_with_logits(logit, x_next)

            # p_s emits the prediction of h_{t+k}. It reads m only.
            pend.append(self.p_s(m))

            # Consume a matured delta_s if one is due now.
            released = rel[0] if len(rel) == rel.maxlen else None
            if released is not None:
                log["n_ds_consumed"] += 1
                log["ds_norm"] += float(released.norm(dim=-1).mean())

            dw_slot = dw if self.use_dw else torch.zeros_like(dw)
            g_in = torch.cat([(2.0 * x_next - 1.0).unsqueeze(-1) * self.write(kn), m,
                              dw_slot, self._ds_slot(B, dev, released)], dim=-1)
            h_new = self.g(g_in, h)

            # The prediction that targeted h_{t+1} was emitted k-1 iterations ago.
            if len(pend) == self.horizon:
                target = h_new.detach()
                pred = pend[0]
                # Stop-grad on the TARGET: gradient reaches p_s and f_phi only.
                mse = ((target - pred) ** 2).mean()
                if self.normalise_aux:
                    # Makes lambda comparable across horizons: a k=8 residual is
                    # systematically larger than a k=1 one, so an unnormalised loss would
                    # silently reweight the auxiliary objective in the arm under test.
                    mse = mse / (target.var(dim=0).mean().detach() + 1e-6)
                loss_s = loss_s + mse
                n_aux += 1
                rel.append((target - pred).detach())

            with torch.no_grad():
                log["dh"] += float((h_new - h).norm(dim=-1).mean())
            if collect:
                traj["dw"].append(dw.squeeze(-1).detach())
                traj["ds"].append(released.detach() if released is not None
                                  else torch.zeros(B, self.h_dim, device=dev))

            dw = (x_next - torch.sigmoid(logit)).detach().unsqueeze(-1)
            h = h_new

        log["dh"] /= W
        if log["n_ds_consumed"]:
            log["ds_norm"] /= log["n_ds_consumed"]
        out_state = {"h": h, "dw": dw, "pend": pend, "rel": rel}
        if collect:
            traj = {k: torch.stack(v, 1) for k, v in traj.items()}
        return loss_w / W, loss_s / max(1, n_aux), out_state, log, traj

    @torch.no_grad()
    def predict_window(self, x_win, k_win, state):
        _, _, st, _, _ = self.forward_window(x_win, k_win, state)
        return None, st

    @torch.no_grad()
    def probs(self, x_win, k_win, state):
        """Eval path: probabilities only, same timing as training."""
        h, dw = state["h"], state["dw"]
        pend, rel = state["pend"], state["rel"]
        W, B, dev = x_win.shape[1] - 1, x_win.shape[0], x_win.device
        out = []
        for j in range(W):
            m = self.f_phi(h)
            kn = k_win[:, j + 1].long()
            logit = self._read_head(h, m, kn)
            out.append(torch.sigmoid(logit))
            pend.append(self.p_s(m))
            released = rel[0] if len(rel) == rel.maxlen else None
            x_next = x_win[:, j + 1]
            dw_slot = dw if self.use_dw else torch.zeros_like(dw)
            g_in = torch.cat([(2.0 * x_next - 1.0).unsqueeze(-1) * self.write(kn), m,
                              dw_slot, self._ds_slot(B, dev, released)], dim=-1)
            h_new = self.g(g_in, h)
            if len(pend) == self.horizon:
                rel.append((h_new - pend[0]))
            dw = (x_next - torch.sigmoid(logit)).unsqueeze(-1)
            h = h_new
        return torch.stack(out, 1), {"h": h, "dw": dw, "pend": pend, "rel": rel}


def build(h_dim, K, condition, seed, horizon=1, delay=None, normalise_aux=True):
    """Identical weights across conditions for a given (h_dim, K, seed)."""
    torch.manual_seed(seed)
    return AgentK2(h_dim, K, condition, horizon, delay, normalise_aux)
