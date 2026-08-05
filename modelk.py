"""B2 agent for the K-chain environment. Calibration only.

Differs from model.py in one respect: the environment reveals which chain will be queried,
so the query index enters both the world-prediction head and the state update as a one-hot.
Without it the agent cannot know which belief to report.

This is deliberately B2-only (self-model + aux loss + delta_w feedback, delta_s slot zeroed).
Calibration never runs a treatment condition. The full conditioned architecture with the
horizon/FIFO machinery is built for Protocol v2, after (K, eps) is frozen.
"""

import math

import torch
import torch.nn as nn


def m_dim_for(h_dim):
    return max(4, h_dim // 2)


class AgentK(nn.Module):
    def __init__(self, h_dim, K):
        super().__init__()
        self.h_dim, self.K = h_dim, K
        m_dim = m_dim_for(h_dim)
        self.m_dim = m_dim

        self.f_phi = nn.Sequential(nn.Linear(h_dim, h_dim), nn.Tanh(), nn.Linear(h_dim, m_dim))
        self.p_s = nn.Sequential(nn.Linear(m_dim, h_dim), nn.Tanh(), nn.Linear(h_dim, h_dim))

        # Content-addressed READ. A Linear over [h, m, onehot(k)] is additive in k, so the
        # query index could only supply a per-chain bias -- it could not select which part
        # of h to read, making the state unreachable at any h. The readout for chain k is
        # instead a linear functional of [h, m] chosen BY k.
        self.read = nn.Embedding(K, h_dim + m_dim)
        self.read_bias = nn.Embedding(K, 1)
        nn.init.normal_(self.read.weight, std=1.0 / math.sqrt(h_dim + m_dim))
        nn.init.zeros_(self.read_bias.weight)

        # Content-addressed WRITE: the signed observation enters along a direction owned by
        # chain k, so the GRU receives address-aligned input instead of having to bind a
        # value to an address through additive input weights.
        self.write_dim = min(16, max(4, h_dim))
        self.write = nn.Embedding(K, self.write_dim)
        nn.init.normal_(self.write.weight, std=1.0 / math.sqrt(self.write_dim))

        in_dim = self.write_dim + m_dim + 1 + (h_dim + 1)  # write, m, dw slot, ds slot
        self.g = nn.GRUCell(in_dim, h_dim)

    def _read(self, h, m, k):
        return (torch.cat([h, m], -1) * self.read(k)).sum(-1) + self.read_bias(k).squeeze(-1)

    def _write(self, x, k):
        return (2.0 * x - 1.0).unsqueeze(-1) * self.write(k)

    def init_state(self, batch, device):
        z = lambda *s: torch.zeros(*s, device=device)
        return z(batch, self.h_dim), z(batch, 1)

    def forward_window(self, x_win, k_win, state):
        """x_win (B, W+1), k_win (B, W+1) one-hot-able ints. Predicts x_win[:, j+1] using
        k_win[:, j+1], which is revealed before the prediction."""
        h, dw = state
        W = x_win.shape[1] - 1
        loss_w = x_win.new_zeros(())
        loss_s = x_win.new_zeros(())
        sum_dh = 0.0
        zeros_ds = None

        for j in range(W):
            m = self.f_phi(h)
            kn = k_win[:, j + 1].long()
            logit = self._read(h, m, kn)
            h_hat = self.p_s(m)

            x_next = x_win[:, j + 1]
            loss_w = loss_w + nn.functional.binary_cross_entropy_with_logits(logit, x_next)

            if zeros_ds is None:
                zeros_ds = torch.zeros(h.shape[0], self.h_dim + 1, device=h.device)
            g_in = torch.cat([self._write(x_next, kn), m, dw, zeros_ds], dim=-1)
            h_new = self.g(g_in, h)

            # Stop-grad on the target: gradient reaches p_s and f_phi only.
            loss_s = loss_s + ((h_new.detach() - h_hat) ** 2).mean()
            with torch.no_grad():
                sum_dh += (h_new - h).norm(dim=-1).mean().item()

            dw = (x_next - torch.sigmoid(logit)).detach().unsqueeze(-1)
            h = h_new
        return loss_w / W, loss_s / W, (h, dw), {"dh": sum_dh / W}

    @torch.no_grad()
    def predict_window(self, x_win, k_win, state):
        h, dw = state
        W = x_win.shape[1] - 1
        out = []
        for j in range(W):
            m = self.f_phi(h)
            kn = k_win[:, j + 1].long()
            logit = self._read(h, m, kn)
            out.append(torch.sigmoid(logit))
            x_next = x_win[:, j + 1]
            zeros_ds = torch.zeros(h.shape[0], self.h_dim + 1, device=h.device)
            g_in = torch.cat([self._write(x_next, kn), m, dw, zeros_ds], dim=-1)
            h = self.g(g_in, h)
            dw = (x_next - torch.sigmoid(logit)).unsqueeze(-1)
        return torch.stack(out, dim=1), (h, dw)


def build(h_dim, K, seed):
    torch.manual_seed(seed)
    return AgentK(h_dim, K)


def lr_at(step, peak=3e-3, floor=3e-4, warmup=500, total=30_000):
    if step < warmup:
        return peak * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * prog))
