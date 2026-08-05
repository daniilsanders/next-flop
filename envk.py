"""K-parallel-chain environment and exact factorized Bayes filter.

Replaces env.py, which was retired after STAGE1_POSTMORTEM.md: its sufficient statistic
was two numbers whose second dimension was worth under 3% of the frozen-to-Bayes range,
so no capacity deficit existed at any emission setting.

Design target here is not "more dimensions" but "each dimension carries large marginal
value". K independent chains give that directly: holding only j of K beliefs leaves the
other (K-j)/K of queried steps predicted at chance.

    K chains z^1..z^K, each flips independently with probability eps each step
    at step t an index k_t ~ Uniform{0..K-1} is drawn and REVEALED
    x_t ~ Bernoulli(p_low + (p_high - p_low) * z^{k_t}_t)

The agent has seen x_0..x_t and k_0..k_{t+1}, and predicts x_{t+1}.

Because each observation touches exactly one chain and the chains evolve independently,
the posterior factorises across chains exactly and forever. Exact filtering is K
independent scalar updates -- O(K) per step, no approximation, so normalized regret keeps
an exact ceiling. Scaling a single chain's state space instead would have broken this.
"""

from dataclasses import dataclass

import numpy as np

LN2 = float(np.log(2.0))

HELDOUT_SEED = 9999
HELDOUT_SEQS = 64
HELDOUT_PRED = 4096
BURN_IN = 512


@dataclass(frozen=True)
class EnvKConfig:
    K: int = 16
    eps: float = 0.03125  # per-chain flip probability; calibration sets eps*K
    p_low: float = 0.1
    p_high: float = 0.9

    def __post_init__(self):
        # Symmetric emissions keep the marginal P(x=1) at exactly 0.5, which is what makes
        # the frozen reference exactly ln 2 and the regret denominator well defined.
        assert abs((self.p_low + self.p_high) - 1.0) < 1e-12, "emissions must be symmetric"
        assert self.K >= 1 and 0.0 < self.eps < 0.5

    @property
    def eps_K(self):
        """How much of a chain's belief decays between visits, roughly."""
        return self.eps * self.K


def generate(n_seq, n_pred, rng, cfg=EnvKConfig()):
    """Generate n_seq sequences of n_pred+1 observations.

    Returns x (n_seq, T), k (n_seq, T), z (n_seq, T, K). Vectorised: per-chain flips are
    independent Bernoulli draws, so each chain is a parity-cumsum rather than a loop.
    """
    T = n_pred + 1
    z0 = rng.integers(0, 2, size=(n_seq, cfg.K))
    flips = rng.random((n_seq, T, cfg.K)) < cfg.eps
    flips[:, 0, :] = False
    z = (z0[:, None, :] + np.cumsum(flips, axis=1)) % 2

    k = rng.integers(0, cfg.K, size=(n_seq, T))
    z_at_k = np.take_along_axis(z, k[:, :, None], axis=2)[:, :, 0]
    p1 = cfg.p_low + (cfg.p_high - cfg.p_low) * z_at_k
    x = (rng.random((n_seq, T)) < p1).astype(np.int8)
    return {"x": x, "k": k.astype(np.int16), "z": z.astype(np.int8)}


def bayes_predict(x, k, cfg=EnvKConfig(), drop=()):
    """Exact factorised filter. Returns p_hat (n_seq, T-1).

    `drop` is a set of chain indices whose belief is replaced by 0.5 at prediction time --
    used to measure the marginal value of holding a chain, not part of the reference.
    """
    n, T = x.shape
    idx = np.arange(n)
    dropped = np.zeros(cfg.K, dtype=bool)
    dropped[list(drop)] = True

    b = np.full((n, cfg.K), 0.5)
    out = np.empty((n, T - 1))
    span = cfg.p_high - cfg.p_low
    for t in range(T - 1):
        kt, xt = k[:, t], x[:, t]
        bk = b[idx, kt]
        lik1 = np.where(xt == 1, cfg.p_high, 1.0 - cfg.p_high)
        lik0 = np.where(xt == 1, cfg.p_low, 1.0 - cfg.p_low)
        b[idx, kt] = bk * lik1 / (bk * lik1 + (1.0 - bk) * lik0)

        b = cfg.eps + b * (1.0 - 2.0 * cfg.eps)  # every chain transitions each step

        kn = k[:, t + 1]
        bn = np.where(dropped[kn], 0.5, b[idx, kn])
        out[:, t] = cfg.p_low + span * bn
    return out


def joint_bayes_predict(x, k, cfg=EnvKConfig()):
    """Exact forward filter over the FULL 2^K joint state. Independent of the factorised
    filter -- it never assumes the posterior factorises. Validation only; O(4^K) per step."""
    n, T = x.shape
    S = 1 << cfg.K
    bits = np.array([[(s >> i) & 1 for i in range(cfg.K)] for s in range(S)])

    A = np.ones((S, S))
    for i in range(cfg.K):
        same = bits[:, i][:, None] == bits[:, i][None, :]
        A *= np.where(same, 1.0 - cfg.eps, cfg.eps)

    p1 = cfg.p_low + (cfg.p_high - cfg.p_low) * bits  # (S, K)
    b = np.full((n, S), 1.0 / S)
    out = np.empty((n, T - 1))
    for t in range(T - 1):
        e = p1[:, k[:, t]].T  # (n, S)
        lik = np.where(x[:, t : t + 1] == 1, e, 1.0 - e)
        b = b * lik
        b /= b.sum(axis=1, keepdims=True)
        b = b @ A
        out[:, t] = (b * p1[:, k[:, t + 1]].T).sum(axis=1)
    return out


def brute_force_predict(x_seq, k_seq, cfg, n_pred):
    """Enumerate every latent path explicitly. Validates joint_bayes_predict itself.
    Exponential in both K and length -- tiny cases only."""
    from itertools import product

    S = 1 << cfg.K
    bits = [[(s >> i) & 1 for i in range(cfg.K)] for s in range(S)]

    def trans(a, b_):
        p = 1.0
        for i in range(cfg.K):
            p *= cfg.eps if bits[a][i] != bits[b_][i] else 1.0 - cfg.eps
        return p

    def emit(s, kk, xx):
        p = cfg.p_high if bits[s][kk] else cfg.p_low
        return p if xx else 1.0 - p

    out = np.empty(n_pred)
    for t in range(n_pred):
        num = den = 0.0
        for path in product(range(S), repeat=t + 2):
            p = 1.0 / S
            for j in range(1, t + 2):
                p *= trans(path[j - 1], path[j])
            for j in range(t + 1):
                p *= emit(path[j], k_seq[j], x_seq[j])
            den += p
            num += p * (cfg.p_high if bits[path[t + 1]][k_seq[t + 1]] else cfg.p_low)
        out[t] = num / den
    return out


def frozen_predict(x):
    return np.full((x.shape[0], x.shape[1] - 1), 0.5)


def oracle_predict(z, k, cfg=EnvKConfig()):
    """Knows the bit of the chain that will be queried next."""
    zn = np.take_along_axis(z[:, 1:], k[:, 1:, None], axis=2)[:, :, 0]
    return np.where(zn == 1, cfg.p_high, cfg.p_low)


def bce_nats(p_hat, x_next, eps=1e-12):
    p = np.clip(p_hat, eps, 1.0 - eps)
    return -(x_next * np.log(p) + (1 - x_next) * np.log(1.0 - p))


def mean_loss(p_hat, x, burn_in=BURN_IN):
    return float(bce_nats(p_hat[:, burn_in:], x[:, 1:][:, burn_in:]).mean())


def regret(loss, bayes_loss, frozen_loss=LN2):
    return (loss - bayes_loss) / (frozen_loss - bayes_loss)


def heldout(cfg=EnvKConfig()):
    return generate(HELDOUT_SEQS, HELDOUT_PRED, np.random.default_rng(HELDOUT_SEED), cfg)
