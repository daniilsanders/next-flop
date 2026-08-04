"""Environment and exact reference predictors for Experiment 1.

Frozen by PROTOCOL.md §1. Generative parameters must not change without a
DEVIATIONS.md entry.

Latent state is s = 2*v + z, so s in {0,1,2,3}:
    0 -> z=0, calm      1 -> z=1, calm
    2 -> z=0, volatile  3 -> z=1, volatile

Index convention: a sequence with `n_pred` predictions holds `n_pred + 1`
observations. At step t the agent has consumed x[0..t] and emits
p_hat[t] = P(x[t+1] = 1), scored against x[t+1]. So t ranges over
0..n_pred-1 and every prediction has a target.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np

LN2 = float(np.log(2.0))

# Protocol constants (PROTOCOL.md §5)
HELDOUT_SEED = 9999
HELDOUT_SEQS = 64
HELDOUT_PRED = 4096
BURN_IN = 512


@dataclass(frozen=True)
class EnvConfig:
    """Generative parameters. p_low/p_high are the only escape-hatch knobs (§6.3)."""

    p_v: float = 0.001  # volatility flip probability per step
    eps_calm: float = 0.005  # regime flip probability when calm
    eps_volatile: float = 0.05  # regime flip probability when volatile
    p_low: float = 0.25  # P(x=1 | z=0)
    p_high: float = 0.75  # P(x=1 | z=1)

    def __post_init__(self):
        # Symmetric emissions keep the marginal P(x=1) at exactly 0.5, which is
        # what makes the frozen reference exactly ln 2. The escape hatch options
        # preserve this; anything else would break the regret denominator.
        assert abs((self.p_low + self.p_high) - 1.0) < 1e-12, "emissions must be symmetric"


def transition_matrix(cfg: EnvConfig) -> np.ndarray:
    """A[s, s'] = P(s_{t+1}=s' | s_t=s). v transitions first, then z using the new v."""
    A = np.zeros((4, 4))
    for v in (0, 1):
        for z in (0, 1):
            s = 2 * v + z
            for v2 in (0, 1):
                p_vv = (1.0 - cfg.p_v) if v2 == v else cfg.p_v
                eps = cfg.eps_volatile if v2 == 1 else cfg.eps_calm
                for z2 in (0, 1):
                    p_zz = (1.0 - eps) if z2 == z else eps
                    A[s, 2 * v2 + z2] = p_vv * p_zz
    assert np.allclose(A.sum(axis=1), 1.0)
    return A


def emission_p1(cfg: EnvConfig) -> np.ndarray:
    """P(x=1 | s) for s = 0..3."""
    return np.array([cfg.p_low, cfg.p_high, cfg.p_low, cfg.p_high])


def generate(n_seq: int, n_pred: int, rng: np.random.Generator, cfg: EnvConfig = EnvConfig()):
    """Generate `n_seq` sequences of `n_pred + 1` observations.

    Vectorised: the flip indicators are independent given v, so both chains are a
    parity-cumsum over pre-sampled Bernoulli draws rather than a Python loop.
    """
    T = n_pred + 1
    v0 = rng.integers(0, 2, size=n_seq)
    z0 = rng.integers(0, 2, size=n_seq)

    flip_v = rng.random((n_seq, T)) < cfg.p_v
    flip_v[:, 0] = False
    v = (v0[:, None] + np.cumsum(flip_v, axis=1)) % 2

    eps = np.where(v == 1, cfg.eps_volatile, cfg.eps_calm)
    flip_z = rng.random((n_seq, T)) < eps
    flip_z[:, 0] = False
    z = (z0[:, None] + np.cumsum(flip_z, axis=1)) % 2

    p1 = np.where(z == 1, cfg.p_high, cfg.p_low)
    x = (rng.random((n_seq, T)) < p1).astype(np.int8)

    return {"x": x, "z": z.astype(np.int8), "v": v.astype(np.int8)}


def bayes_predict(x: np.ndarray, cfg: EnvConfig = EnvConfig()) -> np.ndarray:
    """Exact forward filter over the 4-state latent.

    x: (n_seq, T) -> p_hat: (n_seq, T-1) where p_hat[:, t] = P(x[t+1]=1 | x[0..t]).
    """
    n, T = x.shape
    A = transition_matrix(cfg)
    e1 = emission_p1(cfg)

    b = np.full((n, 4), 0.25)  # b_{0|-1}: z and v both uniform at t=0
    out = np.empty((n, T - 1))
    for t in range(T - 1):
        lik = np.where(x[:, t : t + 1] == 1, e1[None, :], 1.0 - e1[None, :])
        b = b * lik
        b /= b.sum(axis=1, keepdims=True)  # b_{t|t}
        b = b @ A  # b_{t+1|t}
        out[:, t] = b @ e1
    return out


def brute_force_predict(x_seq: np.ndarray, cfg: EnvConfig = EnvConfig()) -> np.ndarray:
    """Independent check on `bayes_predict` by explicit enumeration of latent paths.

    Exponential in sequence length -- for validation on short sequences only.
    Deliberately shares no code path with the filter.
    """
    T = len(x_seq)
    A = transition_matrix(cfg)
    e1 = emission_p1(cfg)
    prior = np.full(4, 0.25)

    out = np.empty(T - 1)
    for t in range(T - 1):
        num = 0.0
        den = 0.0
        for path in product(range(4), repeat=t + 2):
            p = prior[path[0]]
            for k in range(1, t + 2):
                p *= A[path[k - 1], path[k]]
            for k in range(t + 1):
                p *= e1[path[k]] if x_seq[k] else (1.0 - e1[path[k]])
            den += p  # summing over path[t+1] marginalises it out
            num += p * e1[path[t + 1]]
        out[t] = num / den
    return out


def frozen_predict(x: np.ndarray) -> np.ndarray:
    """Best constant predictor. Emissions are symmetric so the marginal is exactly 0.5."""
    return np.full((x.shape[0], x.shape[1] - 1), 0.5)


def oracle_predict(z: np.ndarray, cfg: EnvConfig = EnvConfig()) -> np.ndarray:
    """Knows the regime generating the observation being predicted, i.e. z[t+1]."""
    return np.where(z[:, 1:] == 1, cfg.p_high, cfg.p_low)


def bce_nats(p_hat: np.ndarray, x_next: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p_hat, eps, 1.0 - eps)
    return -(x_next * np.log(p) + (1 - x_next) * np.log(1.0 - p))


def mean_loss(p_hat: np.ndarray, x: np.ndarray, burn_in: int = BURN_IN) -> float:
    """Mean BCE in nats over scored steps. Burn-in is dropped from metrics only."""
    return float(bce_nats(p_hat[:, burn_in:], x[:, 1:][:, burn_in:]).mean())


def regret(loss: float, bayes_loss: float, frozen_loss: float = LN2) -> float:
    """0 = Bayes-optimal, 1 = frozen state (PROTOCOL.md §1)."""
    return (loss - bayes_loss) / (frozen_loss - bayes_loss)


def heldout(cfg: EnvConfig = EnvConfig()):
    """The fixed evaluation set. Identical across every condition, h, and seed."""
    return generate(HELDOUT_SEQS, HELDOUT_PRED, np.random.default_rng(HELDOUT_SEED), cfg)
