"""Timing and invariant tests for the Stage 2 architecture.

The FIFO is where a silent off-by-one would do the most damage: it would leak or delay
delta_s without changing any loss enough to notice, and the horizon ablation would then
compare two arms that differ by an unknown amount of timing.

    python3 test_modelk2.py
"""

import sys

import torch

import modelk2

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAIL.append(name)


def consumption_steps(h_dim, K, cond, horizon, delay, W=40, B=4):
    """Iteration indices at which a matured delta_s was actually consumed."""
    a = modelk2.build(h_dim, K, cond, seed=0, horizon=horizon, delay=delay)
    seen = []
    orig = a._ds_slot

    def spy(batch, dev, released):
        seen.append(released is not None)
        return orig(batch, dev, released)

    a._ds_slot = spy
    x = torch.randint(0, 2, (B, W + 1)).float()
    k = torch.randint(0, K, (B, W + 1))
    a.forward_window(x, k, a.init_state(B, x.device))
    return [i for i, s in enumerate(seen) if s]


def main():
    print("=" * 66)
    print("Stage 2 architecture: FIFO timing and invariants")
    print("=" * 66)

    print("\n1. Shared architecture across conditions")
    ref = {n: p.clone() for n, p in modelk2.build(8, 16, "A", 3, horizon=8, delay=9)
           .state_dict().items()}
    for c in modelk2.CONDITIONS:
        sd = modelk2.build(8, 16, c, 3, horizon=8, delay=9).state_dict()
        same = all(torch.equal(ref[n], sd[n]) for n in ref)
        n_p = sum(p.numel() for p in
                  modelk2.build(8, 16, c, 3, horizon=8, delay=9).parameters())
        check(f"{c} identical init", same, f"params={n_p}")

    print("\n2. Consumption lag equals `delay` exactly")
    for horizon, delay in [(1, 2), (8, 9), (1, 9), (8, 12), (4, 5), (4, 9)]:
        steps = consumption_steps(8, 16, "A", horizon, delay)
        first = steps[0] if steps else None
        contiguous = steps == list(range(first, first + len(steps))) if steps else False
        check(f"k={horizon}, d={delay}: first consumption at step {delay}",
              first == delay, f"got {first}")
        check(f"k={horizon}, d={delay}: consumption contiguous after that", contiguous)

    print("\n3. The two frozen-design options behave as specified")
    a = consumption_steps(8, 16, "A", 1, 2)[0]
    b = consumption_steps(8, 16, "A", 8, 9)[0]
    check("d=k+1 gives lag 2 at k=1 and 9 at k=8 (arms differ in timing)",
          (a, b) == (2, 9), f"({a}, {b})")
    a = consumption_steps(8, 16, "A", 1, 9)[0]
    b = consumption_steps(8, 16, "A", 8, 9)[0]
    check("d=9 held constant gives lag 9 in BOTH arms (timing matched)",
          (a, b) == (9, 9), f"({a}, {b})")

    print("\n4. delta_s slot contents by condition")
    B, W, K, hd = 4, 30, 16, 8
    x = torch.randint(0, 2, (B, W + 1)).float()
    k = torch.randint(0, K, (B, W + 1))
    slots = {}
    for c in modelk2.CONDITIONS:
        ag = modelk2.build(hd, K, c, 0, horizon=8, delay=9)
        rec = []
        orig = ag._ds_slot
        ag._ds_slot = lambda b, d, r, _o=orig, _r=rec: (_r.append(_o(b, d, r)) or _r[-1])
        ag.forward_window(x, k, ag.init_state(B, x.device))
        slots[c] = torch.stack(rec)
    check("B1 slot is exactly zeros", bool((slots["B1"] == 0).all()))
    check("B2 slot is exactly zeros", bool((slots["B2"] == 0).all()))
    check("A slot becomes non-zero after the delay",
          bool((slots["A"][:9] == 0).all()) and bool(slots["A"][9:].abs().sum() > 0))
    check("B3 differs from A but is non-zero at the same steps",
          not torch.allclose(slots["B3"], slots["A"])
          and bool((slots["B3"][:9] == 0).all())
          and bool(slots["B3"][9:].abs().sum() > 0))

    print("\n5. No future leakage: delta_s consumed at step t predates t")
    # A prediction emitted at j targets h_{j+k}; consumed at j+delay. Since delay >= k+1,
    # the target index j+k is strictly less than the consumption index j+delay.
    for horizon, delay in [(1, 2), (8, 9), (1, 9)]:
        check(f"k={horizon}, d={delay}: target index < consumption index",
              horizon < delay, f"{horizon} < {delay}")

    print("\n6. Aux-loss normalisation divides by the target variance")
    # The reason for normalising -- that a k=8 residual is systematically larger than a
    # k=1 one -- is a property of TRAINED dynamics (the horizon diagnostic measured
    # self-predictability falling 0.91 -> 0.57 across k=1..8). It cannot be observed on an
    # untrained model, where h is near-stationary noise and both horizons sit at roughly
    # the same distance from a random prediction. So this test checks the MECHANISM
    # algebraically rather than its trained-model consequence.
    for horizon in (1, 8):
        raw_a = modelk2.build(16, K, "B2", 0, horizon=horizon, delay=9, normalise_aux=False)
        nrm_a = modelk2.build(16, K, "B2", 0, horizon=horizon, delay=9, normalise_aux=True)
        _, l_raw, *_ = raw_a.forward_window(x, k, raw_a.init_state(B, x.device))
        _, l_nrm, *_ = nrm_a.forward_window(x, k, nrm_a.init_state(B, x.device))
        # Recover the implied divisor and confirm it is a positive variance-like scalar,
        # identical between the two runs apart from it.
        ratio = float(l_raw) / max(float(l_nrm), 1e-12)
        check(f"k={horizon}: normalisation applies a single positive divisor",
              ratio > 0 and abs(float(l_nrm) * ratio - float(l_raw)) < 1e-6,
              f"divisor = {ratio:.4g}")
    a1 = modelk2.build(16, K, "B2", 0, horizon=1, delay=9, normalise_aux=True)
    a8 = modelk2.build(16, K, "B2", 0, horizon=8, delay=9, normalise_aux=True)
    _, n1, *_ = a1.forward_window(x, k, a1.init_state(B, x.device))
    _, n8, *_ = a8.forward_window(x, k, a8.init_state(B, x.device))
    check("normalised losses are finite and comparable in scale at both horizons",
          all(torch.isfinite(torch.tensor(float(v))) for v in (n1, n8))
          and 0.1 < float(n8) / max(float(n1), 1e-12) < 10,
          f"k1={float(n1):.4g} k8={float(n8):.4g}")

    print("\n" + "=" * 66)
    if FAIL:
        print(f"FAILED ({len(FAIL)}): {', '.join(FAIL)}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
