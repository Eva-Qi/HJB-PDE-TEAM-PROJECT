"""Convergence diagnostics for the AC closed-form sampler.

The closed-form trajectory uses the continuous-time Almgren-Chriss solution
sampled at N discrete grid points. As N increases, the discrete sum should
converge to the continuous-time expected cost. We sweep N and verify that
the N=50 (default) bias is bounded (<2% vs large-N reference).

Run:
    pytest tests/test_closed_form_convergence.py -v -s
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from shared.params import DEFAULT_PARAMS, almgren_chriss_closed_form


def test_ac_closed_form_converges_as_N_increases(capsys):
    """Expected cost should converge as N -> infinity.

    At default params (kappa*T = 1.5), N=50 should be within 2% of N=2000.
    """
    Ns = [25, 50, 100, 200, 500, 1000, 2000]
    costs = []
    for n in Ns:
        p = replace(DEFAULT_PARAMS, N=n)
        _, _, cost = almgren_chriss_closed_form(p)
        costs.append(cost)

    ref = costs[-1]
    relative_bias = [abs(c - ref) / abs(ref) for c in costs]

    # Print the sweep for the presentation/limitations section
    print("\nN-vs-cost convergence (default params):")
    print(f"  {'N':>6}  {'cost':>14}  {'|bias|/ref':>10}")
    for n, c, b in zip(Ns, costs, relative_bias):
        print(f"  {n:>6d}  {c:>14.4f}  {b:>10.4%}")

    # N=50 (default) should be within 2% of N=2000 reference
    assert relative_bias[1] < 0.02, (
        f"N=50 closed-form bias vs N=2000 = {relative_bias[1]:.4%} exceeds 2%"
    )


def test_ac_closed_form_trajectory_monotone_decreasing():
    """Optimal inventory should decrease monotonically from X0 to 0."""
    for n in (25, 50, 200):
        p = replace(DEFAULT_PARAMS, N=n)
        _, x, _ = almgren_chriss_closed_form(p)
        assert np.all(np.diff(x) <= 1e-10), f"N={n}: trajectory not monotone"
        assert np.isclose(x[0], p.X0)
        assert abs(x[-1]) < 1e-6
