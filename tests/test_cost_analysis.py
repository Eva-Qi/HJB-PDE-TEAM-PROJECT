"""Tests for montecarlo/cost_analysis.py."""

import numpy as np

from montecarlo.cost_analysis import paired_t_test, paired_bootstrap_test


def test_paired_t_test_rejects_when_means_differ():
    """When A is clearly worse than B, the test should reject H_0."""
    rng = np.random.default_rng(0)
    n = 1000

    # Construct: B is better than A by 1 unit on average.
    costs_a = rng.standard_normal(n) + 1.0   # mean ≈ 1
    costs_b = rng.standard_normal(n)         # mean ≈ 0

    t_stat, p_value = paired_t_test(costs_a, costs_b)

    assert p_value < 1e-10
    assert t_stat > 0   # A > B means positive diff means positive t

def test_paired_t_test_acceptss_when_means_same():
    """When A is similar to B, the test should reject H_1."""
    rng = np.random.default_rng(0)
    rng1 = np.random.default_rng(23)
    n = 1000
    costs_a = rng.standard_normal(n)         # mean ≈ 0
    costs_b = rng1.standard_normal(n)         # mean ≈ 0

    t_stat, p_value = paired_t_test(costs_a, costs_b)

    assert p_value > 1e-6

def test_paired_bootstrap_test_rejects_when_means_differ():
    """When A is clearly worse than B, bootstrap test should reject H_0."""
    rng = np.random.default_rng(0)
    n = 1000

    costs_a = rng.standard_normal(n) + 1.0
    costs_b = rng.standard_normal(n)

    mean_diff, p_value = paired_bootstrap_test(costs_a, costs_b)

    assert p_value < 0.01
    assert mean_diff > 0  # A costs more → positive diff


def test_paired_bootstrap_test_fails_to_reject_when_means_same():
    """When A and B have the same mean, bootstrap test should not reject."""
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(23)
    n = 1000

    costs_a = rng_a.standard_normal(n)
    costs_b = rng_b.standard_normal(n)

    _, p_value = paired_bootstrap_test(costs_a, costs_b)

    assert p_value > 1e-6