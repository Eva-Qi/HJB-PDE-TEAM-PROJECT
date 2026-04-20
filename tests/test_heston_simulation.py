"""Tests for Heston stochastic volatility execution simulation.

REWRITE HISTORY: 2026-04-18 — 5 theater tests replaced after code-council
audit. Each replacement's docstring names the mathematical invariant it
tests. Original theater tests (now removed): test_heston_output_shapes,
test_heston_initial_conditions, test_heston_variance_starts_at_v0,
test_heston_reproducible (demoted to one-liner), test_heston_costs_finite.

HARDENING: 2026-04-18 — GLM-5.1 challenger audit tightened 4 tolerances
and added TestHestonZInjection (deterministic Z-injection test).

Still present (kept — these already test real properties):
    test_heston_price_positive — Euler scheme doesn't produce mostly-negative prices
    test_heston_reduces_to_constant_vol — degenerate case invariant (xi=0)
    test_heston_mean_cost_reasonable — order-of-magnitude sanity
"""

import math
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from shared.params import DEFAULT_PARAMS, ACParams
from extensions.heston import HestonParams
from montecarlo.sde_engine import simulate_execution, simulate_heston_execution
from montecarlo.strategies import twap_trajectory


# Reasonable Heston params for sigma ~ 0.3 (v0 = theta = 0.09)
# 2*kappa*theta = 0.36 > xi^2 = 0.25 → Feller satisfied
HESTON_PARAMS = HestonParams(
    kappa=2.0,
    theta=0.09,
    xi=0.5,
    rho=-0.3,
    v0=0.09,
)

TWAP_X = twap_trajectory(DEFAULT_PARAMS)

N_PATHS = 5000
SEED = 42


# ═══════════════════════════════════════════════════════════════════
# REWRITE: test_heston_output_shapes → mean reversion to theta
# ═══════════════════════════════════════════════════════════════════
# Council analysis:
# - Contrarian: "S.shape == (N_PATHS, N+1)" only fails if allocation
#   logic breaks — zero chance of finding a real Heston bug.
# - First Principles: The defining property of CIR variance is
#   mean-reversion to theta. If mean-reversion is broken (kappa not
#   applied, drift wrong), this is THE test that catches it.
# - Executor: one assertion on time-averaged variance vs theta.
class TestHestonVarianceMeanReversion:
    """Variance process mean-reverts to theta (defining CIR property)."""

    def test_variance_mean_reverts_to_theta(self):
        """Over a long horizon with strong mean-reversion (kappa*T >> 1),
        the time-averaged variance in the second half of each path
        should converge to theta within 10%.

        Invariant: E[v_t] → theta as t → ∞ (CIR stationary mean).
        Breaking this means kappa drift is misapplied — a semantic bug
        that array-shape tests cannot catch.
        """
        # Extend horizon to T=2.5yr, N=500 → kappa*T = 5 (strong reversion)
        long_params = replace(DEFAULT_PARAMS, T=2.5, N=500)
        long_twap = twap_trajectory(long_params)
        _, v, _ = simulate_heston_execution(
            long_params, HESTON_PARAMS, long_twap,
            n_paths=3000, seed=SEED,
        )
        burn_in = v.shape[1] // 2
        mean_v = np.mean(v[:, burn_in:])
        rel_err = abs(mean_v - HESTON_PARAMS.theta) / HESTON_PARAMS.theta
        assert rel_err < 0.05, (
            f"Time-avg variance after burn-in ({mean_v:.6f}) should be "
            f"within 5% of theta ({HESTON_PARAMS.theta:.6f}); got "
            f"{rel_err*100:.1f}% error — mean-reversion may be broken."
        )


# ═══════════════════════════════════════════════════════════════════
# REWRITE: test_heston_initial_conditions → leverage effect (rho)
# ═══════════════════════════════════════════════════════════════════
# Council analysis:
# - Contrarian: "S[:,0] == S0" tests a literal assignment statement.
#   If that breaks, EVERY other test fails — this test is pure noise.
# - Outsider: A reviewer asking "does rho work?" can't answer from
#   this test. The rho parameter is the most complex part of Heston;
#   it needs a behavioral test, not a shape check.
# - First Principles: Negative rho (leverage effect) is the ENTIRE
#   reason Heston is used in finance. Test that directly.
class TestHestonLeverageEffect:
    """Negative rho should produce negative empirical correlation between
    price returns and variance changes (the leverage effect)."""

    def test_negative_rho_produces_leverage_effect(self):
        """Simulating with rho=-0.9, the empirical correlation between
        dlogS and dv should be strongly negative.

        Invariant: corr(dW_S, dW_v) = rho is implemented via Cholesky
        as Z_v = rho*W1 + sqrt(1-rho^2)*W2. If Cholesky is wrong or
        rho is dropped, this test catches it. No other test does.
        """
        strong_neg = HestonParams(kappa=2.0, theta=0.09, xi=0.5, rho=-0.9, v0=0.09)
        S, v, _ = simulate_heston_execution(
            DEFAULT_PARAMS, strong_neg, TWAP_X,
            n_paths=5000, seed=SEED,
        )
        # Clamp variance to positive before log-return (Full Truncation
        # can leave tiny negatives in storage)
        dlog_S = np.diff(np.log(np.maximum(S, 1e-10)), axis=1).ravel()
        dv = np.diff(v, axis=1).ravel()
        mask = np.isfinite(dlog_S) & np.isfinite(dv)
        corr = float(np.corrcoef(dlog_S[mask], dv[mask])[0, 1])
        assert corr < -0.7, (
            f"With rho=-0.9, empirical corr(dlogS, dv) should be < -0.7 "
            f"(leverage effect); got {corr:.3f} — Cholesky of correlated "
            f"Brownians may be broken, rho is being ignored, or magnitude "
            f"is heavily damped (e.g. xi halved internally)."
        )


# ═══════════════════════════════════════════════════════════════════
# REWRITE: test_heston_variance_starts_at_v0 → CIR stationary variance
# ═══════════════════════════════════════════════════════════════════
# Council analysis:
# - Contrarian: "v[:,0] == v0" is a duplicate of the above shape test.
#   Two tests for one assignment statement.
# - Expansionist: The deeper property isn't "v starts at v0" but
#   "v ends up with the right SPREAD". Stationary variance is the
#   complete second-moment characterization of CIR.
# - First Principles: Var_stationary[v] = theta*xi^2/(2*kappa) is the
#   closed-form CIR stationary variance. Testing this covers xi AND
#   kappa AND theta jointly, unlike the mean-reversion test which
#   only covers theta+kappa.
class TestHestonCIRStationarity:
    """Long-run variance of the v process matches CIR stationary formula."""

    def test_stationary_variance_matches_cir_formula(self):
        """Empirical Var[v] in stationary regime ≈ theta*xi^2/(2*kappa).

        This is the CLOSED-FORM stationary variance of a CIR process.
        Jointly validates xi (vol-of-vol), kappa (reversion speed),
        and theta (long-run mean). No existing test covers xi behavior.

        Tolerance is 20% (tightened from 30% per GLM audit). If this
        fails due to noise, paths should be increased rather than
        tolerance loosened.
        """
        long_params = replace(DEFAULT_PARAMS, T=5.0, N=1000)
        long_twap = twap_trajectory(long_params)
        _, v, _ = simulate_heston_execution(
            long_params, HESTON_PARAMS, long_twap,
            n_paths=5000, seed=SEED,
        )
        burn_in = v.shape[1] // 2
        v_stationary = v[:, burn_in:].ravel()
        # Clip any truncation artifacts for variance calc
        v_stationary = v_stationary[v_stationary > 0]
        empirical_var = float(np.var(v_stationary))
        theoretical_var = (
            HESTON_PARAMS.theta * HESTON_PARAMS.xi ** 2 / (2.0 * HESTON_PARAMS.kappa)
        )
        rel_err = abs(empirical_var - theoretical_var) / theoretical_var
        assert rel_err < 0.20, (
            f"Empirical Var[v] ({empirical_var:.6e}) should match CIR "
            f"stationary formula theta*xi^2/(2*kappa) ({theoretical_var:.6e}) "
            f"within 20%; got {rel_err*100:.1f}% error — xi or kappa "
            f"behavior may be off."
        )


# ═══════════════════════════════════════════════════════════════════
# REWRITE: test_heston_reproducible → fat tails (core Heston economic story)
# ═══════════════════════════════════════════════════════════════════
# Council analysis:
# - Contrarian: Same-seed-same-output tests numpy's RNG, not Heston.
#   Useful as a one-line smoke but a whole class for it is overkill.
# - First Principles: The ECONOMIC reason to use Heston over GBM is
#   fat cost tails. If Heston doesn't produce fatter tails than
#   constant-vol, using it is a waste of compute.
# - Executor: reduce determinism to one assertion inside another test
#   (at top of the new class) and use the class slot for fat-tail check.
class TestHestonCostDistribution:
    """Heston cost distribution should have fatter tails than constant-vol."""

    def test_heston_fatter_tails_than_constant_vol(self):
        """With non-zero xi and negative rho, Heston should produce
        wider cost tails than equivalent constant-vol GBM.

        Measured as the ratio of (99th percentile - median) — a robust
        tail-width metric that doesn't blow up under fat-tailed
        distributions like raw kurtosis can.

        Invariant: this is the ENTIRE economic rationale for stochastic
        vol. If Heston tails aren't fatter, xi isn't doing anything
        meaningful and the simulation is effectively constant-vol.
        """
        _, _, costs_h = simulate_heston_execution(
            DEFAULT_PARAMS, HESTON_PARAMS, TWAP_X,
            n_paths=20000, seed=SEED,
        )
        _, costs_c = simulate_execution(
            DEFAULT_PARAMS, TWAP_X, n_paths=20000, seed=SEED,
            antithetic=True, scheme="exact",
        )
        # Tail spread: distance from median to 99th percentile
        tail_h = float(np.percentile(costs_h, 99) - np.median(costs_h))
        tail_c = float(np.percentile(costs_c, 99) - np.median(costs_c))
        # Heston should be at least 10% wider — very conservative bound,
        # actual ratio typically 1.2-2x
        assert tail_h > 1.10 * tail_c, (
            f"Heston tail spread ({tail_h:.2f}) should be ≥ 10% wider "
            f"than constant-vol ({tail_c:.2f}); got ratio "
            f"{tail_h/tail_c:.2f} — xi may be too small to matter, or "
            f"stochastic vol wiring is broken."
        )

    def test_seed_determinism(self):
        """One-line smoke: same seed → same output. Former
        test_heston_reproducible compressed to a quick smoke check."""
        r1 = simulate_heston_execution(DEFAULT_PARAMS, HESTON_PARAMS,
                                       TWAP_X, n_paths=100, seed=7)
        r2 = simulate_heston_execution(DEFAULT_PARAMS, HESTON_PARAMS,
                                       TWAP_X, n_paths=100, seed=7)
        np.testing.assert_array_equal(r1[2], r2[2])


# ═══════════════════════════════════════════════════════════════════
# REWRITE: test_heston_costs_finite → Feller-violation boundary handling
# ═══════════════════════════════════════════════════════════════════
# Council analysis:
# - Contrarian: `np.all(isfinite)` only catches NaN/Inf. A model that
#   silently produces wrong but finite costs passes this test.
# - Outsider: Feller-violated params are an important regime the code
#   claims to handle via "Full Truncation". But no test exercises
#   that code path — it's untested despite being documented.
# - First Principles: The documented guarantee is "v may go slightly
#   negative but S stays valid". That's the exact test to write.
class TestHestonFullTruncationAlgorithm:
    """Structural tests for the Full Truncation scheme algorithm itself.

    GLM audit found that the previous stochastic 'no NaN / no deep negatives'
    test was sophisticated theater: a buggy implementation (e.g. `max(v, -0.05)`
    instead of `max(v, 0)`, or missing `max` in drift but present in diffusion)
    could still pass it. These tests verify the algorithm's DEFINING invariant
    via Z-injection: when v_k < 0, the next step's drift = κ·θ·dt EXACTLY
    (because v_plus = max(v, 0) = 0 is used in both drift and diffusion).
    """

    def test_when_v_negative_next_step_drift_equals_kappa_theta_dt(self):
        """STRUCTURAL Z-injection test for Full Truncation.

        Inject a large negative Z_v on step 0 to force v[1] < 0, then inject
        arbitrary Z_v on step 1. By Full Truncation's definition:
            v_plus(v[1]) = max(v[1], 0) = 0
            drift   = κ·(θ - 0)·dt  = κ·θ·dt   (no dependence on v[1])
            diffusion = xi·sqrt(0)·sqrt(dt)·Z = 0   (no dependence on Z)

        So v[2] - v[1] = κ·θ·dt EXACTLY (up to float precision), regardless
        of what Z we inject on step 1. This is the algorithm's core invariant.

        Bugs this test catches that stochastic bound tests cannot:
            • `max(v, -0.05)` instead of `max(v, 0)` → drift would use -0.05
            • v_plus in diffusion but not drift → drift would use v[1] < 0
            • Missing truncation entirely → sqrt(negative) → NaN
            • Off-by-one on Z indexing → drift would respond to Z[1]
        """
        from montecarlo.sde_engine import simulate_heston_execution

        # N=2, dt=0.5 → κθdt = 2·0.04·0.5 = 0.04 (easy to see in output)
        p = ACParams(S0=100.0, sigma=0.2, mu=0.0, X0=100.0, T=1.0, N=2,
                     gamma=0.0, eta=0.0, alpha=1.0, lam=1e-6)
        h = HestonParams(kappa=2.0, theta=0.04, xi=0.8, rho=0.0, v0=0.04)
        traj = np.array([100.0, 50.0, 0.0])

        # Inject: W1 = Z_S (price Brownian), W2 (mixed into Z_v)
        # With rho=0: Z_v = W2. Step 0 uses Z_v[0]=-5 (force v[1] < 0),
        # step 1 uses Z_v[1]=+10 (should be IGNORED due to truncation).
        W1 = np.array([[0.0, 0.0]])
        W2 = np.array([[-5.0, 10.0]])

        calls = iter([W1, W2])
        mock_rng = type("MockRNG", (), {})()
        mock_rng.standard_normal = lambda shape: next(calls).copy()

        with patch("montecarlo.sde_engine.np.random.default_rng",
                   return_value=mock_rng):
            _, v, _ = simulate_heston_execution(p, h, traj, n_paths=1, seed=42)

        # Precondition: v[1] must actually be negative (else the test is vacuous)
        assert v[0, 1] < 0.0, (
            f"Test setup failure: Z=-5 should have pushed v[1] negative; "
            f"got v[1]={v[0,1]:.6e}. Adjust xi, dt, or Z magnitude."
        )

        # THE INVARIANT: v[2] - v[1] = κ·θ·dt EXACTLY when v[1] < 0
        expected_delta = h.kappa * h.theta * p.dt  # = 0.04
        actual_delta = v[0, 2] - v[0, 1]
        assert abs(actual_delta - expected_delta) < 1e-12, (
            f"Full Truncation algorithm broken: when v[1]={v[0,1]:.6e} < 0, "
            f"v[2]-v[1] should equal κ·θ·dt = {expected_delta:.6e} "
            f"(drift with v_plus=0, diffusion × 0 = 0). "
            f"Got {actual_delta:.6e} (diff: {abs(actual_delta-expected_delta):.2e}). "
            f"Likely causes: non-zero clamp floor, asymmetric truncation "
            f"(drift vs diffusion), or Z-index off-by-one."
        )

    def test_feller_violation_emits_warning_and_preserves_price_positivity(self):
        """Output-level sanity (complements the structural test above).

        When Feller is violated, the simulator should:
          (a) emit the documented warning
          (b) never produce NaN/Inf in price or variance paths
          (c) keep S > 0 on ≥ 99.9% of (path, step) pairs — the log-normal
              structure (S update uses sqrt(v_plus)*S, so S stays ≥ 0)
        Structural correctness of the truncation itself is tested above.
        """
        violating = HestonParams(kappa=2.0, theta=0.09, xi=1.0,
                                  rho=-0.3, v0=0.09)
        with pytest.warns(UserWarning, match="Feller condition violated"):
            S, v, costs = simulate_heston_execution(
                DEFAULT_PARAMS, violating, TWAP_X,
                n_paths=2000, seed=SEED,
            )
        assert np.all(np.isfinite(S)), "Price went to NaN/Inf"
        assert np.all(np.isfinite(v)), "Variance went to NaN/Inf"
        assert np.all(np.isfinite(costs)), "Cost went to NaN/Inf"
        assert (S > 0).mean() > 0.999, (
            f"Only {(S>0).mean()*100:.1f}% of prices positive"
        )


# ═══════════════════════════════════════════════════════════════════
# KEPT (already meaningful — see rewrite history docstring at top)
# ═══════════════════════════════════════════════════════════════════

class TestHestonPricePositivity:
    """Check price positivity under Euler-Maruyama (best effort)."""

    def test_heston_price_positive(self):
        """Most paths should have positive prices (Euler may rarely go negative)."""
        S, _, _ = simulate_heston_execution(
            DEFAULT_PARAMS, HESTON_PARAMS, TWAP_X, n_paths=N_PATHS, seed=SEED,
        )
        frac_positive = np.mean(S > 0)
        assert frac_positive > 0.99, (
            f"Only {frac_positive*100:.1f}% of price entries are positive "
            f"(min={S.min():.2f})"
        )


class TestHestonLimitingBehavior:
    """Compare Heston to constant-vol MC in limiting cases."""

    def test_heston_reduces_to_constant_vol(self):
        """When xi=0, kappa large, theta=v0=sigma^2, Heston reduces to
        constant-vol GBM. With 20k paths, CLT precision is ~0.7% — so
        tolerance tightened to 2% (was 15%, flagged by GLM as 'too wide
        for a 20k-path simulation of deterministic limits').

        If this fails: either the degenerate-case wiring is off (e.g.,
        Heston SDE uses sigma instead of sqrt(v)), or the two simulators
        are using different discretization schemes that shouldn't differ.
        """
        sigma = DEFAULT_PARAMS.sigma
        degenerate_heston = HestonParams(
            kappa=100.0,
            theta=sigma ** 2,
            xi=0.0,
            rho=0.0,
            v0=sigma ** 2,
        )

        _, _, costs_heston = simulate_heston_execution(
            DEFAULT_PARAMS, degenerate_heston, TWAP_X,
            n_paths=20000, seed=SEED,
        )
        _, costs_const = simulate_execution(
            DEFAULT_PARAMS, TWAP_X, n_paths=20000, seed=SEED,
            antithetic=True, scheme="euler",
        )

        mean_h = np.mean(costs_heston)
        mean_c = np.mean(costs_const)
        rel_diff = abs(mean_h - mean_c) / abs(mean_c)
        assert rel_diff < 0.02, (
            f"Degenerate Heston mean cost ({mean_h:.2f}) differs from "
            f"constant-vol Euler ({mean_c:.2f}) by {rel_diff*100:.2f}% — "
            f"with 20k paths, degenerate case should agree within 2%. "
            f"Likely cause: SDE discretization inconsistency between "
            f"simulate_heston_execution and simulate_execution."
        )

    # ═══════════════════════════════════════════════════════════════
    # REWRITE: test_heston_mean_cost_reasonable (sophisticated theater)
    #   → test_zero_xi_produces_deterministic_variance_across_paths
    # ═══════════════════════════════════════════════════════════════
    # GLM verdict: "Ratio 0.5-3.0 catches essentially nothing. You could
    # invert the temporary impact sign and this might still pass."
    #
    # Council:
    # - Contrarian: the 0.5-3.0 bound was so wide it tested only that
    #   the function returns a finite number.
    # - First Principles: when xi=0, the variance process is DETERMINISTIC:
    #   v(t) = theta + (v0 - theta)·exp(-kappa·t). Across paths at time k,
    #   Var_across_paths[v_k] should be exactly 0 (up to float precision).
    # - Executor: one-liner assertion on cross-path variance.
    def test_zero_xi_produces_deterministic_variance_across_paths(self):
        """When xi=0 (no vol-of-vol), the variance process is DETERMINISTIC:
            v(t) = theta + (v0 - theta)·exp(-kappa·t)
        So across MC paths, Var_across_paths[v_k] must be ~0 at every k.

        Catches bugs that a 0.5-3.0 ratio on mean cost could never detect:
            • Non-zero xi being added somewhere despite xi=0
            • Random Brownian W2 contaminating v when it shouldn't
            • Path correlation bugs causing per-path variance drift
        """
        zero_xi = HestonParams(kappa=2.0, theta=0.09, xi=0.0,
                               rho=0.0, v0=0.04)
        _, v, _ = simulate_heston_execution(
            DEFAULT_PARAMS, zero_xi, TWAP_X,
            n_paths=1000, seed=SEED,
        )
        # At each time step, all paths should have identical v
        var_across_paths = np.var(v, axis=0)
        max_var = float(var_across_paths.max())
        assert max_var < 1e-20, (
            f"With xi=0, variance across paths should be identically 0 "
            f"(deterministic CIR), got max Var_paths[v_k] = {max_var:.2e}. "
            f"Non-zero cross-path variance means Brownian noise is leaking "
            f"into v despite xi=0."
        )


# ═══════════════════════════════════════════════════════════════════
# NEW (GLM audit): Z-injection deterministic test
# ═══════════════════════════════════════════════════════════════════
# GLM-5.1 finding: "The suite lacks Z-injection tests for Heston.
# Stochastic tests with 10k paths and 15% tolerance cannot catch
# sign errors in the Cholesky decomposition or a misapplied sqrt(dt)."
#
# Approach: monkey-patch rng.standard_normal inside the simulator to
# return a known array of 1.0s. Then hand-compute one Euler step
# and assert the simulator matches to floating-point precision.
class TestHestonZInjection:
    """Deterministic test: inject known Z arrays, assert path matches
    hand-calculated Euler step exactly. This catches bugs that stochastic
    noise-based tests (10k paths, 15% tolerance) cannot detect.

    Implementation scheme: Euler-Maruyama (confirmed by reading sde_engine.py).
    The simulator does NOT use exact log-normal for S under Heston — it uses:
        S[k+1] = S[k] + mu*S[k]*dt + sqrt(v_plus)*S[k]*sqrt(dt)*Z_S - gamma*n_k
        v[k+1] = v[k] + kappa*(theta-v_plus)*dt + xi*sqrt(v_plus)*sqrt(dt)*Z_v
    where Z_v = rho*W1 + sqrt(1-rho^2)*W2, Z_S = W1.
    """

    # Simple param set: no impact interference, one step, round numbers
    _Z_PARAMS = ACParams(
        S0=100.0,
        sigma=0.2,   # not used in Heston S step, but required field
        mu=0.0,
        X0=100.0,
        T=1.0,
        N=1,
        gamma=0.0,
        eta=0.0,
        alpha=1.0,
        lam=1e-6,
    )
    _HESTON = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.5, v0=0.04)

    def test_one_step_euler_matches_hand_calculation(self):
        """With W1=W2=1.0 injected (Z_S=1.0, Z_v=rho+sqrt(1-rho^2)):

        Hand calculations (mu=0, gamma=0, dt=1, v0=0.04, kappa=2, theta=0.04,
        xi=0.3, rho=-0.5, S0=100, n_k=100 liquidate all in one step):

            v_plus  = max(0.04, 0) = 0.04
            sqrt_v  = sqrt(0.04)   = 0.2
            Z_S     = W1           = 1.0
            Z_v     = rho*W1 + sqrt(1-rho^2)*W2
                    = -0.5*1.0 + sqrt(0.75)*1.0
                    = -0.5 + 0.8660254...
                    = 0.3660254...

            S_1 = S0 + mu*S0*dt + sqrt(v_plus)*S0*sqrt(dt)*Z_S - gamma*n_k
                = 100 + 0 + 0.2*100*1.0*1.0 - 0
                = 120.0

            v_1 = v0 + kappa*(theta-v_plus)*dt + xi*sqrt(v_plus)*sqrt(dt)*Z_v
                = 0.04 + 2*(0.04-0.04)*1 + 0.3*0.2*1.0*0.3660254...
                = 0.04 + 0 + 0.021961524...
                = 0.061961524...

        This is the gold standard test the GLM audit flagged as missing.
        Tolerance: 1e-10 (floating-point arithmetic — no approximation).
        """
        params = self._Z_PARAMS
        hp = self._HESTON

        # Trajectory: liquidate all X0=100 in one step → n_k=[100]
        trajectory = np.array([params.X0, 0.0])

        # Hand-computed expected values
        dt = params.T / params.N   # = 1.0
        sqrt_dt = math.sqrt(dt)    # = 1.0
        v_plus = max(hp.v0, 0.0)   # = 0.04
        sqrt_v = math.sqrt(v_plus) # = 0.2
        W1_val = 1.0
        W2_val = 1.0
        Z_S_val = W1_val           # = 1.0
        Z_v_val = hp.rho * W1_val + math.sqrt(1.0 - hp.rho ** 2) * W2_val
        # = -0.5 + sqrt(0.75)

        expected_S1 = (
            params.S0
            + params.mu * params.S0 * dt
            + sqrt_v * params.S0 * sqrt_dt * Z_S_val
            - params.gamma * (trajectory[0] - trajectory[1])
        )
        expected_v1 = (
            hp.v0
            + hp.kappa * (hp.theta - v_plus) * dt
            + hp.xi * sqrt_v * sqrt_dt * Z_v_val
        )

        # Monkey-patch np.random.default_rng inside sde_engine so both W1
        # and W2 are arrays of 1.0s. The simulator calls:
        #   W1 = rng.standard_normal((n_paths, N))  → shape (1,1)
        #   W2 = rng.standard_normal((n_paths, N))  → shape (1,1)
        # numpy Generator's standard_normal is a read-only C method — we
        # cannot assign to it. Instead we return a plain mock object whose
        # standard_normal attribute is a normal Python lambda.
        import montecarlo.sde_engine as sde_module
        from unittest.mock import MagicMock

        def patched_default_rng(seed=None):
            mock_rng = MagicMock()
            mock_rng.standard_normal = lambda shape=None: (
                np.ones(shape) if shape is not None else np.float64(1.0)
            )
            return mock_rng

        with patch.object(sde_module.np.random, "default_rng", patched_default_rng):
            S, v, costs = simulate_heston_execution(
                params, hp, trajectory,
                n_paths=1, seed=0,
            )

        actual_S1 = float(S[0, 1])
        actual_v1 = float(v[0, 1])

        assert abs(actual_S1 - expected_S1) < 1e-10, (
            f"Z-injection: S_1 mismatch. "
            f"Expected {expected_S1:.10f}, got {actual_S1:.10f}. "
            f"Delta = {actual_S1 - expected_S1:.2e}. "
            f"Check Euler-Maruyama price step in sde_engine.py."
        )
        assert abs(actual_v1 - expected_v1) < 1e-10, (
            f"Z-injection: v_1 mismatch. "
            f"Expected {expected_v1:.10f}, got {actual_v1:.10f}. "
            f"Delta = {actual_v1 - expected_v1:.2e}. "
            f"Check Cholesky Z_v construction or variance Euler step."
        )
