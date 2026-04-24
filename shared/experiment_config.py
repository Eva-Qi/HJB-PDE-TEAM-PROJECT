"""Shared experiment configuration — single source of truth.

These constants were previously duplicated across 10+ scripts. Centralizing
them prevents drift (e.g., the T=1/24 unit bug that went unnoticed in 13
scripts because each had its own local T).

Import pattern:
    from shared.experiment_config import N_STEPS, LAM, SEED, T_1H, \
        N_PATHS_FAST, N_PATHS_HIRES, BOOTSTRAP_REPS, ALPHA_STAT

Do NOT add script-specific constants here (e.g., VOL_REGIMES, FALLBACK_*,
X0_VALUES) — those belong in the script that owns them.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Time / calendar
# ---------------------------------------------------------------------------
SECONDS_PER_YEAR = 365.25 * 24 * 3600

# Execution horizons (years). Crypto is 24/7 so 1 year = 365.25 * 24 hours.
T_1H = 1.0 / (365.25 * 24)        # one hour
T_6H = 6.0 / (365.25 * 24)        # six hours
T_1D = 1.0 / 365.25               # one day

# ---------------------------------------------------------------------------
# MC + PDE discretization
# ---------------------------------------------------------------------------
N_STEPS = 250                      # up from 50; convergence test showed N=50
                                   # bias was 0.028% but N=250 keeps numbers
                                   # consistent across all experiments

N_PATHS_FAST  = 10_000
N_PATHS_HIRES = 100_000
BOOTSTRAP_REPS = 5_000

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
SEED       = 42                    # reproducibility seed (Douglas Adams)
ALPHA_STAT = 0.05                  # significance level for hypothesis tests

# ---------------------------------------------------------------------------
# Model defaults (calibrated, not synthetic)
# ---------------------------------------------------------------------------
# Risk aversion calibrated from Binance BTCUSDT data. NOTE: this is
# different from DEFAULT_PARAMS.lam=4e-7 in shared/params.py, which is
# the synthetic textbook value. Scripts doing real-data experiments use
# this LAM=1e-6.
LAM = 1e-6
