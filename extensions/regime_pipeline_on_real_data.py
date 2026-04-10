# Regime-aware execution pipeline on real Binance data.
# Fits a 2-state Gaussian HMM on 5-min log returns, then solves the
# Almgren-Chriss HJB separately for risk-on and risk-off regimes.
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration.data_loader import load_trades, compute_mid_prices
from extensions.regime import fit_hmm, regime_aware_params
from shared.params import DEFAULT_PARAMS
from pde.hjb_solver import solve_hjb, extract_optimal_trajectory


# HMM / regime pipeline overrides
PIPELINE_FREQ = "5min"
PIPELINE_X0 = 10.0         # 10 BTC — calibrated order size
PIPELINE_TERMINAL_PENALTY = 1e4
PIPELINE_GRID_M = 200


def load_real_returns(
    aggtrades_dir: Path,
    freq: str = PIPELINE_FREQ,
) -> tuple[pd.DataFrame, np.ndarray]:
    trades = load_trades(aggtrades_dir)
    mid = compute_mid_prices(trades, freq=freq).copy()

    mid["log_mid"] = np.log(mid["mid_price"])
    mid["log_return"] = mid["log_mid"].diff()
    mid = mid.dropna().reset_index(drop=True)

    returns = mid["log_return"].to_numpy()
    return mid, returns


def annualized_sigma_from_returns(returns: np.ndarray, freq_minutes: int = 5) -> float:
    bars_per_year = int(365 * 24 * 60 / freq_minutes)
    return float(np.std(returns, ddof=1) * np.sqrt(bars_per_year))


def build_base_params(mid: pd.DataFrame, returns: np.ndarray):
    s0 = float(mid["mid_price"].iloc[-1])
    sigma_ann = annualized_sigma_from_returns(returns, freq_minutes=5)

    base_params = replace(
        DEFAULT_PARAMS,
        S0=s0,
        sigma=sigma_ann,
        X0=PIPELINE_X0,
    )
    return base_params


def get_regime_dict(regimes):
    return {reg.label: reg for reg in regimes}


def solve_regime_trajectory(params, M: int = PIPELINE_GRID_M, terminal_penalty: float = PIPELINE_TERMINAL_PENALTY):
    grid, V, v_star = solve_hjb(params, M=M, terminal_penalty=terminal_penalty)
    x_star = extract_optimal_trajectory(grid, v_star, params)
    return grid, V, v_star, x_star


def summarize_trajectory(x: np.ndarray, name: str) -> dict:
    trades = x[:-1] - x[1:]
    first_trade = float(trades[0])
    total_trade = float(trades.sum())

    half_inventory = 0.5 * x[0]
    half_idx = int(np.argmax(x <= half_inventory))
    if x[half_idx] > half_inventory:
        half_idx = len(x) - 1

    return {
        "name": name,
        "initial_inventory": float(x[0]),
        "first_step_trade": first_trade,
        "fraction_traded_first_step": first_trade / x[0],
        "half_liquidation_step": half_idx,
        "terminal_inventory": float(x[-1]),
        "total_traded": total_trade,
    }


def print_regime_summary(regimes, states):
    print("\n=== HMM regimes ===")
    for reg in regimes:
        print(reg)

    print("\n=== State counts ===")
    unique, counts = np.unique(states, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"state {u}: {c}")

    print("\n=== Label summary ===")
    reg_map = {reg.label: reg for reg in regimes}
    print("risk_on sigma scale :", reg_map["risk_on"].sigma)
    print("risk_off sigma scale:", reg_map["risk_off"].sigma)
    print("risk_on gamma scale :", reg_map["risk_on"].gamma)
    print("risk_off gamma scale:", reg_map["risk_off"].gamma)
    print("risk_on eta scale   :", reg_map["risk_on"].eta)
    print("risk_off eta scale  :", reg_map["risk_off"].eta)


def print_trajectory_summary(summary: dict):
    print(f"\n=== {summary['name']} trajectory summary ===")
    print("initial_inventory        =", summary["initial_inventory"])
    print("first_step_trade         =", summary["first_step_trade"])
    print("fraction_traded_first    =", summary["fraction_traded_first_step"])
    print("half_liquidation_step    =", summary["half_liquidation_step"])
    print("terminal_inventory       =", summary["terminal_inventory"])
    print("total_traded             =", summary["total_traded"])


def main():
    aggtrades_dir = PROJECT_ROOT / "data"
    if not aggtrades_dir.exists():
        raise FileNotFoundError(
            f"{aggtrades_dir} does not exist. Download aggTrades first."
        )

    mid, returns = load_real_returns(aggtrades_dir, freq=PIPELINE_FREQ)

    print("=== Real data loaded ===")
    print("rows in mid-price series =", len(mid))
    print("rows in returns series   =", len(returns))
    print("start timestamp          =", mid["timestamp"].iloc[0])
    print("end timestamp            =", mid["timestamp"].iloc[-1])
    print("last mid price           =", float(mid["mid_price"].iloc[-1]))

    base_params = build_base_params(mid, returns)

    print("\n=== Base params used for regime conditioning ===")
    print(base_params)

    regimes, states = fit_hmm(returns)
    print_regime_summary(regimes, states)

    reg_dict = get_regime_dict(regimes)
    risk_on_params = regime_aware_params(base_params, reg_dict["risk_on"])
    risk_off_params = regime_aware_params(base_params, reg_dict["risk_off"])

    print("\n=== Regime-adjusted params ===")
    print("risk_on :", risk_on_params)
    print("risk_off:", risk_off_params)

    _, _, _, x_base = solve_regime_trajectory(
        base_params,
        M=PIPELINE_GRID_M,
        terminal_penalty=PIPELINE_TERMINAL_PENALTY,
    )
    _, _, _, x_on = solve_regime_trajectory(
        risk_on_params,
        M=PIPELINE_GRID_M,
        terminal_penalty=PIPELINE_TERMINAL_PENALTY,
    )
    _, _, _, x_off = solve_regime_trajectory(
        risk_off_params,
        M=PIPELINE_GRID_M,
        terminal_penalty=PIPELINE_TERMINAL_PENALTY,
    )

    base_summary = summarize_trajectory(x_base, "base")
    on_summary = summarize_trajectory(x_on, "risk_on")
    off_summary = summarize_trajectory(x_off, "risk_off")

    print_trajectory_summary(base_summary)
    print_trajectory_summary(on_summary)
    print_trajectory_summary(off_summary)

    print("\n=== Aggressiveness check ===")
    print(
        "risk_off first trade > risk_on first trade:",
        off_summary["first_step_trade"] > on_summary["first_step_trade"],
    )
    print(
        "risk_off half-liquidation earlier than risk_on:",
        off_summary["half_liquidation_step"] < on_summary["half_liquidation_step"],
    )

    t_grid = np.linspace(0.0, base_params.T, base_params.N + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(t_grid, x_base, label="Base")
    plt.plot(t_grid, x_on, label="Risk-on")
    plt.plot(t_grid, x_off, label="Risk-off")
    plt.xlabel("Time")
    plt.ylabel("Inventory")
    plt.title("Regime-aware optimal execution trajectories")
    plt.legend()
    plt.tight_layout()
    plt.show()

    state_df = mid[["timestamp"]].copy()
    state_df["state"] = states
    state_df["mid_price"] = mid["mid_price"].to_numpy()
    state_df["log_return"] = returns

    out_dir = PROJECT_ROOT / "data"
    state_df.to_csv(out_dir / "hmm_states_real_data.csv", index=False)

    traj_df = pd.DataFrame({
        "t": t_grid,
        "x_base": x_base,
        "x_risk_on": x_on,
        "x_risk_off": x_off,
    })
    traj_df.to_csv(out_dir / "regime_trajectories.csv", index=False)

    print("\n=== Files written ===")
    print(out_dir / "hmm_states_real_data.csv")
    print(out_dir / "regime_trajectories.csv")


if __name__ == "__main__":
    main()