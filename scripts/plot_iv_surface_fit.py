import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

from extensions.heston import fft_call_price

# --- Paths ---
CHAIN_PATH = Path(__file__).resolve().parent.parent / "data" / "deribit_btc_option_chain_20260420.json"
PARAMS_PATH = Path(__file__).resolve().parent.parent / "data" / "heston_pmeasure_vs_qmeasure.json"
FIGURE_PATH = Path(__file__).resolve().parent.parent / "figures" / "iv_fit_heatmap.png"

# --- BS helpers ---
def bs_call_price(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_delta(S, K, T, r, q, sigma, is_call=True):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if is_call:
        return float(np.exp(-q * T) * norm.cdf(d1))
    else:
        return float(-np.exp(-q * T) * norm.cdf(-d1))

def bs_iv_from_price(price, S, K, T, r, q, is_call=True):
    from scipy.optimize import brentq
    if T <= 0 or price <= 0:
        return np.nan
    if is_call:
        intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        upper_bound = S * np.exp(-q * T)
        if price <= intrinsic or price >= upper_bound:
            return np.nan
        f = lambda sigma: bs_call_price(S, K, T, r, q, sigma) - price
    else:
        intrinsic = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
        upper_bound = K * np.exp(-r * T)
        if price <= intrinsic or price >= upper_bound:
            return np.nan
        f = lambda sigma: bs_put_price(S, K, T, r, q, sigma) - price
    try:
        return brentq(f, 1e-6, 10.0, xtol=1e-7, maxiter=200)
    except (ValueError, RuntimeError):
        return np.nan

def model_iv_for_strike(S0, K, T, r, q, kappa, theta, xi, rho, v0):
    """Get Heston model IV for a single strike via FFT + BS inversion."""
    alpha = 1.0
    N = 2**12
    B = 1000.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fft_strikes, fft_prices, _ = fft_call_price(
                S0, S0, r, q, T, alpha, N, B, kappa, theta, xi, rho, v0
            )
        log_fft_k = np.log(fft_strikes)
        price = float(np.interp(np.log(K), log_fft_k, fft_prices))
        iv = bs_iv_from_price(price, S0, K, T, r, q, is_call=True)
        return iv
    except Exception:
        return np.nan

def main():
    # Load data
    with open(CHAIN_PATH, "r") as f:
        chain_data = json.load(f)
    with open(PARAMS_PATH, "r") as f:
        params_data = json.load(f)

    S0 = chain_data["underlying_price"]
    q_params = params_data["q_measure"]
    kappa = q_params["kappa"]
    theta = q_params["theta"]
    xi = q_params["xi"]
    rho = q_params["rho"]
    v0 = q_params["v0"]

    r = 0.0
    q = 0.0

    # Filter option chain: calls only, delta in [0.1, 0.9], T in [7, 180] days
    T_min = 7.0 / 365.25
    T_max = 180.0 / 365.25
    filtered = []
    for contract in chain_data["contracts"]:
        if contract["kind"] != "C":
            continue
        T_years = contract["T"]
        if T_years < T_min or T_years > T_max:
            continue
        mark_iv = contract.get("mark_iv")
        if mark_iv is None or mark_iv <= 0:
            continue
        K = contract["strike"]
        delta = bs_delta(S0, K, T_years, r, q, mark_iv, is_call=True)
        if delta < 0.1 or delta > 0.9:
            continue
        filtered.append({
            "K": K,
            "T": T_years,
            "market_iv": mark_iv,
            "K_over_S": K / S0,
        })

    print(f"Filtered {len(filtered)} call options for surface plot")

    if not filtered:
        print("ERROR: No options pass filter. Exiting.")
        return

    # Group by expiry for batch FFT
    from itertools import groupby
    filtered.sort(key=lambda x: (x["T"], x["K"]))
    expiry_groups = {}
    for T_key, group in groupby(filtered, key=lambda x: round(x["T"], 6)):
        expiry_groups[T_key] = list(group)

    # Compute model IV for each contract
    results = []
    for T_key in sorted(expiry_groups.keys()):
        group = expiry_groups[T_key]
        strikes_arr = np.array([c["K"] for c in group])
        market_ivs = np.array([c["market_iv"] for c in group])

        # Batch compute using one FFT per expiry
        alpha = 1.0
        N_fft = 2**12
        B_fft = 1000.0
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fft_strikes, fft_prices, _ = fft_call_price(
                    S0, S0, r, q, T_key, alpha, N_fft, B_fft,
                    kappa, theta, xi, rho, v0
                )
            log_fft_k = np.log(fft_strikes)
            for i, contract in enumerate(group):
                K = contract["K"]
                price = float(np.interp(np.log(K), log_fft_k, fft_prices))
                miv = bs_iv_from_price(price, S0, K, T_key, r, q, is_call=True)
                results.append({
                    "K_over_S": contract["K_over_S"],
                    "T": T_key,
                    "market_iv": contract["market_iv"],
                    "model_iv": miv if not np.isnan(miv) else np.nan,
                })
        except Exception:
            for contract in group:
                results.append({
                    "K_over_S": contract["K_over_S"],
                    "T": T_key,
                    "market_iv": contract["market_iv"],
                    "model_iv": np.nan,
                })

    # Build arrays for plotting
    K_over_S_arr = np.array([r["K_over_S"] for r in results])
    T_arr = np.array([r["T"] for r in results])
    market_iv_arr = np.array([r["market_iv"] for r in results])
    model_iv_arr = np.array([r["model_iv"] for r in results])
    residuals = model_iv_arr - market_iv_arr

    valid = ~np.isnan(model_iv_arr)
    print(f"Valid model IV points: {valid.sum()} / {len(results)}")

    if valid.sum() == 0:
        print("ERROR: No valid model IVs computed. Check Heston params.")
        return

    rmse = np.sqrt(np.mean(residuals[valid] ** 2))
    max_abs_resid = np.max(np.abs(residuals[valid]))
    print(f"RMSE: {rmse:.6f}")
    print(f"Max |residual|: {max_abs_resid:.6f}")

    # Create grid for heatmaps
    K_unique = np.sort(np.unique(K_over_S_arr[valid]))
    T_unique = np.sort(np.unique(T_arr[valid]))

    # Use pivot-like approach for scatter heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Build 2D grids for heatmaps
    K_grid = np.linspace(K_unique.min(), K_unique.max(), 40)
    T_grid = np.linspace(T_unique.min(), T_unique.max(), 40)

    from scipy.interpolate import griddata

    K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)

    # Market IV heatmap
    market_grid = griddata(
        (K_over_S_arr[valid], T_arr[valid]),
        market_iv_arr[valid],
        (K_mesh, T_mesh),
        method="linear"
    )

    # Model IV heatmap
    model_grid = griddata(
        (K_over_S_arr[valid], T_arr[valid]),
        model_iv_arr[valid],
        (K_mesh, T_mesh),
        method="linear"
    )

    # Residual heatmap
    resid_valid = residuals.copy()
    resid_for_grid = resid_valid[valid]
    resid_grid = griddata(
        (K_over_S_arr[valid], T_arr[valid]),
        resid_for_grid,
        (K_mesh, T_mesh),
        method="linear"
    )

    # Plot Market IV
    im0 = axes[0].pcolormesh(K_mesh, T_mesh * 365.25, market_grid, cmap="viridis", shading="auto")
    axes[0].set_title("Market IV (Q-measure)")
    axes[0].set_xlabel("K / S₀")
    axes[0].set_ylabel("T (days)")
    plt.colorbar(im0, ax=axes[0], label="IV")

    # Plot Model IV
    im1 = axes[1].pcolormesh(K_mesh, T_mesh * 365.25, model_grid, cmap="viridis", shading="auto")
    axes[1].set_title("Heston Model IV (Q)")
    axes[1].set_xlabel("K / S₀")
    axes[1].set_ylabel("T (days)")
    plt.colorbar(im1, ax=axes[1], label="IV")

    # Plot Residuals with diverging colormap
    abs_max = max(abs(np.nanmin(resid_grid)), abs(np.nanmax(resid_grid)))
    if abs_max == 0:
        abs_max = 0.01
    im2 = axes[2].pcolormesh(
        K_mesh, T_mesh * 365.25, resid_grid,
        cmap="RdBu_r", shading="auto", vmin=-abs_max, vmax=abs_max
    )
    axes[2].set_title("Residual (Model − Market)")
    axes[2].set_xlabel("K / S₀")
    axes[2].set_ylabel("T (days)")
    plt.colorbar(im2, ax=axes[2], label="IV residual")

    fig.suptitle(
        f"Q-Measure Heston Fit: RMSE={rmse:.4f}, Max|resid|={max_abs_resid:.4f}\n"
        f"κ={kappa:.2f}  θ={theta:.4f}  ξ={xi:.2f}  ρ={rho:.3f}  v₀={v0:.4f}",
        fontsize=11
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {FIGURE_PATH}")

if __name__ == "__main__":
    main()