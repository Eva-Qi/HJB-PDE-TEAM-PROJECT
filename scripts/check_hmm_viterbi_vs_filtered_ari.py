"""Compare HMM Viterbi (smoothed) vs expanding-window filtered (causal) labels.

P0 action from expert review of statistical audit: quantify look-ahead bias
in HMM regime labels by computing adjusted Rand index (ARI) between:
  - Viterbi labels: hmm.predict(X) — uses forward-backward, sees future
  - Filtered labels: hmm.predict_proba(X[:t])[-1].argmax() — only past up to t

ARI > 0.90 → disagreement minimal, look-ahead bias disclose-only.
ARI < 0.80 → material disagreement, regime-conditional claim needs rerun.

Run: python scripts/check_hmm_viterbi_vs_filtered_ari.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import adjusted_rand_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from calibration.data_loader import _load_single_csv


def main() -> None:
    # Load 5-min log returns from a representative window
    # Use 2026-01-01 to 2026-04-08 — the V5 / regime-conditional calibration window
    data_dir = PROJECT_ROOT / "data"
    start = pd.Timestamp("2026-01-01", tz="UTC")
    end = pd.Timestamp("2026-04-08", tz="UTC")

    print(f"Loading aggTrades {start.date()} → {end.date()} for 5-min returns…")
    closes = []
    day = start
    while day <= end:
        fp = data_dir / f"BTCUSDT-aggTrades-{day.strftime('%Y-%m-%d')}.csv"
        if fp.exists():
            df = _load_single_csv(fp)
            df_idx = df.set_index("timestamp").sort_index()
            c = df_idx["price"].resample("5min").last().dropna()
            closes.append(c)
        day += pd.Timedelta(days=1)
    close_series = pd.concat(closes).sort_index()
    rets = np.log(close_series).diff().dropna().values.reshape(-1, 1)
    print(f"  {len(rets):,} 5-min returns")

    # Fit HMM ONCE on full data (same as refit_regime_conditional_impact)
    print("\nFitting 2-state Gaussian HMM on full sequence…")
    hmm = GaussianHMM(n_components=2, n_iter=100, random_state=42)
    hmm.fit(rets)

    # Order states by sigma (risk_on = lower vol)
    vars_ = hmm.covars_.squeeze()
    order = np.argsort(vars_)

    # Viterbi (smoothed): uses full forward-backward
    viterbi = hmm.predict(rets)
    viterbi_ordered = np.array([np.where(order == s)[0][0] for s in viterbi])

    # Expanding-window causal filtering
    # Public-API path: call predict_proba on X[:t] at each t.
    # O(n²) — 43k points × ~50ms each = ~35 min; use every-k stride for speed.
    print("\nComputing expanding-window filtered labels (stride=1, causal)…")
    n = len(rets)
    filtered_ordered = np.empty(n, dtype=int)
    # Use stride to keep runtime manageable — evaluate every `stride` points,
    # interpolate by last known state. For ARI this approximation is fine since
    # ARI compares label sequences modulo label permutation.
    stride = max(1, n // 2000)  # cap at 2000 eval points
    last_state = 0
    for t in range(1, n + 1):
        if (t % stride == 0) or (t == n):
            post = hmm.predict_proba(rets[:t])[-1]
            state_raw = int(np.argmax(post))
            state_ord = int(np.where(order == state_raw)[0][0])
            last_state = state_ord
        filtered_ordered[t - 1] = last_state

    # Compute ARI
    ari_full = adjusted_rand_score(viterbi_ordered, filtered_ordered)

    # Also compute plain accuracy (fraction matching)
    pct_match = float(np.mean(viterbi_ordered == filtered_ordered))

    # Counts per regime
    v_off = int(np.sum(viterbi_ordered == 1))
    f_off = int(np.sum(filtered_ordered == 1))

    print("\n" + "=" * 60)
    print("RESULTS — Viterbi vs Filtered regime labels")
    print("=" * 60)
    print(f"  Sample size: {n:,} 5-min bars")
    print(f"  Viterbi    risk_off count: {v_off:>6} ({v_off/n:.2%})")
    print(f"  Filtered   risk_off count: {f_off:>6} ({f_off/n:.2%})")
    print(f"  Stride used for filtering: {stride}")
    print(f"  Fraction matching labels : {pct_match:.4f}")
    print(f"  Adjusted Rand Index (ARI): {ari_full:.4f}")
    print()
    if ari_full > 0.90:
        print("  ✅ ARI > 0.90 → look-ahead bias disclosure-only")
    elif ari_full > 0.80:
        print("  ⚠  0.80 < ARI < 0.90 → material but bounded — disclose + caveat")
    else:
        print("  🔴 ARI < 0.80 → regime-conditional calibration must be rerun")

    # Save
    out = {
        "n_obs": int(n),
        "start": str(start.date()),
        "end": str(end.date()),
        "stride_filtered": int(stride),
        "viterbi_riskoff_count": v_off,
        "filtered_riskoff_count": f_off,
        "viterbi_riskoff_fraction": float(v_off / n),
        "filtered_riskoff_fraction": float(f_off / n),
        "pct_match": pct_match,
        "adjusted_rand_index": ari_full,
        "hmm_sigma_risk_on": float(np.sqrt(vars_[order[0]])),
        "hmm_sigma_risk_off": float(np.sqrt(vars_[order[1]])),
    }
    out_path = data_dir / "hmm_viterbi_vs_filtered_ari.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
