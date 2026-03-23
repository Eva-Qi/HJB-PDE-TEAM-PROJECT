# Calibration Results — Mar 22, 2026

## Data
- Source: Binance aggTrades (data.binance.vision)
- Symbol: BTCUSDT
- Period: 2026-03-17 → 2026-03-21 (5 days)
- Total trades: 5,020,555
- Price range: $68,571 - $76,000

## Estimated Parameters

| Parameter | Value | Method |
|-----------|-------|--------|
| σ (annualized) | **0.3956 (39.6%)** | Log returns on 5-min VWAP, annualized × sqrt(365.25×24×3600/300) |
| γ (permanent impact) | **1.33e-02** | Kyle's lambda = Cov(ΔP, signed_flow) / Var(signed_flow) via Welford |

## Notes
- σ = 39.6% is reasonable for BTC (higher than equity, typical crypto range 30-80%)
- γ = 0.0133 means each unit of signed flow moves price by ~$0.013
- η (temporary impact) and α (impact exponent) not yet estimated — need order book depth data or power-law regression
- Timestamp unit is microseconds (not milliseconds) — fixed in data_loader.py
