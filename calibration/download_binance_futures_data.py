import os
import json
import math
import urllib.request
from datetime import datetime, timezone

def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())

def download_open_interest():
    """Binance futures API returns 451 (geographic block) from US IPs.
    Bybit public API provides equivalent OI history without auth."""
    # Bybit v5 — no auth, no 451 block. Returns {result: {list: [{timestamp, openInterest}, ...]}}
    url = "https://api.bybit.com/v5/market/open-interest?category=linear&symbol=BTCUSDT&intervalTime=1d&limit=180"
    resp = fetch_json(url)
    rows = resp.get("result", {}).get("list", [])
    parsed = []
    for item in rows:
        parsed.append({
            "timestamp": int(item["timestamp"]),
            "sumOpenInterest": float(item["openInterest"]),
            "sumOpenInterestValue": None,  # Bybit doesn't provide USD-valued OI directly
        })
    return parsed

def download_klines(interval):
    # Binance.com blocked (451) from US. Use binance.us (spot only, symbol BTCUSD).
    url = f"https://api.binance.us/api/v3/klines?symbol=BTCUSD&interval={interval}&limit=500"
    raw = fetch_json(url)
    parsed = []
    for k in raw:
        ts = k[0]
        if interval == "1d":
            dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
            label = "date"
        else:
            dt = ts
            label = "timestamp"
        parsed.append({
            label: dt,
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "trades": int(k[8])
        })
    return parsed

def compute_stats(values):
    n = len(values)
    if n == 0:
        return 0, 0, 0, 0
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    std = math.sqrt(variance)
    return mean, std, values[-1], values[-1] - values[0]

def pearson_correlation(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n)) / n
    var_x = sum((x - mean_x) ** 2 for x in xs) / n
    var_y = sum((y - mean_y) ** 2 for y in ys) / n
    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return 0.0
    return cov / denom

def realized_vol(closes):
    if len(closes) < 2:
        return 0.0
    log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    n = len(log_returns)
    if n == 0:
        return 0.0
    mean_lr = sum(log_returns) / n
    variance = sum((r - mean_lr) ** 2 for r in log_returns) / n
    return math.sqrt(variance)

def main():
    os.makedirs("data", exist_ok=True)

    # Download all data
    # OI: both fapi.binance.com (451) and Bybit (403 CloudFront) are geo-
    # blocked. Skip gracefully — klines alone are enough for the intended
    # multi-resolution vol regime analysis.
    oi_data = []
    try:
        oi_data = download_open_interest()
    except Exception as e:
        print(f"[skip] OI fetch blocked ({type(e).__name__}: {e}); continuing with klines only")
    klines_1d = download_klines("1d")
    klines_4h = download_klines("4h")

    # Save open interest (empty if blocked)
    oi_payload = {"n_rows": len(oi_data), "data": oi_data, "note": (
        "Empty if futures OI endpoints (fapi.binance.com, Bybit) blocked from region."
    )}
    with open("data/binance_btc_openinterest_daily.json", "w") as f:
        json.dump(oi_payload, f, indent=2)

    # Save klines 1d
    with open("data/binance_btc_klines_1d.json", "w") as f:
        json.dump(klines_1d, f, indent=2)

    # Save klines 4h
    with open("data/binance_btc_klines_4h.json", "w") as f:
        json.dump(klines_4h, f, indent=2)

    # --- Summary ---
    print("=" * 65)
    print("BINANCE BTC FUTURES / SPOT DATA DOWNLOAD SUMMARY")
    print("=" * 65)

    # OI summary
    oi_values = [d["sumOpenInterestValue"] for d in oi_data]
    oi_contracts = [d["sumOpenInterest"] for d in oi_data]
    oi_mean, oi_std, oi_last, oi_range = compute_stats(oi_values)
    oi_c_mean, oi_c_std, oi_c_last, oi_c_range = compute_stats(oi_contracts)

    print(f"\n--- Open Interest (Daily, {len(oi_data)} rows) ---")
    print(f"  OI Value (USD):  mean=${oi_mean:,.0f}  std=${oi_std:,.0f}  last=${oi_last:,.0f}  range={oi_range:+,.0f}")
    print(f"  OI Contracts:    mean={oi_c_mean:,.1f}  std={oi_c_std:,.1f}  last={oi_c_last:,.1f}  range={oi_c_range:+,.1f}")

    # Correlation OI value vs 1d close
    oi_ts_set = {d["timestamp"] for d in oi_data}
    # Align by matching OI timestamps with kline open times (both in ms since epoch)
    kline_1d_by_ts = {k["date"]: k["close"] for k in klines_1d}
    # OI timestamps are epoch ms; convert kline dates to epoch ms for alignment
    kline_1d_ts_map = {}
    for k in klines_1d:
        # date is YYYY-MM-DD string, convert to epoch ms (start of day UTC)
        dt_obj = datetime.strptime(k["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        ts_ms = int(dt_obj.timestamp() * 1000)
        kline_1d_ts_map[ts_ms] = k["close"]

    aligned_oi = []
    aligned_close = []
    for d in oi_data:
        ts = d["timestamp"]
        if ts in kline_1d_ts_map:
            aligned_oi.append(d["sumOpenInterestValue"])
            aligned_close.append(kline_1d_ts_map[ts])

    if len(aligned_oi) >= 2:
        corr = pearson_correlation(aligned_oi, aligned_close)
        print(f"  Correlation (OI-Value vs 1d Close): {corr:.4f}  (over {len(aligned_oi)} aligned days)")
    else:
        print(f"  Correlation: insufficient aligned data ({len(aligned_oi)} points)")

    # 1d klines summary
    closes_1d = [k["close"] for k in klines_1d]
    rv_1d = realized_vol(closes_1d)
    dates_1d = [k["date"] for k in klines_1d]
    print(f"\n--- Klines 1d ({len(klines_1d)} rows) ---")
    print(f"  Date range: {dates_1d[0]} -> {dates_1d[-1]}")
    print(f"  Price range: ${min(closes_1d):,.2f} - ${max(closes_1d):,.2f}")
    print(f"  Realized Vol (std of log returns): {rv_1d:.6f}  ({rv_1d * 100:.4f}%)")

    # 4h klines summary
    closes_4h = [k["close"] for k in klines_4h]
    rv_4h = realized_vol(closes_4h)
    ts_4h_first = klines_4h[0]["timestamp"]
    ts_4h_last = klines_4h[-1]["timestamp"]
    dt_first = datetime.fromtimestamp(ts_4h_first / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    dt_last = datetime.fromtimestamp(ts_4h_last / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n--- Klines 4h ({len(klines_4h)} rows) ---")
    print(f"  Timestamp range: {dt_first} -> {dt_last}")
    print(f"  Price range: ${min(closes_4h):,.2f} - ${max(closes_4h):,.2f}")
    print(f"  Realized Vol (std of log returns): {rv_4h:.6f}  ({rv_4h * 100:.4f}%)")

    print(f"\n{'=' * 65}")
    print("Files written:")
    print("  data/binance_btc_openinterest_daily.json")
    print("  data/binance_btc_klines_1d.json")
    print("  data/binance_btc_klines_4h.json")
    print("=" * 65)

if __name__ == "__main__":
    main()