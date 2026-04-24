"""
Analysis: Deribit BTC-PERPETUAL funding rate vs Realized (physical) rho
MF796 Term Project
"""

import json
import math
from collections import defaultdict
from scipy import stats

DATA_DIR = "/Users/evanolott/Desktop/MF796-COURSE PROJECT/mf796_project/data"

# ── Load data ──────────────────────────────────────────────────────────────
with open(f"{DATA_DIR}/realized_rho_daily.json") as f:
    rho_data = json.load(f)

with open(f"{DATA_DIR}/deribit_btc_funding_hourly.json") as f:
    funding_data = json.load(f)

# ── Build daily rho lookup (skip NaN) ─────────────────────────────────────
daily_rho = {}
for row in rho_data["daily_rho"]:
    date = row["date"]
    r30 = row.get("rho_30d")
    r60 = row.get("rho_60d")
    daily_rho[date] = {"rho_30d": r30, "rho_60d": r60}

# ── Aggregate funding to daily mean ───────────────────────────────────────
# timestamp is milliseconds UTC
daily_funding_sums = defaultdict(list)
for row in funding_data["data"]:
    ts_ms = row["timestamp"]
    ts_s = ts_ms / 1000.0
    # Convert to UTC date string YYYY-MM-DD
    import time
    t = time.gmtime(ts_s)
    date_str = f"{t.tm_year:04d}-{t.tm_mon:02d}-{t.tm_mday:02d}"
    daily_funding_sums[date_str].append(row["interest_1h"])

daily_funding = {}
for date, vals in daily_funding_sums.items():
    if vals:
        daily_funding[date] = sum(vals) / len(vals)

# ── TEST 1: Daily-level correlation ──────────────────────────────────────
pairs_30 = []
pairs_60 = []
overlap_dates = sorted(set(daily_rho.keys()) & set(daily_funding.keys()))

for date in overlap_dates:
    f = daily_funding[date]
    r = daily_rho[date]
    if r["rho_30d"] is not None and not math.isnan(r["rho_30d"]):
        pairs_30.append((f, r["rho_30d"]))
    if r["rho_60d"] is not None and not math.isnan(r["rho_60d"]):
        pairs_60.append((f, r["rho_60d"]))

x30, y30 = zip(*pairs_30)
x60, y60 = zip(*pairs_60)

pearson_30, p_pearson_30 = stats.pearsonr(x30, y30)
pearson_60, p_pearson_60 = stats.pearsonr(x60, y60)
spearman_30, p_spearman_30 = stats.spearmanr(x30, y30)
spearman_60, p_spearman_60 = stats.spearmanr(x60, y60)

test1 = {
    "pearson_30d": round(pearson_30, 6),
    "p_pearson_30d": round(p_pearson_30, 6),
    "spearman_30d": round(spearman_30, 6),
    "p_spearman_30d": round(p_spearman_30, 6),
    "pearson_60d": round(pearson_60, 6),
    "p_pearson_60d": round(p_pearson_60, 6),
    "spearman_60d": round(spearman_60, 6),
    "p_spearman_60d": round(p_spearman_60, 6),
    "n_days_30d": len(pairs_30),
    "n_days_60d": len(pairs_60),
}
print("TEST 1:", test1)

# ── TEST 2: Monthly-level correlation ─────────────────────────────────────
# Monthly funding: aggregate daily mean funding by YYYY-MM
monthly_funding = defaultdict(list)
for date, f in daily_funding.items():
    ym = date[:7]  # YYYY-MM
    monthly_funding[ym].append(f)

monthly_funding_mean = {ym: sum(v)/len(v) for ym, v in monthly_funding.items()}

# Use monthly_q_vs_realized, BTC only (month key does not start with ETH_)
monthly_rows = [r for r in rho_data["monthly_q_vs_realized"]
                if not r["month"].startswith("ETH_")]

m_fund = []
m_r30 = []
m_r60 = []
m_sign_fund = []
m_sign_r30 = []

for row in monthly_rows:
    month_date = row["month"]  # e.g. "2025-05-01"
    ym = month_date[:7]
    if ym not in monthly_funding_mean:
        continue
    f = monthly_funding_mean[ym]
    r30 = row.get("realized_rho_30d")
    r60 = row.get("realized_rho_60d")
    if r30 is not None and r60 is not None:
        m_fund.append(f)
        m_r30.append(r30)
        m_r60.append(r60)
        m_sign_fund.append(1 if f >= 0 else -1)
        m_sign_r30.append(1 if r30 >= 0 else -1)

p30_r, p30_p = stats.pearsonr(m_fund, m_r30)
s30_r, s30_p = stats.spearmanr(m_fund, m_r30)
p60_r, p60_p = stats.pearsonr(m_fund, m_r60)
s60_r, s60_p = stats.spearmanr(m_fund, m_r60)

# Point-biserial: sign(funding) vs realized_rho_30d
pb_r, pb_p = stats.pointbiserialr(m_sign_fund, m_r30)

test2 = {
    "pearson_30d": round(p30_r, 6),
    "p_pearson_30d": round(p30_p, 6),
    "spearman_30d": round(s30_r, 6),
    "p_spearman_30d": round(s30_p, 6),
    "pearson_60d": round(p60_r, 6),
    "p_pearson_60d": round(p60_p, 6),
    "spearman_60d": round(s60_r, 6),
    "p_spearman_60d": round(s60_p, 6),
    "point_biserial_sign_fund_vs_r30": round(pb_r, 6),
    "p_point_biserial": round(pb_p, 6),
    "n_months": len(m_fund),
}
print("TEST 2:", test2)

# ── TEST 3: Lag test ─────────────────────────────────────────────────────
# corr(funding_t, realized_rho_{t+k}) for k = -7, -3, 0, 3, 7
# Build ordered list of (date, funding, rho_30d)
import datetime

all_dates = sorted(set(daily_rho.keys()) & set(daily_funding.keys()))
# Convert to date objects for arithmetic
date_objs = [datetime.date.fromisoformat(d) for d in all_dates]

# Build dicts
f_by_date = {datetime.date.fromisoformat(d): daily_funding[d] for d in all_dates}
r30_by_date = {}
for d in all_dates:
    v = daily_rho[d]["rho_30d"]
    if v is not None and not math.isnan(v):
        r30_by_date[datetime.date.fromisoformat(d)] = v

lag_results = []
for k in [-7, -3, 0, 3, 7]:
    pairs = []
    for dt in date_objs:
        target_dt = dt + datetime.timedelta(days=k)
        if dt in f_by_date and target_dt in r30_by_date:
            pairs.append((f_by_date[dt], r30_by_date[target_dt]))
    if len(pairs) > 10:
        fx, ry = zip(*pairs)
        r, p = stats.pearsonr(fx, ry)
        lag_results.append({"k_days": k, "pearson": round(r, 6), "p": round(p, 6), "n": len(pairs)})
    else:
        lag_results.append({"k_days": k, "pearson": None, "p": None, "n": len(pairs)})

print("TEST 3:", lag_results)

# ── TEST 4: Quartile conditional ─────────────────────────────────────────
# Split by |funding| quartile: top 25% vs bottom 25%
abs_funding = [(abs(f), f, d) for d, f in daily_funding.items() if d in daily_rho and daily_rho[d]["rho_30d"] is not None and not math.isnan(daily_rho[d]["rho_30d"])]
abs_funding.sort()
n = len(abs_funding)
q1_end = n // 4
q3_start = 3 * n // 4

bottom_q = abs_funding[:q1_end]
top_q = abs_funding[q3_start:]

bottom_rho = [daily_rho[d]["rho_30d"] for _, _, d in bottom_q]
top_rho = [daily_rho[d]["rho_30d"] for _, _, d in top_q]

bottom_mean = sum(bottom_rho) / len(bottom_rho)
top_mean = sum(top_rho) / len(top_rho)
t_stat, t_p = stats.ttest_ind(top_rho, bottom_rho)

test4 = {
    "top_quartile_abs_funding_rho_mean": round(top_mean, 6),
    "bottom_quartile_abs_funding_rho_mean": round(bottom_mean, 6),
    "diff": round(top_mean - bottom_mean, 6),
    "t_test_p": round(t_p, 6),
    "n_top": len(top_rho),
    "n_bottom": len(bottom_rho),
}
print("TEST 4:", test4)

# ── Verdict ───────────────────────────────────────────────────────────────
# Determine significance based on test results
daily_significant = (abs(pearson_30) > 0.1 and p_pearson_30 < 0.05) or (abs(pearson_60) > 0.1 and p_pearson_60 < 0.05)
monthly_significant = (abs(p30_r) > 0.3 and p30_p < 0.10)
conditional_significant = abs(test4["diff"]) > 0.05 and t_p < 0.05

if daily_significant and monthly_significant:
    verdict = "YES (moderate): funding shows statistically significant correlation with realized rho at daily level; monthly signal weaker at N=12"
elif daily_significant:
    verdict = "YES (weak): daily Pearson significant but small magnitude; monthly N=12 too small to confirm"
elif monthly_significant:
    verdict = "YES (weak): monthly signal exists but daily correlation near zero — may be spurious at N=12"
elif conditional_significant:
    verdict = "WEAK: no linear correlation; extreme funding episodes show different realized rho distribution"
else:
    verdict = "NO: funding rate does not predict realized rho at daily or monthly level across all tests"

results = {
    "test1_daily": test1,
    "test2_monthly": test2,
    "test3_lag": lag_results,
    "test4_quartile": test4,
    "verdict": verdict,
}

out_path = f"{DATA_DIR}/funding_vs_realized_rho_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults written to {out_path}")
print(f"\nVERDICT: {verdict}")
