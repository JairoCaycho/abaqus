# demo.py
from datetime import date
import numpy as np

from fxswap_core import (
    ZeroCurve,
    BasisCurve,
    Market,
    FXSwapTrade,
    add_tenor,
    usd_per_foreign_spot,
    normalize_strike_usd_per_foreign,
    forward_usd_per_foreign,
    fx_swap_pv_usd,
    bucketed_risks_bp,
    pnl_explain,
)

asof = date(2025, 8, 11)

# Pillars
pillars = ["1W", "1M", "3M", "6M", "1Y", "2Y"]


def make_curve(ref, tenor_rates):
    ds = [add_tenor(ref, t) for t in tenor_rates.keys()]
    zs = np.array(list(tenor_rates.values())) / 100.0
    return ZeroCurve(ref, ds, zs, "ACT/360")


def make_basis(ref, tenor_basis_bp):
    ds = [add_tenor(ref, t) for t in tenor_basis_bp.keys()]
    bps = np.array(list(tenor_basis_bp.values()), dtype=float)
    return BasisCurve(ref, ds, bps)


# USD OIS (illustrative)
usd_ois = make_curve(
    asof, {"1W": 5.35, "1M": 5.30, "3M": 5.20, "6M": 4.95, "1Y": 4.60, "2Y": 4.15}
)

# EUR OIS
eur_ois = make_curve(
    asof, {"1W": 3.55, "1M": 3.50, "3M": 3.35, "6M": 3.05, "1Y": 2.75, "2Y": 2.25}
)

# GBP OIS
gbp_ois = make_curve(
    asof, {"1W": 4.75, "1M": 4.70, "3M": 4.55, "6M": 4.25, "1Y": 3.95, "2Y": 3.45}
)

# JPY OIS
jpy_ois = make_curve(
    asof, {"1W": 0.35, "1M": 0.36, "3M": 0.38, "6M": 0.40, "1Y": 0.45, "2Y": 0.55}
)

# Cross-currency basis (bp, foreign vs USD): negative means foreign curve is raised
eurusd_basis = make_basis(
    asof, {"1W": -3, "1M": -5, "3M": -8, "6M": -11, "1Y": -14, "2Y": -18}
)
gbpusd_basis = make_basis(
    asof, {"1W": -2, "1M": -3, "3M": -5, "6M": -7, "1Y": -9, "2Y": -12}
)
usdjpy_basis = make_basis(
    asof, {"1W": -10, "1M": -14, "3M": -18, "6M": -22, "1Y": -28, "2Y": -35}
)
# Note: for USDJPY we’ll still treat foreign=JPY and add basis to JPY curve.

# Spots (street quotes)
spot_eurusd = 1.0950
spot_gbpusd = 1.2850
spot_usdjpy = 157.00
s_usd_per_eur = usd_per_foreign_spot("EURUSD", spot_eurusd)  # 1.0950
s_usd_per_gbp = usd_per_foreign_spot("GBPUSD", spot_gbpusd)  # 1.2850
s_usd_per_jpy = usd_per_foreign_spot("USDJPY", spot_usdjpy)  # ~0.006369

# Build Markets per pair
mkt_eurusd = Market(s_usd_per_eur, usd_ois, eur_ois, eurusd_basis)
mkt_gbpusd = Market(s_usd_per_gbp, usd_ois, gbp_ois, gbpusd_basis)
mkt_usdjpy = Market(s_usd_per_jpy, usd_ois, jpy_ois, usdjpy_basis)

# Trades (one per pair)
far_6m = add_tenor(asof, "6M")
far_3m = add_tenor(asof, "3M")
far_1y = add_tenor(asof, "1Y")
near_spot = asof  # simplifying

# Compute on-market forwards for strikes
F_eur_6m = forward_usd_per_foreign(mkt_eurusd, far_6m)
F_gbp_3m = forward_usd_per_foreign(mkt_gbpusd, far_3m)
F_jpy_1y = forward_usd_per_foreign(mkt_usdjpy, far_1y)

# Create slightly off-market strikes (to get nonzero PV)
K_eur_6m = F_eur_6m - 0.0002  # -2 pips in EURUSD terms
K_gbp_3m = F_gbp_3m + 0.0005  # +5 pips
K_jpy_1y = F_jpy_1y - 0.00003  # ~ -0.3 pip in USD/JPY inverted units

t_eur = FXSwapTrade(
    "EURUSD",
    notional_foreign=10_000_000,
    near_date=near_spot,
    far_date=far_6m,
    strike_usd_per_foreign=K_eur_6m,
)
t_gbp = FXSwapTrade(
    "GBPUSD",
    notional_foreign=8_000_000,
    near_date=near_spot,
    far_date=far_3m,
    strike_usd_per_foreign=K_gbp_3m,
)
t_jpy = FXSwapTrade(
    "USDJPY",
    notional_foreign=1_000_000_000,
    near_date=near_spot,
    far_date=far_1y,
    strike_usd_per_foreign=K_jpy_1y,
)

for name, trade, mkt in [
    ("EURUSD", t_eur, mkt_eurusd),
    ("GBPUSD", t_gbp, mkt_gbpusd),
    ("USDJPY", t_jpy, mkt_usdjpy),
]:
    F = forward_usd_per_foreign(mkt, trade.far_date)
    PV = fx_swap_pv_usd(trade, mkt)
    print(
        f"{name} Forward={F:.6f}, Points={F - mkt.spot_usd_per_foreign:+.6f}, PV_USD={PV:,.2f}"
    )

    risks = bucketed_risks_bp(trade, mkt, 1.0)
    print(f"  IR01 USD buckets (per bp): {np.round(risks['IR01_USD_buckets'], 2)}")
    print(f"  IR01 Foreign buckets (per bp): {np.round(risks['IR01_FOR_buckets'], 2)}")
    print(f"  Basis01 buckets (per bp): {np.round(risks['BASIS01_buckets'], 2)}\n")

# P&L Explain: simulate day+1 curves
asof_t1 = date(2025, 8, 12)
# Small realistic shifts
usd_ois_t1 = usd_ois.parallel_bump(+1.5)  # +1.5 bp
eur_ois_t1 = eur_ois.parallel_bump(-0.5)  # -0.5 bp
gbp_ois_t1 = gbp_ois.parallel_bump(+0.0)
jpy_ois_t1 = jpy_ois.parallel_bump(+2.0)
eurusd_basis_t1 = eurusd_basis.parallel_bump(+0.5)
gbpusd_basis_t1 = gbpusd_basis.parallel_bump(0.0)
usdjpy_basis_t1 = usdjpy_basis.parallel_bump(-0.5)

mkt_eurusd_t1 = Market(s_usd_per_eur, usd_ois_t1, eur_ois_t1, eurusd_basis_t1)
mkt_gbpusd_t1 = Market(s_usd_per_gbp, usd_ois_t1, gbp_ois_t1, gbpusd_basis_t1)
mkt_usdjpy_t1 = Market(s_usd_per_jpy, usd_ois_t1, jpy_ois_t1, usdjpy_basis_t1)

for name, trade, mkt_t, mkt_t1 in [
    ("EURUSD", t_eur, mkt_eurusd, mkt_eurusd_t1),
    ("GBPUSD", t_gbp, mkt_gbpusd, mkt_gbpusd_t1),
    ("USDJPY", t_jpy, mkt_usdjpy, mkt_usdjpy_t1),
]:
    explain = pnl_explain(trade, mkt_t, mkt_t1, asof, asof_t1)
    print(f"{name} P&L Explain (USD):")
    for k, v in explain.items():
        print(f"  {k:>12}: {v:,.2f}")
    print("")
