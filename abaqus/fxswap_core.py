# fxswap_core.py
from dataclasses import dataclass, replace
from datetime import date, timedelta
import math
import numpy as np


# ---------- Utilities ----------
def year_frac(d0: date, d1: date, basis: str = "ACT/360") -> float:
    days = (d1 - d0).days
    if basis.upper() == "ACT/360":
        return days / 360.0
    elif basis.upper() == "ACT/365":
        return days / 365.0
    else:
        raise ValueError("Unsupported day count")


def add_tenor(d: date, tenor: str) -> date:
    # Simple approximation: W=7d, M=30d, Y=365d
    unit = tenor[-1].upper()
    n = int(tenor[:-1])
    if unit == "D":
        return d + timedelta(days=n)
    if unit == "W":
        return d + timedelta(days=7 * n)
    if unit == "M":
        return d + timedelta(days=30 * n)
    if unit == "Y":
        return d + timedelta(days=365 * n)
    raise ValueError("Tenor like 1W, 3M, 6M, 1Y")


def lin_interp(x, xp, fp):
    return float(np.interp(x, xp, fp))


# ---------- Curves ----------
@dataclass
class ZeroCurve:
    ref_date: date
    pillars: list[date]  # strictly increasing
    zeros: np.ndarray  # continuous-comp zero rates per pillar (decimal)
    day_count: str = "ACT/360"

    def times(self):
        return np.array(
            [year_frac(self.ref_date, d, self.day_count) for d in self.pillars],
            dtype=float,
        )

    def zero_at_t(self, t: float) -> float:
        T = self.times()
        Z = self.zeros
        t = max(0.0, min(t, T[-1]))
        return lin_interp(t, T, Z)

    def df(self, T_date: date) -> float:
        t = year_frac(self.ref_date, T_date, self.day_count)
        z = self.zero_at_t(t)
        return math.exp(-z * t)

    def parallel_bump(self, bp: float):
        return replace(self, zeros=self.zeros + bp / 1e4)

    def bucket_bump(self, idx: int, bp: float):
        z = self.zeros.copy()
        z[idx] += bp / 1e4
        return replace(self, zeros=z)


@dataclass
class BasisCurve:
    ref_date: date
    pillars: list[date]
    basis_bp: np.ndarray  # in bp per pillar

    def times(self):
        return np.array(
            [year_frac(self.ref_date, d, "ACT/360") for d in self.pillars], dtype=float
        )

    def basis_at_t(self, t: float) -> float:
        T = self.times()
        B = self.basis_bp
        t = max(0.0, min(t, T[-1]))
        return lin_interp(t, T, B)

    def parallel_bump(self, bp: float):
        return replace(self, basis_bp=self.basis_bp + bp)

    def bucket_bump(self, idx: int, bp: float):
        b = self.basis_bp.copy()
        b[idx] += bp
        return replace(self, basis_bp=b)


def foreign_df_with_basis(
    f_curve: ZeroCurve, b_curve: BasisCurve, T_date: date
) -> float:
    t = year_frac(f_curve.ref_date, T_date, f_curve.day_count)
    zf = f_curve.zero_at_t(t)
    b_dec = b_curve.basis_at_t(t) / 1e4
    return math.exp(-(zf + b_dec) * t)


# ---------- FX helpers ----------
def usd_per_foreign_spot(pair: str, spot_quote: float) -> float:
    pair = pair.upper()
    if pair in ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"):
        return spot_quote  # already USD per foreign
    if pair in ("USDJPY", "USDCNH", "USDMXN", "USDCHF"):
        return 1.0 / spot_quote  # invert to USD per foreign (foreign is JPY, CNH, etc.)
    raise ValueError("Pair not supported in helper")


def normalize_strike_usd_per_foreign(pair: str, K_quote: float) -> float:
    return usd_per_foreign_spot(pair, K_quote)


# ---------- Pricer ----------
@dataclass
class FXSwapTrade:
    pair: str  # e.g., "EURUSD", "GBPUSD", "USDJPY"
    notional_foreign: float  # amount in foreign currency units
    near_date: date
    far_date: date
    strike_usd_per_foreign: float


@dataclass
class Market:
    spot_usd_per_foreign: float
    usd_curve: ZeroCurve
    foreign_curve: ZeroCurve
    basis_curve: BasisCurve


def forward_usd_per_foreign(mkt: Market, far_date: date) -> float:
    Dd = mkt.usd_curve.df(far_date)
    Df_b = foreign_df_with_basis(mkt.foreign_curve, mkt.basis_curve, far_date)
    return mkt.spot_usd_per_foreign * (Df_b / Dd)


def fx_swap_pv_usd(trade: FXSwapTrade, mkt: Market) -> float:
    # PV of off-market forward (spot legs net out economically)
    Dd = mkt.usd_curve.df(trade.far_date)
    F = forward_usd_per_foreign(mkt, trade.far_date)
    return trade.notional_foreign * Dd * (F - trade.strike_usd_per_foreign)


# ---------- Risks ----------
def bucketed_risks_bp(trade: FXSwapTrade, mkt: Market, bump_bp: float = 1.0):
    base = fx_swap_pv_usd(trade, mkt)
    # USD curve buckets
    ir01_usd = []
    for i in range(len(mkt.usd_curve.pillars)):
        bumped = Market(
            mkt.spot_usd_per_foreign,
            mkt.usd_curve.bucket_bump(i, bump_bp),
            mkt.foreign_curve,
            mkt.basis_curve,
        )
        ir01_usd.append((fx_swap_pv_usd(trade, bumped) - base) / bump_bp)
    # Foreign curve buckets
    ir01_for = []
    for i in range(len(mkt.foreign_curve.pillars)):
        bumped = Market(
            mkt.spot_usd_per_foreign,
            mkt.usd_curve,
            mkt.foreign_curve.bucket_bump(i, bump_bp),
            mkt.basis_curve,
        )
        ir01_for.append((fx_swap_pv_usd(trade, bumped) - base) / bump_bp)
    # Basis buckets
    b01 = []
    for i in range(len(mkt.basis_curve.pillars)):
        bumped = Market(
            mkt.spot_usd_per_foreign,
            mkt.usd_curve,
            mkt.foreign_curve,
            mkt.basis_curve.bucket_bump(i, bump_bp),
        )
        b01.append((fx_swap_pv_usd(trade, bumped) - base) / bump_bp)
    return {
        "PV_USD": base,
        "IR01_USD_buckets": np.array(ir01_usd),
        "IR01_FOR_buckets": np.array(ir01_for),
        "BASIS01_buckets": np.array(b01),
    }


# ---------- Carry/Roll and Daily P&L Explain ----------
def carry_roll(trade: FXSwapTrade, mkt_t: Market, asof_t: date, asof_t1: date) -> float:
    # Curves fixed at t; advance valuation date by 1 day (approx via shorter maturity)
    # Re-create a trade with same far_date but we measure PV as of t1 using same curve functions.
    # Approximation: reuse same DFs relative to original ref_date by moving ref_date forward one day.
    def shift_curve_ref(curve: ZeroCurve, new_ref: date) -> ZeroCurve:
        return ZeroCurve(
            ref_date=new_ref,
            pillars=curve.pillars,
            zeros=curve.zeros,
            day_count=curve.day_count,
        )

    def shift_basis_ref(b: BasisCurve, new_ref: date) -> BasisCurve:
        return BasisCurve(ref_date=new_ref, pillars=b.pillars, basis_bp=b.basis_bp)

    mkt_fixed = Market(
        mkt_t.spot_usd_per_foreign,
        shift_curve_ref(mkt_t.usd_curve, asof_t1),
        shift_curve_ref(mkt_t.foreign_curve, asof_t1),
        shift_basis_ref(mkt_t.basis_curve, asof_t1),
    )
    return fx_swap_pv_usd(trade, mkt_fixed) - fx_swap_pv_usd(trade, mkt_t)


def pnl_explain(
    trade: FXSwapTrade, mkt_t: Market, mkt_t1: Market, asof_t: date, asof_t1: date
):
    pv_t = fx_swap_pv_usd(trade, mkt_t)
    pv_t1 = fx_swap_pv_usd(trade, mkt_t1)

    cr = carry_roll(trade, mkt_t, asof_t, asof_t1)

    # USD move
    m_usd = Market(
        mkt_t.spot_usd_per_foreign,
        mkt_t1.usd_curve,
        mkt_t.foreign_curve,
        mkt_t.basis_curve,
    )
    d_usd = fx_swap_pv_usd(trade, m_usd) - fx_swap_pv_usd(trade, mkt_t)

    # Foreign move
    m_for = Market(
        mkt_t.spot_usd_per_foreign,
        mkt_t1.usd_curve,
        mkt_t1.foreign_curve,
        mkt_t.basis_curve,
    )
    d_for = fx_swap_pv_usd(trade, m_for) - fx_swap_pv_usd(trade, m_usd)

    # Basis move
    m_bas = Market(
        mkt_t.spot_usd_per_foreign,
        mkt_t1.usd_curve,
        mkt_t1.foreign_curve,
        mkt_t1.basis_curve,
    )
    d_bas = fx_swap_pv_usd(trade, m_bas) - fx_swap_pv_usd(trade, m_for)

    # Spot residual (should be ~0 for symmetric FX swap; we keep spot fixed in stages)
    # Total residual
    residual = pv_t1 - pv_t - (cr + d_usd + d_for + d_bas)

    return {
        "PV_t": pv_t,
        "PV_t1": pv_t1,
        "CarryRoll": cr,
        "USD_Rates": d_usd,
        "Foreign_Rates": d_for,
        "Basis": d_bas,
        "Residual": residual,
    }
