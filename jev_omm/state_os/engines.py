"""Accounting and clock engines. Paper inputs only.

These are the flows a participant cannot choose to skip: a leverage reset,
a fixed-mix rebalance, a blackout, an expiry roll, a margin clock. Each
function is a formula on caller-supplied numbers. None of them downloads a
calendar, a Fed release, or a lending tape and then pretends the download
happened.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

# Vanguard target-date policy: trade when drift exceeds 200 bp, and only
# back to 175 bp from the strategic weight, not all the way to target.
# https://corporate.vanguard.com/content/dam/corp/research/pdf/the_rebalancing_edge_optimizing_target_date_fund_rebalancing_through_threshold_based_strategies.pdf
TDF_TRIGGER = 0.0200
TDF_DESTINATION = 0.0175

LOOKBACKS = (20, 60, 120, 250)


def letf_rebalance(aum: float, leverage: float, day_return: float) -> float:
    """Dollar rebalance in the underlying. Positive means buy.

    Cheng & Madhavan, Journal of Investment Management, Q4 2009, eq. (11):
    Δ = AUM × (L² − L) × r_day.
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1539120

    The sign is the same for leveraged and inverse products: an up day is a
    buy, a down day is a sell. L = 1 (a plain ETF) rebalances 0.
    """
    return aum * (leverage * leverage - leverage) * day_return


def net_liquidity(fed_assets: float, tga: float, rrp: float) -> float:
    """Fed assets − TGA − ON RRP.

    A level proxy used to separate reserve drain (TGA or RRP up) from the
    balance-sheet headline. Inputs are caller levels, not a FRED pull.
    H.4.1: https://www.federalreserve.gov/releases/h41/
    TGA: https://fiscaldata.treasury.gov/datasets/daily-treasury-statement/operating-cash-balance
    ON RRP: https://www.newyorkfed.org/markets/desk-operations/reverse-repo
    """
    return fed_assets - tga - rrp


def net_liquidity_change(d_fed: float, d_tga: float, d_rrp: float) -> dict[str, float]:
    """ΔNL and the share of the move coming from each account.

    Shares use absolute contributions and sum to 1 when anything moved.
    A zero change returns zero shares.
    """
    pieces = {
        "fed_assets": d_fed,
        "tga": -d_tga,
        "rrp": -d_rrp,
    }
    total = pieces["fed_assets"] + pieces["tga"] + pieces["rrp"]
    denom = abs(pieces["fed_assets"]) + abs(pieces["tga"]) + abs(pieces["rrp"])
    shares = {name: (abs(value) / denom if denom > 0.0 else 0.0) for name, value in pieces.items()}
    return {"delta": total, **shares}


def in_issuer_blackout(
    asof: date,
    quarter_end: date,
    earnings: date,
    pre_days: int = 14,
) -> bool:
    """Issuer-policy window, not a statutory halt.

    SEC Rule 10b-18 is a buyback safe harbor, not an order to stop.
    17 CFR § 240.10b-18:
    https://www.ecfr.gov/current/title-17/chapter-II/part-240/section-240.10b-18

    Many issuers still close the window from about two weeks before quarter-end
    through the earnings release. Bettis, Coles, Lemmon, JFE 2000,
    https://doi.org/10.1016/S0304-405X(00)00055-6 document blackout windows as
    corporate policy. The dates here are the caller's. This function does not
    know a live earnings calendar.
    """
    start = quarter_end - timedelta(days=pre_days)
    return start <= asof <= earnings


def withheld_buyback(paces: list[float], flags: list[bool]) -> float:
    """Dollar pace that does not print while the issuer is in blackout."""
    return sum(pace * (1.0 if flag else 0.0) for pace, flag in zip(paces, flags))


def pension_equity_trade(
    aum: float,
    weight: float,
    equity_return: float,
    bond_return: float,
    *,
    exact: bool = True,
) -> float:
    """Equity dollars to trade to restore a fixed mix. Negative means sell equity.

    Exact: end-of-period weight w(1+Rs)/(1+Rp), trade (w − w_end) × AUM_end.
    Linear: AUM × w(1−w) × (Rb − Rs), the small-return expansion.

    Perold & Sharpe, *Dynamic Strategies for Asset Allocation*, FAJ 1988.
    https://doi.org/10.2469/faj.v44.n1.16
    """
    if exact:
        port = weight * equity_return + (1.0 - weight) * bond_return
        w_end = weight * (1.0 + equity_return) / (1.0 + port)
        return (weight - w_end) * aum * (1.0 + port)
    return aum * weight * (1.0 - weight) * (bond_return - equity_return)


def third_friday(year: int, month: int) -> date:
    """Monthly equity/option expiration Friday. A calendar rule, not a feed."""
    first = date(year, month, 1)
    offset = (4 - first.weekday()) % 7
    return first + timedelta(days=offset + 14)


def gen1_roll_dates(year: int) -> list[date]:
    """Twelve monthly expiries. Gen-1 overwrite (BXM) sells the front ATM call
    and rolls on this date.

    Cboe BXM: https://www.cboe.com/us/indices/dashboard/bxm/
    This list is the expiry rule. It is not a fund's holdings file.
    """
    return [third_friday(year, month) for month in range(1, 13)]


def gen3_coverage(iv: float, iv_ref: float, base: float = 0.50, slope: float = 2.0, cap: float = 1.0) -> float:
    """State-dependent cover. Rich implied vol raises the fraction overwritten.

    Gen-1 is full ATM cover (``coverage = 1``). Gen-3 here is a research rule:
    base cover plus a slope on IV versus a reference. It is not a cloned
    prospectus. Capped at ``cap`` and floored at 0.
    """
    raw = base + slope * (iv - iv_ref)
    if raw < 0.0:
        return 0.0
    if raw > cap:
        return cap
    return raw


def gen3_moneyness(iv: float, iv_ref: float) -> float:
    """Strike as a fraction of spot. Rich IV moves the call slightly OTM."""
    return 1.0 + 0.50 * max(iv - iv_ref, 0.0)


def tdf_trade(
    weight: float,
    target: float,
    aum: float,
    trigger: float = TDF_TRIGGER,
    destination: float = TDF_DESTINATION,
) -> float:
    """Equity dollars to trade under a 200/175-style band.

    No trade while |w − w*| is inside the trigger. Outside it, trade only
    back to ``destination`` away from the target, not to the target itself.
    Vanguard's published 200 bp trigger and 175 bp destination:
    https://corporate.vanguard.com/content/dam/corp/research/pdf/the_rebalancing_edge_optimizing_target_date_fund_rebalancing_through_threshold_based_strategies.pdf
    """
    gap = weight - target
    if abs(gap) <= trigger:
        return 0.0
    dest = target + math.copysign(destination, gap)
    return (dest - weight) * aum


def glide_equity_weight(
    years_to_retirement: float,
    equity_start: float = 0.90,
    equity_end: float = 0.30,
    glide_years: float = 40.0,
) -> float:
    """Linear glide. Far from retirement is ``equity_start``."""
    if glide_years <= 0.0:
        return equity_end
    x = min(max(years_to_retirement / glide_years, 0.0), 1.0)
    return equity_end + (equity_start - equity_end) * x


def realized_vol(returns: list[float], lookback: int, periods_per_year: float = 252.0) -> float:
    """Close-to-close sample volatility, annualized. Lookback shorter than 2 → 0."""
    window = list(returns)[-lookback:]
    n = len(window)
    if n < 2:
        return 0.0
    mean = sum(window) / n
    var = sum((r - mean) ** 2 for r in window) / (n - 1)
    return math.sqrt(max(var, 0.0) * periods_per_year)


def lookback_vols(returns: list[float], lookbacks: tuple[int, ...] = LOOKBACKS) -> dict[int, float]:
    """20 / 60 / 120 / 250 reconstructions. Missing history uses whatever is there."""
    return {lookback: realized_vol(returns, lookback) for lookback in lookbacks}


def vol_target_weight(sigma: float, target: float, cap: float = 1.0) -> float:
    """w = min(cap, σ* / σ). Moreira & Muir, JF 2017. https://doi.org/10.1111/jofi.12513"""
    if sigma <= 0.0:
        return cap
    weight = target / sigma
    return cap if weight > cap else weight


def vol_control_boundary(target: float, cap: float) -> float:
    """Realized vol where exposure leaves the cap and the step begins."""
    if cap <= 0.0:
        return math.inf
    return target / cap


def weight_step(prev_weight: float, new_weight: float, aum: float) -> float:
    """Dollar trade when a vol-control, CTA, or risk-parity weight changes."""
    return (new_weight - prev_weight) * aum


def trend_signal(prices: list[float], lookback: int) -> float:
    """Price / SMA(lookback) − 1. The sign is the trend kernel."""
    window = list(prices)[-lookback:]
    if not window:
        return 0.0
    sma = sum(window) / len(window)
    if sma == 0.0:
        return 0.0
    return window[-1] / sma - 1.0


def cta_weight(signal: float, sigma: float, target: float, cap: float = 1.0) -> float:
    """Sign of the trend times a vol-targeted gross.

    Hurst, Ooi, Pedersen, *A Century of Evidence on Trend-Following Investing*.
    https://doi.org/10.3905/jpm.2017.44.1.015
    Hamill, Rattray, Van Hemert, SSRN 2831926. https://doi.org/10.2139/ssrn.2831926
    """
    if signal > 0.0:
        direction = 1.0
    elif signal < 0.0:
        direction = -1.0
    else:
        direction = 0.0
    return direction * vol_target_weight(sigma, target, cap)


def risk_parity_weights(sigmas: list[float]) -> list[float]:
    """Inverse-vol weights. Asness, Frazzini, Pedersen, FAJ 2012.
    https://doi.org/10.2469/faj.v68.n1.1
    """
    inv = [0.0 if sigma <= 0.0 else 1.0 / sigma for sigma in sigmas]
    total = sum(inv)
    if total <= 0.0:
        return [0.0 for _ in sigmas]
    return [value / total for value in inv]


def borrow_pressure(
    utilization: float,
    fee: float,
    days_to_cover: float,
    delta_lendable: float,
) -> float:
    """Utilization × fee × DTC × (1 + supply that left).

    A securities-lending stub on fixture fields. Not a live Astec/S3 feed.
    D'Avolio, JFE 2002, https://doi.org/10.1016/S0304-405X(02)00206-4
    Duffie, Gârleanu, Pedersen, JFE 2002, https://doi.org/10.1016/S0304-405X(02)00226-X
    """
    supply_out = max(-delta_lendable, 0.0)
    return max(utilization, 0.0) * max(fee, 0.0) * max(days_to_cover, 0.0) * (1.0 + supply_out)


def auction_imbalance(buy_qty: float, sell_qty: float) -> float:
    """(buy − sell) / (buy + sell). Zero when the auction is empty.

    Cushing & Madhavan, *Stock returns and trading at the close*, JFM 2000.
    https://doi.org/10.1016/S1386-4181(99)00014-0
    """
    denom = buy_qty + sell_qty
    if denom <= 0.0:
        return 0.0
    return (buy_qty - sell_qty) / denom


def is_opex(day: date) -> bool:
    return day == third_friday(day.year, day.month)


def quarter_end(year: int, quarter: int) -> date:
    """Calendar quarter end. Quarter is 1..4."""
    month = quarter * 3
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - timedelta(days=1)


def in_slr_window(day: date, days: int = 3) -> bool:
    """Research window before a calendar quarter-end.

    This is not a bank's SLR optimizer. Quarter-end balance-sheet pressure
    is the fact the window stands in for. Du, Tepper, Verdelhan, JF 2018,
    https://doi.org/10.1111/jofi.12620
    """
    q = (day.month - 1) // 3 + 1
    end = quarter_end(day.year, q)
    return 0 <= (end - day).days <= days


def settlement_gap_hours(style: str) -> float:
    """Hours between the cash close and the option fix.

    PM-settled contracts share the 16:00 cash close (gap 0).
    AM-settled SPX standard monthlies fix on the opening print (overnight gap).
    The 17.5 hour figure is a research label for that overnight, not a vendor
    timestamp. SPX contract facts: https://www.cboe.com/tradable_products/sp_500/spx_options/
    """
    if style == "PM":
        return 0.0
    if style == "AM":
        return 17.5
    raise ValueError(f"unknown settlement style: {style}")


def vm_window(hour: float) -> bool:
    """Caller-supplied exchange hour. A morning variation-margin window."""
    return 7.0 <= hour < 9.0


def clock_pressure(
    *,
    opex: bool = False,
    slr: bool = False,
    vm: bool = False,
    im: bool = False,
    am_settlement: bool = False,
) -> float:
    """τ in [0, 1]. Several clocks stacking is a tighter book, not a direction."""
    score = 0.0
    if opex:
        score += 0.35
    if slr:
        score += 0.30
    if vm:
        score += 0.15
    if im:
        score += 0.10
    if am_settlement:
        score += 0.10
    return min(score, 1.0)
