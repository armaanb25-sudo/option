from __future__ import annotations

import numpy as np
import pandas as pd

from pricing import black_scholes


def analyze_chain(
    symbol: str,
    snapshot: dict,
    chain: pd.DataFrame,
    dte: int,
    capital: float,
    lot_size: int,
    risk_free_rate: float,
) -> dict:
    spot = float(snapshot["spot"])
    years = dte / 365.0
    sigma = float(snapshot["realized_vol"])
    rows = []

    for _, row in chain.iterrows():
        strike = float(row["strike"])
        bs = black_scholes(spot, strike, years, sigma, risk_free_rate)
        put_market = _market_price(row.get("put_ltp"), row.get("put_bid"), row.get("put_ask"))
        call_market = _market_price(row.get("call_ltp"), row.get("call_bid"), row.get("call_ask"))

        rows.append(
            _build_row(
                "CSP",
                symbol,
                spot,
                strike,
                put_market,
                row,
                bs.put_value,
                bs.put_delta,
                bs.put_theta,
                dte,
                capital,
                lot_size,
                snapshot,
            )
        )
        rows.append(
            _build_row(
                "Covered Call",
                symbol,
                spot,
                strike,
                call_market,
                row,
                bs.call_value,
                bs.call_delta,
                bs.call_theta,
                dte,
                capital,
                lot_size,
                snapshot,
            )
        )

    results = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    results = results.dropna(subset=["market_price", "model_price"])
    if results.empty:
        raise ValueError("The CSV did not contain usable market prices. Include LTP or bid/ask columns.")

    results["recommendation"] = results["score"].apply(_recommendation)
    csp = results[results["strategy"] == "CSP"].sort_values(["score", "annualized_return_pct"], ascending=False).head(8)
    cc = results[results["strategy"] == "Covered Call"].sort_values(["score", "annualized_return_pct"], ascending=False).head(8)

    return {
        "symbol": symbol,
        "dte": dte,
        "capital": capital,
        "lot_size": lot_size,
        "snapshot": snapshot,
        "top_csp": csp.to_dict(orient="records"),
        "top_covered_calls": cc.to_dict(orient="records"),
        "all_candidates": results.sort_values(["strategy", "strike"]).to_dict(orient="records"),
    }


def _build_row(strategy, symbol, spot, strike, market_price, source, model_price, delta, theta, dte, capital, lot_size, snapshot):
    spread = _spread_pct(
        source.get("put_bid" if strategy == "CSP" else "call_bid"),
        source.get("put_ask" if strategy == "CSP" else "call_ask"),
        market_price,
    )
    premium = market_price * lot_size
    collateral = strike * lot_size if strategy == "CSP" else spot * lot_size
    annualized_return = (premium / collateral) * (365.0 / dte) * 100.0 if collateral > 0 else np.nan
    breakeven = strike - market_price if strategy == "CSP" else spot - market_price
    mispricing = market_price - model_price
    mispricing_pct = (mispricing / model_price * 100.0) if model_price > 0 else np.nan
    score, reasons = _score_candidate(
        strategy,
        spot,
        strike,
        annualized_return,
        spread,
        delta,
        source,
        snapshot,
        capital,
        collateral,
        market_price,
        model_price,
    )

    return {
        "symbol": symbol,
        "strategy": strategy,
        "strike": round(strike, 2),
        "spot": round(spot, 2),
        "market_price": round(market_price, 2),
        "model_price": round(model_price, 2),
        "mispricing": round(mispricing, 2),
        "mispricing_pct": round(mispricing_pct, 2) if np.isfinite(mispricing_pct) else None,
        "delta": round(delta, 3),
        "theta_per_day": round(theta, 3),
        "annualized_return_pct": round(annualized_return, 2),
        "premium": round(premium, 2),
        "collateral": round(collateral, 2),
        "breakeven": round(breakeven, 2),
        "spread_pct": round(spread, 2) if np.isfinite(spread) else None,
        "score": int(score),
        "reasons": reasons,
    }


def _score_candidate(strategy, spot, strike, arr, spread, delta, source, snapshot, capital, collateral, market_price, model_price):
    score = 50
    reasons = []

    if capital and collateral > capital:
        score -= 30
        reasons.append("Needs more capital than provided.")
    else:
        score += 8
        reasons.append("Fits within the selected capital.")

    if market_price > model_price:
        score += 14
        reasons.append("Market premium is above Black-Scholes fair value.")
    else:
        score -= 8
        reasons.append("Market premium is below model value.")

    if arr >= 18:
        score += 14
        reasons.append("Annualized premium return is attractive.")
    elif arr >= 10:
        score += 7
        reasons.append("Annualized premium return is reasonable.")
    else:
        score -= 10
        reasons.append("Premium return is thin for the risk.")

    if np.isfinite(spread):
        if spread <= 5:
            score += 10
            reasons.append("Bid/ask spread looks liquid.")
        elif spread <= 12:
            score += 2
            reasons.append("Bid/ask spread is acceptable but not tight.")
        else:
            score -= 12
            reasons.append("Bid/ask spread is wide.")

    volume_col = "put_volume" if strategy == "CSP" else "call_volume"
    oi_col = "put_oi" if strategy == "CSP" else "call_oi"
    volume = _num(source.get(volume_col))
    oi = _num(source.get(oi_col))
    if (np.isfinite(volume) and volume >= 100) or (np.isfinite(oi) and oi >= 1000):
        score += 8
        reasons.append("Volume or open interest supports tradability.")
    elif np.isfinite(volume) or np.isfinite(oi):
        score -= 5
        reasons.append("Liquidity is present but limited.")

    if strategy == "CSP":
        if strike < spot:
            score += 10
            reasons.append("Put strike is below spot.")
        else:
            score -= 12
            reasons.append("Put strike is at/above spot, raising assignment risk.")
        if -0.35 <= delta <= -0.12:
            score += 10
            reasons.append("Put delta is in a wheel-friendly range.")
        if snapshot["range_pos_90d"] <= 45:
            score += 8
            reasons.append("Stock is not stretched versus its 90-day range.")
    else:
        if strike > spot:
            score += 10
            reasons.append("Call strike is above spot.")
        else:
            score -= 12
            reasons.append("Call strike is at/below spot, so upside is tightly capped.")
        if 0.12 <= delta <= 0.35:
            score += 10
            reasons.append("Call delta is in a wheel-friendly range.")
        if snapshot["range_pos_90d"] >= 55:
            score += 8
            reasons.append("Stock is closer to resistance, which favors covered calls.")

    return max(0, min(100, score)), reasons[:5]


def _market_price(ltp, bid, ask):
    ltp = _num(ltp)
    bid = _num(bid)
    ask = _num(ask)

    if np.isfinite(ltp) and ltp > 0:
        return ltp
    if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return np.nan


def _spread_pct(bid, ask, market):
    bid = _num(bid)
    ask = _num(ask)
    market = _num(market)
    if np.isfinite(bid) and np.isfinite(ask) and np.isfinite(market) and market > 0:
        return ((ask - bid) / market) * 100.0
    return np.nan


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _recommendation(score: int) -> str:
    if score >= 78:
        return "Strong candidate"
    if score >= 62:
        return "Possible, review manually"
    if score >= 45:
        return "Wait / only if thesis is strong"
    return "Avoid"
    if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    if np.isfinite(ltp) and ltp > 0:
        return ltp
    return np.nan


def _spread_pct(bid, ask, market):
    bid = _num(bid)
    ask = _num(ask)
    market = _num(market)
    if np.isfinite(bid) and np.isfinite(ask) and np.isfinite(market) and market > 0:
        return ((ask - bid) / market) * 100.0
    return np.nan


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _recommendation(score: int) -> str:
    if score >= 78:
        return "Strong candidate"
    if score >= 62:
        return "Possible, review manually"
    if score >= 45:
        return "Wait / only if thesis is strong"
    return "Avoid"
