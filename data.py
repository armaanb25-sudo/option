from __future__ import annotations

import re
from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf


_YF_CACHE = Path(".yf_cache")
_YF_CACHE.mkdir(exist_ok=True)
if hasattr(yf, "set_tz_cache_location"):
    yf.set_tz_cache_location(str(_YF_CACHE.resolve()))


def load_history(symbol: str, period: str = "3y") -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    history = ticker.history(period=period, auto_adjust=True)
    if history.empty:
        raise ValueError(f"No historical price data returned for {symbol}. Check the Yahoo Finance symbol.")
    return history.dropna(subset=["Close"])


def market_snapshot(history: pd.DataFrame) -> dict:
    close = history["Close"].astype(float)
    spot = float(close.iloc[-1])
    last_90 = close.tail(90)
    daily_returns = np.log(close / close.shift(1)).dropna()
    realized_vol = float(daily_returns.std() * np.sqrt(252))
    high_52w = float(close.tail(252).max())
    low_52w = float(close.tail(252).min())
    high_90d = float(last_90.max())
    low_90d = float(last_90.min())
    range_pos_90d = ((spot - low_90d) / (high_90d - low_90d) * 100.0) if high_90d != low_90d else 50.0

    return {
        "spot": spot,
        "realized_vol": realized_vol,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "high_90d": high_90d,
        "low_90d": low_90d,
        "pct_from_90d_low": ((spot - low_90d) / low_90d) * 100.0,
        "pct_from_52w_low": ((spot - low_52w) / low_52w) * 100.0,
        "range_pos_90d": range_pos_90d,
    }


def read_option_csv(content: bytes | str) -> pd.DataFrame:
    text = content.decode("utf-8-sig", errors="replace") if isinstance(content, bytes) else content
    first_line = text.splitlines()[0].strip().upper() if text.splitlines() else ""
    if first_line.startswith("CALLS"):
        return pd.read_csv(StringIO(text), header=1)
    return pd.read_csv(StringIO(text))


def normalize_option_chain(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("The uploaded option-chain CSV is empty.")

    lookup = {_clean_col(col): col for col in raw.columns}

    def find(*candidates: str) -> str | None:
        for candidate in candidates:
            cleaned = _clean_col(candidate)
            if cleaned in lookup:
                return lookup[cleaned]
        return None

    strike_col = find("strike", "strike price", "strikeprice")
    if not strike_col:
        raise ValueError("Could not find a strike column. Expected something like 'Strike' or 'Strike Price'.")

    if _looks_like_nse_option_chain(raw, strike_col):
        normalized = pd.DataFrame(
            {
                "strike": _to_number(raw[strike_col]),
                "call_ltp": _to_number(raw.get("LTP")),
                "call_bid": _to_number(raw.get("BID")),
                "call_ask": _to_number(raw.get("ASK")),
                "call_iv": _to_number(raw.get("IV")),
                "call_oi": _to_number(raw.get("OI")),
                "call_volume": _to_number(raw.get("VOLUME")),
                "put_ltp": _to_number(raw.get("LTP.1")),
                "put_bid": _to_number(raw.get("BID.1")),
                "put_ask": _to_number(raw.get("ASK.1")),
                "put_iv": _to_number(raw.get("IV.1")),
                "put_oi": _to_number(raw.get("OI.1")),
                "put_volume": _to_number(raw.get("VOLUME.1")),
            }
        )
        normalized = normalized.dropna(subset=["strike"]).sort_values("strike").reset_index(drop=True)
        if normalized.empty:
            raise ValueError("No valid strike rows found after reading the CSV.")
        return normalized

    mappings = {
        "call_ltp": find("call ltp", "calls ltp", "ce ltp", "call_ltp", "ce_ltp", "ltp ce"),
        "call_bid": find("call bid", "ce bid", "bid ce", "call_bid"),
        "call_ask": find("call ask", "ce ask", "ask ce", "call_ask"),
        "call_iv": find("call iv", "ce iv", "iv ce", "call_iv"),
        "call_oi": find("call oi", "ce oi", "oi ce", "call_oi"),
        "call_volume": find("call volume", "ce volume", "volume ce", "call_volume"),
        "put_ltp": find("put ltp", "puts ltp", "pe ltp", "put_ltp", "pe_ltp", "ltp pe"),
        "put_bid": find("put bid", "pe bid", "bid pe", "put_bid"),
        "put_ask": find("put ask", "pe ask", "ask pe", "put_ask"),
        "put_iv": find("put iv", "pe iv", "iv pe", "put_iv"),
        "put_oi": find("put oi", "pe oi", "oi pe", "put_oi"),
        "put_volume": find("put volume", "pe volume", "volume pe", "put_volume"),
    }

    normalized = pd.DataFrame({"strike": _to_number(raw[strike_col])})
    for output_col, source_col in mappings.items():
        normalized[output_col] = _to_number(raw[source_col]) if source_col else np.nan

    normalized = normalized.dropna(subset=["strike"]).sort_values("strike").reset_index(drop=True)
    if normalized.empty:
        raise ValueError("No valid strike rows found after reading the CSV.")
    return normalized


def _clean_col(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _looks_like_nse_option_chain(raw: pd.DataFrame, strike_col: str) -> bool:
    return (
        strike_col == "STRIKE"
        and {"LTP", "BID", "ASK", "LTP.1", "BID.1", "ASK.1"}.issubset(set(raw.columns))
    )


def _to_number(series: Iterable) -> pd.Series:
    return pd.to_numeric(
        pd.Series(series)
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False),
        errors="coerce",
    )
