import math
from dataclasses import dataclass

from scipy.stats import norm


@dataclass(frozen=True)
class BlackScholesResult:
    call_value: float
    put_value: float
    call_delta: float
    put_delta: float
    call_theta: float
    put_theta: float


def black_scholes(spot: float, strike: float, years: float, sigma: float, risk_free_rate: float) -> BlackScholesResult:
    if spot <= 0 or strike <= 0 or years <= 0 or sigma <= 0:
        return BlackScholesResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * sigma**2) * years) / (sigma * math.sqrt(years))
    d2 = d1 - sigma * math.sqrt(years)

    call_value = spot * norm.cdf(d1) - strike * math.exp(-risk_free_rate * years) * norm.cdf(d2)
    put_value = strike * math.exp(-risk_free_rate * years) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    call_delta = norm.cdf(d1)
    put_delta = call_delta - 1.0
    common_theta = -(spot * norm.pdf(d1) * sigma) / (2 * math.sqrt(years))
    call_theta = (common_theta - risk_free_rate * strike * math.exp(-risk_free_rate * years) * norm.cdf(d2)) / 365.0
    put_theta = (common_theta + risk_free_rate * strike * math.exp(-risk_free_rate * years) * norm.cdf(-d2)) / 365.0

    return BlackScholesResult(call_value, put_value, call_delta, put_delta, call_theta, put_theta)
