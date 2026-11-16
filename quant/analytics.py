"""Honest uncertainty and reporting helpers."""
import math


def wilson_interval(wins: int, trials: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% confidence interval for historical success frequency (not a forecast)."""
    if trials < 1 or not (0 <= wins <= trials):
        return None
    p = wins/trials
    denominator = 1 + z*z/trials
    center = (p + z*z/(2*trials))/denominator
    margin = z/denominator * math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials))
    return max(0., center-margin), min(1., center+margin)


def fmt_percent(value: float | None, decimals: int = 2) -> str:
    return f'{value*100:,.{decimals}f}%' if value is not None and math.isfinite(value) else '—'


def fmt_decimal(value: float | None, decimals: int = 2) -> str:
    return f'{value:,.{decimals}f}' if value is not None and math.isfinite(value) else '—'
