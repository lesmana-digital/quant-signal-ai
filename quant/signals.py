"""Predefined end-of-day signals: no fitted parameters / no future observations."""
from __future__ import annotations

import numpy as np
import pandas as pd

STRATEGY_LABELS = {
    'pullback': 'Trend Pullback — mean reversion dalam tren naik',
    'breakout': 'Momentum Breakout — tembus high 20 hari',
}


def add_signals(df: pd.DataFrame, strategy: str, min_turnover: float = 0.0,
                max_daily_vol: float = .08) -> pd.DataFrame:
    if strategy not in STRATEGY_LABELS:
        raise ValueError('Strategi tidak dikenal.')
    out = df.copy()
    close, high, vol = out['Close'], out['High'], out['Volume']
    out['Return_1D'] = close.pct_change()
    out['Return_3D'] = close.pct_change(3)
    out['Return_10D'] = close.pct_change(10)
    out['SMA_20'] = close.rolling(20, min_periods=20).mean()
    out['SMA_50'] = close.rolling(50, min_periods=50).mean()
    std20 = close.rolling(20, min_periods=20).std(ddof=0).replace(0, np.nan)
    out['Z_20'] = (close - out['SMA_20']) / std20
    out['Volume_Ratio'] = vol / vol.shift(1).rolling(20, min_periods=20).mean().replace(0, np.nan)
    out['Turnover_Median_20'] = (close * vol).shift(1).rolling(20, min_periods=20).median()
    out['Daily_Vol_20'] = out['Return_1D'].rolling(20, min_periods=20).std()
    prev_close = close.shift(1)
    true_range = pd.concat([high-out['Low'], (high-prev_close).abs(),
                            (out['Low']-prev_close).abs()], axis=1).max(axis=1)
    out['ATR_14'] = true_range.rolling(14, min_periods=14).mean()
    out['Prior_High_20'] = high.shift(1).rolling(20, min_periods=20).max()
    regime = close > out['SMA_50']
    liquidity = out['Turnover_Median_20'] >= min_turnover
    risk_ok = (out['Daily_Vol_20'] > 0) & (out['Daily_Vol_20'] <= max_daily_vol)
    valid = (out[['SMA_50', 'ATR_14', 'Daily_Vol_20', 'Volume_Ratio',
                   'Turnover_Median_20']].notna().all(axis=1))
    if strategy == 'pullback':
        factors = pd.DataFrame({
            'Tren > SMA50': regime,
            'Z20 <= -0,6': out['Z_20'] <= -.6,
            'Return 3D < 0': out['Return_3D'] < 0,
            'Vol ratio >= 0,65': out['Volume_Ratio'] >= .65,
            'Risiko volatilitas': risk_ok,
            'Likuiditas minimum': liquidity,
        }, index=out.index)
    else:
        factors = pd.DataFrame({
            'Tren > SMA50': regime,
            'Close > High20 sebelumnya': close > out['Prior_High_20'],
            'Vol ratio >= 1,2': out['Volume_Ratio'] >= 1.2,
            'Return 10D > 0': out['Return_10D'] > 0,
            'Risiko volatilitas': risk_ok,
            'Likuiditas minimum': liquidity,
        }, index=out.index)
    out['Match_Score'] = (factors.sum(axis=1) / len(factors.columns) * 100).where(valid, 0)
    out['Signal'] = (factors.all(axis=1) & valid).fillna(False).astype(bool)
    return out


def signal_factors(row: pd.Series, strategy: str, min_turnover: float,
                   max_daily_vol: float = .08) -> dict[str, bool]:
    """Current-bar component checklist for transparency."""
    shared = {
        'Tren: close > SMA50': bool(row['Close'] > row['SMA_50']),
        'Volatilitas 20D aman': bool(0 < row['Daily_Vol_20'] <= max_daily_vol),
        'Likuiditas memadai': bool(row['Turnover_Median_20'] >= min_turnover),
    }
    if strategy == 'pullback':
        return {**shared,
                'Z-score <= -0,6': bool(row['Z_20'] <= -.6),
                'Return 3D negatif': bool(row['Return_3D'] < 0),
                'Volume >= 0,65x normal': bool(row['Volume_Ratio'] >= .65)}
    return {**shared,
            'Break high 20 hari sebelumnya': bool(row['Close'] > row['Prior_High_20']),
            'Momentum 10 hari positif': bool(row['Return_10D'] > 0),
            'Volume >= 1,2x normal': bool(row['Volume_Ratio'] >= 1.2)}
