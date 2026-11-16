"""Research data adapters and strict OHLCV validation."""
from __future__ import annotations

import io
import re
import numpy as np
import pandas as pd

OHLCV = ['Open', 'High', 'Low', 'Close', 'Volume']


def normalize_symbol(symbol: str, market: str) -> str:
    """Accept stock symbols only; automatically add .JK for Indonesian shares."""
    value = symbol.strip().upper()
    if market == 'IDX' and not value.endswith('.JK'):
        value += '.JK'
    if not re.fullmatch(r'[A-Z0-9^][A-Z0-9.\-^=]{0,24}', value):
        raise ValueError('Kode saham tidak valid. Contoh: BBCA, BBRI, AAPL, MSFT.')
    return value


def validate_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Return clean daily bars; reject bad data rather than silently filling prices."""
    if frame is None or frame.empty:
        raise ValueError('Data OHLCV kosong.')
    df = frame.copy()
    if 'Date' in df.columns:
        df = df.set_index('Date')
    df.columns = [str(c).strip().title() for c in df.columns]
    missing = [name for name in OHLCV if name not in df.columns]
    if missing:
        raise ValueError(f'Kolom CSV tidak lengkap: {", ".join(missing)}.')
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df.loc[~df.index.isna()]
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    if df.index.has_duplicates:
        raise ValueError('Data memiliki tanggal duplikat; mohon perbaiki CSV.')
    df = df.sort_index()[OHLCV]
    for column in OHLCV:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan)
    if df.isna().any().any():
        raise ValueError('Data mengandung nilai kosong/bukan angka pada OHLCV.')
    if (df[['Open', 'High', 'Low', 'Close']] <= 0).any().any():
        raise ValueError('Seluruh harga harus lebih besar dari 0.')
    if (df['Volume'] < 0).any():
        raise ValueError('Volume tidak boleh negatif.')
    if ((df['High'] < df[['Open', 'Close', 'Low']].max(axis=1)) |
        (df['Low'] > df[['Open', 'Close', 'High']].min(axis=1))).any():
        raise ValueError('Terdapat bar dengan High/Low tidak konsisten.')
    if len(df) < 130:
        raise ValueError('Minimal 130 bar harian untuk indikator dan pengujian.')
    df.index.name = 'Date'
    return df


def from_csv(file_bytes: bytes) -> pd.DataFrame:
    return validate_ohlcv(pd.read_csv(io.BytesIO(file_bytes)))


def fetch_yahoo(symbol: str, period: str = '5y') -> pd.DataFrame:
    """Yahoo/yfinance is for personal research; NOT a commercial data license."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError('Paket yfinance belum terpasang.') from exc
    try:
        # auto_adjust=False avoids dividend-back-adjusted execution prices.
        df = yf.Ticker(symbol).history(period=period, interval='1d',
                                        auto_adjust=False, actions=False, timeout=20)
    except Exception as exc:
        raise RuntimeError(f'Gagal mengambil data dari penyedia: {exc}') from exc
    if df is None or df.empty:
        raise ValueError('Penyedia mengembalikan data kosong. Periksa ticker/rate limit.')
    return validate_ohlcv(df)


def make_synthetic(seed: int = 42, market: str = 'IDX', n: int = 900) -> pd.DataFrame:
    """Clearly fictional OHLCV series. Never represent as actual market prices."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2022-01-03', periods=n)
    initial = 6500.0 if market == 'IDX' else 180.0
    # Regime changes + weak mean reversion: synthetic research exercise only.
    r = np.empty(n)
    r[0] = 0.0
    for i in range(1, n):
        drift = 0.00045 if (i // 180) % 3 != 2 else -0.00012
        r[i] = np.clip(drift + rng.normal(0, .017) - .12 * r[i-1], -.11, .11)
    close = initial * np.exp(np.cumsum(r))
    opens = np.r_[initial, close[:-1]] * np.exp(rng.normal(0, .004, n))
    high = np.maximum(opens, close) * (1 + rng.uniform(.002, .025, n))
    low = np.minimum(opens, close) * (1 - rng.uniform(.002, .022, n))
    volume = np.round((rng.lognormal(15.7 if market == 'IDX' else 14.8, .35, n))
                      * (1 + 3*np.abs(r))).astype(int)
    return validate_ohlcv(pd.DataFrame({
        'Date': dates, 'Open': opens, 'High': high, 'Low': low,
        'Close': close, 'Volume': volume
    }))
