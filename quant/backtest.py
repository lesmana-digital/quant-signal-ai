"""Long-only bar-by-bar execution simulator with next-open entries.

No look-ahead: signal, ATR and sizing context come from completed bar t;
entry is at bar t+1 open. Stop intraday is modeled conservatively, and gap
past stop exits at the worse opening price. No leverage, integer shares/lots.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 100_000_000.0
    holding_days: int = 3
    risk_per_trade: float = .01
    max_position: float = .25
    stop_atr: float = 2.0
    fee_bps: float = 15.0
    slippage_bps: float = 10.0
    lot_size: int = 100

    def __post_init__(self):
        if not (self.initial_capital > 0 and 1 <= self.holding_days <= 5 and
                0 < self.risk_per_trade <= .05 and 0 < self.max_position <= 1 and
                .5 <= self.stop_atr <= 6 and 0 <= self.fee_bps <= 200 and
                0 <= self.slippage_bps <= 200 and self.lot_size >= 1):
            raise ValueError('Konfigurasi backtest di luar rentang keamanan.')


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    stats: dict
    open_position: dict | None
    skipped_small: int


def compute_stats(equity: pd.DataFrame, trades: pd.DataFrame,
                  capital: float) -> dict:
    eq = equity['Equity'].astype(float)
    daily = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    total = float(eq.iloc[-1] / capital - 1) if len(eq) else 0.0
    days = max(len(eq) - 1, 1)
    cagr = float((eq.iloc[-1] / capital) ** (252/days) - 1) if len(eq) > 1 and eq.iloc[-1] > 0 else -1.0
    std = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    sharpe = float(daily.mean()/std * math.sqrt(252)) if std > 1e-12 else None
    peak = np.maximum.accumulate(np.r_[capital, eq.to_numpy()])
    dd = np.r_[capital, eq.to_numpy()] / peak - 1
    pnl = trades['Net_PnL'] if len(trades) and 'Net_PnL' in trades else pd.Series(dtype=float)
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]
    gross_gain = float(winners.sum())
    gross_loss = float(-losers.sum())
    profit_factor = gross_gain/gross_loss if gross_loss > 0 else None
    return {
        'Total Return': total,
        'CAGR (perkiraan)': cagr,
        'Sharpe (rf=0)': sharpe,
        'Max Drawdown': float(dd.min()),
        'Jumlah Trade Selesai': int(len(trades)),
        'Win Rate': float((pnl > 0).mean()) if len(pnl) else None,
        'Profit Factor': profit_factor,
        'Rata-rata PnL / Trade': float(pnl.mean()) if len(pnl) else None,
        'Eksposur Harian': float(equity['Invested'].mean()) if len(equity) else 0.0,
    }


def backtest(df: pd.DataFrame, cfg: BacktestConfig,
             start_idx: int = 65) -> BacktestResult:
    """Run without overlapping positions. df requires Signal and ATR_14."""
    required = {'Open', 'High', 'Low', 'Close', 'Signal', 'ATR_14'}
    if not required.issubset(df.columns):
        raise ValueError(f'Data belum siap: {sorted(required - set(df.columns))}')
    if len(df) <= start_idx+1:
        raise ValueError('Data tidak cukup untuk periode backtest.')
    if not (0 <= start_idx < len(df)):
        raise ValueError('start_idx salah.')

    cash = float(cfg.initial_capital)
    position: dict | None = None
    trades: list[dict] = []
    equity_rows: list[dict] = []
    skipped_small = 0
    side_cost = (cfg.fee_bps + cfg.slippage_bps) / 10_000
    last_signal = False
    index = df.index

    def exit_position(px: float, date: pd.Timestamp, reason: str, i: int):
        nonlocal cash, position
        assert position is not None
        proceeds = position['Qty'] * px * (1-side_cost)
        cash += proceeds
        buy_cost = position['Qty'] * position['Entry_Price'] * (1+side_cost)
        pnl = proceeds - buy_cost
        trades.append({
            'Signal_Date': position['Signal_Date'],
            'Entry_Date': position['Entry_Date'], 'Exit_Date': date,
            'Entry_Price': position['Entry_Price'], 'Exit_Price': px,
            'Qty': position['Qty'], 'Holding_Sessions': i-position['Entry_i'],
            'Stop_Price': position['Stop_Price'], 'Exit_Reason': reason,
            'Net_PnL': pnl, 'Net_Return_On_Trade': pnl / buy_cost,
        })
        position = None

    for i in range(start_idx, len(df)):
        bar = df.iloc[i]
        date = index[i]
        op, lo, cl = float(bar['Open']), float(bar['Low']), float(bar['Close'])
        exited_today = False

        if position is not None:
            if op <= position['Stop_Price']:
                exit_position(op, date, 'Stop gap (open)', i)
                exited_today = True
            elif i >= position['Entry_i'] + cfg.holding_days:
                exit_position(op, date, 'Time exit (open)', i)
                exited_today = True

        # last_signal was known at t-1 close; can enter only at t open.
        if position is None and not exited_today and last_signal:
            prev = df.iloc[i-1]
            atr = float(prev['ATR_14'])
            if np.isfinite(atr) and atr > 0:
                stop_distance = max(atr * cfg.stop_atr, op * .005)
                # Side cost included in risk budget and capital allocation.
                risk_per_share = stop_distance + op*side_cost*2
                q_risk = (cash * cfg.risk_per_trade) / risk_per_share
                q_cap = (cash * cfg.max_position) / (op * (1+side_cost))
                q_cash = cash / (op * (1+side_cost))
                qty = int(math.floor(min(q_risk, q_cap, q_cash)/cfg.lot_size)*cfg.lot_size)
                if qty > 0:
                    cash -= qty * op * (1+side_cost)
                    position = {
                        'Signal_Date': index[i-1], 'Entry_Date': date,
                        'Entry_i': i, 'Entry_Price': op, 'Qty': qty,
                        'Stop_Price': max(op-stop_distance, .00001),
                    }
                else:
                    skipped_small += 1

        # Intraday stops apply even to a newly opened trade.
        if position is not None and lo <= position['Stop_Price']:
            exit_position(position['Stop_Price'], date, 'Stop ATR (intraday)', i)
            exited_today = True

        market_value = position['Qty']*cl if position is not None else 0.0
        equity_rows.append({'Date': date, 'Equity': cash+market_value,
                            'Cash': cash, 'Invested': int(position is not None)})
        # New signal generated AFTER today's bar closes, not before.
        last_signal = bool(bar['Signal']) and position is None

    equity = pd.DataFrame(equity_rows).set_index('Date')
    reference_close = float(df['Close'].iloc[start_idx])
    equity['BuyHold'] = cfg.initial_capital * df.loc[equity.index, 'Close'] / reference_close
    trades_df = pd.DataFrame(trades)
    stats = compute_stats(equity, trades_df, cfg.initial_capital)
    return BacktestResult(equity=equity, trades=trades_df, stats=stats,
                          open_position=position, skipped_small=skipped_small)
