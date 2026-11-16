"""Quant Signal AI v1 — executable educational Streamlit dashboard.
Start: streamlit run app.py
"""
from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quant.analytics import fmt_decimal, fmt_percent, wilson_interval
from quant.backtest import BacktestConfig, backtest
from quant.data import fetch_yahoo, from_csv, make_synthetic, normalize_symbol
from quant.signals import STRATEGY_LABELS, add_signals, signal_factors

st.set_page_config(page_title='Quant Signal AI | Research Terminal',
                   page_icon='📈', layout='wide', initial_sidebar_state='expanded')
st.markdown('''
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
[data-testid="stMetric"] {border: 1px solid #28364c; background: #101b2c;
    border-radius: 13px; padding: 16px 15px;}
.small-note {color:#91a4bf;font-size:.82rem;}
</style>
''', unsafe_allow_html=True)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_yahoo(ticker: str, period: str):
    return fetch_yahoo(ticker, period)

@st.cache_data(show_spinner=False)
def cached_demo(market: str):
    return make_synthetic(seed=23 if market == 'IDX' else 61, market=market)


def closed_daily_only(df: pd.DataFrame, market: str) -> tuple[pd.DataFrame, bool]:
    """Conservatively remove in-progress current-day bar from Yahoo daily history."""
    tz = ZoneInfo('Asia/Jakarta' if market == 'IDX' else 'America/New_York')
    local = datetime.now(tz)
    # Conservative same-day cutoff; early sessions should be checked with vendor calendar.
    if len(df) and df.index[-1].date() == local.date() and local.time() < time(17, 0):
        return df.iloc[:-1], True
    return df, False


def m(value: float, currency: str) -> str:
    if currency == 'IDR':
        return 'Rp' + f'{value:,.0f}'.replace(',', '.')
    return '$' + f'{value:,.2f}'


def performance_chart(full, oos, capital):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=full.equity.index, y=full.equity['Equity']/capital,
                             name='Strategi (seluruh periode)', mode='lines',
                             line=dict(color='#36c2ae', width=2.3)))
    fig.add_trace(go.Scatter(x=full.equity.index, y=full.equity['BuyHold']/capital,
                             name='Buy & Hold (harga saja)', mode='lines',
                             line=dict(color='#96a1b8', width=1.8, dash='dash')))
    fig.add_vline(x=oos.equity.index[0].timestamp()*1000,
                  line_color='#e3b462', line_dash='dot', opacity=.85,
                  annotation_text='Awal uji OOS', annotation_position='top')
    fig.update_layout(height=390, margin=dict(l=10,r=10,t=25,b=10),
                      yaxis_title='Kelipatan modal awal',
                      legend=dict(orientation='h',y=-.22), hovermode='x unified')
    return fig


def price_chart(df: pd.DataFrame, trades: pd.DataFrame):
    frame = df.tail(260)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.index, y=frame.Close, mode='lines', name='Close',
                             line=dict(color='#53ccbb',width=2)))
    fig.add_trace(go.Scatter(x=frame.index, y=frame.SMA_20, mode='lines', name='SMA20',
                             line=dict(color='#f0c26a',width=1.4)))
    fig.add_trace(go.Scatter(x=frame.index, y=frame.SMA_50, mode='lines', name='SMA50',
                             line=dict(color='#9f95d9',width=1.4)))
    signals = frame.loc[frame['Signal']]
    if len(signals):
        fig.add_trace(go.Scatter(x=signals.index, y=signals.Close, mode='markers',
                    name='Sinyal EOD', marker=dict(symbol='triangle-up',size=9,color='#f4ac55')))
    if not trades.empty:
        visible = trades[trades['Entry_Date'] >= frame.index[0]]
        if len(visible):
            fig.add_trace(go.Scatter(x=visible.Entry_Date, y=visible.Entry_Price,
                        mode='markers', name='Buy (open T+1)',
                        marker=dict(symbol='circle',size=10,color='#43d8a4')))
            fig.add_trace(go.Scatter(x=visible.Exit_Date, y=visible.Exit_Price,
                        mode='markers', name='Exit',
                        marker=dict(symbol='x',size=10,color='#f17a7a')))
    fig.update_layout(height=420,margin=dict(l=10,r=10,t=25,b=10),
                      yaxis_title='Harga',hovermode='x unified',
                      legend=dict(orientation='h',y=-.23))
    return fig

st.markdown('### 📈 QUANT SIGNAL AI  <span class="small-note">/ RESEARCH TERMINAL v0.1</span>',
            unsafe_allow_html=True)
st.caption('Riset kuantitatif EOD • saham Indonesia & AS • swing trading 1–5 sesi • bukan nasihat investasi')

with st.sidebar:
    st.markdown('## ⚙️ Konfigurasi')
    market = st.selectbox('Pasar', ['IDX','US'], format_func=lambda x: '🇮🇩 Indonesia (IDX)' if x=='IDX' else '🇺🇸 Amerika Serikat')
    currency = 'IDR' if market == 'IDX' else 'USD'
    source = st.radio('Sumber data', ['Demo sintetik', 'Yahoo (riset pribadi)', 'Unggah CSV'],
                      help='Data demo adalah simulasi. Yahoo hanya untuk riset pribadi sesuai ketentuan penyedia.')
    ticker = 'SIM-IDX' if market == 'IDX' else 'SIM-US'
    uploaded = None
    period = '5y'
    if source == 'Yahoo (riset pribadi)':
        input_symbol = st.text_input('Kode saham', 'BBCA' if market=='IDX' else 'AAPL')
        period = st.selectbox('Riwayat data', ['2y','5y','10y'], index=1)
        ticker = input_symbol
    elif source == 'Unggah CSV':
        uploaded = st.file_uploader('Unggah OHLCV (.csv)', type=['csv'])
        ticker = st.text_input('Label saham', 'CSV-IDX' if market=='IDX' else 'CSV-US')
        st.caption('Kolom wajib: Date, Open, High, Low, Close, Volume.')
    strategy = st.selectbox('Model sinyal', list(STRATEGY_LABELS), format_func=lambda x: STRATEGY_LABELS[x])
    holding = st.slider('Durasi maksimum posisi (sesi)',1,5,3)
    min_turnover_default = 1_000_000_000.0 if market=='IDX' else 5_000_000.0
    min_turnover = st.number_input('Median nilai transaksi 20D minimum', min_value=0.0,
                                   value=min_turnover_default, step=100_000_000.0 if market=='IDX' else 1_000_000.0,
                                   help=f'Dalam {currency}; memakai harga × volume, perkiraan nilai transaksi.')
    st.divider()
    st.markdown('#### 🛡️ Risiko & eksekusi')
    default_capital = 100_000_000.0 if market=='IDX' else 50_000.0
    capital = st.number_input(f'Modal simulasi ({currency})', min_value=1000.0,
                              value=default_capital, step=1_000_000.0 if market=='IDX' else 5_000.0)
    risk_pct = st.slider('Risiko target per transaksi (%)', .25, 3.0, 1.0, .25)
    max_position_pct = st.slider('Batas alokasi per saham (%)', 5, 100, 25, 5)
    stop_atr = st.slider('Stop loss (× ATR14)', .5, 5.0, 2.0, .5)
    fee_bps = st.number_input('Fee per sisi (basis point)',0.0,200.0,15.0 if market=='IDX' else 5.0,1.0)
    slip_bps = st.number_input('Slippage per sisi (basis point)',0.0,200.0,10.0 if market=='IDX' else 5.0,1.0)
    st.caption('Nilai fee/slippage merupakan asumsi yang bisa diubah, bukan tarif resmi. 100 bp = 1%.')
    split_pct = st.slider('Proporsi periode awal (eksplorasi)', 55, 85, 70, 5,
                         help='Aturan tetap, tidak dioptimasi otomatis. Periode terakhir = uji out-of-sample.')

try:
    if source == 'Demo sintetik':
        raw = cached_demo(market)
        ticker = 'SIM-IDX' if market=='IDX' else 'SIM-US'
        st.warning('MODE DEMO — seluruh harga dan volume bersifat SINTETIK/fiktif, bukan kutipan BBCA/AAPL atau saham nyata.', icon='🧪')
    elif source == 'Yahoo (riset pribadi)':
        ticker = normalize_symbol(ticker, market)
        with st.spinner(f'Mengambil harga harian {ticker}...'):
            raw, removed = closed_daily_only(cached_yahoo(ticker, period), market)
        if removed:
            st.info('Bar hari ini yang kemungkinan belum selesai telah dikeluarkan dari analisis.')
        st.warning('Data Yahoo adalah sumber riset pribadi, bukan feed real-time atau data berlisensi komersial. Verifikasi aksi korporasi & penyesuaian historis.', icon='⚠️')
    else:
        if uploaded is None:
            st.info('Silakan unggah CSV OHLCV pada panel kiri untuk memulai.')
            st.stop()
        raw = from_csv(uploaded.getvalue())
        st.info('Data dari CSV pengguna. Pastikan data telah disesuaikan untuk split dan memenuhi hak penggunaan.')
    df = add_signals(raw, strategy, min_turnover=min_turnover)
    cfg = BacktestConfig(initial_capital=capital, holding_days=holding,
        risk_per_trade=risk_pct/100, max_position=max_position_pct/100,
        stop_atr=stop_atr, fee_bps=fee_bps, slippage_bps=slip_bps,
        lot_size=100 if market=='IDX' else 1)
    first_trade_idx = max(65, int(len(df)*.10))
    test_idx = max(65, int(len(df)*split_pct/100))
    entire = backtest(df, cfg, start_idx=first_trade_idx)
    oos = backtest(df, cfg, start_idx=test_idx)
except Exception as exc:
    st.error(f'Analisis tidak dapat dilakukan: {exc}')
    st.stop()

latest = df.iloc[-1]
latest_date = df.index[-1].date().isoformat()
label = 'KANDIDAT ENTRY T+1' if latest['Signal'] else 'NO TRADE'
st.markdown(f'#### {ticker}  ·  {market}  ·  Data terakhir {latest_date}')
col1,col2,col3,col4 = st.columns(4)
col1.metric('Harga penutupan terakhir', m(float(latest.Close), currency))
col2.metric('Status sinyal pada close', label)
col3.metric('Kecocokan aturan (bukan probabilitas)',f'{latest.Match_Score:.0f}/100')
col4.metric('Volatilitas harian 20D', fmt_percent(float(latest.Daily_Vol_20)))

panels = st.tabs(['📊 Analisis Sinyal','🧪 Backtest & OOS','🛡️ Kontrol Risiko','📂 Data & Metodologi'])
with panels[0]:
    a,b = st.columns([2,1])
    with a:
        st.plotly_chart(price_chart(df, entire.trades), use_container_width=True)
    with b:
        st.markdown('#### Pemeriksaan faktor (data terakhir)')
        for key, yes in signal_factors(latest,strategy,min_turnover).items():
            st.markdown(('✅ ' if yes else '⬜ ') + key)
        st.divider()
        st.write(f'**Z-score 20D:** {fmt_decimal(float(latest.Z_20))}')
        st.write(f'**Volume relatif:** {fmt_decimal(float(latest.Volume_Ratio))}×')
        st.write(f'**ATR14:** {m(float(latest.ATR_14),currency)}')
        st.caption('Kecocokan aturan = persentase kondisi yang dipenuhi; bukan peluang naik. Sinyal baru diketahui setelah candle harian selesai.')
    st.info('KANDIDAT tidak berarti pasti untung. Sinyal EOD hanya dapat dieksekusi paling cepat pada open sesi berikutnya dan wajib lolos risk sizing.')

with panels[1]:
    st.markdown('#### Hasil backtest model tetap (tidak dioptimasi otomatis)')
    st.caption(f'Awal periode penuh: {entire.equity.index[0].date()} | Awal uji OOS: {oos.equity.index[0].date()} | Akhir: {df.index[-1].date()}')
    x,y = st.columns(2)
    with x:
        st.markdown('**Seluruh periode simulasi**')
        sc = entire.stats
        aa,bb,cc = st.columns(3)
        aa.metric('Total return',fmt_percent(sc['Total Return']))
        bb.metric('Max drawdown',fmt_percent(sc['Max Drawdown']))
        cc.metric('Trade selesai',sc['Jumlah Trade Selesai'])
        st.write(f"Win rate: {fmt_percent(sc['Win Rate'])} · Sharpe: {fmt_decimal(sc['Sharpe (rf=0)'])} · Exposure: {fmt_percent(sc['Eksposur Harian'])}")
    with y:
        st.markdown('**Periode terakhir / out-of-sample**')
        sc = oos.stats
        aa,bb,cc = st.columns(3)
        aa.metric('Total return OOS',fmt_percent(sc['Total Return']))
        bb.metric('Max drawdown OOS',fmt_percent(sc['Max Drawdown']))
        cc.metric('Trade selesai OOS',sc['Jumlah Trade Selesai'])
        st.write(f"Win rate: {fmt_percent(sc['Win Rate'])} · Sharpe: {fmt_decimal(sc['Sharpe (rf=0)'])} · Exposure: {fmt_percent(sc['Eksposur Harian'])}")
    st.plotly_chart(performance_chart(entire,oos,capital), use_container_width=True)
    trials = len(oos.trades)
    if trials:
        winners = int((oos.trades['Net_PnL'] > 0).sum())
        bounds = wilson_interval(winners, trials)
        st.caption(f'Win rate historis OOS: {winners}/{trials}; interval Wilson 95%: {fmt_percent(bounds[0])}–{fmt_percent(bounds[1])}. '
                   'Ini interval frekuensi historis, BUKAN probabilitas profit pada trade berikutnya.')
    if trials < 30:
        st.warning(f'Hanya {trials} trade selesai pada periode OOS; sampel terlalu kecil untuk klaim keunggulan statistik. Perlu uji beberapa rezim dan instrumen.')
    if entire.open_position is not None or oos.open_position is not None:
        st.info('Ada posisi terbuka di ujung data. Equity mencakup mark-to-market, sedangkan metrik win rate hanya menghitung trade yang selesai.')
    st.markdown('#### Jurnal transaksi (periode OOS)')
    if oos.trades.empty:
        st.info('Tidak ada transaksi selesai pada periode OOS dengan parameter ini.')
    else:
        show = oos.trades.copy()
        for c in ['Signal_Date','Entry_Date','Exit_Date']:
            show[c] = pd.to_datetime(show[c]).dt.strftime('%Y-%m-%d')
        st.dataframe(show.sort_values('Entry_Date',ascending=False),use_container_width=True,hide_index=True)
        st.download_button('Unduh trade log OOS (.csv)', show.to_csv(index=False).encode('utf-8'),
                           file_name=f'quant_signal_{ticker.replace(".","_")}_oos_trades.csv',mime='text/csv')

with panels[2]:
    st.markdown('#### Aturan ukuran posisi dan stop')
    st.latex(r'Q=\left\lfloor\frac{\min\left(\frac{E r}{d+2Pc},\,\frac{E a}{P(1+c)},\,\frac{E}{P(1+c)}\right)}{L}\right\rfloor L')
    st.caption('E=ekuitas tersedia; r=risiko per trade; d=jarak stop (ATR × multiplier); P=harga masuk; '
               'c=fee+slippage per sisi; a=alokasi maksimum; L=ukuran lot. Semua eksekusi disimulasikan dengan saham bulat.')
    atr = float(latest['ATR_14'])
    assumed_entry = float(latest['Close'])  # illustrative, not a fill or forecast
    distance = max(atr*stop_atr, assumed_entry*.005)
    cost = (fee_bps+slip_bps)/10_000
    q_risk = (capital*risk_pct/100)/(distance + 2*assumed_entry*cost)
    q_cap = capital*max_position_pct/100/(assumed_entry*(1+cost))
    lot = 100 if market=='IDX' else 1
    illustrative_qty = int(min(q_risk,q_cap,capital/(assumed_entry*(1+cost)))//lot)*lot
    c1,c2,c3 = st.columns(3)
    c1.metric('Jarak stop estimasi',m(distance,currency))
    c2.metric('Batas rugi target (sebelum gap)',m(capital*risk_pct/100,currency))
    c3.metric('Ukuran posisi ilustratif',f'{illustrative_qty:,} saham')
    st.caption('Contoh memakai harga penutupan terakhir, BUKAN harga entry. Entry aktual, jumlah lot, dan stop dihitung lagi pada open T+1; gap dapat menyebabkan rugi melampaui target.')
    st.markdown('#### Proteksi dasar MVP')
    st.markdown('- Tidak ada leverage, short selling, atau auto-order.\n- Risiko per posisi dan batas alokasi diatur sebelum entry.\n- Stop ATR berlaku dari hari entry; gap stop keluar di harga open yang lebih buruk.\n- Fee dan slippage dikenakan pada dua sisi transaksi.\n- Tidak membuka posisi kedua selama posisi pertama aktif.\n- Untuk IDX, jumlah saham dibulatkan ke lot 100.')
    st.info(f'Jumlah sinyal yang dilewati karena modal/lot tidak cukup dalam simulasi OOS: **{oos.skipped_small}**.')

with panels[3]:
    st.markdown('#### Integritas data & desain riset')
    st.write(f'**Sumber:** {source}  ·  **Jumlah candle:** {len(df):,}  ·  **Periode:** {df.index[0].date()} s.d. {df.index[-1].date()}')
    st.write(f'**Model:** {STRATEGY_LABELS[strategy]}  ·  **Holding:** {holding} sesi  ·  **Likuiditas minimum:** {m(min_turnover,currency)}')
    st.markdown('''
**Aturan waktu dan simulasi:**
1. Indikator dan sinyal dihitung setelah penutupan sesi **T**.
2. Perintah beli, jika memenuhi syarat, terjadi pada **open T+1**.
3. Stop dihitung dari ATR yang sudah diketahui pada **T**; stop intraday dieksekusi konservatif.
4. Exit normal pada open setelah durasi 1–5 sesi, kecuali stop lebih dahulu.
5. Rentang OOS adalah bagian kronologis terakhir; **parameter tidak dioptimasi** oleh aplikasi ini.
6. Trade log, ekuitas harian, biaya, dan ukuran lot dapat diperiksa ulang.

**Yang belum tercakup:** dividen tunai dan total return; pemodelan corporate action lengkap,
perubahan lot/tick, bid-ask order book, suspensi/limit harga, price impact, kalender bursa,
perbandingan lintas mata uang dan pajak spesifik akun. Tidak boleh dipakai sebagai simulasi eksekusi presisi.
''')
    csv = raw.reset_index().to_csv(index=False).encode('utf-8')
    st.download_button('Unduh data OHLCV yang dipakai (.csv)',csv,
                       file_name=f'quant_data_{ticker.replace(".","_")}.csv',mime='text/csv')
    st.markdown('**Format CSV:** `Date,Open,High,Low,Close,Volume` (tanggal ISO 8601, satu bar per tanggal).')
    st.caption('Untuk versi komersial, gunakan vendor data berlisensi dan proses validasi aksi korporasi serta kalender resmi di masing-masing pasar.')

st.divider()
st.caption('QUANT SIGNAL AI · Prototipe riset edukasi · Bukan reproduksi algoritma Medallion Fund · Tidak ada jaminan kinerja.')
