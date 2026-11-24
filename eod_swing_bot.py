import os
import io
import sys
import json
import time
import zipfile
import sqlite3
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
from google import genai
from huggingface_hub import HfApi, hf_hub_download

# ==============================================================================
# 🔐 CREDENTIALS & CONFIG
# ==============================================================================
IST = timezone(timedelta(hours=5, minutes=30))
def now_ist() -> datetime:
    return datetime.now(IST).replace(tzinfo=None)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = str(os.environ.get("TELEGRAM_CHAT_ID", "475421574"))

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_DATASET_REPO = "avik1289/phoenix-market-data"

DB_FILE = "market_research.db"
FUND_DB_FILE = "fundamental_master.csv"
EOD_STATE_FILE = "eod_bot_state.json"

NSE_SECTOR_TICKERS = {
    "NIFTY_BANK": "^NSEBANK", "NIFTY_IT": "^CNXIT", "NIFTY_AUTO": "^CNXAUTO",
    "NIFTY_METAL": "^CNXMETAL", "NIFTY_PHARMA": "^CNXPHARMA", "NIFTY_FMCG": "^CNXFMCG",
    "NIFTY_ENERGY": "^CNXENERGY", "NIFTY_INFRA": "^CNXINFRA", "NIFTY_REALTY": "^CNXREALTY",
    "NIFTY_MEDIA": "^CNXMEDIA", "NIFTY_CONSUMPTION": "^CNXCONSUM", "NIFTY_COMMODITIES": "^CNXCMDT"
}

SECTOR_INDEX_MAP = {
    'Automobile and Auto Components': '^CNXAUTO',
    'Financial Services': '^NSEBANK', 'Banking': '^NSEBANK',
    'Information Technology': '^CNXIT', 'IT': '^CNXIT',
    'Healthcare': '^CNXPHARMA', 'Pharmaceuticals': '^CNXPHARMA',
    'Fast Moving Consumer Goods': '^CNXFMCG', 'FMCG': '^CNXFMCG',
    'Metals & Mining': '^CNXMETAL', 'Metal': '^CNXMETAL',
    'Oil Gas & Consumable Fuels': '^CNXENERGY', 'Energy': '^CNXENERGY', 'Power': '^CNXENERGY',
    'Realty': '^CNXREALTY', 'Real Estate': '^CNXREALTY',
    'Media Entertainment & Publication': '^CNXMEDIA', 'Media': '^CNXMEDIA',
    'Capital Goods': '^CNXINFRA', 'Construction': '^CNXINFRA',
    'Construction Materials': '^CNXINFRA', 'Infrastructure': '^CNXINFRA',
    'Consumer Durables': '^CNXCONSUM', 'Consumer Services': '^CNXCONSUM',
    'Telecommunication': '^CNXINFRA', 'Chemicals': '^CNXCMDT', 'Textiles': '^CNXCONSUM',
    'Others': '^NSEI'
}

# ==============================================================================
# ☁️ HUGGING FACE CLOUD SYNC & TELEGRAM HELPERS
# ==============================================================================
def sync_from_hf():
    if not HF_TOKEN:
        return
    for fname in [DB_FILE, EOD_STATE_FILE]:
        try:
            hf_hub_download(
                repo_id=HF_DATASET_REPO, filename=fname, repo_type="dataset",
                token=HF_TOKEN, local_dir=".", force_download=True
            )
            print(f"☁️ Downloaded {fname}")
        except Exception as e:
            print(f"ℹ️ {fname} not found in HF ({e})")

def sync_to_hf(files=None):
    if not HF_TOKEN:
        return
    if files is None:
        files = [DB_FILE, EOD_STATE_FILE]
    api = HfApi(token=HF_TOKEN)
    for fname in files:
        if os.path.exists(fname):
            try:
                api.upload_file(
                    path_or_fileobj=fname, path_in_repo=fname,
                    repo_id=HF_DATASET_REPO, repo_type="dataset",
                    commit_message=f"EOD Sync {fname} [{now_ist().strftime('%Y-%m-%d %H:%M IST')}]"
                )
                print(f"☁️ Uploaded {fname}")
            except Exception as e:
                print(f"⚠️ HF Upload Error ({fname}): {e}")

def send_telegram(msg: str, chat_id: str = TELEGRAM_CHAT_ID):
    if not TELEGRAM_BOT_TOKEN:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for chunk in [msg[i:i+4000] for i in range(0, len(msg), 4000)]:
        try:
            requests.post(url, json={"chat_id": chat_id, "text": chunk}, timeout=15)
        except Exception as e:
            print(f"Telegram Error: {e}")

def get_gemini_response(prompt: str, fallback_text: str = "") -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)
    models = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    last_err = ""
    for attempt in range(4):
        m = models[attempt % len(models)]
        try:
            resp = client.models.generate_content(model=m, contents=prompt)
            if resp and resp.text:
                return resp.text
        except Exception as e:
            last_err = str(e)
            time.sleep(3)
    return fallback_text if fallback_text else f"⚠️ AI সামারি ব্যস্ত ({last_err[:60]}), তবে ডেটা সিঙ্ক সম্পন্ন।"

# ==============================================================================
# ১. NSE Cash Bhavcopy ডাউনলোড ও 200-Day Rolling DB আপডেট
# ==============================================================================
def auto_sync_nse_db(days_back=7) -> int:
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS equity_history (
            trade_date TEXT, symbol TEXT, prev_close REAL, open_price REAL,
            high_price REAL, low_price REAL, close_price REAL, ttq REAL,
            nt REAL, deliv_qty REAL, deliv_per REAL, hpt REAL, money_flow REAL,
            price_chg_per REAL, PRIMARY KEY (trade_date, symbol)
        )
    """)
    existing_dates = set(pd.read_sql("SELECT DISTINCT trade_date FROM equity_history", conn)['trade_date'].tolist())
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Referer": "https://www.nseindia.com/"}

    added = 0
    for i in range(days_back):
        dt = now_ist().date() - timedelta(days=i)
        if dt.weekday() >= 5:
            continue
        dt_str = dt.strftime("%Y-%m-%d")
        if dt_str in existing_dates:
            continue

        url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{dt.strftime('%d%m%Y')}.csv"
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200 and "SYMBOL" in r.text:
                df_day = pd.read_csv(io.StringIO(r.text))
                df_day.columns = df_day.columns.str.strip()
                df_day = df_day[df_day['SERIES'].astype(str).str.strip() == 'EQ'].copy()
                df_day['SYMBOL'] = df_day['SYMBOL'].astype(str).str.strip()
                df_day = df_day[~df_day['SYMBOL'].str.endswith(('ETF', 'BEES', 'IETF', 'LIQUID'))]

                clean_df = pd.DataFrame({
                    'trade_date': dt_str, 'symbol': df_day['SYMBOL'],
                    'prev_close': pd.to_numeric(df_day['PREV_CLOSE'], errors='coerce'),
                    'open_price': pd.to_numeric(df_day['OPEN_PRICE'], errors='coerce'),
                    'high_price': pd.to_numeric(df_day['HIGH_PRICE'], errors='coerce'),
                    'low_price': pd.to_numeric(df_day['LOW_PRICE'], errors='coerce'),
                    'close_price': pd.to_numeric(df_day['CLOSE_PRICE'], errors='coerce'),
                    'ttq': pd.to_numeric(df_day['TTL_TRD_QNTY'], errors='coerce'),
                    'nt': pd.to_numeric(df_day['NO_OF_TRADES'], errors='coerce'),
                    'deliv_qty': pd.to_numeric(df_day['DELIV_QTY'], errors='coerce'),
                    'deliv_per': pd.to_numeric(df_day['DELIV_PER'], errors='coerce')
                })
                clean_df['hpt'] = round(clean_df['ttq'] / clean_df['nt'].replace(0, np.nan), 3)
                clean_df['money_flow'] = round(clean_df['close_price'] * clean_df['ttq'], 2)
                clean_df['price_chg_per'] = round(((clean_df['close_price'] - clean_df['prev_close']) / clean_df['prev_close']) * 100, 2)
                clean_df.to_sql("equity_history", conn, if_exists="append", index=False)
                added += 1
        except Exception as e:
            print(f"Bhavcopy Error ({dt_str}): {e}")

    all_dates = sorted(pd.read_sql("SELECT DISTINCT trade_date FROM equity_history", conn)['trade_date'].tolist())
    if len(all_dates) > 200:
        conn.execute("DELETE FROM equity_history WHERE trade_date < ?", (all_dates[-200],))
        conn.commit()
        conn.execute("VACUUM")
    conn.close()
    return added

# ==============================================================================
# ২. Swing RS + Supertrend & Sector Ranking & F&O Option S/R
# ==============================================================================
def get_sector_ranking() -> dict:
    stats = []
    for name, ticker in NSE_SECTOR_TICKERS.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if len(hist) >= 2:
                pc, cc = float(hist["Close"].iloc[-2]), float(hist["Close"].iloc[-1])
                stats.append({"sector": name, "close": round(cc, 2), "day_chg_%": round(((cc - pc) / pc) * 100, 2)})
        except Exception:
            pass
    stats.sort(key=lambda x: x["day_chg_%"], reverse=True)
    return {"best": stats[:3], "worst": list(reversed(stats[-3:])), "all": stats}

def compute_swing_metrics(df: pd.DataFrame) -> pd.DataFrame:
    recent_dates = sorted(df['trade_date'].unique())[-95:]
    sub = df[df['trade_date'].isin(recent_dates)].sort_values(['symbol', 'trade_date']).reset_index(drop=True).copy()
    unique_tickers = list(set(SECTOR_INDEX_MAP.values()) | {'^NSEI'})
    start_dt = pd.to_datetime(sub['trade_date'].min()) - pd.Timedelta(days=15)

    idx_records = []
    for ticker in unique_tickers:
        try:
            hist = yf.Ticker(ticker).history(start=start_dt.strftime('%Y-%m-%d'))
            if not hist.empty:
                hist = hist.reset_index()
                hist['trade_date'] = pd.to_datetime(hist['Date']).dt.strftime('%Y-%m-%d')
                hist['index_ticker'] = ticker
                hist['idx_close'] = hist['Close']
                idx_records.append(hist[['trade_date', 'index_ticker', 'idx_close']])
        except Exception:
            pass

    idx_df = pd.concat(idx_records, ignore_index=True) if idx_records else pd.DataFrame(columns=['trade_date', 'index_ticker', 'idx_close'])
    sub['sec_ticker'] = sub['sector'].map(SECTOR_INDEX_MAP).fillna('^NSEI') if 'sector' in sub.columns else '^NSEI'
    nifty_df = idx_df[idx_df['index_ticker'] == '^NSEI'][['trade_date', 'idx_close']].rename(columns={'idx_close': 'nifty_close'})
    sub = sub.merge(nifty_df, on='trade_date', how='left')
    sec_df = idx_df[['trade_date', 'index_ticker', 'idx_close']].rename(columns={'index_ticker': 'sec_ticker', 'idx_close': 'sec_close'})
    sub = sub.merge(sec_df, on=['trade_date', 'sec_ticker'], how='left')
    sub['sec_close'] = sub['sec_close'].fillna(sub['nifty_close'])

    sub = sub.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
    max_days = sub.groupby('symbol')['trade_date'].transform('count').max()
    lookback = 55 if max_days >= 56 else max(5, int(max_days) - 2)

    sub['RS_MKT'] = round((sub['close_price'] / sub['nifty_close']).groupby(sub['symbol']).pct_change(periods=lookback) * 100, 2)
    sub['RS_SEC'] = round((sub['close_price'] / sub['sec_close']).groupby(sub['symbol']).pct_change(periods=lookback) * 100, 2)

    def calc_st(g):
        g = g.sort_values('trade_date').copy()
        h, l, c = g['high_price'].values, g['low_price'].values, g['close_price'].values
        n = len(g)
        if n < 12:
            g['SUPERTREND'], g['ST_SIGNAL'] = np.nan, 'N/A'
            return g
        prev_c = np.roll(c, 1); prev_c[0] = c[0]
        tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
        atr = pd.Series(tr).ewm(alpha=1/10, adjust=False).mean().values
        hl2 = (h + l) / 2.0
        ub, lb = hl2 + (3.0 * atr), hl2 - (3.0 * atr)
        uptrend = np.ones(n, dtype=bool); st_val = np.zeros(n)
        for i in range(1, n):
            if c[i] > ub[i - 1]: uptrend[i] = True
            elif c[i] < lb[i - 1]: uptrend[i] = False
            else:
                uptrend[i] = uptrend[i - 1]
                if uptrend[i] and lb[i] < lb[i - 1]: lb[i] = lb[i - 1]
                if not uptrend[i] and ub[i] > ub[i - 1]: ub[i] = ub[i - 1]
            st_val[i] = lb[i] if uptrend[i] else ub[i]
        g['SUPERTREND'], g['ST_SIGNAL'] = np.round(st_val, 2), np.where(uptrend, 'Bullish', 'Bearish')
        return g

    syms_bak = sub['symbol'].copy()
    sub = sub.groupby('symbol', group_keys=False).apply(calc_st).reset_index(drop=True)
    if 'symbol' not in sub.columns:
        sub['symbol'] = syms_bak.values
    return sub

def fetch_fno_eod_sr_and_iv(df: pd.DataFrame, latest_dt: str):
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com/"}
    ymd = pd.to_datetime(latest_dt).strftime("%Y%m%d")
    url = f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{ymd}_F_0000.csv.zip"
    fno_syms = set()
    above_res, below_sup = [], []

    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                fo_df = pd.read_csv(z.open(z.namelist()[0]))
            sto = fo_df[fo_df['FinInstrmTp'] == 'STO'].copy()
            fno_syms = set(sto['TckrSymb'].unique())
            ce_grp = sto[sto['OptnTp'] == 'CE'].groupby(['TckrSymb', 'StrkPric'], as_index=False)['OpnIntrst'].sum()
            pe_grp = sto[sto['OptnTp'] == 'PE'].groupby(['TckrSymb', 'StrkPric'], as_index=False)['OpnIntrst'].sum()
            max_ce = ce_grp.sort_values('OpnIntrst', ascending=False).drop_duplicates('TckrSymb').set_index('TckrSymb')
            max_pe = pe_grp.sort_values('OpnIntrst', ascending=False).drop_duplicates('TckrSymb').set_index('TckrSymb')

            day_spot = df[df['trade_date'] == latest_dt].set_index('symbol')
            for sym in fno_syms:
                if sym in day_spot.index and sym in max_ce.index and sym in max_pe.index:
                    spot = float(day_spot.loc[sym, 'close_price'])
                    ce_stk = float(max_ce.loc[sym, 'StrkPric'])
                    pe_stk = float(max_pe.loc[sym, 'StrkPric'])
                    if spot > ce_stk:
                        above_res.append({"symbol": sym, "close": spot, "Max_Call_OI": ce_stk})
                    elif spot < pe_stk:
                        below_sup.append({"symbol": sym, "close": spot, "Max_Put_OI": pe_stk})
    except Exception:
        pass

    iv_crush_df = pd.DataFrame()
    if fno_syms:
        recent_90 = sorted(df['trade_date'].unique())[-90:]
        fno_hist = df[(df['symbol'].isin(fno_syms)) & (df['trade_date'].isin(recent_90))].sort_values(['symbol', 'trade_date']).copy()
        log_ret = np.log(fno_hist['close_price'] / fno_hist.groupby('symbol')['close_price'].shift(1).replace(0, np.nan))
        fno_hist['IV_%'] = np.round(log_ret.groupby(fno_hist['symbol']).transform(lambda x: x.rolling(14, min_periods=3).std() * np.sqrt(252) * 100), 2)
        fno_hist['IV_CHG'] = np.round(fno_hist['IV_%'] - fno_hist.groupby('symbol')['IV_%'].shift(1), 2)
        iv_min = fno_hist.groupby('symbol')['IV_%'].transform('min')
        iv_max = fno_hist.groupby('symbol')['IV_%'].transform('max')
        fno_hist['IVR_%'] = np.round(np.where((iv_max - iv_min) > 0, ((fno_hist['IV_%'] - iv_min) / (iv_max - iv_min)) * 100, 50.0), 1)
        fno_hist['IVP_%'] = np.round(fno_hist.groupby('symbol')['IV_%'].rank(pct=True) * 100, 1)
        fno_today = fno_hist[fno_hist['trade_date'] == latest_dt]
        iv_crush_df = fno_today[(fno_today['IVR_%'] > 85) & (fno_today['IVP_%'] > 85) & (fno_today['IV_CHG'] < 0)].sort_values(by='IVP_%', ascending=False)

    return pd.DataFrame(above_res), pd.DataFrame(below_sup), iv_crush_df

# ==============================================================================
# ৩. SCREENERS (SWING & FUNDAMENTAL INVESTING) + EOD REPORT
# ==============================================================================
def get_screened_lists():
    if not os.path.exists(DB_FILE):
        return None
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM equity_history", conn)
    conn.close()
    if df.empty:
        return None

    df.columns = [c.lower() for c in df.columns]
    df = df.drop_duplicates(subset=['trade_date', 'symbol']).sort_values(['symbol', 'trade_date'])

    if os.path.exists("nse_stock_metadata.csv"):
        meta = pd.read_csv("nse_stock_metadata.csv")
        meta.columns = [c.strip().lower() for c in meta.columns]
        if 'symbol' in meta.columns:
            keep = [c for c in ['symbol', 'sector', 'market cap'] if c in meta.columns]
            df = df.merge(meta[keep].drop_duplicates(subset=['symbol']), on='symbol', how='left')

    delta = df.groupby('symbol')['close_price'].diff()
    avg_g = delta.clip(lower=0).groupby(df['symbol']).transform(lambda x: x.ewm(alpha=1/14, min_periods=14, adjust=False).mean())
    avg_l = (-delta.clip(upper=0)).groupby(df['symbol']).transform(lambda x: x.ewm(alpha=1/14, min_periods=14, adjust=False).mean())
    df['rsi_14'] = round(100 - (100 / (1 + (avg_g / avg_l.replace(0, np.nan)))), 2)

    latest_dt = df['trade_date'].max()
    day_df = df[df['trade_date'] == latest_dt].copy()

    swing_calc = compute_swing_metrics(df)
    swing_today = swing_calc[swing_calc['trade_date'] == latest_dt]
    screened_swing = swing_today[
        (swing_today['RS_MKT'] > 0) & (swing_today['RS_SEC'] > 0) &
        (swing_today['ST_SIGNAL'] == 'Bullish') & (swing_today['rsi_14'] >= 60) &
        (swing_today['close_price'] >= 50)
    ].sort_values(by=['RS_SEC', 'deliv_per'], ascending=[False, False]).head(15)

    screened_invest = pd.DataFrame()
    if os.path.exists(FUND_DB_FILE):
        fund_df = pd.read_csv(FUND_DB_FILE)
        merged = day_df.merge(fund_df, on='symbol', how='inner')
        screened_invest = merged[
            (merged['Market_Cap_Cr'].fillna(0) > 2000) &
            (merged['Pledged_%'].fillna(0) < 30.0) &
            (merged['EPS_Latest_Q'] > merged['EPS_Preceding_Q']) &
            (merged['Current_PE'] < merged['Historical_PE_5Y']) &
            ((merged['Promoter_Chg_%'].fillna(0) > 0.5) | (merged['FII_Chg_%'].fillna(0) > 0.5))
        ].sort_values(by=['deliv_per', 'Market_Cap_Cr'], ascending=[False, False]).head(12)

    return df, latest_dt, screened_swing, screened_invest

def run_nightly_eod_report():
    added = auto_sync_nse_db(days_back=5)
    res = get_screened_lists()
    if not res:
        return
    df, latest_dt, swing_df, invest_df = res
    above_res, below_sup, iv_crush = fetch_fno_eod_sr_and_iv(df, latest_dt)
    sectors = get_sector_ranking()

    fb_eod = (
        f"• *Swing Cash Picks ({len(swing_df)}):* {', '.join(swing_df['symbol'].head(8).tolist())}\n"
        f"• *Investing Picks ({len(invest_df)}):* {', '.join(invest_df['symbol'].head(8).tolist())}\n"
        f"• *IV Crush Candidates:* {', '.join(iv_crush['symbol'].head(6).tolist()) if not iv_crush.empty else 'None'}"
    )

    prompt = f"""
    You are Phoenix Swing, Fundamental Investing & F&O EOD Research Analyst. Date: {latest_dt}.
    1. SECTORS: Best={json.dumps(sectors['best'])} | Worst={json.dumps(sectors['worst'])}
    2. SWING CASH (RS_MKT>0, RS_SEC>0, Supertrend Bullish, RSI>=60):
    {swing_df[['symbol', 'close_price', 'RS_MKT', 'RS_SEC', 'rsi_14', 'deliv_per']].to_csv(index=False) if not swing_df.empty else "None"}
    3. FUNDAMENTAL INVESTING (PE Re-Rating + QoQ EPS Growth + Promoter/FII Buying):
    {invest_df[['symbol', 'close_price', 'Current_PE', 'Historical_PE_5Y', 'Promoter_Chg_%', 'FII_Chg_%', 'deliv_per']].to_csv(index=False) if not invest_df.empty else "None"}
    4. IV CRUSH: {iv_crush[['symbol', 'close_price', 'IV_%', 'IVR_%', 'IVP_%']].to_csv(index=False) if not iv_crush.empty else "None"}
    5. OPTION OI BREAKOUTS: Above Call OI={above_res.head(8).to_dict('records')} | Below Put OI={below_sup.head(8).to_dict('records')}

    Write a crisp, actionable Nightly EOD Report in Bengali for Telegram.
    """
    report = get_gemini_response(prompt, fallback_text=fb_eod)
    send_telegram(f"🌙 *Phoenix Nightly Swing & Investing EOD [{latest_dt}]*\n📥 DB Sync: +{added}d (200D Saved)\n\n{report}")

# ==============================================================================
# ৪. TELEGRAM Q&A HANDLER (SWING / INVESTING / SPECIFIC STOCK)
# ==============================================================================
def process_telegram_questions():
    if not TELEGRAM_BOT_TOKEN:
        return
    state = {"tg_offset": 0}
    if os.path.exists(EOD_STATE_FILE):
        try:
            with open(EOD_STATE_FILE, "r") as f:
                state = json.load(f)
        except Exception:
            pass

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    try:
        resp = requests.get(url, params={"offset": state.get("tg_offset", 0), "timeout": 2}, timeout=8).json()
        updates = resp.get("result", [])
        if not updates:
            return

        res = get_screened_lists()
        for upd in updates:
            state["tg_offset"] = upd["update_id"] + 1
            msg = upd.get("message", {})
            if str(msg.get("chat", {}).get("id", "")) != TELEGRAM_CHAT_ID:
                continue
            text = msg.get("text", "").strip()
            if not text:
                continue

            if text.lower() == "/eod":
                run_nightly_eod_report()
                continue

            q_upper = text.upper()
            context_payload = {}
            if res:
                df, latest_dt, swing_df, invest_df = res
                context_payload["latest_date"] = latest_dt
                context_payload["top_investing_stocks"] = invest_df.to_dict(orient="records") if not invest_df.empty else []
                context_payload["top_swing_stocks"] = swing_df.to_dict(orient="records") if not swing_df.empty else []

                # Check if user asked about a specific stock symbol
                all_syms = set(df['symbol'].unique())
                matched = [w for w in q_upper.replace("?", "").replace(",", " ").split() if w in all_syms]
                if matched:
                    sym = matched[0]
                    hist_5d = df[df['symbol'] == sym].tail(5).to_dict(orient="records")
                    fund_row = []
                    if os.path.exists(FUND_DB_FILE):
                        fdf = pd.read_csv(FUND_DB_FILE)
                        fund_row = fdf[fdf['symbol'] == sym].to_dict(orient="records")
                    context_payload["specific_stock"] = {"symbol": sym, "recent_5d": hist_5d, "fundamentals": fund_row}

            prompt = f"""
            You are Phoenix Swing & Fundamental Investing AI Research Analyst.
            User Question: "{text}"
            Database Context: {json.dumps(context_payload, indent=2)}
            Answer clearly in Bengali with exact numbers (Close Price, Current PE vs 5Y PE, QoQ EPS, Promoter/FII Chg, RS_MKT, RS_SEC, RSI, Delivery %).
            """
            send_telegram(get_gemini_response(prompt))

        with open(EOD_STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception as e:
        print(f"TG Q&A Error: {e}")

if __name__ == "__main__":
    sync_from_hf()
    process_telegram_questions()
    if "--qa-only" not in sys.argv:
        run_nightly_eod_report()
    sync_to_hf()
