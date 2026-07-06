"""
=====================================================================
🏛️ QUANT AI - SOVEREIGN COCKPIT V60.0 (TEK DOSYA, BİRLEŞİK VERİ KATMANI)
=====================================================================
Bu sürüm V58.0'ın TÜM özelliklerini (6 sekme, tüm osilatörler/
göstergeler, 3'lü konsensüs ML motoru, walk-forward CV, backtest,
risk yönetimi) korur. DEĞİŞEN TEK ŞEY veri çekme katmanıdır:

  ESKİ (V58.0):  yf.download(...) -> SADECE Yahoo, SADECE Yahoo'nun
                 kısıtlı interval seti (4h gibi gerçek interval'lar
                 yoktu, "4 saat" etiketi altında gizlice 1h çekiliyordu)

  YENİ (V60.0):  get_market_data(...) -> KRIPTO için Binance Public
                 API (API KEY GEREKMEZ), gerçek native 1m,3m,5m,15m,
                 30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,1M interval'ları;
                 TR_HISSE/ABD_HISSE/EMTIA için Yahoo'nun tüm native
                 interval'ları (1m,2m,5m,15m,30m,60m,90m,1d,5d,1wk,
                 1mo,3mo). Yahoo'da olmayan interval'lar (örn. 4h)
                 otomatik + AÇIKÇA ETİKETLENMİŞ resample ile üretilir.

ÇALIŞTIRMA:
    pip install streamlit yfinance pandas numpy scikit-learn requests xgboost
    streamlit run dashboard_v60_birlesik.py
=====================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import streamlit as str_ui
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

np.random.seed(42)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


# #####################################################################
# #####################################################################
#  BÖLÜM A — BİRLEŞİK VERİ SAĞLAYICI KATMANI (eski data_provider.py)
# #####################################################################
# #####################################################################

# =====================================================================
# 📚 A.1 NATIVE INTERVAL TANIMLARI
# =====================================================================

YAHOO_NATIVE_INTERVALS = {
    "1m", "2m", "5m", "15m", "30m",
    "60m", "90m",
    "1d", "5d", "1wk", "1mo", "3mo",
}

BINANCE_NATIVE_INTERVALS = {
    "1s", "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d",
    "1w", "1M",
}

BINANCE_BASE_URL = "https://api.binance.com"
BINANCE_KLINES_ENDPOINT = "/api/v3/klines"
BINANCE_MAX_LIMIT = 1000

# ÖNEMLİ: "90m" burada BİLEREK "2h"den SONRAYA konmuştur. Sebebi: 90
# dakika; 4h/6h/8h/12h gibi saat-bazlı interval'lara TAM BÖLÜNMEZ
# (90*N hiçbir zaman 240, 360, 480, 720 dakikaya tam denk gelmez), bu
# yüzden resample kaynağı olarak seçilirse YANLIŞ hizalanmış mumlar
# üretir. "90m" sırada geriye doğru ilk taranan native aday olmasın
# diye bilerek 4h/6h/8h/12h'nin ARDINA alınmıştır; böylece bu
# interval'lar için Yahoo'da her zaman "60m" (60 dakika, tüm saatlik
# hedeflere tam bölünür) native kaynak olarak seçilir.
INTERVAL_ORDER = [
    "1s", "1m", "2m", "3m", "5m", "15m", "30m",
    "60m", "1h", "2h",
    "4h", "6h", "8h", "12h",
    "90m",
    "1d", "3d", "5d",
    "1wk", "1w",
    "1mo", "1M", "3mo",
]


@dataclass
class FetchResult:
    df: pd.DataFrame
    source: str
    requested_interval: str
    actual_native_interval: str
    is_resampled: bool
    warning: Optional[str] = None


# =====================================================================
# 🧮 A.2 RESAMPLE YARDIMCILARI
# =====================================================================

def _interval_to_pandas_rule(interval: str) -> str:
    mapping = {
        "1s": "1s", "1m": "1min", "2m": "2min", "3m": "3min", "5m": "5min",
        "15m": "15min", "30m": "30min",
        "60m": "1h", "1h": "1h", "90m": "90min", "2h": "2h",
        "4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h",
        "1d": "1D", "3d": "3D", "5d": "5D",
        "1wk": "1W", "1w": "1W",
        "1mo": "1MS", "1M": "1MS", "3mo": "3MS",
    }
    if interval not in mapping:
        raise ValueError(f"Bilinmeyen interval, resample kuralı yok: {interval}")
    return mapping[interval]


def _resample_ohlcv(df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
    if df.empty:
        return df
    rule = _interval_to_pandas_rule(target_interval)
    agg_map = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    agg_map = {k: v for k, v in agg_map.items() if k in df.columns}
    out = df.resample(rule).agg(agg_map)
    out = out.dropna(subset=["Close"]) if "Close" in out.columns else out.dropna()
    return out


def _find_best_native_source_interval(target_interval: str, native_set: set, order: list) -> Optional[str]:
    if target_interval in native_set:
        return target_interval
    try:
        target_idx = order.index(target_interval)
    except ValueError:
        return None
    for candidate in reversed(order[:target_idx]):
        if candidate in native_set:
            return candidate
    return None


# =====================================================================
# 🟡 A.3 YAHOO FINANCE VERİ ÇEKİCİ
# =====================================================================

def _period_string_to_days(period: str) -> Optional[int]:
    """
    '30d','60d','1y','2y','3y','4y','5y','max' gibi BİZİM kullandığımız
    serbest period string'lerini gün sayısına çevirir. Yahoo'nun resmi
    whitelist'i (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max) yalnızca
    `period=` parametresi DOĞRUDAN Yahoo'ya geçildiğinde geçerlidir;
    biz bu kısıtı tamamen ortadan kaldırmak için period'u GÜN SAYISINA
    çevirip start/end tarih aralığı ile istek yapıyoruz (yfinance'in
    start/end parametreleri herhangi bir whitelist'e tabi değildir).
    """
    if period is None or period == "":
        return None
    p = period.strip().lower()
    if p == "max":
        return None  # None -> start vermeyip Yahoo'nun mevcut en eski veriyi dönmesini sağlarız
    try:
        if p.endswith("mo"):
            return int(float(p[:-2]) * 30)
        if p.endswith("wk"):
            return int(float(p[:-2]) * 7)
        if p.endswith("y"):
            return int(float(p[:-1]) * 365)
        if p.endswith("d"):
            return int(float(p[:-1]))
    except ValueError:
        return None
    return None


# Yahoo'nun intraday interval'lar için pratik geriye dönük lookback
# sınırları (bu sınırlar Yahoo'nun sunucu tarafı kısıtıdır; aşılırsa
# boş veri/hata gelir). start/end ile istek yapsak bile bu sınırlar
# GEÇERLİ KALIR — bu yüzden gün sayısını burada otomatik tavanlıyoruz.
YAHOO_INTRADAY_LOOKBACK_CAP_DAYS = {
    "1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60,
    "60m": 730, "90m": 60,
}


def _clean_yahoo_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def fetch_yahoo(symbol: str, period: str, interval: str) -> FetchResult:
    native_interval = _find_best_native_source_interval(interval, YAHOO_NATIVE_INTERVALS, INTERVAL_ORDER)

    if native_interval is None:
        return FetchResult(
            df=pd.DataFrame(), source="yahoo", requested_interval=interval,
            actual_native_interval="", is_resampled=False,
            warning=f"'{interval}' için Yahoo'da uygun bir kaynak interval bulunamadı.",
        )

    # 🛡️ KRİTİK DÜZELTME: Yahoo'nun period= whitelist'i (1d,5d,1mo,3mo,
    # 6mo,1y,2y,5y,10y,ytd,max) bizim '30d','45d','1.5y' gibi serbest
    # değerlerimizle ASLA uyuşmaz ve "Period '...' is invalid" hatası
    # verir. Bunun önüne geçmek için period= YERİNE start/end tarih
    # aralığı kullanıyoruz; bu parametre çifti whitelist'e tabi değildir.
    requested_days = _period_string_to_days(period)
    lookback_cap = YAHOO_INTRADAY_LOOKBACK_CAP_DAYS.get(native_interval)
    if lookback_cap is not None and (requested_days is None or requested_days > lookback_cap):
        requested_days = lookback_cap  # Yahoo'nun sunucu tarafı sınırına otomatik tavanla

    download_kwargs = dict(interval=native_interval, progress=False)
    if requested_days is None:
        # 'max' istenmiş ve intraday değilse: period='max' GÜVENLE
        # kullanılabilir (whitelist'te zaten var).
        download_kwargs["period"] = "max"
    else:
        end_dt = pd.Timestamp.now("UTC").tz_localize(None)
        start_dt = end_dt - pd.Timedelta(days=requested_days)
        download_kwargs["start"] = start_dt.strftime("%Y-%m-%d")
        download_kwargs["end"] = (end_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    raw = yf.download(symbol, **download_kwargs)
    if raw.empty:
        return FetchResult(
            df=pd.DataFrame(), source="yahoo", requested_interval=interval,
            actual_native_interval=native_interval, is_resampled=False,
            warning=f"Yahoo '{symbol}' için '{native_interval}' interval'ında veri döndürmedi.",
        )

    raw = _clean_yahoo_columns(raw)

    if native_interval == interval:
        return FetchResult(df=raw, source="yahoo", requested_interval=interval,
                            actual_native_interval=native_interval, is_resampled=False)

    resampled = _resample_ohlcv(raw, interval)
    return FetchResult(
        df=resampled, source="yahoo_resampled", requested_interval=interval,
        actual_native_interval=native_interval, is_resampled=True,
        warning=f"Yahoo native olarak '{interval}' sunmuyor. '{native_interval}' verisinden resample edildi.",
    )


# =====================================================================
# 🟠 A.4 BINANCE PUBLIC API VERİ ÇEKİCİ (API KEY GEREKMEZ)
# =====================================================================

def _binance_symbol_from_yahoo_style(symbol: str) -> str:
    s = symbol.upper().strip()
    if "-" in s:
        base, quote = s.split("-", 1)
        if quote == "USD":
            quote = "USDT"
        return f"{base}{quote}"
    return s


def _period_to_lookback_days(period: str) -> Optional[int]:
    if period in (None, "", "max"):
        return None
    period = period.strip().lower()
    try:
        if period.endswith("d"):
            return int(float(period[:-1]))
        if period.endswith("mo"):
            return int(float(period[:-2]) * 30)
        if period.endswith("y"):
            return int(float(period[:-1]) * 365)
        if period.endswith("wk"):
            return int(float(period[:-2]) * 7)
    except ValueError:
        return None
    return None


def _binance_interval_to_ms(interval: str) -> int:
    units = {"s": 1_000, "m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}
    if interval == "1M":
        return 30 * 86_400_000
    unit = interval[-1]
    value = int(interval[:-1])
    return value * units[unit]


def _fetch_binance_klines_paged(symbol: str, interval: str, lookback_days: Optional[int],
                                 max_candles_cap: int = 20000) -> pd.DataFrame:
    interval_ms = _binance_interval_to_ms(interval)
    end_time_ms = int(time.time() * 1000)

    start_time_ms = end_time_ms - lookback_days * 86_400_000 if lookback_days is not None else None

    all_rows = []
    fetched = 0

    if start_time_ms is not None:
        cursor = start_time_ms
        while cursor < end_time_ms and fetched < max_candles_cap:
            params = {"symbol": symbol, "interval": interval, "startTime": cursor, "limit": BINANCE_MAX_LIMIT}
            resp = requests.get(BINANCE_BASE_URL + BINANCE_KLINES_ENDPOINT, params=params, timeout=10)
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            all_rows.extend(batch)
            fetched += len(batch)
            cursor = batch[-1][0] + interval_ms
            if len(batch) < BINANCE_MAX_LIMIT:
                break
    else:
        cursor_end = end_time_ms
        while fetched < max_candles_cap:
            params = {"symbol": symbol, "interval": interval, "endTime": cursor_end, "limit": BINANCE_MAX_LIMIT}
            resp = requests.get(BINANCE_BASE_URL + BINANCE_KLINES_ENDPOINT, params=params, timeout=10)
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            all_rows = batch + all_rows
            fetched += len(batch)
            cursor_end = batch[0][0] - 1
            if len(batch) < BINANCE_MAX_LIMIT:
                break

    if not all_rows:
        return pd.DataFrame()

    cols = ["open_time", "Open", "High", "Low", "Close", "Volume", "close_time",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(all_rows, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = df[c].astype(float)
    df = df.set_index("open_time")[["Open", "High", "Low", "Close", "Volume"]]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def fetch_binance(symbol: str, period: str, interval: str) -> FetchResult:
    binance_symbol = _binance_symbol_from_yahoo_style(symbol)
    native_interval = _find_best_native_source_interval(interval, BINANCE_NATIVE_INTERVALS, INTERVAL_ORDER)

    if native_interval is None:
        return FetchResult(df=pd.DataFrame(), source="binance", requested_interval=interval,
                            actual_native_interval="", is_resampled=False,
                            warning=f"'{interval}' için Binance'de uygun bir kaynak interval bulunamadı.")

    lookback_days = _period_to_lookback_days(period)

    try:
        raw = _fetch_binance_klines_paged(binance_symbol, native_interval, lookback_days)
    except requests.RequestException as exc:
        return FetchResult(df=pd.DataFrame(), source="binance", requested_interval=interval,
                            actual_native_interval=native_interval, is_resampled=False,
                            warning=f"Binance API'sine bağlanılamadı: {exc}")

    if raw.empty:
        return FetchResult(df=pd.DataFrame(), source="binance", requested_interval=interval,
                            actual_native_interval=native_interval, is_resampled=False,
                            warning=f"Binance '{binance_symbol}' için veri döndürmedi.")

    if native_interval == interval:
        return FetchResult(df=raw, source="binance", requested_interval=interval,
                            actual_native_interval=native_interval, is_resampled=False)

    resampled = _resample_ohlcv(raw, interval)
    return FetchResult(
        df=resampled, source="binance_resampled", requested_interval=interval,
        actual_native_interval=native_interval, is_resampled=True,
        warning=f"Binance native olarak '{interval}' sunmuyor. '{native_interval}' verisinden resample edildi.",
    )


# =====================================================================
# 🧭 A.5 PAZAR -> KAYNAK ROUTER + RİTİM MATRİSİ
# =====================================================================

MARKET_SOURCE_MAP = {
    "KRIPTO": "binance",
    "TR_HISSE": "yahoo",
    "ABD_HISSE": "yahoo",
    "EMTIA": "yahoo",
}


def get_market_data(symbol: str, period: str, interval: str, market: str,
                     prefer_source: Optional[str] = None) -> FetchResult:
    source = prefer_source or MARKET_SOURCE_MAP.get(market, "yahoo")

    if source == "binance":
        result = fetch_binance(symbol, period, interval)
        if result.df.empty and prefer_source is None:
            fallback = fetch_yahoo(symbol, period, interval)
            if not fallback.df.empty:
                fallback.warning = ((result.warning or "") + " | Binance'te bulunamadı, Yahoo'ya düşüldü.").strip(" |")
                return fallback
        return result

    return fetch_yahoo(symbol, period, interval)


# Pazara göre TÜM (native + resample) interval seçenekleri (UI için)
DISPLAY_INTERVALS_BY_MARKET = {
    "KRIPTO": [
        ("15 Dakika", "15m", True), ("30 Dakika", "30m", True),
        ("1 Saat", "1h", True), ("2 Saat", "2h", True), ("4 Saat", "4h", True),
        ("6 Saat", "6h", True), ("8 Saat", "8h", True), ("12 Saat", "12h", True),
        ("1 Gün", "1d", True), ("1 Hafta", "1w", True),
    ],
    "TR_HISSE": [
        ("15 Dakika", "15m", True), ("30 Dakika", "30m", True),
        ("1 Saat", "60m", True), ("90 Dakika", "90m", True),
        ("4 Saat", "4h", False), ("1 Gün", "1d", True),
        ("1 Hafta", "1wk", True), ("1 Ay", "1mo", True),
    ],
    "ABD_HISSE": [
        ("15 Dakika", "15m", True), ("30 Dakika", "30m", True),
        ("1 Saat", "60m", True), ("90 Dakika", "90m", True),
        ("4 Saat", "4h", False), ("1 Gün", "1d", True),
        ("1 Hafta", "1wk", True), ("1 Ay", "1mo", True),
    ],
    "EMTIA": [
        ("15 Dakika", "15m", True), ("30 Dakika", "30m", True),
        ("1 Saat", "60m", True), ("90 Dakika", "90m", True),
        ("4 Saat", "4h", False), ("1 Gün", "1d", True),
        ("1 Hafta", "1wk", True), ("1 Ay", "1mo", True),
    ],
}

# ⏱️ RİTİM MATRİSİ — pazar bazlı 1d hedefleri: KRIPTO 3y, TR_HISSE 1y,
# ABD_HISSE 5y, EMTIA 4y (sizin talebiniz üzerine sabitlenmiştir).
RITIM_MATRISI = {
    "15m":  {"KRIPTO": "60d", "TR_HISSE": "30d", "ABD_HISSE": "60d", "EMTIA": "45d"},
    "30m":  {"KRIPTO": "60d", "TR_HISSE": "30d", "ABD_HISSE": "60d", "EMTIA": "45d"},
    "60m":  {"KRIPTO": "2y",  "TR_HISSE": "1y",  "ABD_HISSE": "2y",  "EMTIA": "1y"},
    "1h":   {"KRIPTO": "2y",  "TR_HISSE": "1y",  "ABD_HISSE": "2y",  "EMTIA": "1y"},
    "90m":  {"KRIPTO": "60d", "TR_HISSE": "30d", "ABD_HISSE": "60d", "EMTIA": "45d"},
    "2h":   {"KRIPTO": "2y",  "TR_HISSE": "1y",  "ABD_HISSE": "2y",  "EMTIA": "1y"},
    "4h":   {"KRIPTO": "2y",  "TR_HISSE": "1y",  "ABD_HISSE": "2y",  "EMTIA": "2y"},
    "6h":   {"KRIPTO": "2y",  "TR_HISSE": "1y",  "ABD_HISSE": "2y",  "EMTIA": "2y"},
    "8h":   {"KRIPTO": "2y",  "TR_HISSE": "1y",  "ABD_HISSE": "2y",  "EMTIA": "2y"},
    "12h":  {"KRIPTO": "2y",  "TR_HISSE": "1y",  "ABD_HISSE": "2y",  "EMTIA": "2y"},
    "1d":   {"KRIPTO": "3y",  "TR_HISSE": "1y",  "ABD_HISSE": "5y",  "EMTIA": "4y"},
    "1w":   {"KRIPTO": "max", "TR_HISSE": "5y",  "ABD_HISSE": "max", "EMTIA": "max"},
    "1wk":  {"KRIPTO": "max", "TR_HISSE": "5y",  "ABD_HISSE": "max", "EMTIA": "max"},
    "1mo":  {"KRIPTO": "max", "TR_HISSE": "5y",  "ABD_HISSE": "max", "EMTIA": "max"},
}

INTRADAY_INTERVALS = {"15m", "30m", "60m", "1h", "90m", "2h", "4h", "6h", "8h", "12h"}


def suggest_period(market: str, interval: str) -> str:
    row = RITIM_MATRISI.get(interval)
    if row is not None and market in row:
        return row[market]
    return "60d" if interval in INTRADAY_INTERVALS else "1y"


# Annual factor (Sharpe hesabı için) — V58.0'daki ritim_matrisi mantığı
ANNUAL_FACTOR_BY_INTERVAL = {
    "15m": 252 * 6.5 * 4, "30m": 252 * 6.5 * 2,
    "60m": 252 * 6.5, "1h": 252 * 6.5, "90m": 252 * 6.5,
    "2h": 252 * 3.25, "4h": 252 * 1.625, "6h": 252 * 1.08,
    "8h": 252 * 0.8125, "12h": 252 * 0.54,
    "1d": 252, "1w": 52, "1wk": 52, "1mo": 12,
}


# 🔐 ENTITLEMENT HOOK (şu an pasif — istek üzerine kısıt eklenmedi)
# Aboneliğe gore kisit eklemek isterseniz su sablonu kullanin:
#
# FREE_TIER_ALLOWED_INTERVALS = {"1d", "1wk", "1mo"}
# FREE_TIER_ALLOWED_MARKETS = {"KRIPTO", "ABD_HISSE"}
#
# def check_entitlement(user_is_premium: bool, market: str, interval: str):
#     if user_is_premium:
#         return True, ""
#     if market not in FREE_TIER_ALLOWED_MARKETS:
#         return False, f"'{market}' pazarı sadece Premium üyelere açıktır."
#     if interval not in FREE_TIER_ALLOWED_INTERVALS:
#         return False, f"'{interval}' periyodu sadece Premium üyelere açıktır."
#     return True, ""


# #####################################################################
# #####################################################################
#  BÖLÜM B — V58.0 DASHBOARD (TÜM SEKME / GÖSTERGE / ML / BACKTEST)
# #####################################################################
# #####################################################################

str_ui.set_page_config(page_title="Quant AI - Sovereign Pro V60.0", page_icon="🏛️", layout="wide")

str_ui.markdown("""
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
    h1 {color: #1A1A1A; font-weight: 700;}
    h2 {color: #2C3E50; font-weight: 600; margin-top: 1rem;}
    .indicator-card {background: #FFFFFF; border: 1px solid #E0E0E0; border-radius: 12px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); height: 100%; text-align: center;}
    .ind-title {font-size: 14px; font-weight: 700; color: #333333; margin-bottom: 2px;}
    .ind-value {font-size: 26px; font-weight: 800; color: #111111; margin: 10px 0;}
    .ind-desc {font-size: 11px; color: #555555; text-align: left; line-height: 1.4;}
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 🎛️ B.1 SOL MENÜ — PAZAR / VARLIK / İNTERVAL (TÜM İNTERVAL'LAR AÇIK)
# =====================================================================
str_ui.sidebar.header("🏛️ Sovereign V60.0 Pro")
pazar = str_ui.sidebar.selectbox("1. Pazar Seçimi", ["KRIPTO", "TR_HISSE", "ABD_HISSE", "EMTIA"])

varlik_havuzu = {
    "KRIPTO": ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "XRP-USD", "DOGE-USD", "PEPE-USD", "FIL-USD", "LINK-USD"],
    "TR_HISSE": ["THYAO.IS", "SOKM.IS", "ASELS.IS", "EREGL.IS", "BIMAS.IS", "GARAN.IS", "TUPRS.IS", "SISE.IS"],
    "ABD_HISSE": ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "META", "GOOGL"],
    "EMTIA": ["GC=F", "SI=F", "CL=F", "NG=F"]
}
sembol = str_ui.sidebar.selectbox("2. Varlık Seçimi / Ara", varlik_havuzu[pazar])

interval_options = DISPLAY_INTERVALS_BY_MARKET[pazar]
interval_labels = [etiket + ("" if is_native else " (sentetik/resample)") for etiket, _, is_native in interval_options]
secim_idx = str_ui.sidebar.selectbox(
    "3. Grafik Mum Periyodu",
    options=list(range(len(interval_options))),
    format_func=lambda i: interval_labels[i],
    index=min(len(interval_options) - 1, len(interval_options) // 2),
)
secilen_periyot_etiket, aktif_interval, interval_native_mi = interval_options[secim_idx]

aktif_period = suggest_period(pazar, aktif_interval)
annual_factor = ANNUAL_FACTOR_BY_INTERVAL.get(aktif_interval, 252)

# 🔐 ENTITLEMENT HOOK çağrı noktası (şu an pasif):
# allowed, reason = check_entitlement(user.is_premium, pazar, aktif_interval)
# if not allowed:
#     str_ui.sidebar.error(f"🔒 {reason}")
#     str_ui.stop()

str_ui.sidebar.markdown("---")
kaynak_etiketi = "🟠 Binance (Public API)" if MARKET_SOURCE_MAP[pazar] == "binance" else "🟡 Yahoo Finance"
str_ui.sidebar.caption(
    f"📡 **Veri Kaynağı:** {kaynak_etiketi}\n\n"
    f"🛡️ **Akademik Ritim Bilgisi:** Veri penceresi **`{aktif_period}`** olarak kilitlendi."
    + ("" if interval_native_mi else "\n\n🧪 *Bu interval alt periyottan sentezlendi (resample).*")
)

# =====================================================================
# 🧮 B.2 MATEMATİKSEL GÖSTERGE MOTORLARI (V58.0 ile birebir aynı)
# =====================================================================

def rsi_calc(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    return 100 - (100 / (1 + (gain / (loss + 1e-10)) + 1e-10))

def macd_calc(series, fast=12, slow=26, signal=9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def stoch_calc(high, low, close, k=14, d=3):
    low_min = low.rolling(window=k).min()
    high_max = high.rolling(window=k).max()
    pk = ((close - low_min) / (high_max - low_min + 1e-10)) * 100
    return pk, pk.rolling(window=d).mean()

def wavetrend_calc(high, low, close, n1=10, n2=21):
    esa = (high + low + close) / 3
    esa_ema = esa.ewm(span=n1, adjust=False).mean()
    d_ema = (esa - esa_ema).abs().ewm(span=n1, adjust=False).mean()
    ci = (esa - esa_ema) / (0.015 * d_ema + 1e-10)
    wt1 = ci.ewm(span=n2, adjust=False).mean()
    return wt1, wt1.rolling(window=4).mean()

def cci_calc(high, low, close, period=20):
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = (tp - sma_tp).abs().rolling(window=period).mean()
    return (tp - sma_tp) / (0.015 * mad + 1e-10)

def atr_calc(high, low, close, period=14):
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_metrics(df):
    if df.empty or len(df) < 35: return pd.DataFrame()
    df_out = df.copy()
    c = df_out['Close'].squeeze()
    h = df_out['High'].squeeze()
    l = df_out['Low'].squeeze()
    v = df_out['Volume'].squeeze()

    df_out['EMA_20'] = c.ewm(span=20, adjust=False).mean()
    df_out['EMA_50'] = c.ewm(span=50, adjust=False).mean()
    df_out['SMA_200'] = c.rolling(window=min(200, len(df_out) // 2)).mean()
    df_out['SMA_Volume_20'] = v.rolling(window=min(20, len(df_out))).mean()

    sma_20 = c.rolling(20).mean()
    std_20 = c.rolling(20).std()
    df_out['Bollinger_Orta'] = sma_20
    df_out['Bollinger_Ust'] = sma_20 + (std_20 * 2)
    df_out['Bollinger_Alt'] = sma_20 - (std_20 * 2)
    df_out['Bollinger_Width'] = (df_out['Bollinger_Ust'] - df_out['Bollinger_Alt']) / (sma_20 + 1e-10)

    df_out['RSI'] = rsi_calc(c)
    df_out['MACD'], df_out['MACD_Sig'], df_out['MACD_Hist'] = macd_calc(c)
    df_out['Stoch_K'], df_out['Stoch_D'] = stoch_calc(h, l, c)
    df_out['WT1'], df_out['WT2'] = wavetrend_calc(h, l, c)
    df_out['CCI'] = cci_calc(h, l, c)
    df_out['ATR'] = atr_calc(h, l, c)
    df_out['Getiri_1G'] = c.pct_change(1)
    df_out['Volatilite_5G'] = df_out['Getiri_1G'].rolling(5).std()
    df_out['Hacim_ROC_5'] = v.pct_change(5)
    df_out['Fiyat_SMA200_Orani'] = c / (df_out['SMA_200'] + 1e-10)
    df_out['Trend_Gucu'] = (h.rolling(14).max() - l.rolling(14).min()) / (c + 1e-10)

    width_mean = df_out['Bollinger_Width'].rolling(20).mean()
    df_out['Regime_Sideways'] = np.where(df_out['Bollinger_Width'] < width_mean * 0.8, 1.0, 0.0)
    df_out['Regime_Bull'] = np.where((df_out['Regime_Sideways'] == 0) & (c > df_out['SMA_200']), 1.0, 0.0)
    df_out['Regime_Bear'] = np.where((df_out['Regime_Sideways'] == 0) & (c <= df_out['SMA_200']), 1.0, 0.0)

    df_out['Yuzde_Getiri_3G'] = c.pct_change(3).shift(-3)
    df_out['Hedef'] = (df_out['Yuzde_Getiri_3G'] > 0.002).astype(int)
    return df_out.replace([np.inf, -np.inf], np.nan).dropna()


# =====================================================================
# 🔌 B.3 VERİ ÇEKME — Eski yf.download YERİNE birleşik katman (cache'li)
# =====================================================================
@str_ui.cache_data(ttl=300, show_spinner="📡 Veri çekiliyor...")
def cached_fetch(symbol, period, interval, market):
    result = get_market_data(symbol, period, interval, market)
    return result.df, result.source, result.is_resampled, result.warning


df_raw, veri_kaynagi, sentetik_mi, uyari_metni = cached_fetch(sembol, aktif_period, aktif_interval, pazar)
if uyari_metni:
    str_ui.sidebar.info(f"ℹ️ {uyari_metni}")

df_active = calculate_metrics(df_raw)
veri_yetersiz = df_active.empty or len(df_active) < 30

# =====================================================================
# 🧠 B.4 ÖNBELLEKLİ MODEL FABRİKASI (V58.0 ile birebir aynı)
# =====================================================================
@str_ui.cache_resource(show_spinner=False)
def egit_klasik_modelleri(_X_tr_sc, _y_tr, scale_pos_weight_value, cache_anahtari):
    m_gbm = GradientBoostingClassifier(n_estimators=40, max_depth=3, random_state=42).fit(_X_tr_sc, _y_tr)
    m_rf = RandomForestClassifier(n_estimators=40, max_depth=5, class_weight='balanced', random_state=42).fit(_X_tr_sc, _y_tr)
    m_xgb = XGBClassifier(n_estimators=40, max_depth=3, learning_rate=0.03, subsample=0.8, scale_pos_weight=scale_pos_weight_value, random_state=42, eval_metric='logloss') if XGB_AVAILABLE else LogisticRegression(class_weight='balanced')
    m_xgb.fit(_X_tr_sc, _y_tr)
    return m_gbm, m_rf, m_xgb

@str_ui.cache_resource(show_spinner=False)
def egit_uretim_modellerini(_X_train_sc, _y_train, scale_pos_weight_value, cache_anahtari):
    model_gbm = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42).fit(_X_train_sc, _y_train)
    model_rf = RandomForestClassifier(n_estimators=50, max_depth=5, class_weight='balanced', random_state=42).fit(_X_train_sc, _y_train)
    model_xgb = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.03, subsample=0.8, scale_pos_weight=scale_pos_weight_value, random_state=42, eval_metric='logloss') if XGB_AVAILABLE else LogisticRegression(class_weight='balanced')
    model_xgb.fit(_X_train_sc, _y_train)
    return model_gbm, model_rf, model_xgb


str_ui.title("🏛️ QUANT AI - SOVEREIGN COCKPIT V60.0")
str_ui.markdown(
    f"### {sembol} | %100 Güvenli Katman Kontrollü Altyapı Sürümü 👑 — 3'lü Konsensüs Çekirdeği ⚡ "
    f"| Kaynak: **{veri_kaynagi}** {'🧪' if sentetik_mi else '✅'}"
)

# =====================================================================
# 🧠 B.5 ML PIPELINE (V58.0 ile birebir aynı — walk-forward CV, PF+Acc
#        karma skor, dinamik ensemble ağırlıklandırma, class imbalance)
# =====================================================================
if not veri_yetersiz:
    fiyat_su_an = float(df_active['Close'].squeeze().iloc[-1])
    degisim_24s = ((fiyat_su_an - df_active['Close'].squeeze().iloc[-2]) / df_active['Close'].squeeze().iloc[-2]) * 100
    atr_gucu = float(df_active['ATR'].iloc[-1])
    sma_200_degeri = float(df_active['SMA_200'].iloc[-1])
    para_birimi = "TL" if sembol.endswith(".IS") else "USD"

    ozellikler = ['Close', 'EMA_20', 'EMA_50', 'SMA_200', 'Bollinger_Width', 'RSI', 'MACD', 'ATR', 'Getiri_1G', 'Volatilite_5G', 'Fiyat_SMA200_Orani', 'Trend_Gucu', 'Regime_Sideways', 'Regime_Bull', 'Regime_Bear']
    df_ml = df_active.iloc[:-3].copy()
    X_all = df_ml[ozellikler].copy()
    y_all = df_ml['Hedef']

    split_idx = int(len(X_all) * 0.80)
    X_train, X_test = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
    y_train, y_test = y_all.iloc[:split_idx], y_all.iloc[split_idx:]

    counts_tr = y_train.value_counts(normalize=True)
    scale_pos_weight_value = max(0.1, counts_tr.get(0, 0.5) / (counts_tr.get(1, 0.5) + 1e-10))

    cache_anahtari = f"{sembol}_{secilen_periyot_etiket}_{pazar}_{len(X_all)}"

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    bugunun_sc = scaler.transform(df_active.iloc[[-1]][ozellikler])

    tscv = TimeSeriesSplit(n_splits=3)
    final_scores = {"gbm": [], "rf": [], "xgb": []}
    acc_pure_scores = {"gbm": [], "rf": [], "xgb": []}

    def ic_dongu_karma_skor_hesapla(preds, forward_returns_sliced, y_true_sliced):
        s_ret = np.where(preds == 1, forward_returns_sliced, -forward_returns_sliced)
        g = np.sum(s_ret[s_ret > 0])
        l = np.abs(np.sum(s_ret[s_ret < 0]))
        pf = np.clip(g / (l + 1e-10), 0.1, 5.0)
        pf_norm = pf / 5.0
        acc = accuracy_score(y_true_sliced, preds)
        return (0.5 * pf_norm) + (0.5 * acc), acc

    for fold_no, (tr_idx, te_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_te = X_train.iloc[tr_idx], X_train.iloc[te_idx]
        y_tr, y_te = y_train.iloc[tr_idx], y_train.iloc[te_idx]

        sc_inner = StandardScaler()
        X_tr_sc = sc_inner.fit_transform(X_tr)
        X_te_sc = sc_inner.transform(X_te)
        te_forward_returns = df_ml['Yuzde_Getiri_3G'].iloc[:split_idx].values[te_idx]

        fold_cache_anahtari = f"{cache_anahtari}_fold{fold_no}"
        m_gbm, m_rf, m_xgb = egit_klasik_modelleri(X_tr_sc, y_tr, scale_pos_weight_value, fold_cache_anahtari)

        for name, model in [("gbm", m_gbm), ("rf", m_rf), ("xgb", m_xgb)]:
            scr, ac = ic_dongu_karma_skor_hesapla(model.predict(X_te_sc), te_forward_returns, y_te)
            final_scores[name].append(scr)
            acc_pure_scores[name].append(ac)

    cv_gbm = max(0.1, np.mean(final_scores["gbm"])) if final_scores["gbm"] else 1.0
    cv_rf = max(0.1, np.mean(final_scores["rf"])) if final_scores["rf"] else 1.0
    cv_xgb = max(0.1, np.mean(final_scores["xgb"])) if final_scores["xgb"] else 1.0

    log_gbm = np.log1p(cv_gbm)
    log_rf = np.log1p(cv_rf)
    log_xgb = np.log1p(cv_xgb)

    toplam_log_pf = log_gbm + log_rf + log_xgb
    w_gbm = log_gbm / toplam_log_pf
    w_rf = log_rf / toplam_log_pf
    w_xgb = log_xgb / toplam_log_pf

    model_gbm, model_rf, model_xgb = egit_uretim_modellerini(X_train_sc, y_train, scale_pos_weight_value, cache_anahtari)

    prob_gbm = model_gbm.predict_proba(bugunun_sc)[0][1]
    prob_rf = model_rf.predict_proba(bugunun_sc)[0][1]
    prob_xgb = model_xgb.predict_proba(bugunun_sc)[0][1]

    boga_ihtimali = ((prob_gbm * w_gbm) + (prob_rf * w_rf) + (prob_xgb * w_xgb)) * 100
    ayi_ihtimali = 100 - boga_ihtimali
    karar = "ARTIŞ (YÜKSELİŞ)" if boga_ihtimali >= 50 else "AZALIŞ (DÜŞÜŞ)"

    t_p_gbm_sliced = model_gbm.predict_proba(X_test_sc)[:, 1]
    t_p_rf_sliced = model_rf.predict_proba(X_test_sc)[:, 1]
    t_p_xgb_sliced = model_xgb.predict_proba(X_test_sc)[:, 1]
    ens_probs_sliced = (t_p_gbm_sliced * w_gbm) + (t_p_rf_sliced * w_rf) + (t_p_xgb_sliced * w_xgb)

    test_signals_sliced = (ens_probs_sliced >= 0.50).astype(int)
    test_returns_sliced = df_ml['Yuzde_Getiri_3G'].iloc[split_idx:].values

    profit_factor, max_dd, sharpe = 1.0, 0.0, 0.0

    if len(test_signals_sliced) > 0 and len(test_returns_sliced) == len(test_signals_sliced):
        raw_strat_returns = np.where(test_signals_sliced == 1, test_returns_sliced, -test_returns_sliced)
        signal_changes = np.diff(test_signals_sliced, prepend=test_signals_sliced[0])
        strategy_returns = raw_strat_returns - np.where(signal_changes != 0, 0.0005, 0.0)

        gains = np.sum(strategy_returns[strategy_returns > 0])
        losses = np.abs(np.sum(strategy_returns[strategy_returns < 0]))
        if losses > 0:
            profit_factor = gains / losses

        cum_r = np.cumsum(strategy_returns)
        if len(cum_r) > 0:
            max_dd = np.max(np.maximum.accumulate(cum_r) - cum_r) * 100
        if len(strategy_returns) > 1 and np.std(strategy_returns) > 0:
            sharpe = (np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(annual_factor)

# =====================================================================
# 📊 B.6 6'LI ANA SEKME YAPISI (V58.0 ile birebir aynı)
# =====================================================================
sekme_ozet, sekme_teknik, sekme_zincir, sekme_grafik, sekme_performans, sekme_maliyet = str_ui.tabs([
    "🔮 Yapay Zeka Özet Raporu", "📊 Teknik Gösterge Odası", "⛓️ Trend & Volatilite Hattı",
    "📈 Canlı Grafik Odası", "🎯 Backtest / Performans", "💸 Maliyet & Risk Analizi"
])

if veri_yetersiz:
    for tab in [sekme_ozet, sekme_teknik, sekme_zincir, sekme_grafik, sekme_performans, sekme_maliyet]:
        with tab:
            str_ui.error(f"⚠️ **Veri Derinliği Yetersiz:** Süzgeçten sonra kalan mum sayısı ({len(df_active)}) analize elvermiyor hoca.")
            str_ui.warning("💡 Lütfen sol menüden daha uzun bir mum periyodu (Örn: 1 Gün) seçerek havuzu genişletin.")
else:
    with sekme_ozet:
        str_ui.subheader(f"🔮 Doğrulanmış Ortak Akıl Tahmin Raporu ({secilen_periyot_etiket})")
        str_ui.markdown(f"""
        <div style="background-color: #0F172A; border: 1px solid #334155; border-radius: 12px; padding: 12px 20px; margin-bottom: 20px;">
            <span style="font-size: 11px; font-weight: 700; color: #94A3B8; letter-spacing: 0.05em;">📡 KONSENSÜS ÇEKİRDEĞİ MONİTÖRÜ</span>
            <div style="display: flex; gap: 40px; margin-top: 6px;">
                <div><span style="font-size: 11px; color: #64748B;">Mevcut Train Verisi:</span> <strong style="font-size: 14px; color: #F1F5F9;">{len(X_train)} Satır</strong></div>
                <div><span style="font-size: 11px; color: #64748B;">Model Mimarisi:</span> <strong style="font-size: 14px; color: #38BDF8;">3'lü Konsensüs (GBM + RF + XGBoost)</strong></div>
                <div><span style="font-size: 11px; color: #64748B;">Veri Kaynağı:</span> <strong style="font-size: 14px; color: #10B981;">{veri_kaynagi.upper()} ⚡</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        o_c1, o_c2, o_c3 = str_ui.columns(3)
        with o_c1:
            bg_color = '#F4F9EA' if boga_ihtimali >= 50 else '#FFF5F5'
            border_color = '#97C459' if boga_ihtimali >= 50 else '#F3C6C6'
            text_color = '#3B6D11' if boga_ihtimali >= 50 else '#E24B4A'
            str_ui.markdown(f"""<div style="border:2px solid {border_color}; border-radius:12px; padding:20px; text-align:center; background:{bg_color};"><div style="font-size:12px; font-weight:600; color:#555;">KONSENSÜS ANA KARARI</div><div style="font-size:28px; font-weight:800; color:{text_color}; margin:8px 0;">{karar}</div></div>""", unsafe_allow_html=True)
        with o_c2:
            str_ui.markdown(f"""<div style="border:1px solid #E0E0E0; border-radius:12px; padding:20px; text-align:center; background:#FFF;"><div style="font-size:12px; font-weight:600; color:#555;">YÜKSELİŞ İHTİMALİ</div><div style="font-size:28px; font-weight:800; color:#3B6D11; margin:8px 0;">%{boga_ihtimali:.1f}</div></div>""", unsafe_allow_html=True)
        with o_c3:
            str_ui.markdown(f"""<div style="border:1px solid #E0E0E0; border-radius:12px; padding:20px; text-align:center; background:#FFF;"><div style="font-size:12px; font-weight:600; color:#555;">DÜŞÜŞ İHTİMALİ</div><div style="font-size:28px; font-weight:800; color:#E24B4A; margin:8px 0;">%{ayi_ihtimali:.1f}</div></div>""", unsafe_allow_html=True)

        str_ui.write("---")
        last_row_data = df_active.iloc[-1]
        card_c1, card_c2, card_c3, card_c4, card_c5, card_c6 = str_ui.columns(6)
        with card_c1:
            str_ui.markdown(f"""<div class="indicator-card"><div class="ind-title">RSI (14)</div><div class="ind-value">{float(last_row_data['RSI']):.1f}</div></div>""", unsafe_allow_html=True)
        with card_c2:
            str_ui.markdown(f"""<div class="indicator-card"><div class="ind-title">MACD</div><div class="ind-value">{float(last_row_data['MACD']):.2f}</div></div>""", unsafe_allow_html=True)
        with card_c3:
            str_ui.markdown(f"""<div class="indicator-card"><div class="ind-title">Stochastic</div><div class="ind-value">{float(last_row_data['Stoch_K']):.1f}</div></div>""", unsafe_allow_html=True)
        with card_c4:
            str_ui.markdown(f"""<div class="indicator-card"><div class="ind-title">WaveTrend</div><div class="ind-value">{float(last_row_data['WT1']):.1f}</div></div>""", unsafe_allow_html=True)
        with card_c5:
            str_ui.markdown(f"""<div class="indicator-card"><div class="ind-title">CCI</div><div class="ind-value">{float(last_row_data['CCI']):.1f}</div></div>""", unsafe_allow_html=True)
        with card_c6:
            str_ui.markdown(f"""<div class="indicator-card"><div class="ind-title">ATR (14)</div><div class="ind-value">{float(last_row_data['ATR']):.4f}</div></div>""", unsafe_allow_html=True)

    with sekme_teknik:
        str_ui.subheader("📊 Canlı Teknik Gösterge ve Osilatör Laboratuvarı")

        with str_ui.expander(f"📋 Ham Veri Seti Kesiti (Pencere derinliği: {aktif_period})", expanded=False):
            str_ui.write(df_active[['Close', 'EMA_20', 'EMA_50', 'SMA_200', 'Bollinger_Ust', 'Bollinger_Alt', 'RSI', 'MACD', 'ATR', 'Hacim_ROC_5']])

        str_ui.write("---")
        secilen_grafikler = str_ui.multiselect(
            "🔍 Grafik Odasına Eklenecek Göstergeleri Seçin:",
            options=["Dinamik Fiyat ve Ortalamalar", "Bollinger Bantları (Alt, Orta, Üst)",
                     "RSI (Göreceli Güç Endeksi)", "MACD & Sinyal Hattı", "Stochastic Osilatör",
                     "WaveTrend (WT1/WT2)", "CCI (Commodity Channel Index)", "ATR (Volatilite Gücü)"],
            default=["Dinamik Fiyat ve Ortalamalar"]
        )
        str_ui.write("---")

        if "Dinamik Fiyat ve Ortalamalar" in secilen_grafikler:
            str_ui.markdown("**📈 Dinamik Fiyat Eğilimi ve Hareketli Ortalamalar**")
            str_ui.line_chart(df_active[['Close', 'EMA_20', 'EMA_50', 'SMA_200']])
            str_ui.caption("💡 **Nasıl Yorumlanır?** Fiyatın hareketli ortalamaların üzerinde olması genel yükseliş trendini, altında olması düşüş eğilimini gösterir. Kısa vadeli ortalamanın uzun vadeliyi yukarı kesmesi Golden Cross (alım) sinyalidir.")
            str_ui.write("")

        if "Bollinger Bantları (Alt, Orta, Üst)" in secilen_grafikler:
            str_ui.markdown("**🌪️ Bollinger Bantları - Fiyat Oynaklık Kanalları**")
            str_ui.line_chart(df_active[['Close', 'Bollinger_Ust', 'Bollinger_Orta', 'Bollinger_Alt']])
            str_ui.caption("💡 **Nasıl Yorumlanır?** Fiyatın üst çizgiye yaklaşması aşırı alım, alt çizgiye yaklaşması aşırı satım bölgesini gösterir. Bantların daralması sert bir hareketin yaklaştığına işarettir.")
            str_ui.write("")

        if "RSI (Göreceli Güç Endeksi)" in secilen_grafikler:
            str_ui.markdown("**🔮 RSI - Aşırı Alım / Satım Osilatörü**")
            str_ui.line_chart(df_active['RSI'])
            str_ui.caption("💡 **Nasıl Yorumlanır?** RSI 70 üzerinde aşırı alım (düzeltme riski), 30 altında aşırı satım (dönüş potansiyeli) bölgesidir.")
            str_ui.write("")

        if "MACD & Sinyal Hattı" in secilen_grafikler:
            str_ui.markdown("**⛓️ MACD (Trend Takip ve Momentum Osilatörü)**")
            str_ui.line_chart(df_active[['MACD', 'MACD_Sig']])
            str_ui.caption("💡 **Nasıl Yorumlanır?** MACD çizgisi Sinyal çizgisini yukarı keserse alım, aşağı keserse satım sinyali olarak yorumlanır.")
            str_ui.write("")

        if "Stochastic Osilatör" in secilen_grafikler:
            str_ui.markdown("**🎯 Stochastic Osilatör (%K / %D)**")
            str_ui.line_chart(df_active[['Stoch_K', 'Stoch_D']])
            str_ui.caption("💡 **Nasıl Yorumlanır?** 80 üzeri aşırı alım, 20 altı aşırı satım. %K'nın %D'yi kesmesi kısa vadeli dönüş sinyali verir.")
            str_ui.write("")

        if "WaveTrend (WT1/WT2)" in secilen_grafikler:
            str_ui.markdown("**🌊 WaveTrend Osilatörü (WT1 / WT2)**")
            str_ui.line_chart(df_active[['WT1', 'WT2']])
            str_ui.caption("💡 **Nasıl Yorumlanır?** WT1'in WT2'yi aşırı satım bölgesinde yukarı kesmesi alım, aşırı alım bölgesinde aşağı kesmesi satım sinyali olarak değerlendirilir.")
            str_ui.write("")

        if "CCI (Commodity Channel Index)" in secilen_grafikler:
            str_ui.markdown("**📐 CCI - Commodity Channel Index**")
            str_ui.line_chart(df_active['CCI'])
            str_ui.caption("💡 **Nasıl Yorumlanır?** +100 üzeri güçlü yükseliş trendi, -100 altı güçlü düşüş trendini işaret eder.")
            str_ui.write("")

        if "ATR (Volatilite Gücü)" in secilen_grafikler:
            str_ui.markdown("**🌪️ ATR (Average True Range) - Piyasa Volatilite Grafiği**")
            str_ui.line_chart(df_active['ATR'])
            str_ui.caption("💡 **Nasıl Yorumlanır?** ATR yükselmesi piyasada volatilitenin arttığını gösterir; risk yönetiminde stop mesafesi belirlemek için kullanılır.")

    with sekme_zincir:
        str_ui.subheader("⛓️ Makro Trend Gücü ve Rejim İzleme Hattı")
        son_satir = df_active.iloc[-1]
        rejim_str = "YATAY PİYASA 🔒" if son_satir['Regime_Sideways'] == 1 else ("BOĞA REJİMİ 🐂" if son_satir['Regime_Bull'] == 1 else "AYI REJİMİ 🐻")
        str_ui.warning(f"Piyasa Yapısı: {rejim_str}")

        rz_c1, rz_c2, rz_c3 = str_ui.columns(3)
        rz_c1.metric("Mevcut ATR Gücü", f"{atr_gucu:.4f}")
        rz_c2.metric("Makro SMA_200", f"{sma_200_degeri:,.2f}")
        rz_c3.metric("Trend Gücü (14)", f"{son_satir['Trend_Gucu']:.4f}")

        str_ui.write("---")
        str_ui.markdown("**📉 Bollinger Bant Genişliği (Rejim Tespiti için Volatilite Sıkışması)**")
        str_ui.line_chart(df_active['Bollinger_Width'])
        str_ui.caption("💡 Bant genişliğinin daralması (Sıkışma) yakında sert bir kırılım olabileceğine işaret eder; bu bölgeler 'Yatay Piyasa' rejimi olarak sınıflandırılır.")

    with sekme_grafik:
        str_ui.subheader("📈 Canlı Fiyat Akış Hatları")
        str_ui.line_chart(df_active['Close'])
        str_ui.write("---")
        str_ui.markdown("**📊 Hacim (Volume)**")
        str_ui.bar_chart(df_active['Volume'] if 'Volume' in df_active.columns else df_active['SMA_Volume_20'])

    with sekme_performans:
        str_ui.subheader("🎯 Kurul Üyelerinin Oy Güçleri")
        ind_c1, ind_c2, ind_c3 = str_ui.columns(3)
        ind_c1.metric("GBM Ağırlığı", f"%{w_gbm*100:.1f}", f"CV Skor: {cv_gbm:.2f}")
        ind_c2.metric("RF Ağırlığı", f"%{w_rf*100:.1f}", f"CV Skor: {cv_rf:.2f}")
        ind_c3.metric("XGBoost Ağırlığı", f"%{w_xgb*100:.1f}", f"CV Skor: {cv_xgb:.2f}")

        str_ui.write("---")
        str_ui.markdown("### 🗳️ İç Döngü Saf Doğruluk (Accuracy) İstatistikleri")
        a_c1, a_c2, a_c3 = str_ui.columns(3)
        a_c1.metric("GBM Fold Ortalaması", f"%{np.mean(acc_pure_scores['gbm'])*100:.1f}" if acc_pure_scores['gbm'] else "%0.0")
        a_c2.metric("RF Fold Ortalaması", f"%{np.mean(acc_pure_scores['rf'])*100:.1f}" if acc_pure_scores['rf'] else "%0.0")
        a_c3.metric("XGB Fold Ortalaması", f"%{np.mean(acc_pure_scores['xgb'])*100:.1f}" if acc_pure_scores['xgb'] else "%0.0")

        str_ui.write("---")
        str_ui.markdown("### 📉 Backtest Performans Özeti (Test Seti)")
        bt_c1, bt_c2, bt_c3 = str_ui.columns(3)
        bt_c1.metric("Profit Factor (3G)", f"{profit_factor:.2f}")
        bt_c2.metric("Maks. Drawdown", f"%{max_dd:.1f}")
        bt_c3.metric("Sharpe Oranı (Yıllıklandırılmış)", f"{sharpe:.2f}")

    with sekme_maliyet:
        str_ui.subheader("💸 Volatilite Tabanlı Risk ve Kasa Yönetimi Kokpiti")
        kullanici_kasasi = str_ui.number_input(f"💰 Kasa Büyüklüğü ({para_birimi})", min_value=100.0, value=2000.0)
        risk_yuzdesi = str_ui.slider("🔥 Risk Yüzdesi (%)", 0.5, 5.0, 1.0)

        risk_basi_stop = atr_gucu * 1.5
        hedef_kar_al = atr_gucu * 3.0
        max_pozisyon = (kullanici_kasasi * (risk_yuzdesi / 100.0)) / (risk_basi_stop + 1e-10)

        stop_fiyat = fiyat_su_an - risk_basi_stop if karar == "ARTIŞ (YÜKSELİŞ)" else fiyat_su_an + risk_basi_stop
        kar_fiyat = fiyat_su_an + hedef_kar_al if karar == "ARTIŞ (YÜKSELİŞ)" else fiyat_su_an - hedef_kar_al

        st_c1, st_c2, st_c3 = str_ui.columns(3)
        st_c1.error(f"🚨 Stop-Loss: {stop_fiyat:,.2f}")
        st_c2.success(f"🎯 Kâr-Al Hedefi: {kar_fiyat:,.2f}")
        st_c3.warning(f"💼 Maks. Pozisyon: {max_pozisyon:,.4f} Adet")

        göze_alinan_para = kullanici_kasasi * (risk_yuzdesi / 100.0)
        str_ui.markdown(f"""<div style="background-color: #F8F9FA; border-left: 5px solid #2980B9; padding: 15px; border-radius: 4px;"><ul><li><b>Realist Kârlılık Faktörü (3G PF):</b> {profit_factor:.2f}</li><li><b>Maksimum Çöküş (Max DD):</b> %{max_dd:.1f}</li><li><b>Sharpe Oranı:</b> {sharpe:.2f}</li><li><b>Göze Alınan Risk Tutarı:</b> {göze_alinan_para:,.2f} {para_birimi}</li><li><b>Risk / Ödül Oranı (R:R):</b> 1 : 2.0</li></ul></div>""", unsafe_allow_html=True)