"""
=====================================================================
🏛️ QUANT AI - SOVEREIGN CORE (quant_ml_core.py)
=====================================================================
Bu dosya, Streamlit dashboard V61.2'nin TÜM motor mantığını
(veri çekme, gösterge hesaplama, ML pipeline, kalibrasyon, backtest)
Streamlit'ten TAMAMEN BAĞIMSIZ hale getirir. Django view'ları (veya
ileride herhangi bir başka arayüz: FastAPI, CLI, Celery task) bu
dosyadaki `analiz_yap()` fonksiyonunu çağırarak aynı analiz motoruna
ulaşır.

NEDEN AYRI DOSYA?
Streamlit'e özel her şey (st.sidebar, st.cache_data, st.tabs, HTML
kartları) bilerek ÇIKARILDI. Geriye kalan: saf Python + pandas/numpy/
sklearn mantığı. Bu sayede aynı motor Django'da, bir cron job'da, ya
da gelecekte bir mobil API'de tekrar tekrar kullanılabilir.

YAPISAL BÖLÜMLER (V61 dashboard'undaki A/B bölümlerinin doğrudan
karşılığıdır, isimlendirme bilerek aynı tutuldu ki karşılaştırma
kolay olsun):

  BÖLÜM A — Birleşik veri sağlayıcı katmanı (Yahoo + Binance)
  BÖLÜM B — Gösterge motorları + feature engineering
  BÖLÜM C — ML pipeline (walk-forward CV, kalibrasyon, ensemble)
  BÖLÜM D — Backtest / risk metrikleri
  BÖLÜM E — Dışa açılan tek giriş noktası: analiz_yap()

NOT (V61.2 -> quant_ml_core geçişinde düzeltilen bug, bilgi amaçlı):
calculate_metrics() içindeki genel dropna() çağrısı, ileriye-dönük
(shift(-3)) hedef sütunundan kaynaklanan NaN'ları TÜM tabloyla
birlikte siliyordu; bu da görüntüleme verisinin her zaman "bugünden
3 gün geride" kesilmesine sebep oluyordu. Bu dosyada o düzeltme
(dropna'nın sadece görüntüleme sütunlarına uygulanması) DAHİLDİR.

KURULUM:
    pip install yfinance pandas numpy scikit-learn requests xgboost
=====================================================================
"""

from __future__ import annotations

import contextlib
import io
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

np.random.seed(42)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


# #####################################################################
# #####################################################################
#  BÖLÜM A — BİRLEŞİK VERİ SAĞLAYICI KATMANI
# #####################################################################
# #####################################################################

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
# dakika; 4h/6h/8h/12h gibi saat-bazlı interval'lara TAM BÖLÜNMEZ, bu
# yüzden resample kaynağı olarak seçilirse yanlış hizalanmış mumlar
# üretir. Bu sıralama Yahoo'da bu interval'lar için her zaman "60m"
# native kaynak seçilmesini garanti eder.
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


def _period_string_to_days(period: str) -> Optional[int]:
    if period is None or period == "":
        return None
    p = period.strip().lower()
    if p == "max":
        return None
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

    requested_days = _period_string_to_days(period)
    lookback_cap = YAHOO_INTRADAY_LOOKBACK_CAP_DAYS.get(native_interval)
    if lookback_cap is not None and (requested_days is None or requested_days > lookback_cap):
        requested_days = lookback_cap

    download_kwargs = dict(interval=native_interval, progress=False)
    if requested_days is None:
        download_kwargs["period"] = "max"
    else:
        end_dt = pd.Timestamp.now("UTC").tz_localize(None)
        start_dt = end_dt - pd.Timedelta(days=requested_days)
        download_kwargs["start"] = start_dt.strftime("%Y-%m-%d")
        download_kwargs["end"] = (end_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # 🆕 yfinance'in "possibly delisted" / "no price data found" gibi terminal
    # uyarılarını susturma. Hem logging hem direkt print() çağrılarını yakalar.
    # os.devnull kullanımı thread-safe — her thread kendi file descriptor'ını alır.
    import logging as _logging
    _yf_logger = _logging.getLogger("yfinance")
    _onceki_seviye = _yf_logger.level
    _yf_logger.setLevel(_logging.CRITICAL)
    try:
        with open(os.devnull, 'w') as _devnull:
            with contextlib.redirect_stdout(_devnull):
                raw = yf.download(symbol, **download_kwargs)
    finally:
        _yf_logger.setLevel(_onceki_seviye)
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


_BINANCE_SESSION = requests.Session()
_BINANCE_RETRY = Retry(
    total=3,
    backoff_factor=0.6,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=frozenset(["GET"]),
)
_BINANCE_SESSION.mount("https://", HTTPAdapter(max_retries=_BINANCE_RETRY))


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
            resp = _BINANCE_SESSION.get(BINANCE_BASE_URL + BINANCE_KLINES_ENDPOINT, params=params, timeout=10)
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
            resp = _BINANCE_SESSION.get(BINANCE_BASE_URL + BINANCE_KLINES_ENDPOINT, params=params, timeout=10)
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
        return FetchResult(
            df=pd.DataFrame(), source="binance", requested_interval=interval,
            actual_native_interval=native_interval, is_resampled=False,
            warning=(
                f"Binance API'sine 3 deneme sonrasında bağlanılamadı: {exc}. "
                f"'{binance_symbol}' sembolü Binance'de mevcut olmayabilir veya geçici bir ağ sorunu olabilir."
            ),
        )

    if raw.empty:
        return FetchResult(
            df=pd.DataFrame(), source="binance", requested_interval=interval,
            actual_native_interval=native_interval, is_resampled=False,
            warning=(
                f"Binance '{binance_symbol}' için veri döndürmedi. Bu sembol Binance Spot'ta "
                f"işlem görmüyor olabilir (delisting veya hiç listelenmemiş)."
            ),
        )

    if native_interval == interval:
        return FetchResult(df=raw, source="binance", requested_interval=interval,
                            actual_native_interval=native_interval, is_resampled=False)

    resampled = _resample_ohlcv(raw, interval)
    return FetchResult(
        df=resampled, source="binance_resampled", requested_interval=interval,
        actual_native_interval=native_interval, is_resampled=True,
        warning=f"Binance native olarak '{interval}' sunmuyor. '{native_interval}' verisinden resample edildi.",
    )


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


@dataclass
class HafifFiyatSonucu:
    """
    🆕 ANASAYFA / TARAYICI ekranları için "hafif" fiyat sonucu — ML
    pipeline'ı (Triple Barrier, model eğitimi, Purged CV) ÇALIŞTIRILMAZ,
    sadece en güncel OHLCV barından fiyat/hacim/değişim bilgisi çıkarılır.

    NEDEN AYRI BİR FONKSİYON: Ana sayfa, 4 pazardan 20 varlığın fiyatını
    AYNI ANDA göstermesi gerekiyor. Eğer her varlık için tam analiz
    (calistir_ml_pipeline) çalıştırılsaydı, sayfa açılışı 20 × birkaç
    saniye sürerdi — kullanıcı deneyimi için kabul edilemez. Bu fonksiyon
    SADECE veri çekme katmanını (get_market_data) kullanır, ML'i atlar.
    """
    sembol: str
    pazar: str
    basarili: bool
    hata_mesaji: Optional[str] = None
    fiyat: float = 0.0
    acilis: float = 0.0
    degisim_yuzde: float = 0.0
    hacim: float = 0.0
    hacim_degisim_yuzde: float = 0.0  # 🆕 önceki bara göre hacim % değişimi (ARTAN/AZALAN sıralaması için)
    yuksek: float = 0.0
    dusuk: float = 0.0


def binance_toplu_ticker_getir() -> dict:
    """
    🆕 Binance'in TEK ÇAĞRIDA TÜM sembollerin 24 saatlik fiyat/değişim/
    hacim verisini döndüren toplu endpoint'ini (/api/v3/ticker/24hr,
    parametresiz) kullanır. KRIPTO_TAM_LISTE gibi büyük (örn. 500 coin)
    listelerde, coin başına ayrı çağrı yapmak yerine bu fonksiyon
    KULLANILMALIDIR — aksi halde 500 ayrı HTTP isteği gerekir.

    Dönüş: {"BTCUSDT": {"fiyat": ..., "degisim_yuzde": ..., "hacim": ...,
                          "acilis": ..., "yuksek": ..., "dusuk": ...}, ...}
    Binance sembol formatında (örn. "BTCUSDT") anahtarlanır — çağıran
    kod, kendi "BTC-USD" formatını _binance_symbol_from_yahoo_style ile
    çevirip bu sözlükten arama yapmalıdır.

    Hata durumunda boş sözlük döner (çağıran kod bunu "veri yok" olarak
    ele almalıdır).
    """
    try:
        resp = _BINANCE_SESSION.get(
            BINANCE_BASE_URL + "/api/v3/ticker/24hr", timeout=10
        )
        resp.raise_for_status()
        veri = resp.json()
    except Exception:
        return {}

    sonuc = {}
    for item in veri:
        try:
            sonuc[item["symbol"]] = {
                "fiyat": float(item["lastPrice"]),
                "degisim_yuzde": float(item["priceChangePercent"]),
                "hacim": float(item["quoteVolume"]),  # USDT cinsinden hacim (daha anlamlı karşılaştırma)
                "acilis": float(item["openPrice"]),
                "yuksek": float(item["highPrice"]),
                "dusuk": float(item["lowPrice"]),
            }
        except (KeyError, ValueError, TypeError):
            continue
    return sonuc


def hafif_fiyat_getir(sembol: str, pazar: str, interval: str = "1d") -> HafifFiyatSonucu:
    """
    Tek bir varlık için, ML ÇALIŞTIRMADAN, sadece son barın fiyat/hacim
    bilgisini döner. Anasayfa ve Tarama ekranlarında kullanılır.
    """
    try:
        period = suggest_period(pazar, interval)
        fetch_result = get_market_data(sembol, period, interval, pazar)
        df = fetch_result.df

        if df.empty or len(df) < 2:
            return HafifFiyatSonucu(sembol=sembol, pazar=pazar, basarili=False,
                                     hata_mesaji=fetch_result.warning or "Veri bulunamadı.")

        son_bar = df.iloc[-1]
        onceki_bar = df.iloc[-2]

        fiyat = float(son_bar['Close'])
        onceki_kapanis = float(onceki_bar['Close'])
        degisim_yuzde = ((fiyat - onceki_kapanis) / onceki_kapanis * 100) if onceki_kapanis != 0 else 0.0

        son_hacim = float(son_bar['Volume']) if 'Volume' in son_bar else 0.0
        onceki_hacim = float(onceki_bar['Volume']) if 'Volume' in onceki_bar else 0.0
        hacim_degisim_yuzde = ((son_hacim - onceki_hacim) / onceki_hacim * 100) if onceki_hacim != 0 else 0.0

        return HafifFiyatSonucu(
            sembol=sembol, pazar=pazar, basarili=True,
            fiyat=fiyat,
            acilis=float(son_bar['Open']),
            degisim_yuzde=degisim_yuzde,
            hacim=son_hacim,
            hacim_degisim_yuzde=hacim_degisim_yuzde,
            yuksek=float(son_bar['High']),
            dusuk=float(son_bar['Low']),
        )
    except Exception as exc:
        return HafifFiyatSonucu(sembol=sembol, pazar=pazar, basarili=False,
                                 hata_mesaji=f"Beklenmeyen hata: {exc}")


@dataclass
class GostergeSerileriSonucu:
    """
    🆕 Market sayfası "Analiz" sekmesi için — OHLCV + TÜM görsel
    göstergeleri (RSI, MACD, Bollinger, ATR, Stochastic, WaveTrend, CCI,
    EMA 20/50/100, SMA 200) zaman serisi olarak döner. ML pipeline'ı
    (Triple Barrier, model eğitimi, Purged CV) HİÇ ÇALIŞTIRILMAZ —
    sadece calculate_metrics() çağrılır, bu da hızlıdır (saniyenin
    altında).
    """
    sembol: str
    pazar: str
    basarili: bool
    hata_mesaji: Optional[str] = None
    tarihler: list = field(default_factory=list)
    open: list = field(default_factory=list)
    high: list = field(default_factory=list)
    low: list = field(default_factory=list)
    close: list = field(default_factory=list)
    volume: list = field(default_factory=list)
    ema_20: list = field(default_factory=list)
    ema_50: list = field(default_factory=list)
    ema_100: list = field(default_factory=list)
    sma_200: list = field(default_factory=list)
    bollinger_ust: list = field(default_factory=list)
    bollinger_orta: list = field(default_factory=list)
    bollinger_alt: list = field(default_factory=list)
    rsi: list = field(default_factory=list)
    macd: list = field(default_factory=list)
    macd_sinyal: list = field(default_factory=list)
    stoch_k: list = field(default_factory=list)
    stoch_d: list = field(default_factory=list)
    wt1: list = field(default_factory=list)
    wt2: list = field(default_factory=list)
    cci: list = field(default_factory=list)
    atr: list = field(default_factory=list)


# Görüntülemede kullanılacak son bar sayısı — tüm geçmişi göndermek
# gereksiz veri taşır, son N bar grafik için yeterlidir.
GOSTERGE_GORUNUM_BAR_SAYISI = 200


def gosterge_serileri_getir(sembol: str, pazar: str, interval: str = "1d") -> GostergeSerileriSonucu:
    """Market sayfası Analiz sekmesindeki grafikler için OHLCV + tüm
    görsel göstergeleri (ML ÇALIŞTIRMADAN) zaman serisi olarak döner."""
    try:
        period = suggest_period(pazar, interval)
        fetch_result = get_market_data(sembol, period, interval, pazar)
        df_raw = fetch_result.df

        if df_raw.empty:
            return GostergeSerileriSonucu(sembol=sembol, pazar=pazar, basarili=False,
                                           hata_mesaji=fetch_result.warning or "Veri bulunamadı.")

        df_active, _ = calculate_metrics(df_raw)
        if df_active.empty:
            return GostergeSerileriSonucu(sembol=sembol, pazar=pazar, basarili=False,
                                           hata_mesaji="Gösterge hesaplaması için yeterli veri yok.")

        gorunum_df = df_active.tail(GOSTERGE_GORUNUM_BAR_SAYISI)

        def _liste(kolon):
            return gorunum_df[kolon].round(6).tolist() if kolon in gorunum_df.columns else []

        return GostergeSerileriSonucu(
            sembol=sembol, pazar=pazar, basarili=True,
            tarihler=[str(t) for t in gorunum_df.index],
            open=_liste('Open'), high=_liste('High'), low=_liste('Low'),
            close=_liste('Close'), volume=_liste('Volume'),
            ema_20=_liste('EMA_20'), ema_50=_liste('EMA_50'), ema_100=_liste('EMA_100'),
            sma_200=_liste('SMA_200'),
            bollinger_ust=_liste('Bollinger_Ust'), bollinger_orta=_liste('Bollinger_Orta'),
            bollinger_alt=_liste('Bollinger_Alt'),
            rsi=_liste('RSI'), macd=_liste('MACD'), macd_sinyal=_liste('MACD_Sig'),
            stoch_k=_liste('Stoch_K'), stoch_d=_liste('Stoch_D'),
            wt1=_liste('WT1'), wt2=_liste('WT2'),
            cci=_liste('CCI'), atr=_liste('ATR'),
        )
    except Exception as exc:
        return GostergeSerileriSonucu(sembol=sembol, pazar=pazar, basarili=False,
                                       hata_mesaji=f"Beklenmeyen hata: {exc}")


VARLIK_HAVUZU = {
    "KRIPTO": [
        "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ZEC-USD", "BNB-USD", "DOGE-USD", "NEAR-USD",
        "WLD-USD", "AAVE-USD", "TRX-USD", "XPL-USD", "SYN-USD", "HEI-USD", "U-USD", "SUI-USD",
        "ENA-USD", "TAO-USD", "ADA-USD", "PEPE-USD", "SPCXB-USD", "RE-USD", "LTC-USD", "AVAX-USD",
        "TRUMP-USD", "LINK-USD", "PAXG-USD", "XAUT-USD", "XLM-USD", "G-USD", "HYPER-USD", "JTO-USD",
        "UNI-USD", "INJ-USD", "UTK-USD", "FOGO-USD", "TNSR-USD", "BCH-USD", "SNDKB-USD", "PUMP-USD",
        "ASTER-USD", "MUB-USD", "APE-USD", "FET-USD", "FIL-USD", "ONDO-USD", "WLFI-USD", "SEI-USD",
        "AGLD-USD", "ALLO-USD", "PORTAL-USD", "DOT-USD", "MEGA-USD", "DASH-USD", "OP-USD", "PENGU-USD",
        "ATM-USD", "RESOLV-USD", "HBAR-USD", "CELO-USD", "HMSTR-USD", "KAITO-USD", "LUNC-USD", "WBTC-USD",
        "BEL-USD", "ICP-USD", "POL-USD", "ZRO-USD", "BIO-USD", "APT-USD", "OPG-USD", "ETHFI-USD",
        "TON-USD", "RENDER-USD", "TIA-USD", "STO-USD", "W-USD", "ARB-USD", "ID-USD", "DEXE-USD",
        "CITY-USD", "BICO-USD", "SHIB-USD", "CRCLB-USD", "ORDI-USD", "DYDX-USD", "ESP-USD", "LRC-USD",
        "CHZ-USD", "SPK-USD", "ETC-USD", "MORPHO-USD", "PENDLE-USD", "ALGO-USD", "EDEN-USD", "JST-USD",
        "VIRTUAL-USD", "WIF-USD", "NFP-USD", "CHIP-USD", "LUMIA-USD", "CHR-USD", "ZAMA-USD", "LDO-USD",
        "OPN-USD", "ZK-USD", "EIGEN-USD", "AXS-USD", "BONK-USD", "ATOM-USD", "RAY-USD", "HUMA-USD",
        "MITO-USD", "SAHARA-USD", "CAKE-USD", "BABY-USD", "STRK-USD", "GIGGLE-USD", "D-USD", "IO-USD",
        "0G-USD", "KAIA-USD", "AR-USD", "NEIRO-USD", "FLOKI-USD", "CRV-USD", "OGN-USD", "PUNDIX-USD",
        "RNDR-USD", "BANANAS31-USD", "AVNT-USD", "OPEN-USD", "MANTA-USD", "COCOS-USD", "POLY-USD", "PHB-USD",
        "RPL-USD", "NIGHT-USD", "PSG-USD", "STG-USD", "MOVE-USD", "EPIC-USD", "HOME-USD", "PHA-USD",
        "AWE-USD", "FTM-USD", "MET-USD", "MEME-USD", "KITE-USD", "GAL-USD", "ZEN-USD", "TUT-USD",
        "HIGH-USD", "PYTH-USD", "BERA-USD", "SKY-USD", "GALA-USD", "ALICE-USD", "S-USD", "AT-USD",
        "A2Z-USD", "DENT-USD", "MBL-USD", "OMNI-USD", "BAR-USD", "JASMY-USD", "RDNT-USD", "TOMO-USD",
        "PNUT-USD", "CFG-USD", "RIF-USD", "MATIC-USD", "STRAX-USD", "COMP-USD", "MMT-USD", "QNT-USD",
        "ATA-USD", "TRB-USD", "EDU-USD", "TVK-USD", "FIO-USD", "ZBT-USD", "SAND-USD", "WAXP-USD",
        "LAYER-USD", "IOTA-USD", "HEMI-USD", "GENIUS-USD", "WCT-USD", "RUNE-USD", "EOS-USD", "FF-USD",
        "ENSO-USD", "LOKA-USD", "BLZ-USD", "LA-USD", "BREV-USD", "HIFI-USD", "COS-USD", "MSTRB-USD",
        "CYBER-USD", "SYS-USD", "KLAY-USD", "HNT-USD", "AIGENSYN-USD", "MBOX-USD", "PARTI-USD", "HFT-USD",
        "ROBO-USD", "POND-USD", "SXP-USD", "A-USD", "RED-USD", "KAT-USD", "JUV-USD", "XTZ-USD",
        "PLA-USD", "SUN-USD", "LIT-USD", "MANTRA-USD", "AGIX-USD", "FIDA-USD", "SNX-USD", "TRU-USD",
        "DNT-USD", "BNX-USD", "ORCA-USD", "YFII-USD", "VET-USD", "NOT-USD", "PLUME-USD", "MOVR-USD",
        "MFT-USD", "GUN-USD", "NMR-USD", "BEAM-USD", "SAGA-USD", "ENS-USD", "TST-USD", "SENT-USD",
        "BTG-USD", "DAR-USD", "NOM-USD", "VIC-USD", "OOKI-USD", "BAKE-USD", "BOND-USD", "NEWT-USD",
        "OMG-USD", "DEGO-USD", "SCRT-USD", "VANA-USD", "BTCST-USD", "GRT-USD", "ANC-USD", "LAZIO-USD",
        "CFX-USD", "OXT-USD", "1000CHEEMS-USD", "RARE-USD", "ERN-USD", "ELF-USD", "MC-USD", "ASR-USD",
        "ALCX-USD", "NEXO-USD", "FUN-USD", "GPS-USD", "ACM-USD", "PROVE-USD", "LOOM-USD", "DYM-USD",
        "USUAL-USD", "XMR-USD", "AUTO-USD", "OM-USD", "AION-USD", "NBS-USD", "NIL-USD", "BSW-USD",
        "BIFI-USD", "NEO-USD", "CGPT-USD", "ALT-USD", "PORTO-USD", "SUPER-USD", "ANT-USD", "FRONT-USD",
        "SOMI-USD", "ARKM-USD", "XAI-USD", "KAVA-USD", "REZ-USD", "GTO-USD", "PDA-USD", "GFT-USD",
        "SCR-USD", "MIR-USD", "IMX-USD", "NEBL-USD", "DCR-USD", "SUSHI-USD", "VGX-USD", "INTCB-USD",
        "BANK-USD", "ENJ-USD", "TCT-USD", "GMT-USD", "OG-USD", "HOOK-USD", "ACX-USD", "ORN-USD",
        "ROSE-USD", "TOWNS-USD", "TORN-USD", "AIXBT-USD", "WTC-USD", "TRIBE-USD", "NVDAB-USD", "REN-USD",
        "REP-USD", "FLOW-USD", "SRM-USD", "API3-USD", "2Z-USD", "DOCK-USD", "ME-USD", "SANTOS-USD",
        "STPT-USD", "FXS-USD", "THETA-USD", "CVX-USD", "LSK-USD", "KMNO-USD", "STX-USD", "ZIL-USD",
        "BOME-USD", "EUL-USD", "PIXEL-USD", "SOLV-USD", "MKR-USD", "TREE-USD", "WAVES-USD", "AUCTION-USD",
        "FORTH-USD", "SPELL-USD", "C-USD", "CVP-USD", "QUICK-USD", "ACT-USD", "ALPHA-USD", "FIRO-USD",
        "ACH-USD", "MDT-USD", "MITH-USD", "HOLO-USD", "LTO-USD", "TWT-USD", "OSMO-USD", "DREP-USD",
        "IDEX-USD", "NTRN-USD", "WRX-USD", "LUNA-USD", "HIVE-USD", "OCEAN-USD", "CVC-USD", "BROCCOLI714-USD",
        "REEF-USD", "MUBARAK-USD", "UNFI-USD", "LEVER-USD", "ANIME-USD", "FORM-USD", "ACE-USD", "VIB-USD",
        "IRIS-USD", "1INCH-USD", "ACA-USD", "FOR-USD", "AMB-USD", "DOGS-USD", "KP3R-USD", "PERL-USD",
        "ADX-USD", "PIVX-USD", "WAL-USD", "BB-USD", "KSM-USD", "MOB-USD", "CLV-USD", "EGLD-USD",
        "ZKP-USD", "KDA-USD", "MULTI-USD", "AXL-USD", "DODO-USD", "LINEA-USD", "SNT-USD", "MLN-USD",
        "MASK-USD", "WING-USD", "PYR-USD", "RSR-USD", "SOPH-USD", "ALPINE-USD", "COOKIE-USD", "SAPIEN-USD",
        "VELODROME-USD", "SKL-USD", "COMBO-USD", "EPX-USD", "ONG-USD", "BNT-USD", "MDX-USD", "1MBABYDOGE-USD",
        "LISTA-USD", "TURBO-USD", "SLF-USD", "RONIN-USD", "1000SATS-USD", "RVN-USD", "FARM-USD", "XEM-USD",
        "BAT-USD", "BETA-USD", "BEAMX-USD", "STMX-USD", "INIT-USD", "NXPC-USD", "PEOPLE-USD", "POLS-USD",
        "MANA-USD", "MIRA-USD", "QKC-USD", "AUDIO-USD", "CREAM-USD", "ALPACA-USD", "MINA-USD", "CELR-USD",
        "CKB-USD", "CTXC-USD", "BANANA-USD", "SIGN-USD", "GAS-USD", "BARD-USD", "BTS-USD", "PNT-USD",
        "ARDR-USD", "CATI-USD", "FLM-USD", "AMP-USD", "YGG-USD", "C98-USD", "YB-USD", "ZKC-USD",
        "GLMR-USD", "LPT-USD", "AEVO-USD", "VITE-USD", "GTC-USD", "DOLO-USD", "BAL-USD", "T-USD",
        "SXT-USD", "ONT-USD", "EWYB-USD", "ASTR-USD", "KERNEL-USD", "BURGER-USD", "MAGIC-USD", "KMD-USD",
        "FTT-USD", "GMX-USD", "F-USD", "WAN-USD", "ONE-USD", "NBT-USD", "THE-USD", "AKRO-USD",
        "FIS-USD", "TURTLE-USD", "OAX-USD", "RAD-USD", "DUSK-USD", "BLUR-USD", "YFI-USD", "TSLAB-USD",
        "POLYX-USD", "SSV-USD", "VTHO-USD", "FLUX-USD", "COTI-USD", "LQTY-USD", "ERA-USD", "TLM-USD",
        "ILV-USD", "VIDT-USD", "BADGER-USD", "NKN-USD",
    ],
    "TR_HISSE": [
        "THYAO.IS", "ASELS.IS", "KCHOL.IS", "EREGL.IS", "AKBNK.IS", "GARAN.IS", "BIMAS.IS", "SAHOL.IS",
        "TUPRS.IS", "SISE.IS", "FROTO.IS", "TCELL.IS", "PGSUS.IS", "YKBNK.IS", "ISCTR.IS", "A1CAP.IS",
        "ACSEL.IS", "ADEL.IS", "ADGYO.IS", "AEFES.IS", "AFYON.IS", "AGESA.IS", "AGHOL.IS", "AGROT.IS",
        "AHGAZ.IS", "AKCNS.IS", "AKENR.IS", "AKFGY.IS", "AKFYE.IS", "AKGRT.IS", "AKMGY.IS", "AKSA.IS",
        "AKSEN.IS", "AKSGY.IS", "ALARK.IS", "ALBRK.IS", "ALCTL.IS", "ALFAS.IS", "ALGYO.IS", "ALKA.IS",
        "ALKIM.IS", "ALTNY.IS", "ALVES.IS", "ANELE.IS", "ANGEN.IS", "ANHYT.IS", "ANSGR.IS", "ARCLK.IS",
        "ARDYZ.IS", "ARENA.IS", "ARSAN.IS", "ASGYO.IS", "ASTOR.IS", "ASUZU.IS", "ATAKP.IS", "ATATP.IS",
        "AVOD.IS", "AYDEM.IS", "AYGAZ.IS", "BAGFS.IS", "BAKAB.IS", "BALAT.IS", "BANVT.IS", "BARMA.IS",
        "BAYRK.IS", "BERA.IS", "BEYAZ.IS", "BFREN.IS", "BIENY.IS", "BIGCH.IS", "BINHO.IS", "BIOEN.IS",
        "BIZIM.IS", "BLCYT.IS", "BMSCH.IS", "BMSTL.IS", "BNTAS.IS", "BOBET.IS", "BORLS.IS", "BORSK.IS",
        "BOSSA.IS", "BRISA.IS", "BRKO.IS", "BRKSN.IS", "BRMEN.IS", "BRSAN.IS", "BRYAT.IS", "BSOKE.IS",
        "BTCIM.IS", "BUCIM.IS", "BURCE.IS", "BURVA.IS", "BVSAN.IS", "BYDNR.IS", "CASA.IS", "CATES.IS",
        "CCOLA.IS", "CELHA.IS", "CEMAS.IS", "CEMTS.IS", "CEOEM.IS", "CIMSA.IS", "CLEBI.IS", "CMBTN.IS",
        "CMENT.IS", "CONSE.IS", "COSMO.IS", "CRDFA.IS", "CUSAN.IS", "CVKMD.IS", "CWENE.IS", "DAGI.IS",
        "DAPGM.IS", "DESA.IS", "DESPC.IS", "DEVA.IS", "DGGYO.IS", "DGNMO.IS", "DIRIT.IS", "DITAS.IS",
        "DMSAS.IS", "DNISI.IS", "DOAS.IS", "DOCO.IS", "DOGUB.IS", "DOHOL.IS", "DOKTA.IS", "DURDO.IS",
        "DYOBY.IS", "DZGYO.IS", "EBEBK.IS", "ECILC.IS", "ECZYT.IS", "EDATA.IS", "EDIP.IS", "EGEEN.IS",
        "EGEPO.IS", "EGGUB.IS", "EGSER.IS", "EKGYO.IS", "EKIZ.IS", "EKOS.IS", "ELITE.IS", "EMKEL.IS",
        "ENERY.IS", "ENJSA.IS", "ENKAI.IS", "EPLAS.IS", "ERBOS.IS", "ERCB.IS", "ERSU.IS", "ESCAR.IS",
        "ESCOM.IS", "ESEN.IS", "ETILR.IS", "EUPWR.IS", "EYGYO.IS", "FADE.IS", "FENER.IS", "FLAP.IS",
        "FMIZP.IS", "FONET.IS", "FORMT.IS", "FRIGO.IS", "GENIL.IS", "GENTS.IS", "GEREL.IS", "GESAN.IS",
        "GIPTA.IS", "GLBMD.IS", "GLCVY.IS", "GLRYH.IS", "GLYHO.IS", "GOODY.IS", "GOZDE.IS", "GRNYO.IS",
        "GRSEL.IS", "GSDHO.IS", "GSDDE.IS", "GUBRF.IS", "GWIND.IS", "GZNMI.IS", "HALKB.IS", "HATEK.IS",
        "HEDEF.IS", "HEKTS.IS", "HKTM.IS", "HLGYO.IS", "HTTBT.IS", "HUBVC.IS", "HUNER.IS", "HURGZ.IS",
        "ICBCT.IS", "IDGYO.IS", "IEYHO.IS", "IHAAS.IS", "IHEVA.IS", "IHGZT.IS", "IHLAS.IS", "IHYAY.IS",
        "IMASM.IS", "INDES.IS", "INFO.IS", "INTEM.IS", "ISATR.IS", "ISBTR.IS", "ISDMR.IS", "ISFIN.IS",
        "ISGSY.IS", "ISMEN.IS", "ISSEN.IS", "IZENR.IS", "IZFAS.IS", "IZMDC.IS", "JANTS.IS", "KAPLM.IS",
        "KATMR.IS", "KAYSE.IS", "KCAER.IS", "KENT.IS", "KFEIN.IS", "KGYO.IS", "KIMMR.IS", "KLGYO.IS",
        "KLMSN.IS", "KLNMA.IS", "KLRHO.IS", "KLSER.IS", "KNFRT.IS", "KONTR.IS", "KONYA.IS", "KORDS.IS",
        "KOTON.IS", "KRVGD.IS", "KRTEK.IS", "KRONT.IS", "KSTUR.IS", "KUTPO.IS", "KUVVA.IS", "LIDFA.IS",
        "LINK.IS", "LKMNH.IS", "LMKDC.IS", "LOGO.IS", "LUKSK.IS", "MAALT.IS", "MACKO.IS", "MAGEN.IS",
        "MAKIM.IS", "MAKTK.IS", "MANAS.IS", "MARKA.IS", "MARTI.IS", "MAVI.IS", "MEDTR.IS", "MEGAP.IS",
        "MEGMT.IS", "MEPET.IS", "MERCN.IS", "MERKO.IS", "METRO.IS", "MGROS.IS", "MIATK.IS", "MNDRS.IS",
        "MNDTR.IS", "MOBTL.IS", "MOGAN.IS", "MPARK.IS", "MRGYO.IS", "MRSHL.IS", "MSGYO.IS", "MTRKS.IS",
        "MTRYO.IS", "MZHLD.IS", "NATEN.IS", "NETAS.IS", "NIBAS.IS", "NTGAZ.IS", "NUGYO.IS", "NUHCM.IS",
        "OBASE.IS", "ODAS.IS", "OFSYM.IS", "ONCSM.IS", "ORGE.IS", "ORMA.IS", "OSMEN.IS", "OSTIM.IS",
        "OTKAR.IS", "OYAKC.IS", "OYAYO.IS", "OYLUM.IS", "OYYAT.IS", "OZGYO.IS", "OZKGY.IS", "OZSUB.IS",
        "PAGYO.IS", "PAMEL.IS", "PARSN.IS", "PASEU.IS", "PEKGY.IS", "PENGD.IS", "PENTA.IS", "PETKM.IS",
        "PETUN.IS", "PINSU.IS", "PKART.IS", "PKENT.IS", "PLTUR.IS", "PNLSN.IS", "PNSUT.IS", "POLHO.IS",
        "POLTK.IS", "PRKAB.IS", "PRKME.IS", "PRZMA.IS", "PSDTC.IS", "QUAGR.IS", "RALYH.IS", "RAYSG.IS",
        "RNPOL.IS", "RODRG.IS", "RYSAS.IS", "RYGYO.IS", "SAMAT.IS", "SANEL.IS", "SANKO.IS", "SARKY.IS",
        "SASA.IS", "SAYAS.IS", "SDTTR.IS", "SEKFK.IS", "SELEC.IS", "SELVA.IS", "SEYKM.IS", "SILVR.IS",
        "SKTAS.IS", "SMART.IS", "SMRTG.IS", "SNGYO.IS", "SNICA.IS", "SNPAM.IS", "SONME.IS", "SRVGY.IS",
        "SUMAS.IS", "SUNTK.IS", "SURGY.IS", "TABGD.IS", "TATEN.IS", "TATGD.IS", "TAVHL.IS", "TBORG.IS",
        "TDGYO.IS", "TEKTU.IS", "TERA.IS", "TEZOL.IS", "TKFEN.IS", "TKNSA.IS", "TLMAN.IS", "TMPOL.IS",
        "TMSN.IS", "TOASO.IS", "TRCAS.IS", "TRGYO.IS", "TRILC.IS", "TSKB.IS", "TSPOR.IS", "TTKOM.IS",
        "TTRAK.IS", "TUCLK.IS", "TUKAS.IS", "TURGG.IS", "TURSG.IS", "UFUK.IS", "ULAS.IS", "ULKER.IS",
        "ULUFA.IS", "ULUSE.IS", "VAKBN.IS", "VAKFN.IS", "VAKKO.IS", "VANGD.IS", "VBTYZ.IS", "VESTL.IS",
        "VESBE.IS", "VKGYO.IS", "VKING.IS", "YAPRK.IS", "YATAS.IS", "YAYLA.IS", "YGGYO.IS", "YONGA.IS",
        "YUNSA.IS", "YYAPI.IS", "ZEDUR.IS", "ZOREN.IS", "ZRGYO.IS",
    ],
    "ABD_HISSE": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "BRK-B",
        "LLY", "AVGO", "TSLA", "JPM", "WMT", "V", "UNH", "XOM",
        "MA", "PG", "JNJ", "HD", "COST", "ABBV", "MRK", "NFLX",
        "CRM", "BAC", "ORCL", "CVX", "KO", "AMD", "PEP", "ADBE",
        "TMO", "LIN", "WFC", "MCD", "CSCO", "ABT", "ACN", "DIS",
        "INTU", "DHR", "GE", "CAT", "VZ", "IBM", "NOW", "PFE",
        "TXN", "QCOM", "AMAT", "AXP", "PM", "NEE", "UNP", "RTX",
        "LOW", "SPGI", "AMGN", "HON", "COP", "BX", "UPS", "T",
        "ETN", "BSX", "SYK", "GS", "ELV", "BLK", "PGR", "UBER",
        "MS", "ISRG", "LMT", "SCHW", "TJX", "NKE", "REGN", "C",
        "VRTX", "ADP", "MU", "CB", "BA", "PLD", "PANW", "MDT",
        "ANET", "SBUX", "DE", "SO", "BMY", "KLAC", "GILD", "ADI",
        "LRCX", "WM", "DUK", "SHW", "EQIX", "ICE", "CMG", "CME",
        "PYPL", "MO", "ZTS", "TT", "CDNS", "SNPS", "NOC", "APH",
        "CI", "FDX", "ECL", "CL", "ITW", "MCK", "PH", "WELL",
        "AON", "MCO", "USB", "TGT", "EOG", "CSX", "MAR", "MSI",
        "NSC", "PNC", "EMR", "CRWD", "AJG", "ROP", "APD", "CARR",
        "ORLY", "GD", "PSA", "AFL", "WMB", "ADSK", "TFC", "SLB",
        "HLT", "PCAR", "SRE", "COF", "MET", "FTNT", "DLR", "NXPI",
        "TRV", "O", "ALL", "FCX", "PSX", "KMB", "AEP", "SPG",
        "JCI", "AMP", "URI", "FAST", "KMI", "CPRT", "D", "GWW",
        "PWR", "AIG", "PCG", "PEG", "XEL", "KDP", "MNST", "CMI",
        "OKE", "AME", "ROST", "CTAS", "RSG", "DHI", "KVUE", "VRSK",
        "NDAQ", "EW", "YUM", "FANG", "CTVA", "IT", "HSY", "A",
        "EA", "DOW", "WTW", "CHTR", "GIS", "DD", "LULU", "EXC",
        "ODFL", "ON", "HPQ", "ACGL", "VICI", "DAL", "EFX", "IDXX",
        "MLM", "MPWR", "XYL", "HAL", "VMC", "WAB", "FITB", "CCI",
        "GLW", "MTD", "RMD", "ED", "WEC", "NUE", "TSCO", "FTV",
        "AVB", "STZ", "TYL", "HIG", "BR", "DXCM", "HUM", "CAH",
        "CSGP", "ETR", "GPN", "SYY", "BIIB", "STT", "TROW", "WST",
        "FE", "CHD", "NTAP", "DOV", "DTE", "INVH", "VLTO", "AEE",
        "PPG", "RJF", "FSLR", "TDY", "CNP", "VTR", "PPL", "ROK",
        "WY", "EQR", "ZBH", "KEYS", "CBOE", "ES", "COO", "EBAY",
        "HPE", "LH", "HBAN", "WAT", "PHM", "MAA", "STE", "SBAC",
        "LVS", "BALL", "IFF", "ULTA", "MOH", "TER", "FOXA", "FOX",
        "NTRS", "SWKS", "MKC", "EXR", "ARE", "PFG", "CINF", "CFG",
        "RF", "TXT", "JBHT", "OMC", "ALGN", "SYF", "BBY", "DGX",
        "CE", "CMS", "DRI", "CDW", "LDOS", "GEN", "NRG", "BG",
        "LYB", "ATO", "EXPE", "IRM", "EXPD", "AKAM", "TRMB", "L",
        "PODD", "INCY", "MAS", "JKHY", "CAG", "WBD", "LKQ", "AMCR",
        "NDSN", "UAL", "TPL", "LUV", "KIM", "UDR", "SWK", "EVRG",
        "CPT", "BXP", "NWSA", "NWS", "POOL", "PNR", "WRB", "AIZ",
        "HST", "FFIV", "TAP", "APTV", "CHRW", "MGM", "RVTY", "TECH",
        "PAYC", "SJM", "MRNA", "ALLE", "GL", "DVA", "EMN", "NI",
        "VTRS", "WYNN", "DPZ", "DOC", "FRT", "BWA", "AOS", "HSIC",
        "HII", "MTCH", "HRL", "CRL", "PNW", "CPB", "BEN", "MOS",
        "GNRC", "TFX", "MKTX", "IP", "CZR", "EPAM", "ENPH", "FMC",
        "QRVO", "PDD", "ASML", "ARM", "TTD", "MDB", "DDOG", "ZS",
        "TEAM", "WDAY", "ABNB", "MELI", "ILMN", "LCID", "SIRI", "MRVL",
        "CEG", "GFS", "DASH", "ROKU", "ZM", "KHC", "CCEP", "BIDU",
        "JD", "NTES", "DOCU",
    ],
    "EMTIA": ["GC=F", "SI=F", "CL=F", "BZ=F", "NG=F"],  # Gold, Silver, WTI, Brent, Natural Gas
}

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

RITIM_MATRISI = {
    "15m":  {"KRIPTO": "60d", "TR_HISSE": "30d", "ABD_HISSE": "60d", "EMTIA": "45d"},
    "30m":  {"KRIPTO": "60d", "TR_HISSE": "30d", "ABD_HISSE": "60d", "EMTIA": "45d"},
    "60m":  {"KRIPTO": "2y",  "TR_HISSE": "1y",  "ABD_HISSE": "2y",  "EMTIA": "1y"},
    "1h":   {"KRIPTO": "2y",  "TR_HISSE": "1y",  "ABD_HISSE": "1y",  "EMTIA": "1y"},
    "90m":  {"KRIPTO": "60d", "TR_HISSE": "30d", "ABD_HISSE": "60d", "EMTIA": "45d"},
    "2h":   {"KRIPTO": "2y",  "TR_HISSE": "1y",  "ABD_HISSE": "1y",  "EMTIA": "1y"},
    "4h":   {"KRIPTO": "2y",  "TR_HISSE": "1y",  "ABD_HISSE": "1y",  "EMTIA": "1y"},
    "6h":   {"KRIPTO": "2y",  "TR_HISSE": "1y",  "ABD_HISSE": "1y",  "EMTIA": "1y"},
    "8h":   {"KRIPTO": "2y",  "TR_HISSE": "1y",  "ABD_HISSE": "1y",  "EMTIA": "1y"},
    "12h":  {"KRIPTO": "2y",  "TR_HISSE": "1y",  "ABD_HISSE": "1y",  "EMTIA": "1y"},
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


TRADING_DAYS_PER_YEAR_BY_MARKET = {
    "KRIPTO": 365, "TR_HISSE": 252, "ABD_HISSE": 252, "EMTIA": 252,
}

TRADING_HOURS_PER_DAY_BY_MARKET = {
    "KRIPTO": 24.0, "TR_HISSE": 6.5, "ABD_HISSE": 6.5, "EMTIA": 23.0,
}

_INTERVAL_TO_HOURS = {
    "15m": 0.25, "30m": 0.5, "60m": 1.0, "1h": 1.0, "90m": 1.5,
    "2h": 2.0, "4h": 4.0, "6h": 6.0, "8h": 8.0, "12h": 12.0,
}


def get_annual_factor(market: str, interval: str) -> float:
    trading_days = TRADING_DAYS_PER_YEAR_BY_MARKET.get(market, 252)
    if interval in ("1d",):
        return float(trading_days)
    if interval in ("1w", "1wk"):
        return float(trading_days / 7.0)
    if interval in ("1mo",):
        return float(trading_days / 30.0)
    hours_per_day = TRADING_HOURS_PER_DAY_BY_MARKET.get(market, 6.5)
    interval_hours = _INTERVAL_TO_HOURS.get(interval)
    if interval_hours is None:
        return float(trading_days)
    bars_per_day = hours_per_day / interval_hours
    return float(trading_days * bars_per_day)


COMMISSION_RATE_BY_MARKET = {
    "KRIPTO": 0.0010, "TR_HISSE": 0.0020, "ABD_HISSE": 0.0005, "EMTIA": 0.0008,
}


def get_commission_rate(market: str) -> float:
    return COMMISSION_RATE_BY_MARKET.get(market, 0.0005)


MAX_ACCEPTABLE_STALENESS_DAYS_BY_MARKET = {
    "KRIPTO": 2, "TR_HISSE": 3, "ABD_HISSE": 3, "EMTIA": 4,
}

DAILY_AND_ABOVE_INTERVALS = {"1d", "3d", "5d", "1wk", "1w", "1mo", "1M", "3mo"}


def check_data_freshness(df: pd.DataFrame, market: str, interval: str) -> Optional[str]:
    if df.empty or interval not in DAILY_AND_ABOVE_INTERVALS:
        return None

    son_mum_tarihi = pd.Timestamp(df.index.max())
    if son_mum_tarihi.tzinfo is not None:
        son_mum_tarihi = son_mum_tarihi.tz_localize(None)

    simdi = pd.Timestamp.now()
    gecikme_gun = (simdi - son_mum_tarihi).days
    esik = MAX_ACCEPTABLE_STALENESS_DAYS_BY_MARKET.get(market, 5)

    if gecikme_gun > esik:
        return (
            f"Son veri noktası {son_mum_tarihi.strftime('%d %B %Y')} tarihli — bugünden "
            f"{gecikme_gun} gün geride. '{market}' pazarı için normalin (≤{esik} gün) üzerinde."
        )
    return None


# #####################################################################
# #####################################################################
#  BÖLÜM B — GÖSTERGE MOTORLARI + FEATURE ENGINEERING
# #####################################################################
# #####################################################################

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


# =====================================================================
# 🛡️ TRIPLE BARRIER LABELING (V62 — araştırma serisinden üretime taşındı)
# =====================================================================
# V61'deki naif etiketleme ("3 bar sonra %0.2 üstü mü") YERİNE geçti.
# 6 turluk bir araştırma serisinde (bkz. arastirma/ klasörü) bu yöntem
# kapsamlı şekilde test edildi: Triple Barrier + Purged CV ile genel
# motorda istatistiksel olarak anlamlı bir "edge" BULUNAMADI (47
# TR_HISSE kombinasyonu dahil tüm pazarlarda permütasyon testi p>0.05).
#
# BU YÖNTEMİ NEDEN HÂLÂ KULLANIYORUZ (edge bulunamamasına rağmen)?
# Çünkü Triple Barrier + Purged CV, NAİF yönteme göre METODOLOJİK
# OLARAK kesinlikle daha sağlamdır (volatiliteye duyarlı bariyerler,
# bilgi sızıntısı önleme) — "edge yok" sonucu motorun KÖTÜ olduğunu
# göstermiyor, dürüstçe ölçüldüğünde piyasanın bu şekilde
# tahmin edilemediğini gösteriyor. Bu yüzden ürün artık "kazanç
# vaadi olmayan, şeffaf bir karar destek aracı" olarak konumlandırılıyor
# (bkz. AnalizSonucu.arastirma_notu alanı ve dashboard'daki uyarılar).
def triple_barrier_etiketle(
    close: pd.Series, high: pd.Series, low: pd.Series, atr: pd.Series,
    kar_al_katsayisi: float = 2.0, zarar_kes_katsayisi: float = 1.5, max_bar: int = 20,
):
    """
    Her zaman noktası için ATR'ye göre ölçeklenmiş kâr-al/zarar-kes
    bariyerlerinin hangisinin önce tetiklendiğine (veya max_bar içinde
    hiçbiri tetiklenmezse zaman aşımındaki getiriye) bakarak etiketler.
    Döner: (hedef, gercek_getiri) — ikisi de pd.Series, df ile aynı index.
    """
    n = len(close)
    hedef = np.full(n, np.nan)
    gercek_getiri = np.full(n, np.nan)

    close_arr = close.values
    high_arr = high.values
    low_arr = low.values
    atr_arr = atr.values

    for i in range(n - 1):
        giris_fiyat = close_arr[i]
        atr_degeri = atr_arr[i]
        if np.isnan(atr_degeri) or atr_degeri <= 0:
            continue

        ust_bariyer = giris_fiyat + kar_al_katsayisi * atr_degeri
        alt_bariyer = giris_fiyat - zarar_kes_katsayisi * atr_degeri
        bitis_idx = min(i + 1 + max_bar, n)
        tetiklendi = False

        for j in range(i + 1, bitis_idx):
            # Aynı barda hem üst hem alt tetiklenebilir; KONSERVATİF
            # yaklaşımla önce zarar-kesin tetiklendiğini varsayıyoruz.
            if low_arr[j] <= alt_bariyer:
                hedef[i] = 0
                gercek_getiri[i] = (alt_bariyer - giris_fiyat) / giris_fiyat
                tetiklendi = True
                break
            if high_arr[j] >= ust_bariyer:
                hedef[i] = 1
                gercek_getiri[i] = (ust_bariyer - giris_fiyat) / giris_fiyat
                tetiklendi = True
                break

        if not tetiklendi:
            son_idx = bitis_idx - 1
            if son_idx > i:
                zaman_asimi_getiri = (close_arr[son_idx] - giris_fiyat) / giris_fiyat
                hedef[i] = 1 if zaman_asimi_getiri > 0 else 0
                gercek_getiri[i] = zaman_asimi_getiri

    return pd.Series(hedef, index=close.index), pd.Series(gercek_getiri, index=close.index)


class PurgedEmbargoCV:
    """
    🛡️ PURGED + EMBARGO CROSS-VALIDATION (V62 — araştırma serisinden
    üretime taşındı). TimeSeriesSplit'in YERİNE geçti.

    NEDEN: Her gözlemin etiketi ileriye (max_bar kadar) bakarak
    hesaplandığı için, normal TimeSeriesSplit train/test arasında
    BİLGİ SIZINTISI yaratabilir (train setinin son satırlarının etiketi,
    test setinin ilk satırlarındaki fiyatlara bakarak hesaplanmış olabilir).
    Purging, bu çakışan gözlemleri train setinden çıkarır.
    """

    def __init__(self, n_splits: int = 3, etiket_ufku: int = 20):
        self.n_splits = n_splits
        self.etiket_ufku = etiket_ufku

    def split(self, X):
        n = len(X)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1

        current = 0
        fold_bounds = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            fold_bounds.append((start, stop))
            current = stop

        for fold_idx in range(1, self.n_splits):
            test_start, test_stop = fold_bounds[fold_idx]
            test_idx = np.arange(test_start, test_stop)

            train_adaylari = np.arange(0, test_start)
            purge_baslangic = max(0, test_start - self.etiket_ufku)
            train_idx = train_adaylari[train_adaylari < purge_baslangic]

            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        return self.n_splits - 1


def calculate_metrics(df: pd.DataFrame):
    """
    Tüm teknik göstergeleri + ham fiyat feature'larını hesaplar.
    Döner: (df_active, sma_200_guvenilir)

    🐛 KRİTİK DÜZELTME (V61.2'den taşındı): `Yuzde_Getiri_3G` ileriye
    dönük (shift(-3)) hesaplandığı için son 3 satırda HER ZAMAN NaN
    olur. dropna() bu yüzden SADECE görüntüleme dışı (ileriye-dönük)
    sütunlara uygulanır; Close/EMA/RSI gibi görüntüleme sütunları son
    satırlarda korunur. ML eğitimi zaten ayrıca son 3 satırı
    `df_ml = df_active.iloc[:-3]` ile dışlar (bkz. BÖLÜM C).
    """
    if df.empty or len(df) < 35:
        return pd.DataFrame(), False

    df_out = df.copy()
    c = df_out['Close'].squeeze()
    h = df_out['High'].squeeze()
    l = df_out['Low'].squeeze()
    v = df_out['Volume'].squeeze()

    df_out['EMA_20'] = c.ewm(span=20, adjust=False).mean()
    df_out['EMA_50'] = c.ewm(span=50, adjust=False).mean()
    df_out['EMA_100'] = c.ewm(span=100, adjust=False).mean()

    sma_200_pencere = min(200, len(df_out) // 2)
    sma_200_guvenilir = sma_200_pencere >= 200
    df_out['SMA_200'] = c.rolling(window=sma_200_pencere).mean()
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

    df_out['return_1'] = c.pct_change(1)
    df_out['return_5'] = c.pct_change(5)
    df_out['return_20'] = c.pct_change(20)
    df_out['range_pct'] = (h - l) / (c + 1e-10)
    df_out['volume_change_5'] = v.pct_change(5)

    width_mean = df_out['Bollinger_Width'].rolling(20).mean()
    df_out['Regime_Sideways'] = np.where(df_out['Bollinger_Width'] < width_mean * 0.8, 1.0, 0.0)
    df_out['Regime_Bull'] = np.where((df_out['Regime_Sideways'] == 0) & (c > df_out['SMA_200']), 1.0, 0.0)
    df_out['Regime_Bear'] = np.where((df_out['Regime_Sideways'] == 0) & (c <= df_out['SMA_200']), 1.0, 0.0)

    df_out['Hedef'], df_out['Yuzde_Getiri_3G'] = triple_barrier_etiketle(
        close=c, high=h, low=l, atr=df_out['ATR'],
        kar_al_katsayisi=2.0, zarar_kes_katsayisi=1.5, max_bar=20,
    )
    # NOT: Sütun adı 'Yuzde_Getiri_3G' geriye dönük uyumluluk için
    # korunmuştur (görüntüleme/backtest kodu bu adı kullanıyor) — ama
    # artık "3 gün" anlamına gelmiyor, Triple Barrier'ın gerçekleştiği
    # andaki getiriyi temsil ediyor (bkz. triple_barrier_etiketle).
    #
    # ⚠️ ÖNEMLİ: 'Hedef' burada BİLEREK NaN olarak bırakılıyor (ATR henüz
    # hesaplanmamışsa veya son max_bar satırında bariyer hesaplanamazsa).
    # Bu NaN'ları sessizce 0'a çevirmek YANLIŞ olur — "hesaplanamadı"
    # ile "AZALIŞ etiketi" birbirine karışır. ML eğitimi (calistir_ml_pipeline)
    # bu NaN'lı satırları kendi içinde ayrıca filtreler.

    ileriye_donuk_kolonlar = ['Yuzde_Getiri_3G', 'Hedef']
    goruntuleme_kolonlari = [col for col in df_out.columns if col not in ileriye_donuk_kolonlar]

    df_out[goruntuleme_kolonlari] = df_out[goruntuleme_kolonlari].replace([np.inf, -np.inf], np.nan)
    df_out = df_out.dropna(subset=goruntuleme_kolonlari)
    return df_out, sma_200_guvenilir


FEATURE_KOLONLARI = [
    'Close', 'EMA_20', 'EMA_50', 'SMA_200', 'Bollinger_Width', 'RSI', 'MACD', 'ATR',
    'Getiri_1G', 'Volatilite_5G', 'Fiyat_SMA200_Orani', 'Trend_Gucu',
    'Regime_Sideways', 'Regime_Bull', 'Regime_Bear',
    'return_1', 'return_5', 'return_20', 'range_pct', 'volume_change_5',
]


# #####################################################################
# #####################################################################
#  BÖLÜM C — ML PIPELINE (walk-forward CV, kalibrasyon, ensemble)
# #####################################################################
# #####################################################################

def _build_base_models(n_estimators: int, scale_pos_weight_value: float):
    gbm = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=3, random_state=42)
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=5, class_weight='balanced', random_state=42)
    if XGB_AVAILABLE:
        xgb = XGBClassifier(
            n_estimators=n_estimators, max_depth=3, learning_rate=0.03, subsample=0.8,
            scale_pos_weight=scale_pos_weight_value, random_state=42, eval_metric='logloss',
        )
    else:
        xgb = LogisticRegression(class_weight='balanced')
    return gbm, rf, xgb


def egit_model_seti(X_tr_sc, y_tr, scale_pos_weight_value, n_estimators, kalibre_et=True):
    """
    NOT: Streamlit'teki @st.cache_resource burada YOK — Django'da her
    istek için modeller sıfırdan eğitilir (seçenek 2: anlık hesaplama).
    İleride performans gerekirse Django cache framework'ü (örn.
    django.core.cache) ile benzer bir önbellekleme eklenebilir.
    """
    gbm, rf, xgb = _build_base_models(n_estimators, scale_pos_weight_value)

    if kalibre_et:
        n_pos = int(np.sum(y_tr))
        n_neg = len(y_tr) - n_pos
        cv_folds = 3 if min(n_pos, n_neg) >= 3 else 2 if min(n_pos, n_neg) >= 2 else 0

        if cv_folds >= 2:
            gbm = CalibratedClassifierCV(gbm, method='isotonic', cv=cv_folds).fit(X_tr_sc, y_tr)
            rf = CalibratedClassifierCV(rf, method='isotonic', cv=cv_folds).fit(X_tr_sc, y_tr)
            xgb = CalibratedClassifierCV(xgb, method='isotonic', cv=cv_folds).fit(X_tr_sc, y_tr)
            return gbm, rf, xgb, True

    gbm.fit(X_tr_sc, y_tr)
    rf.fit(X_tr_sc, y_tr)
    xgb.fit(X_tr_sc, y_tr)
    return gbm, rf, xgb, False


def _ic_dongu_karma_skor_hesapla(preds, forward_returns_sliced, y_true_sliced):
    s_ret = np.where(preds == 1, forward_returns_sliced, -forward_returns_sliced)
    g = np.sum(s_ret[s_ret > 0])
    l = np.abs(np.sum(s_ret[s_ret < 0]))
    pf = np.clip(g / (l + 1e-10), 0.1, 5.0)
    pf_norm = pf / 5.0
    acc = accuracy_score(y_true_sliced, preds)
    return (0.5 * pf_norm) + (0.5 * acc), acc


@dataclass
class MLSonuc:
    boga_ihtimali: float
    ayi_ihtimali: float
    karar: str
    kalibrasyon_aktif: bool
    w_gbm: float
    w_rf: float
    w_xgb: float
    cv_gbm: float
    cv_rf: float
    cv_xgb: float
    acc_gbm: float
    acc_rf: float
    acc_xgb: float
    train_satir_sayisi: int
    profit_factor: float
    max_dd: float
    sharpe: float
    expectancy: float
    win_rate: float
    avg_win: float
    avg_loss: float


def calistir_ml_pipeline(df_active: pd.DataFrame, annual_factor: float, komisyon_orani: float) -> MLSonuc:
    """
    🛡️ V62 — Triple Barrier + Purged CV ile güncellendi (bkz. araştırma
    serisi, arastirma/ klasörü). V61'deki naif "3 bar sonra %0.2 üstü"
    etiketlemesi ve sızıntıya açık TimeSeriesSplit yerine geçti.

    df_active, calculate_metrics() çıktısı olmalı (Hedef/Yuzde_Getiri_3G
    artık Triple Barrier'dan geliyor, Hedef bazı satırlarda NaN olabilir
    — bu satırlar aşağıda ayrıca filtrelenir).
    """
    MAX_BAR = 20  # calculate_metrics() içindeki triple_barrier_etiketle çağrısıyla AYNI değer olmalı

    # Son MAX_BAR satırın etiketi her zaman eksik/güvenilmez olabilir
    # (bariyer hesaplaması için ileriye yeterli bar yok) — bunları ML
    # eğitiminden çıkarıyoruz (görüntüleme verisinde HÂLÂ kalıyorlar).
    df_ml = df_active.iloc[:-MAX_BAR].copy() if len(df_active) > MAX_BAR else df_active.iloc[0:0].copy()
    # Triple Barrier'ın NaN bıraktığı (örn. ATR henüz oturmamış) satırları da çıkar.
    df_ml = df_ml.dropna(subset=['Hedef', 'Yuzde_Getiri_3G'])
    df_ml['Hedef'] = df_ml['Hedef'].astype(int)

    X_all = df_ml[FEATURE_KOLONLARI].copy()
    y_all = df_ml['Hedef']

    split_idx = int(len(X_all) * 0.80)
    X_train, X_test = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
    y_train, y_test = y_all.iloc[:split_idx], y_all.iloc[split_idx:]

    counts_tr = y_train.value_counts(normalize=True)
    scale_pos_weight_value = max(0.1, counts_tr.get(0, 0.5) / (counts_tr.get(1, 0.5) + 1e-10))

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    bugunun_sc = scaler.transform(df_active.iloc[[-1]][FEATURE_KOLONLARI])

    # 🛡️ TimeSeriesSplit -> PurgedEmbargoCV: train/test arasında, etiket
    # ufkuyla (MAX_BAR) çakışan gözlemler artık train setinden temizleniyor.
    pcv = PurgedEmbargoCV(n_splits=4, etiket_ufku=MAX_BAR)
    final_scores = {"gbm": [], "rf": [], "xgb": []}
    acc_pure_scores = {"gbm": [], "rf": [], "xgb": []}

    for tr_idx, te_idx in pcv.split(X_train):
        if len(tr_idx) < 30 or len(te_idx) < 10:
            continue  # Purging sonrası çok küçük kalan fold'ları atla

        X_tr, X_te = X_train.iloc[tr_idx], X_train.iloc[te_idx]
        y_tr, y_te = y_train.iloc[tr_idx], y_train.iloc[te_idx]

        sc_inner = StandardScaler()
        X_tr_sc = sc_inner.fit_transform(X_tr)
        X_te_sc = sc_inner.transform(X_te)
        te_forward_returns = df_ml['Yuzde_Getiri_3G'].iloc[:split_idx].values[te_idx]

        m_gbm, m_rf, m_xgb, _ = egit_model_seti(
            X_tr_sc, y_tr, scale_pos_weight_value, 40, kalibre_et=False
        )

        for name, model in [("gbm", m_gbm), ("rf", m_rf), ("xgb", m_xgb)]:
            scr, ac = _ic_dongu_karma_skor_hesapla(model.predict(X_te_sc), te_forward_returns, y_te)
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

    model_gbm, model_rf, model_xgb, kalibrasyon_aktif = egit_model_seti(
        X_train_sc, y_train, scale_pos_weight_value, 50, kalibre_et=True
    )

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
    expectancy, win_rate, avg_win, avg_loss = 0.0, 0.0, 0.0, 0.0

    if len(test_signals_sliced) > 0 and len(test_returns_sliced) == len(test_signals_sliced):
        raw_strat_returns = np.where(test_signals_sliced == 1, test_returns_sliced, -test_returns_sliced)
        signal_changes = np.diff(test_signals_sliced, prepend=test_signals_sliced[0])
        strategy_returns = raw_strat_returns - np.where(signal_changes != 0, komisyon_orani, 0.0)

        gains_mask = strategy_returns > 0
        losses_mask = strategy_returns < 0
        gains = np.sum(strategy_returns[gains_mask])
        losses = np.abs(np.sum(strategy_returns[losses_mask]))
        if losses > 0:
            profit_factor = gains / losses

        win_rate = float(np.mean(gains_mask)) * 100
        avg_win = float(np.mean(strategy_returns[gains_mask])) if gains_mask.any() else 0.0
        avg_loss = float(np.mean(strategy_returns[losses_mask])) if losses_mask.any() else 0.0
        expectancy = float(np.mean(strategy_returns)) * 100

        # 🐛 KRİTİK DÜZELTME: Max Drawdown önceden np.cumsum (ADDITIVE/toplamsal)
        # ile hesaplanıyordu. Bu, art arda gelen kayıpları kasanın O ANKİ
        # büyüklüğüne göre değil, hep BAŞLANGIÇ değerine göre saydığı için
        # gerçekçi olmayan, %100'ü AŞAN drawdown'lar üretiyordu (örn. %190.6
        # gibi bir rakam, long-only/kaldıraçsız bir stratejide matematiksel
        # olarak imkansızdır — kasanın en fazla %100'ü erir). Doğru yöntem
        # np.cumprod ile BİLEŞİK (compounding) kümülatif getiriyi hesaplayıp
        # drawdown'ı bu eğrinin tepe noktasına göre ORANSAL almaktır.
        cum_r = np.cumprod(1 + strategy_returns)
        if len(cum_r) > 0:
            running_max = np.maximum.accumulate(cum_r)
            drawdown_series = (running_max - cum_r) / running_max
            max_dd = np.max(drawdown_series) * 100
        if len(strategy_returns) > 1 and np.std(strategy_returns) > 0:
            sharpe = (np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(annual_factor)

    return MLSonuc(
        boga_ihtimali=float(boga_ihtimali), ayi_ihtimali=float(ayi_ihtimali), karar=karar,
        kalibrasyon_aktif=kalibrasyon_aktif,
        w_gbm=float(w_gbm), w_rf=float(w_rf), w_xgb=float(w_xgb),
        cv_gbm=float(cv_gbm), cv_rf=float(cv_rf), cv_xgb=float(cv_xgb),
        acc_gbm=float(np.mean(acc_pure_scores['gbm'])) if acc_pure_scores['gbm'] else 0.0,
        acc_rf=float(np.mean(acc_pure_scores['rf'])) if acc_pure_scores['rf'] else 0.0,
        acc_xgb=float(np.mean(acc_pure_scores['xgb'])) if acc_pure_scores['xgb'] else 0.0,
        train_satir_sayisi=len(X_train),
        profit_factor=float(profit_factor), max_dd=float(max_dd), sharpe=float(sharpe),
        expectancy=float(expectancy), win_rate=float(win_rate),
        avg_win=float(avg_win), avg_loss=float(avg_loss),
    )


# #####################################################################
# #####################################################################
#  BÖLÜM E — DIŞA AÇILAN TEK GİRİŞ NOKTASI
# #####################################################################
# #####################################################################

@dataclass
class AnalizSonucu:
    """Django view'larının (veya başka herhangi bir çağıranın) tüketeceği,
    her şeyi tek bir objede toplayan nihai sonuç paketi."""
    basarili: bool
    hata_mesaji: Optional[str] = None
    veri_yetersiz: bool = False

    sembol: str = ""
    pazar: str = ""
    interval: str = ""
    period: str = ""
    veri_kaynagi: str = ""
    sentetik_mi: bool = False
    uyari_metni: Optional[str] = None
    tazelik_uyarisi: Optional[str] = None
    sma_200_guvenilir: bool = True

    # 🆕 SİDEBAR / TEŞHİS BİLGİSİ — V61 Streamlit'teki "Akademik Ritim
    # Bilgisi" ve "Teşhis Bilgisi" panellerinin Django karşılığı. Bu
    # alanlar hesaplanıyordu ama önceden AnalizSonucu'na hiç eklenmemişti.
    annual_factor: float = 0.0
    komisyon_orani: float = 0.0
    df_raw_satir_sayisi: int = 0
    df_raw_ilk_tarih: str = ""
    df_raw_son_tarih: str = ""

    fiyat_su_an: float = 0.0
    atr_gucu: float = 0.0
    sma_200_degeri: float = 0.0
    para_birimi: str = "USD"

    rsi: float = 0.0
    macd: float = 0.0
    stoch_k: float = 0.0
    wt1: float = 0.0
    cci: float = 0.0

    rejim: str = ""

    ml: Optional[MLSonuc] = None

    # Grafik için ham seriler (tarih -> değer listeleri, JSON'a kolayca çevrilir)
    grafik_tarihler: list = field(default_factory=list)
    grafik_open: list = field(default_factory=list)
    grafik_high: list = field(default_factory=list)
    grafik_low: list = field(default_factory=list)
    grafik_close: list = field(default_factory=list)
    grafik_ema20: list = field(default_factory=list)
    grafik_ema50: list = field(default_factory=list)
    grafik_sma200: list = field(default_factory=list)
    grafik_rsi: list = field(default_factory=list)
    grafik_macd: list = field(default_factory=list)
    grafik_macd_sig: list = field(default_factory=list)
    grafik_stoch_k: list = field(default_factory=list)
    grafik_wt1: list = field(default_factory=list)
    grafik_cci: list = field(default_factory=list)
    grafik_volume: list = field(default_factory=list)


def analiz_yap(sembol: str, pazar: str, interval: str) -> AnalizSonucu:
    """
    Tüm motoru tek çağrıda çalıştırır. Django view'ı sadece bunu
    çağırıp sonucu template'e/JSON'a aktarır.

    Örnek kullanım (Django view içinde):
        from .quant_ml_core import analiz_yap
        sonuc = analiz_yap("BTC-USD", "KRIPTO", "1d")
        if not sonuc.basarili:
            ...hata göster...
    """
    period = suggest_period(pazar, interval)
    annual_factor = get_annual_factor(pazar, interval)
    komisyon_orani = get_commission_rate(pazar)

    fetch_result = get_market_data(sembol, period, interval, pazar)
    df_raw = fetch_result.df

    if df_raw.empty:
        return AnalizSonucu(
            basarili=False,
            hata_mesaji=fetch_result.warning or "Veri kaynağından sonuç alınamadı.",
            sembol=sembol, pazar=pazar, interval=interval, period=period,
            annual_factor=annual_factor, komisyon_orani=komisyon_orani,
        )

    tazelik_uyarisi = check_data_freshness(df_raw, pazar, interval)

    df_active, sma_200_guvenilir = calculate_metrics(df_raw)
    # 🛡️ V62: Eşik 30'dan 100'e yükseltildi. Triple Barrier artık son 20
    # satırı (MAX_BAR) ML eğitiminden çıkarıyor; ayrıca PurgedEmbargoCV
    # ile 4 fold'a bölünüp her fold'da etiket ufkuna göre purge ediliyor.
    # Eski eşik (30) ile pratikte eğitime yetecek satır kalmıyordu.
    veri_yetersiz = df_active.empty or len(df_active) < 100

    if veri_yetersiz:
        return AnalizSonucu(
            basarili=True, veri_yetersiz=True,
            sembol=sembol, pazar=pazar, interval=interval, period=period,
            veri_kaynagi=fetch_result.source, sentetik_mi=fetch_result.is_resampled,
            uyari_metni=fetch_result.warning, tazelik_uyarisi=tazelik_uyarisi,
            annual_factor=annual_factor, komisyon_orani=komisyon_orani,
            df_raw_satir_sayisi=len(df_raw),
            df_raw_ilk_tarih=str(df_raw.index.min()) if not df_raw.empty else "",
            df_raw_son_tarih=str(df_raw.index.max()) if not df_raw.empty else "",
        )

    fiyat_su_an = float(df_active['Close'].squeeze().iloc[-1])
    atr_gucu = float(df_active['ATR'].iloc[-1])
    sma_200_degeri = float(df_active['SMA_200'].iloc[-1])
    para_birimi = "TL" if sembol.endswith(".IS") else "USD"

    last_row = df_active.iloc[-1]
    rejim = (
        "YATAY PİYASA" if last_row['Regime_Sideways'] == 1
        else ("BOĞA REJİMİ" if last_row['Regime_Bull'] == 1 else "AYI REJİMİ")
    )

    ml_sonuc = calistir_ml_pipeline(df_active, annual_factor, komisyon_orani)

    # Grafik için son 200 mumu (ya da hepsini, hangisi azsa) gönderiyoruz —
    # tarayıcıya devasa diziler göndermemek için pratik bir sınır.
    grafik_df = df_active.tail(200)

    return AnalizSonucu(
        basarili=True, veri_yetersiz=False,
        sembol=sembol, pazar=pazar, interval=interval, period=period,
        veri_kaynagi=fetch_result.source, sentetik_mi=fetch_result.is_resampled,
        uyari_metni=fetch_result.warning, tazelik_uyarisi=tazelik_uyarisi,
        sma_200_guvenilir=sma_200_guvenilir,
        annual_factor=annual_factor, komisyon_orani=komisyon_orani,
        df_raw_satir_sayisi=len(df_raw),
        df_raw_ilk_tarih=str(df_raw.index.min()),
        df_raw_son_tarih=str(df_raw.index.max()),
        fiyat_su_an=fiyat_su_an, atr_gucu=atr_gucu, sma_200_degeri=sma_200_degeri,
        para_birimi=para_birimi,
        rsi=float(last_row['RSI']), macd=float(last_row['MACD']),
        stoch_k=float(last_row['Stoch_K']), wt1=float(last_row['WT1']), cci=float(last_row['CCI']),
        rejim=rejim,
        ml=ml_sonuc,
        grafik_tarihler=[ts.strftime("%Y-%m-%d %H:%M") for ts in grafik_df.index],
        grafik_open=grafik_df['Open'].round(4).tolist(),
        grafik_high=grafik_df['High'].round(4).tolist(),
        grafik_low=grafik_df['Low'].round(4).tolist(),
        grafik_close=grafik_df['Close'].round(4).tolist(),
        grafik_ema20=grafik_df['EMA_20'].round(4).tolist(),
        grafik_ema50=grafik_df['EMA_50'].round(4).tolist(),
        grafik_sma200=grafik_df['SMA_200'].round(4).tolist(),
        grafik_rsi=grafik_df['RSI'].round(2).tolist(),
        grafik_macd=grafik_df['MACD'].round(4).tolist(),
        grafik_macd_sig=grafik_df['MACD_Sig'].round(4).tolist(),
        grafik_stoch_k=grafik_df['Stoch_K'].round(2).tolist(),
        grafik_wt1=grafik_df['WT1'].round(2).tolist(),
        grafik_cci=grafik_df['CCI'].round(2).tolist(),
        grafik_volume=grafik_df['Volume'].round(2).tolist(),
    )
