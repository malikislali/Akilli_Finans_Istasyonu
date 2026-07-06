"""
=====================================================================
🔌 QUANT AI — BİRLEŞİK VERİ SAĞLAYICI KATMANI (data_provider.py)
=====================================================================
Bu modül iki kaynağı tek bir arayüz altında birleştirir:

  1) YAHOO FINANCE (yfinance)  -> TR_HISSE, ABD_HISSE, EMTIA, ve
     isteğe bağlı KRIPTO (Yahoo'nun -USD sembolleri ile)
  2) BINANCE PUBLIC REST API   -> KRIPTO (gerçek native 4h, 6h, 8h,
     12h, 2h gibi Yahoo'da OLMAYAN interval'lar dahil)

Tasarım kararları:
  - Binance için API key/secret GEREKMİYOR (public klines endpoint'i
    kimlik doğrulama istemez, sadece rate-limit'e tabidir).
  - Her iki kaynağın "native" (gerçekten borsanın/saglayicinin
    desteklediği) interval listesi ayrı ayrı tanımlanır. Native
    olmayan bir interval istenirse, en yakın alt native interval'dan
    resample (yeniden örnekleme) ile sentezlenir. Böylece kullanıcıya
    "deniyor gibi görünüp" yanlış veri vermek yerine açıkça
    SOURCE='resampled' etiketi döner.
  - Şu an için herhangi bir abonelik/free-tier kısıtı YOK. Bu katman
    sadece "tüm interval + tüm pazar" özelliğini sağlar. Kısıtlama
    daha sonra ayrı bir yetkilendirme katmanında (örn. bir
    `check_entitlement(user, interval, market)` fonksiyonu ile)
    devreye alınabilir — bkz. dosya sonundaki ENTITLEMENT NOTU.
=====================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# =====================================================================
# 📚 1. NATIVE INTERVAL TANIMLARI
# =====================================================================

# Yahoo Finance'in GERÇEKTEN desteklediği interval'lar (yfinance docs)
YAHOO_NATIVE_INTERVALS = {
    "1m", "2m", "5m", "15m", "30m",
    "60m", "90m",
    "1d", "5d", "1wk", "1mo", "3mo",
}

# Yahoo'da intraday interval'lar için geriye dönük veri penceresi limitleri
# (Yahoo bu limitleri zorluyor; aşılırsa boş veri ya da hata dönebilir)
YAHOO_INTRADAY_MAX_LOOKBACK = {
    "1m": "7d",
    "2m": "60d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "60m": "730d",
    "90m": "60d",
}

# Binance'in GERÇEKTEN desteklediği kline interval'ları (Binance REST API docs)
BINANCE_NATIVE_INTERVALS = {
    "1s", "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d",
    "1w", "1M",
}

BINANCE_BASE_URL = "https://api.binance.com"
BINANCE_KLINES_ENDPOINT = "/api/v3/klines"
BINANCE_MAX_LIMIT = 1000  # Binance tek istekte en fazla 1000 mum döner


@dataclass
class FetchResult:
    """Tek bir veri çekme işleminin sonucunu ve meta bilgisini taşır."""
    df: pd.DataFrame
    source: str          # "yahoo" | "binance" | "yahoo_resampled" | "binance_resampled"
    requested_interval: str
    actual_native_interval: str
    is_resampled: bool
    warning: Optional[str] = None


# =====================================================================
# 🧮 2. YARDIMCI: PANDAS RESAMPLE KURALI ÜRETİCİ
# =====================================================================

def _interval_to_pandas_rule(interval: str) -> str:
    """
    Bizim interval string'lerimizi ('4h', '1d', '1wk' gibi) pandas'ın
    resample() fonksiyonunun beklediği offset alias'larına çevirir.
    """
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
    """
    Daha küçük bir native interval'dan (örn. 1h) daha büyük bir hedef
    interval'a (örn. 4h) OHLCV agregasyonu yapar. Borsa/finans
    standardına uygun agregasyon kuralları kullanılır:
        Open   -> ilk değer
        High   -> maksimum
        Low    -> minimum
        Close  -> son değer
        Volume -> toplam
    """
    if df.empty:
        return df

    rule = _interval_to_pandas_rule(target_interval)
    agg_map = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    # Sadece DataFrame'de mevcut olan kolonları agregasyona dahil et
    agg_map = {k: v for k, v in agg_map.items() if k in df.columns}

    out = df.resample(rule).agg(agg_map)
    out = out.dropna(subset=["Close"]) if "Close" in out.columns else out.dropna()
    return out


def _find_best_native_source_interval(
    target_interval: str, native_set: set, order: list[str]
) -> Optional[str]:
    """
    target_interval native değilse, resample için kullanılabilecek
    EN YAKIN ALT (daha küçük zaman dilimli) native interval'ı bulur.
    `order` listesi küçükten büyüğe sıralı olmalıdır.
    """
    if target_interval in native_set:
        return target_interval

    try:
        target_idx = order.index(target_interval)
    except ValueError:
        return None  # Hiyerarşide tanımlı değil, resample edilemez

    # Hedeften küçük (idx'ten önceki) native interval'lar arasında en büyüğünü seç
    for candidate in reversed(order[:target_idx]):
        if candidate in native_set:
            return candidate
    return None


# Zaman dilimi hiyerarşisi (küçükten büyüğe) — resample kaynağı seçiminde kullanılır
INTERVAL_ORDER = [
    "1s", "1m", "2m", "3m", "5m", "15m", "30m",
    "60m", "1h", "90m", "2h",
    "4h", "6h", "8h", "12h",
    "1d", "3d", "5d",
    "1wk", "1w",
    "1mo", "1M", "3mo",
]


# =====================================================================
# 🟡 3. YAHOO FINANCE VERİ ÇEKİCİ (TÜM NATIVE INTERVAL'LAR)
# =====================================================================

def _clean_yahoo_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def fetch_yahoo(symbol: str, period: str, interval: str) -> FetchResult:
    """
    Yahoo Finance'ten veri çeker. interval Yahoo'nun native setinde
    değilse, en yakın alt native interval'dan otomatik resample eder.
    """
    native_interval = _find_best_native_source_interval(
        interval, YAHOO_NATIVE_INTERVALS, INTERVAL_ORDER
    )

    if native_interval is None:
        return FetchResult(
            df=pd.DataFrame(), source="yahoo", requested_interval=interval,
            actual_native_interval="", is_resampled=False,
            warning=f"'{interval}' için Yahoo'da uygun bir kaynak interval bulunamadı.",
        )

    fetch_period = period
    if native_interval in YAHOO_INTRADAY_MAX_LOOKBACK:
        # Yahoo'nun intraday lookback limitini aşmamak için period'u sınırla.
        # (Kullanıcı daha uzun period istese de Yahoo zaten kabul etmeyecektir.)
        fetch_period = period  # Limit aşımı Yahoo tarafında hata/boş veri ile kendini gösterir

    raw = yf.download(symbol, period=fetch_period, interval=native_interval, progress=False)
    if raw.empty:
        return FetchResult(
            df=pd.DataFrame(), source="yahoo", requested_interval=interval,
            actual_native_interval=native_interval, is_resampled=False,
            warning=f"Yahoo '{symbol}' için '{native_interval}' interval'ında veri döndürmedi.",
        )

    raw = _clean_yahoo_columns(raw)

    if native_interval == interval:
        return FetchResult(
            df=raw, source="yahoo", requested_interval=interval,
            actual_native_interval=native_interval, is_resampled=False,
        )

    resampled = _resample_ohlcv(raw, interval)
    return FetchResult(
        df=resampled, source="yahoo_resampled", requested_interval=interval,
        actual_native_interval=native_interval, is_resampled=True,
        warning=(
            f"Yahoo native olarak '{interval}' sunmuyor. "
            f"'{native_interval}' verisinden resample edildi."
        ),
    )


# =====================================================================
# 🟠 4. BINANCE PUBLIC API VERİ ÇEKİCİ (TÜM NATIVE INTERVAL'LAR)
# =====================================================================

def _binance_symbol_from_yahoo_style(symbol: str) -> str:
    """
    'BTC-USD' gibi Yahoo-stili kripto sembollerini Binance'in
    beklediği 'BTCUSDT' formatına çevirir. Zaten Binance formatında
    gelen sembolleri (örn. 'BTCUSDT') olduğu gibi geçirir.
    """
    s = symbol.upper().strip()
    if "-" in s:
        base, quote = s.split("-", 1)
        if quote == "USD":
            quote = "USDT"  # Binance spot'ta USD çifti yerine USDT standarttır
        return f"{base}{quote}"
    return s


def _period_to_lookback_days(period: str) -> Optional[int]:
    """
    Bizim 'period' string'lerimizi ('60d', '2y', '3y', 'max' vb.)
    gün sayısına çevirir. 'max' için None döner (Binance'de mevcut
    olan en eski veriden başla anlamına gelir; bu durumda startTime
    göndermeyip sayfalama ile tüm geçmişi çekmek gerekir).
    """
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
    units = {"s": 1_000, "m": 60_000, "h": 3_600_000, "d": 86_400_000,
             "w": 604_800_000}
    if interval == "1M":
        return 30 * 86_400_000  # yaklaşık; ay sayfalamasında kullanılır
    unit = interval[-1]
    value = int(interval[:-1])
    return value * units[unit]


def _fetch_binance_klines_paged(
    symbol: str, interval: str, lookback_days: Optional[int],
    max_candles_cap: int = 20000,
) -> pd.DataFrame:
    """
    Binance public klines endpoint'inden, gerekirse 1000'lik sayfalar
    halinde, istenen geçmişe kadar veri çeker. API key gerektirmez.
    """
    interval_ms = _binance_interval_to_ms(interval)
    end_time_ms = int(time.time() * 1000)

    if lookback_days is not None:
        start_time_ms = end_time_ms - lookback_days * 86_400_000
    else:
        start_time_ms = None  # 'max': en baştan, sayfalayarak ileri gideceğiz

    all_rows = []
    fetched = 0

    if start_time_ms is not None:
        cursor = start_time_ms
        while cursor < end_time_ms and fetched < max_candles_cap:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "limit": BINANCE_MAX_LIMIT,
            }
            resp = requests.get(
                BINANCE_BASE_URL + BINANCE_KLINES_ENDPOINT, params=params, timeout=10
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            all_rows.extend(batch)
            fetched += len(batch)
            last_open_time = batch[-1][0]
            cursor = last_open_time + interval_ms
            if len(batch) < BINANCE_MAX_LIMIT:
                break
    else:
        # 'max' istendi: en eski veriye ulaşana kadar GERİYE sayfalama.
        # Binance startTime vermeden istek atınca en SON mumları döner;
        # bu yüzden endTime'ı kademeli geriye çekerek ilerliyoruz.
        cursor_end = end_time_ms
        while fetched < max_candles_cap:
            params = {
                "symbol": symbol,
                "interval": interval,
                "endTime": cursor_end,
                "limit": BINANCE_MAX_LIMIT,
            }
            resp = requests.get(
                BINANCE_BASE_URL + BINANCE_KLINES_ENDPOINT, params=params, timeout=10
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            all_rows = batch + all_rows
            fetched += len(batch)
            first_open_time = batch[0][0]
            cursor_end = first_open_time - 1
            if len(batch) < BINANCE_MAX_LIMIT:
                break

    if not all_rows:
        return pd.DataFrame()

    cols = [
        "open_time", "Open", "High", "Low", "Close", "Volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(all_rows, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = df[c].astype(float)
    df = df.set_index("open_time")[["Open", "High", "Low", "Close", "Volume"]]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def fetch_binance(symbol: str, period: str, interval: str) -> FetchResult:
    """
    Binance public API'sinden veri çeker (API key GEREKMEZ).
    interval Binance'in native setinde değilse (örn. '90m' gibi
    Binance'de hiç olmayan bir değer), en yakın alt native
    interval'dan resample edilir.
    """
    binance_symbol = _binance_symbol_from_yahoo_style(symbol)

    native_interval = _find_best_native_source_interval(
        interval, BINANCE_NATIVE_INTERVALS, INTERVAL_ORDER
    )
    if native_interval is None:
        return FetchResult(
            df=pd.DataFrame(), source="binance", requested_interval=interval,
            actual_native_interval="", is_resampled=False,
            warning=f"'{interval}' için Binance'de uygun bir kaynak interval bulunamadı.",
        )

    lookback_days = _period_to_lookback_days(period)

    try:
        raw = _fetch_binance_klines_paged(binance_symbol, native_interval, lookback_days)
    except requests.HTTPError as exc:
        return FetchResult(
            df=pd.DataFrame(), source="binance", requested_interval=interval,
            actual_native_interval=native_interval, is_resampled=False,
            warning=f"Binance API hatası: {exc}",
        )
    except requests.RequestException as exc:
        return FetchResult(
            df=pd.DataFrame(), source="binance", requested_interval=interval,
            actual_native_interval=native_interval, is_resampled=False,
            warning=f"Binance API'sine bağlanılamadı: {exc}",
        )

    if raw.empty:
        return FetchResult(
            df=pd.DataFrame(), source="binance", requested_interval=interval,
            actual_native_interval=native_interval, is_resampled=False,
            warning=f"Binance '{binance_symbol}' için veri döndürmedi (sembol yanlış olabilir).",
        )

    if native_interval == interval:
        return FetchResult(
            df=raw, source="binance", requested_interval=interval,
            actual_native_interval=native_interval, is_resampled=False,
        )

    resampled = _resample_ohlcv(raw, interval)
    return FetchResult(
        df=resampled, source="binance_resampled", requested_interval=interval,
        actual_native_interval=native_interval, is_resampled=True,
        warning=(
            f"Binance native olarak '{interval}' sunmuyor. "
            f"'{native_interval}' verisinden resample edildi."
        ),
    )


# =====================================================================
# 🧭 5. PAZAR -> KAYNAK YÖNLENDİRME (ROUTER)
# =====================================================================

# Hangi pazar tipi hangi kaynağı kullanır. KRIPTO'da Binance birincil
# kaynaktır (gerçek 4h/6h/8h/12h/2h sağlar); Yahoo ise KRIPTO için
# yedek/karşılaştırma kaynağı olarak kullanılabilir.
MARKET_SOURCE_MAP = {
    "KRIPTO": "binance",
    "TR_HISSE": "yahoo",
    "ABD_HISSE": "yahoo",
    "EMTIA": "yahoo",
}


def get_market_data(
    symbol: str,
    period: str,
    interval: str,
    market: str,
    prefer_source: Optional[str] = None,
) -> FetchResult:
    """
    Tüm dashboard'un kullanacağı TEK giriş noktası.

    Parametreler
    ------------
    symbol   : "BTC-USD", "THYAO.IS", "AAPL", "GC=F" gibi sembol
    period   : "60d", "1y", "2y", "max" gibi geriye dönük pencere
    interval : "1m","5m","15m","30m","1h","2h","4h","6h","8h","12h",
               "1d","1wk","1mo" ... (Yahoo veya Binance native/resample)
    market   : "KRIPTO" | "TR_HISSE" | "ABD_HISSE" | "EMTIA"
    prefer_source : "yahoo" | "binance" ile router'ı manuel ezmek
                    isterseniz (örn. KRIPTO için Yahoo karşılaştırması)

    Dönen FetchResult.df DataFrame'i Open/High/Low/Close/Volume
    kolonlarını içerir, index'i datetime'dır.

    NOT (abonelik/kısıt entegrasyonu için): Bu fonksiyon şu an HİÇBİR
    kullanıcı/plan kısıtı uygulamaz — tüm interval + tüm pazar her
    çağrıda serbesttir. Ücretli/ücretsiz ayrımı eklemek isterseniz,
    bu fonksiyonu çağırmadan ÖNCE kendi yetkilendirme kontrolünüzü
    yapın (örn. `if not user.is_premium and interval not in FREE_INTERVALS: ...`).
    Dosya sonundaki ENTITLEMENT NOTU bölümüne örnek iskelet eklendi.
    """
    source = prefer_source or MARKET_SOURCE_MAP.get(market, "yahoo")

    if source == "binance":
        result = fetch_binance(symbol, period, interval)
        if result.df.empty and prefer_source is None:
            # Binance'te bulunamadıysa (örn. nadir bir kripto paritesi)
            # Yahoo'ya otomatik düş (best-effort fallback)
            fallback = fetch_yahoo(symbol, period, interval)
            if not fallback.df.empty:
                fallback.warning = (
                    (result.warning or "") +
                    " | Binance'te bulunamadı, Yahoo Finance'e düşüldü."
                ).strip(" |")
                return fallback
        return result

    return fetch_yahoo(symbol, period, interval)


# =====================================================================
# 🎛️ 6. ARAYÜZ İÇİN: PAZAR BAZLI TAM INTERVAL LİSTESİ
# =====================================================================

# Streamlit selectbox'ında kullanıcıya gösterilecek, pazar bazlı TÜM
# (native + resample edilebilir) interval seçenekleri ve etiketleri.
# is_native=True  -> kaynaktan birebir gelir
# is_native=False -> alt native interval'dan resample edilir
DISPLAY_INTERVALS_BY_MARKET = {
    "KRIPTO": [
        # (etiket, interval_kodu, native_mi)
        ("1 Dakika", "1m", True),
        ("3 Dakika", "3m", True),
        ("5 Dakika", "5m", True),
        ("15 Dakika", "15m", True),
        ("30 Dakika", "30m", True),
        ("1 Saat", "1h", True),
        ("2 Saat", "2h", True),
        ("4 Saat", "4h", True),
        ("6 Saat", "6h", True),
        ("8 Saat", "8h", True),
        ("12 Saat", "12h", True),
        ("1 Gün", "1d", True),
        ("3 Gün", "3d", True),
        ("1 Hafta", "1w", True),
        ("1 Ay", "1M", True),
    ],
    "TR_HISSE": [
        ("1 Dakika", "1m", True),
        ("2 Dakika", "2m", True),
        ("5 Dakika", "5m", True),
        ("15 Dakika", "15m", True),
        ("30 Dakika", "30m", True),
        ("1 Saat", "60m", True),
        ("90 Dakika", "90m", True),
        ("4 Saat", "4h", False),     # Yahoo'da yok -> 60m'den resample
        ("1 Gün", "1d", True),
        ("5 Gün", "5d", True),
        ("1 Hafta", "1wk", True),
        ("1 Ay", "1mo", True),
        ("3 Ay", "3mo", True),
    ],
    "ABD_HISSE": [
        ("1 Dakika", "1m", True),
        ("2 Dakika", "2m", True),
        ("5 Dakika", "5m", True),
        ("15 Dakika", "15m", True),
        ("30 Dakika", "30m", True),
        ("1 Saat", "60m", True),
        ("90 Dakika", "90m", True),
        ("4 Saat", "4h", False),
        ("1 Gün", "1d", True),
        ("5 Gün", "5d", True),
        ("1 Hafta", "1wk", True),
        ("1 Ay", "1mo", True),
        ("3 Ay", "3mo", True),
    ],
    "EMTIA": [
        ("15 Dakika", "15m", True),
        ("30 Dakika", "30m", True),
        ("1 Saat", "60m", True),
        ("90 Dakika", "90m", True),
        ("4 Saat", "4h", False),
        ("1 Gün", "1d", True),
        ("1 Hafta", "1wk", True),
        ("1 Ay", "1mo", True),
    ],
}

# Pazar + interval bazlı varsayılan geriye dönük veri penceresi (period)
DEFAULT_PERIOD_BY_MARKET_INTERVAL = {
    "KRIPTO":    {"default": "1y", "intraday_short": "60d"},
    "TR_HISSE":  {"default": "1y", "intraday_short": "30d"},
    "ABD_HISSE": {"default": "2y", "intraday_short": "60d"},
    "EMTIA":     {"default": "1y", "intraday_short": "45d"},
}

INTRADAY_INTERVALS = {
    "1m", "2m", "3m", "5m", "15m", "30m",
    "60m", "1h", "90m", "2h", "4h", "6h", "8h", "12h",
}


def suggest_period(market: str, interval: str) -> str:
    """Pazar ve interval'a göre akıllı bir varsayılan period önerir."""
    cfg = DEFAULT_PERIOD_BY_MARKET_INTERVAL.get(market, {"default": "1y", "intraday_short": "60d"})
    if interval in INTRADAY_INTERVALS:
        return cfg["intraday_short"]
    return cfg["default"]


# =====================================================================
# 🔐 7. ENTİTLEMENT (ABONELİK KISITI) NOTU — ŞU AN PASİF
# =====================================================================
"""
Şu anda bu modülde hiçbir kısıt YOK; tüm fonksiyonlar herkese tüm
interval + tüm pazar erişimini açık tutar (talebiniz üzerine).

İleride paralı/free ayrımı eklemek isterseniz, bu dosyaya dokunmadan
ÇAĞIRAN KOD (örn. Streamlit dashboard'unuz) tarafında şöyle bir kontrol
fonksiyonu yazmanız yeterli olur — veri katmanından tamamen ayrı kalır:

    FREE_TIER_ALLOWED_INTERVALS = {"1d", "1wk", "1mo"}
    FREE_TIER_ALLOWED_MARKETS = {"KRIPTO", "ABD_HISSE"}

    def check_entitlement(user_is_premium: bool, market: str, interval: str) -> tuple[bool, str]:
        if user_is_premium:
            return True, ""
        if market not in FREE_TIER_ALLOWED_MARKETS:
            return False, f"'{market}' pazarı sadece Premium üyelere açıktır."
        if interval not in FREE_TIER_ALLOWED_INTERVALS:
            return False, f"'{interval}' periyodu sadece Premium üyelere açıktır."
        return True, ""

    # Dashboard'da kullanım:
    allowed, reason = check_entitlement(user.is_premium, pazar, interval)
    if not allowed:
        str_ui.warning(f"🔒 {reason} Premium'a yükseltin.")
        str_ui.stop()
    result = get_market_data(sembol, period, interval, pazar)

Bu sayede veri katmanı (bu dosya) hep "tam yetkili" kalır, kısıt
mantığı tamamen UI/iş mantığı tarafında yönetilir — istediğiniz zaman
plan kurallarını değiştirebilirsiniz, veri çekme kodunu bozmadan.
"""
