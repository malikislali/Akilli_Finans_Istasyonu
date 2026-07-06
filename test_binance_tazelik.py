"""
Bu betigi SIZIN ortaminizda calistirin (network gerekiyor, Binance'e ulasmali).
Amac: BTC-USD (BTCUSDT) icin Binance'in gercekte ne donduruyu gormek.
"""
import time
import requests
import pandas as pd

BINANCE_BASE_URL = "https://api.binance.com"
BINANCE_KLINES_ENDPOINT = "/api/v3/klines"
BINANCE_MAX_LIMIT = 1000


def fetch_all(symbol="BTCUSDT", interval="1d", lookback_days=1095):
    interval_ms = 86_400_000  # 1d
    end_time_ms = int(time.time() * 1000)
    start_time_ms = end_time_ms - lookback_days * 86_400_000

    all_rows = []
    cursor = start_time_ms
    sayfa_no = 0

    while cursor < end_time_ms:
        sayfa_no += 1
        params = {"symbol": symbol, "interval": interval, "startTime": cursor, "limit": BINANCE_MAX_LIMIT}
        resp = requests.get(BINANCE_BASE_URL + BINANCE_KLINES_ENDPOINT, params=params, timeout=10)
        resp.raise_for_status()
        batch = resp.json()
        print(f"  Sayfa {sayfa_no}: {len(batch)} mum geldi. "
              f"Ilk={pd.to_datetime(batch[0][0], unit='ms') if batch else '-'} "
              f"Son={pd.to_datetime(batch[-1][0], unit='ms') if batch else '-'}")
        if not batch:
            break
        all_rows.extend(batch)
        cursor = batch[-1][0] + interval_ms
        if len(batch) < BINANCE_MAX_LIMIT:
            print(f"  -> Sayfa {len(batch)} < {BINANCE_MAX_LIMIT}, dongu burada durdu (bu normal, son sayfa demektir).")
            break

    return all_rows


print(f"Su an (UTC): {pd.Timestamp.now('UTC')}")
print(f"Su an (yerel): {pd.Timestamp.now()}")
print()
print("BTCUSDT 1d, 3 yil (1095 gun) cekiliyor...")
rows = fetch_all()
print()
print(f"TOPLAM mum sayisi: {len(rows)}")
if rows:
    print(f"Ilk mum tarihi: {pd.to_datetime(rows[0][0], unit='ms')}")
    print(f"SON mum tarihi: {pd.to_datetime(rows[-1][0], unit='ms')}")
    print()
    print("Son 5 mum:")
    for r in rows[-5:]:
        print(" ", pd.to_datetime(r[0], unit='ms'), "Close:", r[4])