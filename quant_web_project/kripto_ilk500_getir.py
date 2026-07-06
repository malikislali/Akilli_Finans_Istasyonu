"""
Binance'den İlk 500 USDT Çifti (Hacme Göre Sıralı)
=====================================================
Bu script, Binance'in toplu ticker endpoint'ini kullanarak TÜM USDT
çiftlerini çeker, 24 saatlik USDT hacmine göre büyükten küçüğe sıralar
ve ilk 500'ünü (Yahoo-stili "BTC-USD" formatına çevirerek) kaydeder.

ÇALIŞTIRMA:
    python3 kripto_ilk500_getir.py

Bu, kendi makinende (gerçek internet erişimi olan) çalıştırılmalı —
sandbox ortamında api.binance.com'a erişim yok.

ÇIKTI: kripto_ilk500.json (Yahoo-stili sembol listesi, Python listesi
       formatında, VARLIK_HAVUZU'na eklenmeye hazır)
"""

import json
import requests

BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr"
N_COIN = 500

# 🐛 DÜZELTME: İlk çalıştırmada listeye stablecoin'ler (USDC, BUSD, USDE
# gibi — fiyatı sabit ~1, "değişim %" göstergesi anlamsız), fiat para
# çiftleri (EUR, AUD, GBP — kripto değil, Binance'in forex ürünleri) ve
# ASCII olmayan karakterli bir Binance şaka/test sembolü ("币安人生")
# karışmıştı. Bunlar artık BİLİNÇLİ OLARAK filtreleniyor — yerlerine
# listenin altındaki (501., 502., ... sıradaki) gerçek kriptolar gelir.
STABLECOIN_VE_TUREV_TABANLAR = {
    "USDC", "USD1", "RLUSD", "FDUSD", "BUSD", "USDE", "XUSD", "USDS",
    "USTC", "BFUSD", "BETH", "WBETH", "BNSOL", "TUSD", "DAI", "PAX",
    "GUSD", "EURI",  # EURI: Euro stablecoin
}
FIAT_PARA_TABANLARI = {"EUR", "AUD", "GBP", "TRY", "BRL", "RUB", "ZAR", "JPY"}


def gecerli_kripto_mu(binance_sembol: str) -> bool:
    """Stablecoin, fiat çifti, leveraged token, ASCII-olmayan karakter
    içeren sembolleri eler — sadece 'gerçek', fiyatı dalgalanan
    kriptoları kabul eder."""
    if not binance_sembol.isascii():
        return False
    if not binance_sembol.endswith("USDT"):
        return False
    if binance_sembol.endswith(("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")):
        return False
    base = binance_sembol[:-4]
    if base in STABLECOIN_VE_TUREV_TABANLAR:
        return False
    if base in FIAT_PARA_TABANLARI:
        return False
    return True


def binance_yahoo_stiline_cevir(binance_sembol: str) -> str:
    """'BTCUSDT' -> 'BTC-USD' (quant_ml_core.py'nin beklediği format)."""
    if binance_sembol.endswith("USDT"):
        base = binance_sembol[:-4]
        return f"{base}-USD"
    return binance_sembol


def main():
    print("Binance'den tüm ticker verisi çekiliyor...")
    resp = requests.get(BINANCE_TICKER_URL, timeout=15)
    resp.raise_for_status()
    veri = resp.json()
    print(f"Toplam {len(veri)} sembol alındı.")

    # gecerli_kripto_mu(): leveraged token, stablecoin, fiat çifti ve
    # ASCII-olmayan karakterli sembolleri tek bir yerden filtreler.
    usdt_ciftleri = [item for item in veri if gecerli_kripto_mu(item["symbol"])]
    print(f"Geçerli kripto/USDT çiftleri (leveraged/stablecoin/fiat hariç): {len(usdt_ciftleri)}")

    # 24 saatlik USDT hacmine göre büyükten küçüğe sırala
    usdt_ciftleri.sort(key=lambda x: float(x.get("quoteVolume", 0)), reverse=True)

    ilk_n = usdt_ciftleri[:N_COIN]
    yahoo_stili_liste = [binance_yahoo_stiline_cevir(item["symbol"]) for item in ilk_n]

    print(f"\nİlk 10 (hacme göre):")
    for item in ilk_n[:10]:
        print(f"  {item['symbol']:<15} hacim(USDT)={float(item['quoteVolume']):,.0f}")

    with open("kripto_ilk500.json", "w", encoding="utf-8") as f:
        json.dump(yahoo_stili_liste, f, indent=2, ensure_ascii=False)

    print(f"\n📁 {len(yahoo_stili_liste)} sembol kripto_ilk500.json'a kaydedildi.")


if __name__ == "__main__":
    main()
