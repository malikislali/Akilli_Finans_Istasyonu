"""
=====================================================================
🔬 ARAŞTIRMA TUR 12 (NİHAİ DOĞRULAMA) — KRİPTO ÖRNEKLEM BÜYÜTME
=====================================================================
Tur 11 sonucu: Cross-Sectional Ranking, KRİPTO pazarında (n_varlık=9)
en güçlü pozitif sinyali verdi (Spearman fark: +0.1255), ama 4 pazarlık
toplam örneklemde istatistiksel anlamlılığa ulaşılamadı (p=0.253) —
çünkü "pazar" düzeyinde örneklem büyütmenin bir yolu yok (sadece 4
pazar kategorisi mevcut).

BU TUR: KRİPTO'nun KENDİ İÇİNDEKİ varlık sayısını 9'dan ~21'e
çıkararak, o PAZARIN İÇİNDEKİ günlük sıralamanın istatistiksel gücünü
artırıyoruz. Bu, Tur 5-6'daki TR_HİSSE örneklem büyütmesiyle AYNI
mantık — "umut verici görünen küçük örneklem, büyüyünce ne olur?"
sorusuna nihai cevap.

YENİ COINLER (Binance'de likit, farklı sektörlerden, mevcut 9 ile
SIFIR çakışma): Layer1 (ADA, DOT, ATOM, NEAR), Layer2 (MATIC, ARB,
OP), DeFi (UNI, AAVE), Exchange token (BNB), diğer büyük cap (LTC, TRX).
=====================================================================
"""

from __future__ import annotations

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import quant_web_project.quant_app.quant_ml_coreBinanceli as core
from arastirma.tur11_cross_sectional_ranking import (
    ranking_test_et, pazar_capraz_kesit_veri_olustur
)

# 🆕 KRİPTO havuzunu GEÇİCİ olarak (sadece bu araştırma scripti için)
# genişletiyoruz — üretim kodundaki VARLIK_HAVUZU'nu KALICI olarak
# değiştirmiyoruz, sadece core.VARLIK_HAVUZU["KRIPTO"] referansını bu
# scriptin çalışması sırasında override ediyoruz.
GENISLETILMIS_KRIPTO_HAVUZU = [
    "BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "XRP-USD", "DOGE-USD",
    "PEPE-USD", "FIL-USD", "LINK-USD",  # Orijinal 9 (Tur 11 ile aynı)
    "ADA-USD", "DOT-USD", "ATOM-USD", "NEAR-USD",  # Layer 1 (yeni, 4)
    "MATIC-USD", "ARB-USD", "OP-USD",  # Layer 2 (yeni, 3)
    "UNI-USD", "AAVE-USD",  # DeFi (yeni, 2)
    "BNB-USD", "LTC-USD", "TRX-USD",  # Diğer büyük cap (yeni, 3)
]  # Toplam: 9 + 12 = 21 coin


def tur12_calistir():
    print(f"Tur 12 (KRİPTO örneklem büyütme) başlıyor — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Yeni KRİPTO havuzu boyutu: {len(GENISLETILMIS_KRIPTO_HAVUZU)} coin (önceki: 9)")
    print()

    orijinal_havuz = core.VARLIK_HAVUZU["KRIPTO"]
    core.VARLIK_HAVUZU["KRIPTO"] = GENISLETILMIS_KRIPTO_HAVUZU
    try:
        sonuc = ranking_test_et("KRIPTO", n_splits=4)
    finally:
        core.VARLIK_HAVUZU["KRIPTO"] = orijinal_havuz  # ⚠️ Yan etkiyi geri al

    print("=" * 70)
    print("📊 TUR 12 SONUCU — KRİPTO (genişletilmiş örneklem)")
    print("=" * 70)
    print("Başarılı mı?:", sonuc.basarili)
    if not sonuc.basarili:
        print("Hata:", sonuc.hata_mesaji)
        return sonuc

    print(f"n_varlık: {sonuc.n_varlik}  n_gün: {sonuc.n_gun}  n_gözlem: {sonuc.n_gozlem}")
    print(f"Spearman gerçek:      {sonuc.spearman_gercek:+.4f}")
    print(f"Spearman permütasyon: {sonuc.spearman_permutasyon:+.4f}")
    fark = sonuc.spearman_gercek - sonuc.spearman_permutasyon
    print(f"Fark: {fark:+.4f}")
    print()
    print("🎯 KARŞILAŞTIRMA:")
    print("  Tur 11 KRİPTO (n_varlık=9):  Spearman fark = +0.1255")
    print(f"  Tur 12 KRİPTO (n_varlık={sonuc.n_varlik}): Spearman fark = {fark:+.4f}")
    if fark < 0.1255 * 0.5:
        print("  -> Fark BELİRGİN ŞEKİLDE KÜÇÜLDÜ. TR_HİSSE'deki ile aynı desen:")
        print("     küçük örneklemdeki sinyal, büyüyünce zayıflıyor/kayboluyor.")
    elif fark > 0.1255 * 0.8:
        print("  -> Fark BENZER KALDI (büyük ölçüde korundu). Bu, ÖNCEKİ turlardan")
        print("     farklı bir sonuç — KRİPTO'daki bu sinyal gerçek olabilir.")
    else:
        print("  -> Fark KISMEN küçüldü. Belirsiz/sınırda bir durum.")
    print("=" * 70)

    cikti_dosyasi = os.path.join(os.path.dirname(__file__), 'tur12_kripto_genis_ornekiem_sonuclari.json')
    with open(cikti_dosyasi, 'w', encoding='utf-8') as f:
        json.dump({
            'n_varlik': sonuc.n_varlik, 'n_gun': sonuc.n_gun, 'n_gozlem': sonuc.n_gozlem,
            'spearman_gercek': sonuc.spearman_gercek,
            'spearman_permutasyon': sonuc.spearman_permutasyon,
            'fark': fark,
            'tur11_kripto_fark_karsilastirma': 0.1255,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Sonuçlar şuraya kaydedildi: {cikti_dosyasi}")
    return sonuc


if __name__ == "__main__":
    tur12_calistir()
