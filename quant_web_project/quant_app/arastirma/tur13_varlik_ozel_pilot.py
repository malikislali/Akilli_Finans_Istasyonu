"""
=====================================================================
🔬 ARAŞTIRMA TUR 13 — VARLIĞA ÖZEL MODELLER (İlk Pilot)
=====================================================================
Tur 1-12 sonucu: GENEL bir motor (tüm varlıklara aynı şekilde uygulanan
tek bir model/feature seti) hiçbir konfigürasyonda istatistiksel olarak
güvenilir bir edge üretmedi. Cross-Sectional Ranking'de (Tur 11-12)
KRİPTO'da örneklem büyüyünce zayıflayan ama tam sıfırlanmayan bir
sinyal görüldü — bu, "genel model yok, VARLIĞA ÖZEL bir şey olabilir"
hipotezini akla getirdi (ChatGPT'nin de önerisi).

BU TUR: 4 ana pazardan birer TEMSİLCİ varlık (BTC-USD, THYAO.IS, AAPL,
GC=F) seçilip, HER BİRİ İÇİN AYRI VE BAĞIMSIZ bir model eğitilip test
edilir. Amaç: "Bu SPESİFİK varlıkta, bu motorun güvenilir bir edge'i
var mı?" sorusuna, varlık başına ayrı ayrı cevap aramak.

⚠️ KRİTİK METODOLOJİK UYARI (Tur 5-6'nın dersi):
"Varlığa özel model" yaklaşımı ÇOKLU KARŞILAŞTIRMA RİSKİNİ BÜYÜTÜR.
4 varlıktan 1'inin rastgele iyi çıkması istatistiksel olarak BEKLENEN
bir şeydir — bu, "edge bulduk" anlamına gelmez. Bu yüzden HER varlık
için, standart permütasyon testine EK OLARAK, bir "FARKLI DÖNEM
DOĞRULAMASI" yapılır (Tur 3'teki TR_HİSSE doğrulamasıyla aynı mantık):
aynı varlık, 1 yıl ÖNCEKİ bir veri penceresiyle TEKRAR test edilir.
SADECE HER İKİ TESTTE DE (orijinal dönem + geçmiş dönem) tutarlı
pozitif sonuç veren bir varlık, "araştırmaya değer" sayılır — tek
dönemde iyi çıkan bir sonuç YETERSİZ kanıttır.
=====================================================================
"""

from __future__ import annotations

import sys
import os
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import quant_web_project.quant_app.quant_ml_coreBinanceli as core
from arastirma.coklu_test import tek_kombinasyon_test_et

PILOT_VARLIKLAR = [
    ("BTC-USD", "KRIPTO", "1d"),
    ("THYAO.IS", "TR_HISSE", "1d"),
    ("AAPL", "ABD_HISSE", "1d"),
    ("GC=F", "EMTIA", "1d"),
]


def gecmis_donem_fetcher_olustur(gun_kaydirma: int = 365):
    """Tur 3'teki _geçmis_donem_fetcher ile AYNI mantık — AYNI uzunlukta
    ama GEÇMİŞE kaymış bir test penceresi üretir (yfinance ile doğrudan
    start/end aralığı verilerek)."""
    import yfinance as yf

    def fetcher(symbol, period, interval, market, prefer_source=None):
        end_dt = pd.Timestamp.now().normalize() - pd.Timedelta(days=gun_kaydirma)
        start_dt = end_dt - pd.Timedelta(days=365 * 2)  # 2 yıllık geçmiş pencere

        raw = yf.download(symbol, start=start_dt.strftime("%Y-%m-%d"),
                           end=end_dt.strftime("%Y-%m-%d"), interval="1d", progress=False)

        if raw.empty:
            return core.FetchResult(df=pd.DataFrame(), source="gecmis_donem", requested_interval=interval,
                                     actual_native_interval=interval, is_resampled=False,
                                     warning=f"Geçmiş dönem verisi boş döndü ({symbol}).")

        raw = core._clean_yahoo_columns(raw)
        return core.FetchResult(df=raw, source="gecmis_donem", requested_interval=interval,
                                 actual_native_interval=interval, is_resampled=False)

    return fetcher


@dataclass
class VarlikOzelSonuc:
    sembol: str
    pazar: str
    orijinal_donem_fark: Optional[float] = None
    orijinal_donem_sharpe: Optional[float] = None
    orijinal_donem_basarili: bool = False
    orijinal_donem_hata: Optional[str] = None
    gecmis_donem_fark: Optional[float] = None
    gecmis_donem_sharpe: Optional[float] = None
    gecmis_donem_basarili: bool = False
    gecmis_donem_hata: Optional[str] = None

    @property
    def her_iki_donemde_tutarli_pozitif(self) -> bool:
        if not (self.orijinal_donem_basarili and self.gecmis_donem_basarili):
            return False
        return (self.orijinal_donem_fark is not None and self.orijinal_donem_fark > 2.0
                and self.gecmis_donem_fark is not None and self.gecmis_donem_fark > 2.0)


def tek_varlik_cift_donem_test_et(sembol: str, pazar: str, interval: str) -> VarlikOzelSonuc:
    """Bir varlığı HEM orijinal (en güncel) dönemde HEM 1 yıl öncesine
    kaydırılmış bir dönemde test eder — tek dönemde iyi çıkmanın
    yeterli olmadığı ilkesiyle (Tur 3'ün dersi)."""
    sonuc = VarlikOzelSonuc(sembol=sembol, pazar=pazar)

    # ---- Orijinal (güncel) dönem ----
    try:
        orijinal_test_sonucu = tek_kombinasyon_test_et(sembol, pazar, interval, max_bar=20)
        sonuc.orijinal_donem_basarili = orijinal_test_sonucu.basarili
        if orijinal_test_sonucu.basarili:
            sonuc.orijinal_donem_fark = (
                orijinal_test_sonucu.win_rate_gercek - orijinal_test_sonucu.win_rate_permutasyon
            )
            sonuc.orijinal_donem_sharpe = orijinal_test_sonucu.sharpe_gercek
        else:
            sonuc.orijinal_donem_hata = orijinal_test_sonucu.hata_mesaji
    except Exception as exc:
        sonuc.orijinal_donem_hata = f"Beklenmeyen hata: {exc}"

    # ---- Geçmiş dönem (1 yıl önceye kaydırılmış) ----
    orijinal_fetch = core.get_market_data
    core.get_market_data = gecmis_donem_fetcher_olustur(gun_kaydirma=365)
    try:
        gecmis_test_sonucu = tek_kombinasyon_test_et(sembol, pazar, interval, max_bar=20)
        sonuc.gecmis_donem_basarili = gecmis_test_sonucu.basarili
        if gecmis_test_sonucu.basarili:
            sonuc.gecmis_donem_fark = (
                gecmis_test_sonucu.win_rate_gercek - gecmis_test_sonucu.win_rate_permutasyon
            )
            sonuc.gecmis_donem_sharpe = gecmis_test_sonucu.sharpe_gercek
        else:
            sonuc.gecmis_donem_hata = gecmis_test_sonucu.hata_mesaji
    except Exception as exc:
        sonuc.gecmis_donem_hata = f"Beklenmeyen hata: {exc}"
    finally:
        core.get_market_data = orijinal_fetch  # ⚠️ Yan etkiyi geri al

    return sonuc


def tum_pilot_varliklari_test_et(varliklar=None, ilerleme_yazdir=True) -> list:
    if varliklar is None:
        varliklar = PILOT_VARLIKLAR

    sonuclar = []
    for sembol, pazar, interval in varliklar:
        if ilerleme_yazdir:
            print(f"\n[{sembol} ({pazar})] çift dönem testi başlıyor...")
        sonuc = tek_varlik_cift_donem_test_et(sembol, pazar, interval)
        sonuclar.append(sonuc)
        if ilerleme_yazdir:
            print(f"  Orijinal dönem: başarılı={sonuc.orijinal_donem_basarili}  "
                  f"fark={sonuc.orijinal_donem_fark}  sharpe={sonuc.orijinal_donem_sharpe}")
            if not sonuc.orijinal_donem_basarili:
                print(f"    hata: {sonuc.orijinal_donem_hata}")
            print(f"  Geçmiş dönem:   başarılı={sonuc.gecmis_donem_basarili}  "
                  f"fark={sonuc.gecmis_donem_fark}  sharpe={sonuc.gecmis_donem_sharpe}")
            if not sonuc.gecmis_donem_basarili:
                print(f"    hata: {sonuc.gecmis_donem_hata}")
            print(f"  ✅ Her iki dönemde tutarlı pozitif mi?: {sonuc.her_iki_donemde_tutarli_pozitif}")

    return sonuclar


def ozet_yazdir(sonuclar: list):
    print()
    print("=" * 70)
    print("📊 TUR 13 — VARLIĞA ÖZEL MODEL PİLOTU SONUÇ ÖZETİ")
    print("=" * 70)
    print(f"{'Varlık':<12} {'Orijinal Fark':<16} {'Geçmiş Fark':<16} {'Tutarlı mı?'}")
    print("-" * 70)
    for s in sonuclar:
        orij = f"{s.orijinal_donem_fark:+.1f}" if s.orijinal_donem_fark is not None else "BAŞARISIZ"
        gecmis = f"{s.gecmis_donem_fark:+.1f}" if s.gecmis_donem_fark is not None else "BAŞARISIZ"
        tutarli = "✅ EVET" if s.her_iki_donemde_tutarli_pozitif else "❌ hayır"
        print(f"{s.sembol:<12} {orij:<16} {gecmis:<16} {tutarli}")

    print("-" * 70)
    tutarli_varliklar = [s.sembol for s in sonuclar if s.her_iki_donemde_tutarli_pozitif]
    print()
    if tutarli_varliklar:
        print(f"🎯 Her iki dönemde TUTARLI pozitif çıkan varlıklar: {', '.join(tutarli_varliklar)}")
        print("   -> Bu varlık(lar) için varlığa-özel model yaklaşımı ARAŞTIRMAYA DEĞER.")
        print("   -> ANCAK: 4 varlıktan 1-2'sinin tutarlı çıkması, çoklu karşılaştırma")
        print("      riski nedeniyle HALA kesin kanıt DEĞİLDİR — daha fazla dönem/parametre")
        print("      ile ek doğrulama önerilir.")
    else:
        print("❌ Hiçbir varlık her iki dönemde de tutarlı pozitif çıkmadı.")
        print("   -> Varlığa özel model hipotezi bu 4 varlık için DESTEKLENMEDİ.")
    print("=" * 70)


if __name__ == "__main__":
    print(f"Tur 13 (Varlığa Özel Modeller, İlk Pilot) başlıyor — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    sonuclar = tum_pilot_varliklari_test_et()
    ozet_yazdir(sonuclar)

    cikti_dosyasi = os.path.join(os.path.dirname(__file__), 'tur13_varlik_ozel_pilot_sonuclar.json')
    with open(cikti_dosyasi, 'w', encoding='utf-8') as f:
        json.dump([
            {
                'sembol': s.sembol, 'pazar': s.pazar,
                'orijinal_donem_fark': s.orijinal_donem_fark, 'orijinal_donem_sharpe': s.orijinal_donem_sharpe,
                'orijinal_donem_basarili': s.orijinal_donem_basarili, 'orijinal_donem_hata': s.orijinal_donem_hata,
                'gecmis_donem_fark': s.gecmis_donem_fark, 'gecmis_donem_sharpe': s.gecmis_donem_sharpe,
                'gecmis_donem_basarili': s.gecmis_donem_basarili, 'gecmis_donem_hata': s.gecmis_donem_hata,
                'tutarli_pozitif': s.her_iki_donemde_tutarli_pozitif,
            }
            for s in sonuclar
        ], f, indent=2, ensure_ascii=False)
    print(f"\n📁 Sonuçlar şuraya kaydedildi: {cikti_dosyasi}")
