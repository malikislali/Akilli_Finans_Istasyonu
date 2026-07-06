"""
=====================================================================
🔬 ARAŞTIRMA TUR 4 — VERİ KALİTESİ TESTİ
=====================================================================
Doğrulama testi (Tur 3) sonucu: TR_HISSE'de pazar-geneli zayıf ama
TUTARLI bir pozitif eğilim bulundu (3 bağımsız testte de +1.94/+2.53/
+3.51 puan, hepsi kendi baseline'ını aştı). Ama hisse-bazında çok
yüksek varyans var (PETKM -27.4, SISE dönem değişince +12.9'dan
-14.5'e geçti).

SORU: Bu pozitif eğilim GERÇEK bir piyasa sinyali mi, yoksa BIST
verisinin Yahoo Finance üzerinden çekilme şeklindeki bir ARTEFAKT mı?

Bu modül, TR_HISSE verisini diğer 3 pazarla (KRIPTO, ABD_HISSE, EMTIA)
4 boyutta karşılaştırır:

  1. EKSİK MUM ORANI: Beklenen takvim günü sayısına göre kaç mum
     gerçekten geldi? BIST'te resmi tatil/yarım gün farkı olabilir.
  2. SIFIR HAREKETLİ BAR ORANI: Close == Open VEYA High == Low == Close
     olan "donmuş" barların oranı — düşük likiditeli hisselerde Yahoo
     bazı günleri hatalı/boş doldurabilir.
  3. AŞIRI SIÇRAMA ORANI: Bir bardan diğerine %20'den büyük fiyat
     değişimi (stock split/temettü ayarlama hatası şüphesi).
  4. ORTALAMA GÜNLÜK VOLATİLİTE: ATR/Close oranı — TR_HISSE'nin
     volatilite KARAKTERİ diğer pazarlardan sistematik olarak farklı
     mı (bu, Triple Barrier'ın davranışını dolaylı etkiler).

Eğer TR_HISSE bu 4 boyutta da diğer pazarlardan ÇARPICI ŞEKİLDE farklı
çıkarsa, Tur 3'teki "pozitif eğilim" in en azından kısmen bir VERİ
ARTEFAKTI olma ihtimali güçlenir. Farklı çıkmazsa, sinyalin gerçek
olma ihtimaline daha çok güvenebiliriz.
=====================================================================
"""

from __future__ import annotations

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import quant_web_project.quant_app.quant_ml_coreBinanceli as core


# Her pazardan birkaç temsilci varlık — Tur 1-3'te kullanılanlarla tutarlı.
KARSILASTIRMA_VARLIKLARI = {
    "TR_HISSE": ["THYAO.IS", "GARAN.IS", "BIMAS.IS", "EREGL.IS", "SISE.IS", "ASELS.IS"],
    "KRIPTO": ["BTC-USD", "ETH-USD", "SOL-USD"],
    "ABD_HISSE": ["AAPL", "MSFT", "TSLA"],
    "EMTIA": ["GC=F", "CL=F"],
}


def tek_varlik_veri_kalitesi_olc(sembol: str, pazar: str, interval: str = "1d") -> dict:
    """Tek bir varlık için 4 veri kalitesi metriğini hesaplar."""
    period = core.suggest_period(pazar, interval)
    fetch_result = core.get_market_data(sembol, period, interval, pazar)
    df = fetch_result.df

    if df.empty or len(df) < 30:
        return {'sembol': sembol, 'pazar': pazar, 'basarili': False,
                'hata': f"Veri yetersiz ({len(df)} satır)."}

    # ---- 1. Eksik mum oranı ----
    takvim_gunu_sayisi = (df.index.max() - df.index.min()).days + 1
    if pazar == "KRIPTO":
        beklenen_mum = takvim_gunu_sayisi  # 7/24 işlem
    else:
        # Kabaca: yılda ~252 iş günü / 365 takvim günü oranı
        beklenen_mum = takvim_gunu_sayisi * (252 / 365)
    gercek_mum = len(df)
    eksik_mum_orani = max(0.0, 1 - (gercek_mum / beklenen_mum)) if beklenen_mum > 0 else None

    # ---- 2. "Donmuş" bar oranı (Close == Open, sıfır günlük hareket) ----
    donmus_bar_sayisi = int((df['Close'] == df['Open']).sum())
    donmus_bar_orani = donmus_bar_sayisi / len(df)

    # ---- 3. Aşırı sıçrama oranı (%20'den büyük bar-to-bar değişim) ----
    getiri = df['Close'].pct_change().dropna()
    asiri_sicrama_sayisi = int((getiri.abs() > 0.20).sum())
    asiri_sicrama_orani = asiri_sicrama_sayisi / len(getiri) if len(getiri) > 0 else 0.0

    # ---- 4. Ortalama günlük volatilite (ATR/Close) ----
    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    atr_close_orani = (atr / close).dropna()
    ortalama_volatilite = float(atr_close_orani.mean()) if len(atr_close_orani) > 0 else None

    return {
        'sembol': sembol, 'pazar': pazar, 'basarili': True,
        'gercek_mum_sayisi': gercek_mum,
        'beklenen_mum_sayisi_kabaca': round(beklenen_mum, 1),
        'eksik_mum_orani': round(eksik_mum_orani, 4) if eksik_mum_orani is not None else None,
        'donmus_bar_orani': round(donmus_bar_orani, 4),
        'asiri_sicrama_orani': round(asiri_sicrama_orani, 4),
        'asiri_sicrama_sayisi': asiri_sicrama_sayisi,
        'ortalama_gunluk_volatilite_pct': round(ortalama_volatilite * 100, 3) if ortalama_volatilite else None,
    }


def pazarlari_karsilastir(varlik_haritasi: dict = None) -> dict:
    """Tüm pazarlardaki varlıkları tek tek ölçer, sonra pazar bazında özetler."""
    if varlik_haritasi is None:
        varlik_haritasi = KARSILASTIRMA_VARLIKLARI

    tum_sonuclar = []
    for pazar, semboller in varlik_haritasi.items():
        print(f"\n[{pazar}]")
        for sembol in semboller:
            print(f"  {sembol} ölçülüyor...")
            sonuc = tek_varlik_veri_kalitesi_olc(sembol, pazar)
            tum_sonuclar.append(sonuc)
            if sonuc['basarili']:
                print(f"    eksik_mum=%{sonuc['eksik_mum_orani']*100:.1f}  "
                      f"donmus_bar=%{sonuc['donmus_bar_orani']*100:.1f}  "
                      f"asiri_sicrama=%{sonuc['asiri_sicrama_orani']*100:.2f}  "
                      f"volatilite=%{sonuc['ortalama_gunluk_volatilite_pct']:.2f}")
            else:
                print(f"    BAŞARISIZ: {sonuc['hata']}")

    # Pazar bazında özet (ortalama)
    pazar_ozet = {}
    for pazar in varlik_haritasi:
        basarili_olanlar = [s for s in tum_sonuclar if s['pazar'] == pazar and s['basarili']]
        if not basarili_olanlar:
            pazar_ozet[pazar] = {'hata': 'Hiçbir varlık başarılı olmadı.'}
            continue

        pazar_ozet[pazar] = {
            'varlik_sayisi': len(basarili_olanlar),
            'ortalama_eksik_mum_orani': float(np.mean([s['eksik_mum_orani'] for s in basarili_olanlar if s['eksik_mum_orani'] is not None])),
            'ortalama_donmus_bar_orani': float(np.mean([s['donmus_bar_orani'] for s in basarili_olanlar])),
            'ortalama_asiri_sicrama_orani': float(np.mean([s['asiri_sicrama_orani'] for s in basarili_olanlar])),
            'ortalama_volatilite_pct': float(np.mean([s['ortalama_gunluk_volatilite_pct'] for s in basarili_olanlar if s['ortalama_gunluk_volatilite_pct'] is not None])),
        }

    return {
        'detayli_sonuclar': tum_sonuclar,
        'pazar_ozet': pazar_ozet,
    }


def ozet_yazdir(sonuc: dict):
    print()
    print("=" * 78)
    print("📊 VERİ KALİTESİ KARŞILAŞTIRMASI — PAZAR BAZINDA ÖZET")
    print("=" * 78)
    print(f"{'Pazar':<12} {'Eksik Mum':<12} {'Donmuş Bar':<12} {'Aşırı Sıçrama':<15} {'Volatilite'}")
    print("-" * 78)

    pazar_ozet = sonuc['pazar_ozet']
    for pazar, ozet in pazar_ozet.items():
        if 'hata' in ozet:
            print(f"{pazar:<12} HATA: {ozet['hata']}")
            continue
        print(f"{pazar:<12} %{ozet['ortalama_eksik_mum_orani']*100:<11.2f} "
              f"%{ozet['ortalama_donmus_bar_orani']*100:<11.2f} "
              f"%{ozet['ortalama_asiri_sicrama_orani']*100:<14.3f} "
              f"%{ozet['ortalama_volatilite_pct']:<.3f}")

    print("-" * 78)
    print()
    print("🎯 YORUM REHBERİ:")
    print("  - Eksik Mum: TR_HISSE diğerlerinden ÇOK yüksekse -> veri boşlukları")
    print("    Triple Barrier'ın bar sayımını bozabilir (zaman aşımı sıklığı değişir).")
    print("  - Donmuş Bar: TR_HISSE'de yüksekse -> düşük likidite, yapay 'durağanlık'")
    print("    sinyali; bu, ATR'nin küçük çıkmasına ve bariyerlerin daha kolay")
    print("    tetiklenmesine yol açabilir (yapay 'kolay kazanç' görünümü).")
    print("  - Aşırı Sıçrama: TR_HISSE'de yüksekse -> split/temettü ayarlama hataları")
    print("    şüphesi; bu sıçramalar Triple Barrier'ı yanlış tetikleyebilir.")
    print("  - Volatilite: TR_HISSE sistematik olarak farklıysa (özellikle yüksekse),")
    print("    kar_al/zarar_kes katsayılarının TR_HISSE için YENİDEN kalibre")
    print("    edilmesi gerekebilir (şu an tüm pazarlar için aynı katsayı kullanılıyor).")
    print("=" * 78)


if __name__ == "__main__":
    print(f"Veri kalitesi testi başlıyor — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    sonuc = pazarlari_karsilastir()
    ozet_yazdir(sonuc)

    cikti_dosyasi = os.path.join(os.path.dirname(__file__), 'veri_kalitesi_sonuclari.json')
    with open(cikti_dosyasi, 'w', encoding='utf-8') as f:
        json.dump(sonuc, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Tam sonuçlar şuraya kaydedildi: {cikti_dosyasi}")
