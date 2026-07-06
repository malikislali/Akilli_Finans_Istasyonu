"""
=====================================================================
🔬 ARAŞTIRMA TUR 2 — HİPOTEZ TESTLERİ
=====================================================================
Tur 1 sonucu: gerçek piyasa farkı (+1.05 puan), rastgele yürüyüş
baseline'ını (+2.07 puan) AŞAMADI — yani şu anki kurulumla (max_bar=10,
kar_al=2.0xATR, zarar_kes=1.5xATR, 19 klasik gösterge feature'ı)
anlamlı bir edge bulunamadı.

Bu turda 3 ayrı, gerekçeli hipotezi TEK TEK test ediyoruz:

  HİPOTEZ 1 (bu dosya): Etiketleme ufku (max_bar) yanlış seçilmiş
    olabilir. Çok kısa ufuklar gürültüye, çok uzun ufuklar rejim
    değişimine karışabilir. 5/10/20 bar karşılaştırılır.

  HİPOTEZ 2 (gelecek dosya): Feature seti zayıf/eksik olabilir.

  HİPOTEZ 3 (gelecek dosya): Veri miktarı yetersiz olabilir.

Her hipotez AYRI ayrı test edilir ki hangi değişikliğin (varsa) etkili
olduğu net görülsün — hepsini birden değiştirip "iyileşti" demek,
neyin işe yaradığını bilmemizi engeller.
=====================================================================
"""

from __future__ import annotations

import sys
import os
from datetime import datetime
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arastirma.coklu_test import (
    TEST_KOMBINASYONLARI,
    tek_kombinasyon_test_et,
    sonuclari_analiz_et,
    rastgele_yuruyus_baseline_olustur,
)


def hipotez_1_etiketleme_ufku_tara(
    max_bar_degerleri: list = [5, 10, 20],
    kombinasyonlar: list = None,
    baseline_senaryo_sayisi: int = 10,
) -> dict:
    """
    HİPOTEZ 1: Etiketleme ufku (max_bar) sonucu önemli ölçüde değiştirir.

    Her max_bar değeri için AYRI bir baseline (çünkü baseline'ın kendisi
    de max_bar'a bağlı — etiket_ufku, purging'de kullanılıyor) ve AYRI
    bir piyasa testi çalıştırılır. Sonuçta, hangi max_bar değerinde
    (varsa) gerçek piyasa farkı baseline'ı en çok aştığı görülür.
    """
    if kombinasyonlar is None:
        kombinasyonlar = TEST_KOMBINASYONLARI

    sonuclar_max_bar_bazinda = {}

    for max_bar in max_bar_degerleri:
        print()
        print("=" * 70)
        print(f"🔬 HİPOTEZ 1 — max_bar = {max_bar} test ediliyor")
        print("=" * 70)

        print(f"\n[Baseline, max_bar={max_bar}]")
        baseline_sonuclari = []
        for i in range(baseline_senaryo_sayisi):
            import numpy as np
            import pandas as pd
            import quant_web_project.quant_app.quant_ml_coreBinanceli as core

            idx = pd.date_range('2022-01-01', periods=1000, freq='D')
            rng = np.random.default_rng(10_000 + i)
            close = 100 + np.cumsum(rng.normal(0, 2, 1000))
            close = np.clip(close, 10, None)
            df_raw = pd.DataFrame({
                'Open': close - 0.5, 'High': close + 1, 'Low': close - 1,
                'Close': close, 'Volume': rng.integers(1000, 5000, 1000)
            }, index=idx)

            def sahte_fetch(symbol, period, interval, market, prefer_source=None, _df=df_raw):
                return core.FetchResult(df=_df, source="sentetik", requested_interval=interval,
                                         actual_native_interval=interval, is_resampled=False)

            orijinal_fetch = core.get_market_data
            core.get_market_data = sahte_fetch
            try:
                sonuc = tek_kombinasyon_test_et(
                    f"SENTETIK_{i}", "KRIPTO", "1d", max_bar=max_bar, rastgele_seed=i
                )
            finally:
                core.get_market_data = orijinal_fetch
            baseline_sonuclari.append(sonuc)
            if sonuc.basarili:
                print(f"  [{i+1}/{baseline_senaryo_sayisi}] fark: {sonuc.win_rate_gercek - sonuc.win_rate_permutasyon:+.1f}")

        baseline_raporu = sonuclari_analiz_et(baseline_sonuclari)

        print(f"\n[Gerçek piyasa, max_bar={max_bar}]")
        piyasa_sonuclari = []
        for i, (sembol, pazar, interval) in enumerate(kombinasyonlar):
            print(f"  [{i+1}/{len(kombinasyonlar)}] {sembol} ({pazar}, {interval})...")
            sonuc = tek_kombinasyon_test_et(sembol, pazar, interval, max_bar=max_bar)
            piyasa_sonuclari.append(sonuc)
            if sonuc.basarili:
                print(f"      fark: {sonuc.win_rate_gercek - sonuc.win_rate_permutasyon:+.1f}")
            else:
                print(f"      BAŞARISIZ: {sonuc.hata_mesaji}")

        piyasa_raporu = sonuclari_analiz_et(piyasa_sonuclari, rastgele_yuruyus_baseline=baseline_raporu)

        sonuclar_max_bar_bazinda[max_bar] = {
            'baseline': baseline_raporu,
            'piyasa': piyasa_raporu,
        }

    return sonuclar_max_bar_bazinda


def hipotez_1_ozet_yazdir(sonuclar_max_bar_bazinda: dict):
    print()
    print("=" * 70)
    print("📊 HİPOTEZ 1 — NİHAİ KARŞILAŞTIRMA (hangi max_bar daha iyi?)")
    print("=" * 70)
    print(f"{'max_bar':<10} {'baseline_fark':<15} {'piyasa_fark':<15} {'asim':<10} {'aşıldı mı?'}")
    print("-" * 70)

    en_iyi_max_bar = None
    en_iyi_asim = -999

    for max_bar, sonuc in sonuclar_max_bar_bazinda.items():
        piyasa = sonuc.get('piyasa', {})
        if 'hata' in piyasa:
            print(f"{max_bar:<10} ⚠️ YETERSİZ VERİ: {piyasa['hata']}")
            continue
        bk = piyasa.get('test_3_baseline_karsilastirma', {})
        baseline_fark = bk.get('rastgele_yuruyus_baseline_farki', None)
        piyasa_fark = bk.get('gercek_piyasa_farki', None)
        asim = bk.get('asim_miktari', None)
        asildi = bk.get('baseline_asildi_mi', False)

        if baseline_fark is not None:
            print(f"{max_bar:<10} {baseline_fark:+.2f}{'':<9} {piyasa_fark:+.2f}{'':<9} {asim:+.2f}{'':<5} {'✅ EVET' if asildi else '❌ hayır'}")
            if asim is not None and asim > en_iyi_asim:
                en_iyi_asim = asim
                en_iyi_max_bar = max_bar
        else:
            print(f"{max_bar:<10} ⚠️ Karşılaştırma verisi oluşturulamadı (baseline raporu eksik olabilir).")

    print("-" * 70)
    if en_iyi_max_bar is not None:
        print(f"\nEn iyi sonuç: max_bar={en_iyi_max_bar} (aşım: {en_iyi_asim:+.2f} puan)")
        if en_iyi_asim > 0:
            print("  -> Bu max_bar değerinde gerçek piyasa baseline'ı AŞTI — araştırmaya değer bir sinyal adayı.")
        else:
            print("  -> Hiçbir max_bar değerinde baseline aşılamadı. Hipotez 1 DOĞRULANMADI.")
    print("=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("🔬 HİPOTEZ 1 TESTİ BAŞLIYOR — Etiketleme Ufku Taraması")
    print(f"   Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    sonuclar = hipotez_1_etiketleme_ufku_tara(
        max_bar_degerleri=[5, 10, 20],
        baseline_senaryo_sayisi=10,
    )
    hipotez_1_ozet_yazdir(sonuclar)

    cikti_dosyasi = os.path.join(os.path.dirname(__file__), 'hipotez_1_sonuclari.json')
    with open(cikti_dosyasi, 'w', encoding='utf-8') as f:
        json.dump(sonuclar, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Tam sonuçlar şuraya kaydedildi: {cikti_dosyasi}")
