"""
=====================================================================
🔬 ARAŞTIRMA — ANA ÇALIŞTIRMA SCRIPT'İ
=====================================================================
Bu script, "Bu sistem gerçekten kazandırabilir mi?" sorusuna cevap
aramak için kurduğumuz TÜM araştırma adımlarını sırayla çalıştırır:

  ADIM 1: Rastgele yürüyüş BASELINE'ı oluştur (gerçek edge'in
          MATEMATİKSEL OLARAK MÜMKÜN OLMADIĞI senaryolarda, metodolojinin
          kendisinin ne kadar "doğal" sapma ürettiğini ölç).

  ADIM 2: Gerçek piyasa verisiyle TEST_KOMBINASYONLARI'nı çalıştır
          (varsayılan: 32 varlık/pazar/periyot kombinasyonu).

  ADIM 3: Gerçek piyasa sonucunu, ADIM 1'deki baseline'a karşı kıyasla.
          Eğer gerçek piyasa farkı baseline'ı aşmıyorsa, gözlemlenen
          her "iyi" sonuç muhtemelen şans/metodolojik artefakttır.

  ADIM 4: Sonuçları hem konsola hem bir JSON dosyasına yaz (raporlama,
          ileride tekrar incelemek için).

ÇALIŞTIRMA:
    cd quant_web_project   (quant_ml_core.py'nin bulunduğu dizin)
    python -m arastirma.calistir_arastirma

NOT: Bu script GERÇEK ağ çağrıları yapar (Binance/Yahoo). Tüm test
evreni (32 kombinasyon) + baseline (10 senaryo) çalıştırmak BİRKAÇ
DAKİKA sürebilir — her kombinasyon birden fazla CV fold'unda model
eğitimi içerir.
=====================================================================
"""

from __future__ import annotations

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arastirma.coklu_test import (
    TEST_KOMBINASYONLARI,
    tum_kombinasyonlari_test_et,
    sonuclari_analiz_et,
    rastgele_yuruyus_baseline_olustur,
)


def ana_raporu_yazdir(rapor: dict):
    print()
    print("=" * 70)
    print("📊 NİHAİ ARAŞTIRMA RAPORU")
    print("=" * 70)

    baseline = rapor['baseline']
    piyasa = rapor['piyasa']

    print()
    print(f"Rastgele Yürüyüş Baseline (n={baseline.get('basarili_test_sayisi', '?')} senaryo):")
    print(f"  Ortalama fark (gerçek - permütasyon): {baseline.get('ortalama_fark_gercek_eksi_permutasyon', 0):+.2f} puan")
    print()
    print(f"Gerçek Piyasa Sonucu (n={piyasa.get('basarili_test_sayisi', '?')} kombinasyon, "
          f"{piyasa.get('basarisiz_test_sayisi', 0)} başarısız):")
    print(f"  Ortalama Win Rate (gerçek etiket): %{piyasa.get('win_rate_gercek_ortalama', 0):.1f}")
    print(f"  Ortalama Win Rate (permütasyon):   %{piyasa.get('win_rate_permutasyon_ortalama', 0):.1f}")
    print(f"  Ortalama fark (gerçek - permütasyon): {piyasa.get('ortalama_fark_gercek_eksi_permutasyon', 0):+.2f} puan")

    bk = piyasa.get('test_3_baseline_karsilastirma', {})
    print()
    print("🎯 ASIL SORUNUN CEVABI:")
    if bk:
        print(f"  Rastgele yürüyüş baseline farkı: {bk['rastgele_yuruyus_baseline_farki']:+.2f} puan")
        print(f"  Gerçek piyasa farkı:             {bk['gercek_piyasa_farki']:+.2f} puan")
        if bk['baseline_asildi_mi']:
            print(f"  ✅ Gerçek piyasa farkı baseline'ı {bk['asim_miktari']:+.2f} puan AŞIYOR.")
            print("     -> Bu, gerçek bir sinyal ADAYI olabilir (kesin kanıt değil, ama araştırmaya değer).")
        else:
            print(f"  ❌ Gerçek piyasa farkı baseline'ı AŞMIYOR (fark: {bk['asim_miktari']:+.2f} puan).")
            print("     -> Gözlemlenen performans muhtemelen GERÇEK BİR EDGE DEĞİL, sadece")
            print("        göstergelerin ve etiketin aynı fiyat serisinden türetilmesinden")
            print("        kaynaklanan matematiksel bir artefakt.")
    else:
        print("  (Baseline karşılaştırması yapılamadı — yetersiz veri.)")

    print()
    print("İstatistiksel testler:")
    t1 = piyasa.get('test_1_gercek_vs_50yuzde', {})
    t2 = piyasa.get('test_2_gercek_vs_permutasyon', {})
    print(f"  Test 1 (gerçek vs %50): p={t1.get('p_degeri', 1):.4f}  "
          f"{'ANLAMLI' if t1.get('anlamli_mi_0.05') else 'anlamlı değil'}")
    print(f"  Test 2 (gerçek vs permütasyon): p={t2.get('p_degeri', 1):.4f}  "
          f"{'ANLAMLI' if t2.get('anlamli_mi_0.05') else 'anlamlı değil'}")
    print()
    print("=" * 70)


def calistir(test_kombinasyonlari=None, baseline_senaryo_sayisi: int = 10):
    """Tüm araştırma akışını çalıştırır ve nihai raporu döner."""
    if test_kombinasyonlari is None:
        test_kombinasyonlari = TEST_KOMBINASYONLARI

    print("=" * 70)
    print("🔬 SOVEREIGN COCKPIT — MOTOR ARAŞTIRMASI BAŞLIYOR")
    print(f"   Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"   Test edilecek kombinasyon sayısı: {len(test_kombinasyonlari)}")
    print(f"   Baseline senaryo sayısı: {baseline_senaryo_sayisi}")
    print("=" * 70)

    print()
    print("### ADIM 1/3: Rastgele yürüyüş baseline'ı oluşturuluyor ###")
    baseline_raporu = rastgele_yuruyus_baseline_olustur(n_senaryo=baseline_senaryo_sayisi, n_bar=1000)

    print()
    print("### ADIM 2/3: Gerçek piyasa verisiyle test ediliyor ###")
    piyasa_sonuclari = tum_kombinasyonlari_test_et(test_kombinasyonlari)
    piyasa_raporu = sonuclari_analiz_et(piyasa_sonuclari, rastgele_yuruyus_baseline=baseline_raporu)

    print()
    print("### ADIM 3/3: Sonuçlar karşılaştırılıyor ve raporlanıyor ###")
    nihai_rapor = {
        'olusturulma_tarihi': datetime.now().isoformat(),
        'baseline': baseline_raporu,
        'piyasa': piyasa_raporu,
    }

    ana_raporu_yazdir(nihai_rapor)

    cikti_dosyasi = os.path.join(os.path.dirname(__file__), 'arastirma_sonuclari.json')
    with open(cikti_dosyasi, 'w', encoding='utf-8') as f:
        json.dump(nihai_rapor, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Tam sonuçlar şuraya kaydedildi: {cikti_dosyasi}")

    return nihai_rapor


if __name__ == "__main__":
    calistir()
