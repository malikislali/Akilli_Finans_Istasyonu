"""
=====================================================================
🔬 ARAŞTIRMA TUR 6 (SON TUR) — NİHAİ ÖRNEKLEM BÜYÜTME
=====================================================================
Tur 5 sonucu: 29 benzersiz TR_HISSE kombinasyonuyla
  - Test 1 (vs %50): p=0.0062 ANLAMLI
  - Test 2 (vs permütasyon): p=0.0559 SINIRDA (eşiğin az üstünde,
    anlamlı değil ama yakın)
  - 18/29 (%62) pozitif, ortalama +4.87 puan, ama yüksek varyans
    (PETKM -27.5, AKSEN -16.1 gibi aşırı değerler var)

BU TUR (SON TUR): 18 YENİ TR hissesi eklenerek toplam örneklem ~47'ye
çıkarılıyor. Amaç: Test 2'nin p-değerini 0.05 eşiğinin altına düşürüp
düşürmediğini görmek — bu, araştırma serisinin SONUÇLANDIRICI adımıdır.

ÖNCEKİ TURLARDA TEST EDİLEN 29 HİSSE (ÇAKIŞMA YOK):
  THYAO, GARAN, ASELS, EREGL, BIMAS, SISE, TUPRS, KCHOL, AKBNK, PETKM,
  KOZAL, PGSUS, VESTL, TOASO, ENKAI, YKBNK, ISCTR, HALKB, SAHOL, DOAS,
  MGROS, SOKM, AKSEN, ARCLK, ULKER, CCOLA, KOZAA, TTKOM, AGESA

YENİ 18 HİSSE (BIST100'den, çakışma yok, sektör çeşitliliği korunarak):
  bankacılık, holding, perakende/gıda, enerji, sanayi, inşaat,
  teknoloji, sağlık, ulaştırma kategorilerinden.
=====================================================================
"""

from __future__ import annotations

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arastirma.hipotez_testleri import hipotez_1_etiketleme_ufku_tara
from arastirma.coklu_test import sonuclari_analiz_et


# 18 YENİ hisse — önceki 29 ile SIFIR çakışma.
TUR6_YENI_HISSELER = [
    ("VAKBN.IS", "TR_HISSE", "1d"),   # VakıfBank (bankacılık)
    ("TSKB.IS",  "TR_HISSE", "1d"),   # TSKB (bankacılık/kalkınma)
    ("ALARK.IS", "TR_HISSE", "1d"),   # Alarko Holding (holding/enerji)
    ("ENJSA.IS", "TR_HISSE", "1d"),   # Enerjisa (enerji)
    ("TAVHL.IS", "TR_HISSE", "1d"),   # TAV Havalimanları (ulaştırma)
    ("PSGYO.IS", "TR_HISSE", "1d"),   # Pasifik GYO (gayrimenkul)
    ("KORDS.IS", "TR_HISSE", "1d"),   # Kordsa (sanayi/tekstil)
    ("OTKAR.IS", "TR_HISSE", "1d"),   # Otokar (otomotiv/sanayi)
    ("FROTO.IS", "TR_HISSE", "1d"),   # Ford Otosan (otomotiv)
    ("TKFEN.IS", "TR_HISSE", "1d"),   # Tekfen Holding (inşaat/holding)
    ("ALBRK.IS", "TR_HISSE", "1d"),   # Albaraka Türk (katılım bankacılığı)
    ("KARSN.IS", "TR_HISSE", "1d"),   # Karsan (otomotiv/sanayi)
    ("DOHOL.IS", "TR_HISSE", "1d"),   # Doğan Holding (holding/medya)
    ("ECILC.IS", "TR_HISSE", "1d"),   # Eczacıbaşı İlaç (sağlık/ilaç)
    ("DEVA.IS",  "TR_HISSE", "1d"),   # Deva Holding (sağlık/ilaç)
    ("ISGYO.IS", "TR_HISSE", "1d"),   # İş GYO (gayrimenkul)
    ("AEFES.IS", "TR_HISSE", "1d"),   # Anadolu Efes (gıda/içecek)
    ("MAVI.IS",  "TR_HISSE", "1d"),   # Mavi Giyim (perakende/tekstil)
]


def tur6_calistir(baseline_senaryo_sayisi: int = 10) -> dict:
    print("=" * 70)
    print("🔬 TUR 6 (SON TUR) — 18 YENİ TR hissesi, max_bar=20")
    print(f"   Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    sonuc = hipotez_1_etiketleme_ufku_tara(
        max_bar_degerleri=[20],
        kombinasyonlar=TUR6_YENI_HISSELER,
        baseline_senaryo_sayisi=baseline_senaryo_sayisi,
    )
    return sonuc[20]


def tum_turlari_birlestir(
    tur6_sonuc: dict,
    onceki_json_yollari: list,
) -> dict:
    """Tur 2, 3, 5 ve 6'nın TÜM TR_HISSE sonuçlarını tek bir nihai
    örneklemde birleştirir."""
    from dataclasses import dataclass

    @dataclass
    class _SahteTestSonucu:
        sembol: str
        pazar: str
        interval: str
        basarili: bool
        win_rate_gercek: float
        win_rate_permutasyon: float
        sharpe_gercek: float

    tum_detaylar = []
    tum_detaylar.extend(tur6_sonuc.get('piyasa', {}).get('kombinasyon_detaylari', []))

    for yol in onceki_json_yollari:
        if not os.path.exists(yol):
            print(f"⚠️ Uyarı: {yol} bulunamadı, atlanıyor.")
            continue
        with open(yol, encoding='utf-8') as f:
            veri = json.load(f)

        if "20" in veri:  # hipotez_1_sonuclari.json formatı
            tum_detaylar.extend(veri["20"].get("piyasa", {}).get("kombinasyon_detaylari", []))
        elif "test_a_farkli_hisseler" in veri:  # dogrulama_sonuclari.json formatı
            tum_detaylar.extend(veri["test_a_farkli_hisseler"].get("piyasa", {}).get("kombinasyon_detaylari", []))
        elif "birlesik_nihai_rapor" in veri:  # tur5_genis_ornekiem_sonuclari.json formatı
            tum_detaylar.extend(veri["birlesik_nihai_rapor"].get("kombinasyon_detaylari", []))
        elif "bu_turun_sonucu" in veri:  # tur5'in kendi bu_turun_sonucu kısmı (aynı dosyada)
            tum_detaylar.extend(veri["bu_turun_sonucu"].get("piyasa", {}).get("kombinasyon_detaylari", []))

    gorulen = set()
    benzersiz_detaylar = []
    for d in tum_detaylar:
        sembol = d.get('sembol', '')
        if not sembol.endswith('.IS'):
            continue
        anahtar = (sembol, d.get('interval', '1d'))
        if anahtar not in gorulen:
            gorulen.add(anahtar)
            benzersiz_detaylar.append(d)

    print(f"\n📊 NİHAİ BİRLEŞİK ÖRNEKLEM: {len(benzersiz_detaylar)} benzersiz hisse/interval kombinasyonu")

    sahte_sonuclar = [
        _SahteTestSonucu(
            sembol=d['sembol'], pazar='TR_HISSE', interval=d.get('interval', '1d'),
            basarili=True, win_rate_gercek=d['win_rate_gercek'],
            win_rate_permutasyon=d['win_rate_permutasyon'], sharpe_gercek=d['sharpe_gercek'],
        )
        for d in benzersiz_detaylar
    ]

    return sonuclari_analiz_et(sahte_sonuclar)


def nihai_arastirma_sonucu_yazdir(rapor: dict):
    print()
    print("=" * 70)
    print("🏁 NİHAİ ARAŞTIRMA SONUCU — TÜM TURLAR BİRLEŞTİRİLMİŞ")
    print("=" * 70)

    if 'hata' in rapor:
        print(f"  HATA: {rapor['hata']}")
        return

    n = rapor['basarili_test_sayisi']
    print(f"\nToplam benzersiz TR_HISSE kombinasyonu: {n}")
    print(f"Ortalama Win Rate (gerçek):       %{rapor['win_rate_gercek_ortalama']:.2f}")
    print(f"Ortalama Win Rate (permütasyon):   %{rapor['win_rate_permutasyon_ortalama']:.2f}")
    print(f"Ortalama fark:                     {rapor['ortalama_fark_gercek_eksi_permutasyon']:+.2f} puan")

    t1 = rapor['test_1_gercek_vs_50yuzde']
    t2 = rapor['test_2_gercek_vs_permutasyon']
    print(f"\nTest 1 (gerçek vs %50):         p={t1['p_degeri']:.4f}  {'✅ ANLAMLI' if t1['anlamli_mi_0.05'] else '❌ anlamlı değil'}")
    print(f"Test 2 (gerçek vs permütasyon):  p={t2['p_degeri']:.4f}  {'✅ ANLAMLI' if t2['anlamli_mi_0.05'] else '❌ anlamlı değil'}")

    detaylar = sorted(rapor['kombinasyon_detaylari'], key=lambda x: -x['fark'])
    pozitif_sayisi = sum(1 for d in detaylar if d['fark'] > 0)
    print(f"\nPozitif fark veren hisse sayısı: {pozitif_sayisi} / {n} (%{pozitif_sayisi/n*100:.0f})")

    print()
    print("=" * 70)
    print("🎯 SERİNİN NİHAİ KARARI:")
    if t1['anlamli_mi_0.05'] and t2['anlamli_mi_0.05']:
        print("  ✅ HER İKİ TEST DE ANLAMLI (p<0.05). TR_HISSE'de istatistiksel olarak")
        print("     anlamlı bir sinyal ADAYI bulundu. Bu HÂLÂ kesin bir kazanç garantisi")
        print("     DEĞİLDİR (hisse-bazında çok yüksek varyans devam ediyor, bkz. PETKM/AKSEN")
        print("     gibi aşırı kayıplar) — ama TR_HISSE'ye özel, SIKI risk yönetimiyle")
        print("     (pozisyon başına düşük risk, çeşitlendirme) kontrollü bir pilot")
        print("     uygulamaya değer bir bulgudur.")
    elif t1['anlamli_mi_0.05']:
        print("  ⚠️  KISMEN ANLAMLI: Test 1 anlamlı ama Test 2 (asıl kritik test) DEĞİL.")
        print("     Bu, 6 tur ve ~47 hisselik araştırma sonrasında bile sinyalin")
        print("     istatistiksel olarak KESİNLEŞTİRİLEMEDİĞİ anlamına gelir. Daha fazla")
        print("     örneklem büyütme muhtemelen yardımcı olmayacaktır — sinyal varsa bile")
        print("     çok zayıf ve gürültü içinde kaybolacak kadar küçüktür.")
    else:
        print("  ❌ ANLAMLI DEĞİL: TR_HISSE'de istatistiksel olarak güvenilir bir edge")
        print("     bulunamadı. Bu noktada B seçeneğine (karar destek aracı, kazanç")
        print("     vaadi olmayan konumlandırma) geçmek en dürüst yoldur.")
    print("=" * 70)


if __name__ == "__main__":
    tur6_sonuc = tur6_calistir(baseline_senaryo_sayisi=10)

    bu_dizin = os.path.dirname(__file__)
    nihai_rapor = tum_turlari_birlestir(
        tur6_sonuc,
        onceki_json_yollari=[
            os.path.join(bu_dizin, 'hipotez_1_sonuclari.json'),
            os.path.join(bu_dizin, 'dogrulama_sonuclari.json'),
            os.path.join(bu_dizin, 'tur5_genis_ornekiem_sonuclari.json'),
        ],
    )
    nihai_arastirma_sonucu_yazdir(nihai_rapor)

    cikti_dosyasi = os.path.join(bu_dizin, 'tur6_nihai_sonuclar.json')
    with open(cikti_dosyasi, 'w', encoding='utf-8') as f:
        json.dump({
            'bu_turun_sonucu': tur6_sonuc,
            'nihai_birlesik_rapor': nihai_rapor,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Tam sonuçlar şuraya kaydedildi: {cikti_dosyasi}")
