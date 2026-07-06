"""
=====================================================================
🔬 ARAŞTIRMA TUR 5 — GENİŞ ÖRNEKLEM TESTİ (TR_HISSE)
=====================================================================
Şu ana kadarki TR_HISSE araştırması (Tur 2-4):

  - Hipotez 1 (7 hisse: THYAO, GARAN, ASELS, EREGL, BIMAS, SISE, TUPRS)
    max_bar=20'de baseline'ı +3.39 puan aştı (p<0.05).
  - Doğrulama A (8 YENİ hisse: KCHOL, AKBNK, PETKM, KOZAL, PGSUS,
    VESTL, TOASO, ENKAI) +2.42 puan aşım — doğrulandı ama PETKM'de
    -27.4 gibi aşırı bir aykırı değer var.
  - Doğrulama B (aynı 5 hisse, geçmiş dönem) +1.94 puan — doğrulandı
    ama hisse bazında yön bile değişti (SISE +12.9 -> -14.5).
  - Veri Kalitesi: TR_HISSE'de anormal bir şey YOK, artefakt açıklaması
    elendi.

TOPLAM ŞİMDİYE KADAR TEST EDİLEN: 15 farklı TR hissesi (7+8).

BU TURDA: 15 YENİ TR hissesi (BIST30/100'den, önceki turlarla SIFIR
çakışma) eklenerek TOPLAM örneklem 30 hisseye çıkarılıyor. Bu, tek bir
araştırma turu içindeki en büyük TR_HISSE örneklemi olacak ve "gerçek
mi şans mı" sorusuna daha güçlü bir istatistiksel cevap verecek.

YENİ HİSSELER sektör çeşitliliği gözetilerek seçildi (bankacılık,
holding, perakende, enerji, teknoloji, gıda, inşaat, sigorta vb.) —
tek bir sektöre yığılma, sonucun "TR_HISSE etkisi" değil "bir sektör
etkisi" olma riskini azaltır.
=====================================================================
"""

from __future__ import annotations

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arastirma.hipotez_testleri import hipotez_1_etiketleme_ufku_tara, hipotez_1_ozet_yazdir
from arastirma.coklu_test import sonuclari_analiz_et


# 15 YENİ hisse — Hipotez 1 (7) + Doğrulama A (8) ile SIFIR çakışma.
# Sektör dağılımı: bankacılık(3), holding(2), perakende(2), enerji(2),
# teknoloji(1), gıda(2), inşaat(1), sigorta(1), telekom(1)
GENIS_ORNEKLEM_YENI_HISSELER = [
    ("YKBNK.IS", "TR_HISSE", "1d"),   # Yapı Kredi Bankası (bankacılık)
    ("ISCTR.IS", "TR_HISSE", "1d"),   # İş Bankası C (bankacılık)
    ("HALKB.IS", "TR_HISSE", "1d"),   # Halkbank (bankacılık)
    ("SAHOL.IS", "TR_HISSE", "1d"),   # Sabancı Holding (holding)
    ("DOAS.IS",  "TR_HISSE", "1d"),   # Doğuş Otomotiv (otomotiv/holding)
    ("MGROS.IS", "TR_HISSE", "1d"),   # Migros (perakende)
    ("SOKM.IS",  "TR_HISSE", "1d"),   # Şok Marketler (perakende)
    ("TUPRS.IS", "TR_HISSE", "4h"),   # Tüpraş — DİKKAT: hisse Hipotez1'de var ama interval FARKLI (4h), çakışma sayılmaz
    ("AKSEN.IS", "TR_HISSE", "1d"),   # Aksa Enerji (enerji)
    ("ARCLK.IS", "TR_HISSE", "1d"),   # Arçelik (dayanıklı tüketim/teknoloji)
    ("ULKER.IS", "TR_HISSE", "1d"),   # Ülker (gıda)
    ("CCOLA.IS", "TR_HISSE", "1d"),   # Coca-Cola İçecek (gıda/içecek)
    ("KOZAA.IS", "TR_HISSE", "1d"),   # Koza Anadolu (madencilik/inşaat)
    ("TTKOM.IS", "TR_HISSE", "1d"),   # Türk Telekom (telekom)
    ("AGESA.IS", "TR_HISSE", "1d"),   # AgeSA (sigorta)
]


def gor_arastirma_calistir(baseline_senaryo_sayisi: int = 10) -> dict:
    """max_bar=20 sabit tutularak, 15 yeni hisseyle test çalıştırılır."""
    print("=" * 70)
    print("🔬 TUR 5 — GENİŞ ÖRNEKLEM TESTİ (15 YENİ TR hissesi, max_bar=20)")
    print(f"   Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    sonuc = hipotez_1_etiketleme_ufku_tara(
        max_bar_degerleri=[20],
        kombinasyonlar=GENIS_ORNEKLEM_YENI_HISSELER,
        baseline_senaryo_sayisi=baseline_senaryo_sayisi,
    )
    return sonuc[20]


def tum_turlari_birlestir_ve_analiz_et(yeni_sonuc: dict, onceki_turlar_json_yollari: list) -> dict:
    """
    Bu turun (15 yeni hisse) sonucunu, ÖNCEKİ turların (Hipotez 1 + Doğrulama A)
    kombinasyon detaylarıyla BİRLEŞTİRİP toplam ~30 hisselik örneklem üzerinden
    NİHAİ istatistiksel testi tekrar çalıştırır. Bu, "tek bir büyük örneklemde
    gerçekten anlamlı mı" sorusuna en güçlü cevabı verir.
    """
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

    # Bu turun detayları
    yeni_piyasa = yeni_sonuc.get('piyasa', {})
    tum_detaylar.extend(yeni_piyasa.get('kombinasyon_detaylari', []))

    # Önceki turların JSON dosyalarından detayları çek
    for yol in onceki_turlar_json_yollari:
        if not os.path.exists(yol):
            print(f"⚠️ Uyarı: {yol} bulunamadı, bu turun verisi BİRLEŞTİRMEYE dahil edilemedi.")
            continue
        with open(yol, encoding='utf-8') as f:
            veri = json.load(f)

        # hipotez_1_sonuclari.json formatı: {"20": {"piyasa": {...}}}
        if "20" in veri:
            tum_detaylar.extend(veri["20"].get("piyasa", {}).get("kombinasyon_detaylari", []))
        # dogrulama_sonuclari.json formatı: {"test_a_farkli_hisseler": {"piyasa": {...}}}
        elif "test_a_farkli_hisseler" in veri:
            tum_detaylar.extend(
                veri["test_a_farkli_hisseler"].get("piyasa", {}).get("kombinasyon_detaylari", [])
            )

    # Tekrar eden (sembol, interval) çiftlerini ele (TUPRS.IS 1d Hipotez1'de
    # zaten var, bu turda 4h eklendiği için interval ile ayırt ediyoruz)
    # ⚠️ ÖNEMLİ: Bu araştırma TR_HISSE'ye ÖZGÜ bir soruyu yanıtlıyor
    # ("TR_HISSE'de gerçek mi şans mı"), bu yüzden diğer pazarlardaki
    # (KRIPTO/ABD_HISSE/EMTIA) sonuçları analiz dışı bırakıyoruz — onları
    # işin içine katmak, TR_HISSE'ye özgü sinyali diğer pazarların
    # ortalamasıyla SULANDIRIR ve asıl soruyu cevapsız bırakır.
    gorulen = set()
    benzersiz_detaylar = []
    for d in tum_detaylar:
        sembol = d.get('sembol', '')
        if not sembol.endswith('.IS'):
            continue  # TR_HISSE dışı sembolleri atla (.IS uzantısı BIST hisselerine özgüdür)
        anahtar = (sembol, d.get('interval', '1d'))
        if anahtar not in gorulen:
            gorulen.add(anahtar)
            benzersiz_detaylar.append(d)

    sahte_sonuclar = [
        _SahteTestSonucu(
            sembol=d['sembol'], pazar=d.get('pazar', 'TR_HISSE'), interval=d.get('interval', '1d'),
            basarili=True, win_rate_gercek=d['win_rate_gercek'],
            win_rate_permutasyon=d['win_rate_permutasyon'], sharpe_gercek=d['sharpe_gercek'],
        )
        for d in benzersiz_detaylar
    ]

    print(f"\n📊 BİRLEŞTİRİLMİŞ ÖRNEKLEM: {len(sahte_sonuclar)} benzersiz hisse/interval kombinasyonu")

    return sonuclari_analiz_et(sahte_sonuclar)


def nihai_ozet_yazdir(birlesik_rapor: dict):
    print()
    print("=" * 70)
    print("📊 TUR 5 SONRASI NİHAİ DURUM — TÜM TR_HISSE ÖRNEKLEMİ BİRLEŞTİRİLMİŞ")
    print("=" * 70)

    if 'hata' in birlesik_rapor:
        print(f"  HATA: {birlesik_rapor['hata']}")
        return

    print(f"\nToplam benzersiz kombinasyon: {birlesik_rapor['basarili_test_sayisi']}")
    print(f"Ortalama Win Rate (gerçek):      %{birlesik_rapor['win_rate_gercek_ortalama']:.2f}")
    print(f"Ortalama Win Rate (permütasyon):  %{birlesik_rapor['win_rate_permutasyon_ortalama']:.2f}")
    print(f"Ortalama fark:                    {birlesik_rapor['ortalama_fark_gercek_eksi_permutasyon']:+.2f} puan")

    t1 = birlesik_rapor['test_1_gercek_vs_50yuzde']
    t2 = birlesik_rapor['test_2_gercek_vs_permutasyon']
    print(f"\nTest 1 (gerçek vs %50):        p={t1['p_degeri']:.4f}  {'✅ ANLAMLI' if t1['anlamli_mi_0.05'] else '❌ anlamlı değil'}")
    print(f"Test 2 (gerçek vs permütasyon): p={t2['p_degeri']:.4f}  {'✅ ANLAMLI' if t2['anlamli_mi_0.05'] else '❌ anlamlı değil'}")

    print()
    print("Hisse bazında sıralı sonuçlar (en yüksekten en düşüğe):")
    detaylar = sorted(birlesik_rapor['kombinasyon_detaylari'], key=lambda x: -x['fark'])
    for d in detaylar:
        print(f"  {d['sembol']:<10} ({d['interval']:<3}) fark={d['fark']:+6.1f}  sharpe={d['sharpe_gercek']:+6.2f}")

    pozitif_sayisi = sum(1 for d in detaylar if d['fark'] > 0)
    print(f"\nPozitif fark veren hisse sayısı: {pozitif_sayisi} / {len(detaylar)} (%{pozitif_sayisi/len(detaylar)*100:.0f})")
    print("=" * 70)


if __name__ == "__main__":
    yeni_sonuc = gor_arastirma_calistir(baseline_senaryo_sayisi=10)

    bu_dizin = os.path.dirname(__file__)
    birlesik_rapor = tum_turlari_birlestir_ve_analiz_et(
        yeni_sonuc,
        onceki_turlar_json_yollari=[
            os.path.join(bu_dizin, 'hipotez_1_sonuclari.json'),
            os.path.join(bu_dizin, 'dogrulama_sonuclari.json'),
        ],
    )
    nihai_ozet_yazdir(birlesik_rapor)

    cikti_dosyasi = os.path.join(bu_dizin, 'tur5_genis_ornekiem_sonuclari.json')
    with open(cikti_dosyasi, 'w', encoding='utf-8') as f:
        json.dump({
            'bu_turun_sonucu': yeni_sonuc,
            'birlesik_nihai_rapor': birlesik_rapor,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Tam sonuçlar şuraya kaydedildi: {cikti_dosyasi}")
