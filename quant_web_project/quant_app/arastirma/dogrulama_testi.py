"""
=====================================================================
🔬 ARAŞTIRMA TUR 3 — DOĞRULAMA TESTİ (Out-of-Sample Validation)
=====================================================================
Hipotez 1 testinde max_bar=20'de TR_HISSE pazarında çok güçlü bir
kümelenme bulundu (BIMAS +24.2, THYAO +19.4, GARAN +17.7, EREGL +14.5,
SISE +12.9 puan fark, Sharpe 7-8 aralığında).

SORU: Bu gerçek bir TR_HISSE'ye özgü etki mi, yoksa şans eseri (çoklu
karşılaştırma problemi) mi?

DOĞRULAMA YÖNTEMİ — İKİ AYRI DEĞİŞİKLİK, AYRI AYRI TEST EDİLİR:

  TEST A — FARKLI HİSSELER: Hipotez 1'de test edilmemiş 8 yeni TR
    hissesi (farklı sektörlerden) ile max_bar=20 tekrar çalıştırılır.
    Eğer etki gerçekse, yeni hisselerde de benzer bir kümelenme/pozitif
    sapma görülmeli.

  TEST B — FARKLI ZAMAN ARALIĞI: AYNI hisseler (BIMAS, THYAO, GARAN,
    EREGL, SISE), ama 1 YIL ÖNCEKİ bir veri penceresiyle (yani bugünden
    değil, ~1 yıl öncesinden başlayan 1 yıllık pencere) test edilir.
    Eğer etki gerçekten o hisselerin/dönemin yapısal bir özelliğiyse,
    farklı zaman diliminde de benzer güç görülmeli; sadece o ANKİ
    döneme özgüyse (rejim etkisi, tek seferlik haber vs.) kaybolmalı.

Her iki testin sonucu, asıl Hipotez 1 sonucuyla (max_bar=20, TR_HISSE
kombinasyonları) karşılaştırılır.
=====================================================================
"""

from __future__ import annotations

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import quant_web_project.quant_app.quant_ml_coreBinanceli as core
from arastirma.coklu_test import tek_kombinasyon_test_et, sonuclari_analiz_et
from arastirma.hipotez_testleri import hipotez_1_etiketleme_ufku_tara


# Hipotez 1'de TEST EDİLMEMİŞ, farklı sektörlerden 8 yeni TR hissesi.
# (Hipotez 1'deki TR_HISSE kümesi: THYAO, GARAN, ASELS, EREGL, BIMAS, SISE, TUPRS idi)
DOGRULAMA_HISSE_LISTESI_A = [
    ("KCHOL.IS", "TR_HISSE", "1d"),   # Koç Holding (holding)
    ("AKBNK.IS", "TR_HISSE", "1d"),   # Akbank (bankacılık, GARAN'a benzer sektör)
    ("PETKM.IS", "TR_HISSE", "1d"),   # Petkim (petrokimya)
    ("KOZAL.IS", "TR_HISSE", "1d"),   # Koza Altın (madencilik)
    ("PGSUS.IS", "TR_HISSE", "1d"),   # Pegasus (havayolu, THYAO'ya benzer sektör)
    ("VESTL.IS", "TR_HISSE", "1d"),   # Vestel (dayanıklı tüketim)
    ("TOASO.IS", "TR_HISSE", "1d"),   # Tofaş (otomotiv)
    ("ENKAI.IS", "TR_HISSE", "1d"),   # Enka İnşaat (inşaat)
]

# Test B için: Hipotez 1'deki ORİJİNAL 5 güçlü sonucu veren hisseler.
DOGRULAMA_HISSE_LISTESI_B_AYNI_HISSELER = [
    ("BIMAS.IS", "TR_HISSE", "1d"),
    ("THYAO.IS", "TR_HISSE", "1d"),
    ("GARAN.IS", "TR_HISSE", "1d"),
    ("EREGL.IS", "TR_HISSE", "1d"),
    ("SISE.IS", "TR_HISSE", "1d"),
]

# 🛡️ Modül yüklenirken, henüz hiçbir mock uygulanmadan, ORİJİNAL
# get_market_data fonksiyonuna bir referans saklıyoruz. _geçmis_donem_fetcher
# içindeki güvenlik düşüşü (TR_HISSE olmayan pazarlar için) bunu kullanır —
# core.get_market_data DEĞİL, çünkü test sırasında o isim geçici olarak
# başka bir fonksiyona işaret edecek.
_ORIJINAL_GET_MARKET_DATA = core.get_market_data


def _geçmis_donem_fetcher(gun_kaydirma: int = 365):
    """
    quant_ml_core.get_market_data'nın YERİNE geçici olarak konacak bir
    fonksiyon üretir: normal akış 'bugünden 1 yıl öncesi'ni çekerken,
    bu fonksiyon 'bugünden (365+gun_kaydirma) gün önce başlayıp
    gun_kaydirma gün önce biten' bir pencere çeker — yani AYNI uzunlukta
    ama GEÇMİŞE kaymış bir test penceresi.
    """
    import yfinance as yf

    def fetcher(symbol, period, interval, market, prefer_source=None):
        if market != "TR_HISSE":
            # Sadece TR_HISSE için özelleştirildi; başka pazar gelirse
            # ORİJİNAL (modül yüklenirken saklanan) fonksiyona düş — bu
            # doğrulama testinde kullanılmayacak ama güvenlik için bırakıldı.
            return _ORIJINAL_GET_MARKET_DATA(symbol, period, interval, market, prefer_source)

        end_dt = pd.Timestamp.now().normalize() - pd.Timedelta(days=gun_kaydirma)
        start_dt = end_dt - pd.Timedelta(days=365)

        raw = yf.download(symbol, start=start_dt.strftime("%Y-%m-%d"),
                           end=end_dt.strftime("%Y-%m-%d"), interval="1d", progress=False)
        if raw.empty:
            return core.FetchResult(df=pd.DataFrame(), source="yahoo_gecmis", requested_interval=interval,
                                     actual_native_interval=interval, is_resampled=False,
                                     warning=f"Geçmiş dönem verisi boş döndü ({symbol}).")

        raw = core._clean_yahoo_columns(raw)
        return core.FetchResult(df=raw, source="yahoo_gecmis_donem", requested_interval=interval,
                                 actual_native_interval=interval, is_resampled=False)

    return fetcher


def test_a_farkli_hisseler(max_bar: int = 20, baseline_senaryo_sayisi: int = 10) -> dict:
    """TEST A: Hipotez 1'de hiç görülmemiş 8 yeni TR hissesiyle aynı max_bar'ı test eder."""
    print("=" * 70)
    print("🔬 TEST A — FARKLI TR HİSSELERİ (max_bar=20, YENİ 8 hisse)")
    print("=" * 70)

    sonuc = hipotez_1_etiketleme_ufku_tara(
        max_bar_degerleri=[max_bar],
        kombinasyonlar=DOGRULAMA_HISSE_LISTESI_A,
        baseline_senaryo_sayisi=baseline_senaryo_sayisi,
    )
    return sonuc[max_bar]


def test_b_farkli_zaman_araligi(max_bar: int = 20, gun_kaydirma: int = 365) -> dict:
    """
    TEST B: AYNI 5 hisseyi, 1 yıl ÖNCEKİ bir veri penceresiyle test eder.
    Bu fonksiyon core.get_market_data'yı GEÇİCİ olarak değiştirir ve
    test bitince orijinaline geri döndürür (yan etki bırakmaz).
    """
    print("=" * 70)
    print(f"🔬 TEST B — FARKLI ZAMAN ARALIĞI (max_bar={max_bar}, {gun_kaydirma} gün geçmişe kaydırılmış)")
    print("=" * 70)

    orijinal_fetch = core.get_market_data
    core.get_market_data = _geçmis_donem_fetcher(gun_kaydirma=gun_kaydirma)
    try:
        piyasa_sonuclari = []
        for sembol, pazar, interval in DOGRULAMA_HISSE_LISTESI_B_AYNI_HISSELER:
            print(f"  {sembol} ({pazar}, {interval}, geçmiş dönem) test ediliyor...")
            sonuc = tek_kombinasyon_test_et(sembol, pazar, interval, max_bar=max_bar)
            piyasa_sonuclari.append(sonuc)
            if sonuc.basarili:
                print(f"      fark: {sonuc.win_rate_gercek - sonuc.win_rate_permutasyon:+.1f}  "
                      f"sharpe: {sonuc.sharpe_gercek:.2f}")
            else:
                print(f"      BAŞARISIZ: {sonuc.hata_mesaji}")
    finally:
        core.get_market_data = orijinal_fetch  # ⚠️ Yan etkiyi geri al — kritik adım

    return sonuclari_analiz_et(piyasa_sonuclari)


def dogrulama_ozet_yazdir(hipotez1_max20_piyasa: dict, test_a: dict, test_b: dict):
    print()
    print("=" * 70)
    print("📊 DOĞRULAMA TESTİ — NİHAİ KARŞILAŞTIRMA")
    print("=" * 70)

    def fark_al(rapor):
        return rapor.get('ortalama_fark_gercek_eksi_permutasyon', None)

    orijinal_fark = fark_al(hipotez1_max20_piyasa)
    test_a_fark = fark_al(test_a.get('piyasa', test_a))
    test_b_fark = fark_al(test_b)

    print(f"\nHipotez 1 orijinal sonuç (5 hisse, BIMAS/THYAO/GARAN/EREGL/SISE):")
    print(f"  Ortalama fark (gerçek-permütasyon): {orijinal_fark:+.2f} puan" if orijinal_fark is not None else "  (veri yok)")

    print(f"\nTest A — FARKLI 8 hisse, AYNI dönem:")
    print(f"  Ortalama fark: {test_a_fark:+.2f} puan" if test_a_fark is not None else "  (veri yok)")
    if test_a_fark is not None and orijinal_fark:
        if test_a_fark > 2.0:
            print("  -> ✅ Yeni hisselerde de pozitif sapma görüldü; etki TR_HISSE'ye genel olabilir.")
        else:
            print("  -> ❌ Yeni hisselerde benzer bir güç YOK; orijinal sonuç o hisselere özgü olabilir (şans).")

    print(f"\nTest B — AYNI 5 hisse, FARKLI (geçmiş) dönem:")
    print(f"  Ortalama fark: {test_b_fark:+.2f} puan" if test_b_fark is not None else "  (veri yok)")
    if test_b_fark is not None and orijinal_fark:
        if test_b_fark > 2.0:
            print("  -> ✅ Aynı hisseler farklı dönemde de güçlü; etki muhtemelen YAPISAL (gerçek).")
        else:
            print("  -> ❌ Aynı hisseler farklı dönemde güçlü DEĞİL; orijinal sonuç o döneme özgü olabilir (rejim/şans).")

    print()
    print("🎯 GENEL SONUÇ:")
    a_basarili = test_a_fark is not None and test_a_fark > 2.0
    b_basarili = test_b_fark is not None and test_b_fark > 2.0
    if a_basarili and b_basarili:
        print("  Her iki doğrulama testi de POZİTİF — bulgu gerçek bir sinyal adayı olarak güçlendi.")
    elif a_basarili or b_basarili:
        print("  Doğrulama testlerinden SADECE BİRİ pozitif — bulgu kısmen destekleniyor, temkinli olunmalı.")
    else:
        print("  Her iki doğrulama testi de NEGATİF — orijinal sonuç büyük olasılıkla ŞANS ESERİ")
        print("  (çoklu karşılaştırma problemi). TR_HISSE'ye özel bir motor kurmak için YETERSİZ kanıt.")
    print("=" * 70)


if __name__ == "__main__":
    print(f"Doğrulama testi başlıyor — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    test_a_sonuc = test_a_farkli_hisseler(max_bar=20, baseline_senaryo_sayisi=10)
    print()
    test_b_sonuc = test_b_farkli_zaman_araligi(max_bar=20, gun_kaydirma=365)

    # Hipotez 1'in orijinal max_bar=20 piyasa sonucunu, kullanıcının
    # gönderdiği hipotez_1_sonuclari.json'dan okumak gerekir — burada
    # script'in kendi kaydettiği dosyadan okunmaya çalışılır, yoksa atlanır.
    hipotez1_dosya = os.path.join(os.path.dirname(__file__), 'hipotez_1_sonuclari.json')
    hipotez1_max20_piyasa = {}
    if os.path.exists(hipotez1_dosya):
        with open(hipotez1_dosya, encoding='utf-8') as f:
            h1 = json.load(f)
            hipotez1_max20_piyasa = h1.get('20', {}).get('piyasa', {})

    dogrulama_ozet_yazdir(hipotez1_max20_piyasa, test_a_sonuc, test_b_sonuc)

    cikti = {
        'olusturulma_tarihi': datetime.now().isoformat(),
        'hipotez1_orijinal_max20': hipotez1_max20_piyasa,
        'test_a_farkli_hisseler': test_a_sonuc,
        'test_b_farkli_donem': test_b_sonuc,
    }
    cikti_dosyasi = os.path.join(os.path.dirname(__file__), 'dogrulama_sonuclari.json')
    with open(cikti_dosyasi, 'w', encoding='utf-8') as f:
        json.dump(cikti, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Tam sonuçlar şuraya kaydedildi: {cikti_dosyasi}")
