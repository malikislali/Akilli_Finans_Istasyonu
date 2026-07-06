"""
=====================================================================
🔬 ARAŞTIRMA TUR 15 (SON DOĞRULAMA) — BASİT VOLATİLİTE BASELINE TESTİ
=====================================================================
Tur 14 sonucu: ML modeli (GradientBoostingRegressor + 19 klasik
gösterge), basit "bugünün volatilitesi = yarının volatilitesi"
kuralından İSTATİSTİKSEL OLARAK ANLAMLI ŞEKİLDE DAHA KÖTÜ performans
gösterdi (Spearman: model +0.079, baseline +0.133). Bu, "ML'e gerek
yok, basit kural yeterli" ihtimalini gündeme getirdi.

BU TUR: O "basit kural"ın KENDİSİNİN gerçekten anlamlı olup olmadığını
(rastgele yürüyüş baseline'ına karşı, ML hiç karışmadan) doğruluyoruz.
Eğer basit kural GERÇEKTEN anlamlıysa, dashboard'a ML olmadan, şeffaf
bir "Volatilite Rejimi" göstergesi eklemek HAKLI bir ürün kararı olur.

YÖNTEM:
  1. Her kombinasyon için, sadece İKİ seri hesaplanır: BASELINE_VOL
     (geçmişe bakan, şu anki gerçekleşen volatilite) ve GERCEK_VOL
     (ileriye bakan, gelecekte gerçekleşen volatilite) — Tur 14 ile
     AYNI tanımlar.
  2. GERÇEK Spearman korelasyonu hesaplanır: korelasyon(BASELINE_VOL,
     GERCEK_VOL).
  3. PERMÜTASYON: BASELINE_VOL değerleri rastgele karıştırılır (hangi
     günün baseline'ı hangi güne ait olduğu bozulur), sonra korelasyon
     TEKRAR hesaplanır. Bu, "volatilite otokorelasyonu olmasaydı bu
     korelasyon ne olurdu" sorusuna cevap verir.
  4. Bu işlem ÇOK SAYIDA (200) kez tekrarlanır — gerçek korelasyonun,
     bu permütasyon dağılımının NERESİNDE durduğuna bakılır (ampirik
     p-değeri).

NOT: Bu turda Purged CV YOKTUR çünkü model eğitimi yok — sadece iki
zaman serisi arasındaki ham korelasyon ölçülüyor.
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
from scipy.stats import spearmanr

import quant_web_project.quant_app.quant_ml_coreBinanceli as core
from arastirma.tur14_volatilite_tahmini import (
    gerceklesen_volatilite_hesapla, simdiki_volatilite_baseline_hesapla, ILERI_BAR_SAYISI
)

N_PERMUTASYON = 200


@dataclass
class BaselineTestSonucu:
    sembol: str
    pazar: str
    basarili: bool
    hata_mesaji: Optional[str] = None
    n_gozlem: int = 0
    spearman_gercek: float = 0.0
    spearman_permutasyon_ortalama: float = 0.0
    spearman_permutasyon_std: float = 0.0
    ampirik_p_degeri: float = 1.0


def baseline_permutasyon_test_et(
    sembol: str, pazar: str, interval: str, rastgele_seed: int = 42,
) -> BaselineTestSonucu:
    try:
        period = core.suggest_period(pazar, interval)
        if pazar == "TR_HISSE" and interval == "1d":
            period = "3y"
        fetch_result = core.get_market_data(sembol, period, interval, pazar)
        df_raw = fetch_result.df

        if df_raw.empty:
            return BaselineTestSonucu(sembol=sembol, pazar=pazar, basarili=False,
                                       hata_mesaji="Veri çekilemedi.")

        df_active, _ = core.calculate_metrics(df_raw)
        if df_active.empty or len(df_active) < 100:
            return BaselineTestSonucu(sembol=sembol, pazar=pazar, basarili=False,
                                       hata_mesaji=f"Veri çok az ({len(df_active)} satır).")

        gercek_vol = gerceklesen_volatilite_hesapla(df_active['Close'], ILERI_BAR_SAYISI)
        baseline_vol = simdiki_volatilite_baseline_hesapla(df_active['Close'], ILERI_BAR_SAYISI)

        ortak = pd.DataFrame({'gercek': gercek_vol, 'baseline': baseline_vol}).dropna()
        if len(ortak) < 80:
            return BaselineTestSonucu(sembol=sembol, pazar=pazar, basarili=False,
                                       hata_mesaji=f"Ortak gözlem sayısı yetersiz ({len(ortak)}).")

        gercek_korelasyon, _ = spearmanr(ortak['baseline'], ortak['gercek'])
        if np.isnan(gercek_korelasyon):
            return BaselineTestSonucu(sembol=sembol, pazar=pazar, basarili=False,
                                       hata_mesaji="Korelasyon hesaplanamadı (sabit seri olabilir).")

        rng = np.random.default_rng(rastgele_seed)
        permutasyon_korelasyonlari = []
        for _ in range(N_PERMUTASYON):
            karisik_baseline = rng.permutation(ortak['baseline'].values)
            korelasyon_p, _ = spearmanr(karisik_baseline, ortak['gercek'].values)
            if not np.isnan(korelasyon_p):
                permutasyon_korelasyonlari.append(korelasyon_p)

        permutasyon_korelasyonlari = np.array(permutasyon_korelasyonlari)
        ampirik_p = float(np.mean(permutasyon_korelasyonlari >= gercek_korelasyon))

        return BaselineTestSonucu(
            sembol=sembol, pazar=pazar, basarili=True, n_gozlem=len(ortak),
            spearman_gercek=float(gercek_korelasyon),
            spearman_permutasyon_ortalama=float(np.mean(permutasyon_korelasyonlari)),
            spearman_permutasyon_std=float(np.std(permutasyon_korelasyonlari)),
            ampirik_p_degeri=ampirik_p,
        )

    except Exception as exc:
        return BaselineTestSonucu(sembol=sembol, pazar=pazar, basarili=False,
                                   hata_mesaji=f"Beklenmeyen hata: {exc}")


TEST_KOMBINASYONLARI = [
    ("BTC-USD", "KRIPTO", "1d"), ("ETH-USD", "KRIPTO", "1d"), ("SOL-USD", "KRIPTO", "1d"),
    ("AVAX-USD", "KRIPTO", "1d"), ("XRP-USD", "KRIPTO", "1d"), ("DOGE-USD", "KRIPTO", "1d"),
    ("LINK-USD", "KRIPTO", "1d"), ("FIL-USD", "KRIPTO", "1d"),
    ("THYAO.IS", "TR_HISSE", "1d"), ("GARAN.IS", "TR_HISSE", "1d"), ("BIMAS.IS", "TR_HISSE", "1d"),
    ("ASELS.IS", "TR_HISSE", "1d"), ("EREGL.IS", "TR_HISSE", "1d"), ("SISE.IS", "TR_HISSE", "1d"),
    ("TUPRS.IS", "TR_HISSE", "1d"), ("KCHOL.IS", "TR_HISSE", "1d"), ("AKBNK.IS", "TR_HISSE", "1d"),
    ("AAPL", "ABD_HISSE", "1d"), ("MSFT", "ABD_HISSE", "1d"), ("TSLA", "ABD_HISSE", "1d"),
    ("AMZN", "ABD_HISSE", "1d"), ("META", "ABD_HISSE", "1d"), ("GOOGL", "ABD_HISSE", "1d"),
    ("NVDA", "ABD_HISSE", "1d"),
    ("GC=F", "EMTIA", "1d"), ("CL=F", "EMTIA", "1d"), ("SI=F", "EMTIA", "1d"), ("NG=F", "EMTIA", "1d"),
]


def tum_kombinasyonlari_test_et(kombinasyonlar=None, ilerleme_yazdir=True) -> list:
    if kombinasyonlar is None:
        kombinasyonlar = TEST_KOMBINASYONLARI
    sonuclar = []
    for i, (sembol, pazar, interval) in enumerate(kombinasyonlar):
        if ilerleme_yazdir:
            print(f"[{i+1}/{len(kombinasyonlar)}] {sembol} ({pazar}) baseline permütasyon testi...")
        sonuc = baseline_permutasyon_test_et(sembol, pazar, interval)
        sonuclar.append(sonuc)
        if ilerleme_yazdir:
            if sonuc.basarili:
                print(f"    -> Spearman gerçek={sonuc.spearman_gercek:+.3f}  "
                      f"permütasyon ort.={sonuc.spearman_permutasyon_ortalama:+.3f}  "
                      f"ampirik p={sonuc.ampirik_p_degeri:.4f}")
            else:
                print(f"    -> BAŞARISIZ: {sonuc.hata_mesaji}")
    return sonuclar


def sonuclari_analiz_et(sonuclar: list) -> dict:
    from scipy import stats

    basarili = [s for s in sonuclar if s.basarili]
    if len(basarili) < 3:
        return {'hata': f"Yeterli başarılı test yok ({len(basarili)} adet)."}

    spearman_g = np.array([s.spearman_gercek for s in basarili])
    ampirik_p = np.array([s.ampirik_p_degeri for s in basarili])

    t1, p1 = stats.ttest_1samp(spearman_g, 0.0)
    anlamli_varlik_sayisi = int(np.sum(ampirik_p < 0.05))

    return {
        'basarili_test_sayisi': len(basarili),
        'toplam_test_sayisi': len(sonuclar),
        'spearman_gercek_ortalama': float(np.mean(spearman_g)),
        'spearman_gercek_std': float(np.std(spearman_g)),
        'anlamli_varlik_sayisi': anlamli_varlik_sayisi,
        'anlamli_varlik_orani': anlamli_varlik_sayisi / len(basarili),
        'test_1_ortalama_spearman_vs_sifir': {
            'aciklama': "Tüm varlıklardaki ortalama Spearman korelasyonu sıfırdan anlamlı şekilde farklı mı?",
            'p_degeri': float(p1), 'anlamli_mi': bool(p1 < 0.05),
        },
        'detaylar': [
            {'sembol': s.sembol, 'pazar': s.pazar, 'n_gozlem': s.n_gozlem,
             'spearman_gercek': round(s.spearman_gercek, 3),
             'spearman_permutasyon_ort': round(s.spearman_permutasyon_ortalama, 3),
             'ampirik_p': round(s.ampirik_p_degeri, 4),
             'varlik_bazinda_anlamli_mi': s.ampirik_p_degeri < 0.05}
            for s in basarili
        ],
    }


def ozet_yazdir(rapor: dict):
    print()
    print("=" * 70)
    print("📊 TUR 15 (SON DOĞRULAMA) — BASİT VOLATİLİTE BASELINE SONUÇ ÖZETİ")
    print("=" * 70)
    if 'hata' in rapor:
        print("HATA:", rapor['hata'])
        return
    print(f"Başarılı/Toplam: {rapor['basarili_test_sayisi']}/{rapor['toplam_test_sayisi']}")
    print(f"Ortalama Spearman korelasyonu: {rapor['spearman_gercek_ortalama']:+.4f} "
          f"(std={rapor['spearman_gercek_std']:.4f})")
    print(f"Varlık bazında anlamlı (p<0.05) sayısı: {rapor['anlamli_varlik_sayisi']}/{rapor['basarili_test_sayisi']} "
          f"(%{rapor['anlamli_varlik_orani']*100:.0f})")
    print()
    t1 = rapor['test_1_ortalama_spearman_vs_sifir']
    print(f"Genel test (ortalama Spearman vs 0): p={t1['p_degeri']:.6f}  "
          f"{'ANLAMLI' if t1['anlamli_mi'] else 'anlamlı değil'}")
    print()
    print("Detaylar (Spearman'a göre sıralı):")
    for d in sorted(rapor['detaylar'], key=lambda x: -x['spearman_gercek']):
        isaret = "✅" if d['varlik_bazinda_anlamli_mi'] else "  "
        print(f"  {isaret} {d['sembol']:<10} Spearman={d['spearman_gercek']:+.3f}  "
              f"ampirik_p={d['ampirik_p']:.4f}")
    print("=" * 70)
    print()
    if t1['anlamli_mi'] and rapor['anlamli_varlik_orani'] > 0.5:
        print("🎯 SONUÇ: Basit volatilite baseline'ı GERÇEKTEN anlamlı. Volatilite kümelenmesi")
        print("   bu veri setinde GERÇEK bir özellik — ML'siz, şeffaf bir 'Volatilite Rejimi'")
        print("   göstergesi olarak ürüne eklenmesi istatistiksel olarak HAKLIDIR.")
    else:
        print("🎯 SONUÇ: Basit volatilite baseline'ı bu testte yeterince güçlü/tutarlı çıkmadı.")
        print("   Volatilite göstergesi eklenebilir ama 'istatistiksel olarak kanıtlanmış' yerine")
        print("   'standart bir teknik gösterge' olarak (RSI/MACD gibi) sunulmalıdır.")


if __name__ == "__main__":
    print(f"Tur 15 (Basit Baseline Doğrulaması) başlıyor — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Permütasyon sayısı: {N_PERMUTASYON}")
    print()

    sonuclar = tum_kombinasyonlari_test_et()
    rapor = sonuclari_analiz_et(sonuclar)
    ozet_yazdir(rapor)

    cikti_dosyasi = os.path.join(os.path.dirname(__file__), 'tur15_baseline_dogrulama_sonuclar.json')
    with open(cikti_dosyasi, 'w', encoding='utf-8') as f:
        json.dump(rapor, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Sonuçlar şuraya kaydedildi: {cikti_dosyasi}")
