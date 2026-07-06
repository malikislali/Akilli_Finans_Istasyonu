"""
=====================================================================
🔬 ARAŞTIRMA TUR 14 — VOLATİLİTE TAHMİNİ (Risk Ölçümüne Geçiş)
=====================================================================
Tur 1-13 sonucu: "OHLCV + teknik göstergeler + ML ile kalıcı YÖN
tahmini yapılabilir" hipotezi, 7 bağımsız metodolojik yaklaşımla
desteklenmedi. ChatGPT'nin son değerlendirmesi: "Yön tahmini yerine
RİSK tahmini dene — volatilite kümelenmesi (volatility clustering)
finans literatüründe çok daha güçlü ampirik desteğe sahip bir
özelliktir."

BU TUR — KATEGORİK FARK: Artık "yukarı mı aşağı mı" (binary
sınıflandırma) tahmin ETMİYORUZ. Önümüzdeki N barlık GERÇEKLEŞEN
VOLATİLİTEYİ (sürekli bir sayı, regresyon problemi) tahmin ediyoruz.

YÖNTEM:
  1. Hedef değişken: önümüzdeki ILERI_BAR_SAYISI barın günlük
     getirilerinin standart sapması (gerçekleşen volatilite).
  2. BASELINE (kritik karşılaştırma noktası): "Yarının volatilitesi,
     bugünün ATR'sine yakın olacak" — bu, GARCH'ın da temel sezgisi
     ve İSTATİSTİKSEL OLARAK ZATEN GÜÇLÜ bir tahmincidir. Modelimizin
     asıl sınanması gereken soru: ML, bu BASİT BASELINE'DAN daha mı
     iyi, yoksa baseline kendisi zaten yeterli mi?
  3. Model: GradientBoostingRegressor, aynı 19 klasik feature ile.
  4. Permütasyon testi: gerçek volatilite hedefiyle vs RASTGELE
     karıştırılmış volatilite hedefiyle eğitilen modelin R²/MAE
     karşılaştırması.
  5. Purged CV: aynı disiplin (etiket ufku = ILERI_BAR_SAYISI).

BAŞARI KRİTERİ (yön tahmininden FARKLI ve daha sıkı):
  Sadece "permütasyondan iyi" yetmez — model ayrıca BASİT BASELINE'I
  (bugünün ATR'si = yarının tahmini volatilitesi) da geçmeli. Aksi
  halde "ML kullanmaya değer" diyemeyiz, sadece "volatilite zaten
  kendi içinde otokorelasyonlu" deriz (ki bu zaten bilinen bir şey).
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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr

import quant_web_project.quant_app.quant_ml_coreBinanceli as core
from arastirma.purged_cv import PurgedEmbargoCV

FEATURE_KOLONLARI = core.FEATURE_KOLONLARI
ILERI_BAR_SAYISI = 10  # Tahmin edilen volatilite ufku


def gerceklesen_volatilite_hesapla(close: pd.Series, ileri_bar: int = ILERI_BAR_SAYISI) -> pd.Series:
    """Her zaman noktası için, ÖNÜMÜZDEKİ ileri_bar barın günlük
    getirilerinin standart sapmasını (gerçekleşen volatilite) hesaplar."""
    log_getiri = np.log(close / close.shift(1))
    # shift(-ileri_bar) ile ileriye kaydırılmış rolling std — "bugünden
    # itibaren önümüzdeki ileri_bar barın volatilitesi"
    ileri_vol = log_getiri.shift(-1).rolling(window=ileri_bar).std().shift(-(ileri_bar - 1))
    return ileri_vol


def simdiki_volatilite_baseline_hesapla(close: pd.Series, gecmis_bar: int = ILERI_BAR_SAYISI) -> pd.Series:
    """BASELINE: şu ana kadarki (geçmişe bakan) gerçekleşen volatilite
    — 'yarın bugüne benzer olacak' sezgisinin basit ölçümü."""
    log_getiri = np.log(close / close.shift(1))
    return log_getiri.rolling(window=gecmis_bar).std()


@dataclass
class VolatiliteTestSonucu:
    sembol: str
    pazar: str
    interval: str
    basarili: bool
    hata_mesaji: Optional[str] = None
    n_gozlem: int = 0
    r2_gercek: float = 0.0
    r2_permutasyon: float = 0.0
    r2_baseline: float = 0.0  # Basit "geçmiş volatilite = gelecek volatilite" baseline'ının R²'si
    mae_gercek: float = 0.0
    mae_baseline: float = 0.0
    # 🆕 R², sistematik ölçek/bias kaymasına çok hassastır — model "yönü"
    # (trendi) doğru yakalasa bile R² kötü çıkabilir (bkz. araştırma notları).
    # Bu yüzden Spearman korelasyonu da ayrıca raporlanır — bu, MUTLAK
    # değerden bağımsız, sadece SIRALAMA/TREND doğruluğunu ölçer.
    spearman_gercek: float = 0.0
    spearman_baseline: float = 0.0


def volatilite_tek_kombinasyon_test_et(
    sembol: str, pazar: str, interval: str,
    n_splits: int = 4, rastgele_seed: int = 42,
) -> VolatiliteTestSonucu:
    try:
        period = core.suggest_period(pazar, interval)
        if pazar == "TR_HISSE" and interval == "1d":
            period = "3y"
        fetch_result = core.get_market_data(sembol, period, interval, pazar)
        df_raw = fetch_result.df

        if df_raw.empty:
            return VolatiliteTestSonucu(sembol=sembol, pazar=pazar, interval=interval,
                                         basarili=False, hata_mesaji="Veri çekilemedi.")

        df_active, _ = core.calculate_metrics(df_raw)
        if df_active.empty or len(df_active) < 150:
            return VolatiliteTestSonucu(sembol=sembol, pazar=pazar, interval=interval,
                                         basarili=False, hata_mesaji=f"Veri çok az ({len(df_active)} satır).")

        df_active = df_active.copy()
        df_active['gercek_vol'] = gerceklesen_volatilite_hesapla(df_active['Close'], ILERI_BAR_SAYISI)
        df_active['baseline_vol'] = simdiki_volatilite_baseline_hesapla(df_active['Close'], ILERI_BAR_SAYISI)

        df_ml = df_active.dropna(subset=['gercek_vol', 'baseline_vol'] + FEATURE_KOLONLARI).copy()
        if len(df_ml) < 100:
            return VolatiliteTestSonucu(sembol=sembol, pazar=pazar, interval=interval,
                                         basarili=False, hata_mesaji=f"Veri (etiketleme sonrası) çok az ({len(df_ml)}).")

        X_all = df_ml[FEATURE_KOLONLARI].copy()
        y_all = df_ml['gercek_vol'].values
        baseline_all = df_ml['baseline_vol'].values

        cv = PurgedEmbargoCV(n_splits=n_splits, etiket_ufku=ILERI_BAR_SAYISI)

        def _calistir(y_kullanilacak):
            r2_listesi, mae_listesi, spearman_listesi = [], [], []
            for tr_idx, te_idx in cv.split(X_all):
                if len(tr_idx) < 30 or len(te_idx) < 10:
                    continue
                X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
                y_tr, y_te = y_kullanilacak[tr_idx], y_kullanilacak[te_idx]

                scaler = StandardScaler()
                X_tr_sc = scaler.fit_transform(X_tr)
                X_te_sc = scaler.transform(X_te)

                model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
                model.fit(X_tr_sc, y_tr)
                tahmin = model.predict(X_te_sc)

                r2_listesi.append(r2_score(y_te, tahmin))
                mae_listesi.append(mean_absolute_error(y_te, tahmin))
                korelasyon, _ = spearmanr(tahmin, y_te)
                if not np.isnan(korelasyon):
                    spearman_listesi.append(korelasyon)
            return r2_listesi, mae_listesi, spearman_listesi

        r2_gercek_l, mae_gercek_l, spearman_gercek_l = _calistir(y_all)
        if not r2_gercek_l:
            return VolatiliteTestSonucu(sembol=sembol, pazar=pazar, interval=interval,
                                         basarili=False, hata_mesaji="Hiçbir CV fold'u yeterli veri içermedi.")

        # Permütasyon testi
        rng = np.random.default_rng(rastgele_seed)
        y_permute = rng.permutation(y_all)
        r2_permute_l, _, _ = _calistir(y_permute)

        # Baseline performansı: AYNI fold'larda, "geçmiş volatilite = gelecek volatilite" tahmini
        r2_baseline_l, mae_baseline_l, spearman_baseline_l = [], [], []
        for tr_idx, te_idx in cv.split(X_all):
            if len(tr_idx) < 30 or len(te_idx) < 10:
                continue
            y_te = y_all[te_idx]
            baseline_tahmin = baseline_all[te_idx]
            r2_baseline_l.append(r2_score(y_te, baseline_tahmin))
            mae_baseline_l.append(mean_absolute_error(y_te, baseline_tahmin))
            korelasyon_b, _ = spearmanr(baseline_tahmin, y_te)
            if not np.isnan(korelasyon_b):
                spearman_baseline_l.append(korelasyon_b)

        return VolatiliteTestSonucu(
            sembol=sembol, pazar=pazar, interval=interval, basarili=True,
            n_gozlem=len(df_ml),
            r2_gercek=float(np.mean(r2_gercek_l)),
            r2_permutasyon=float(np.mean(r2_permute_l)) if r2_permute_l else 0.0,
            r2_baseline=float(np.mean(r2_baseline_l)) if r2_baseline_l else 0.0,
            mae_gercek=float(np.mean(mae_gercek_l)),
            mae_baseline=float(np.mean(mae_baseline_l)) if mae_baseline_l else 0.0,
            spearman_gercek=float(np.mean(spearman_gercek_l)) if spearman_gercek_l else 0.0,
            spearman_baseline=float(np.mean(spearman_baseline_l)) if spearman_baseline_l else 0.0,
        )

    except Exception as exc:
        return VolatiliteTestSonucu(sembol=sembol, pazar=pazar, interval=interval,
                                     basarili=False, hata_mesaji=f"Beklenmeyen hata: {exc}")


# Tur 7-13 ile AYNI 28 kombinasyonluk evren (karşılaştırılabilirlik için).
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
            print(f"[{i+1}/{len(kombinasyonlar)}] {sembol} ({pazar}, {interval}) volatilite tahmini test ediliyor...")
        sonuc = volatilite_tek_kombinasyon_test_et(sembol, pazar, interval)
        sonuclar.append(sonuc)
        if ilerleme_yazdir:
            if sonuc.basarili:
                print(f"    -> R² gerçek={sonuc.r2_gercek:.3f}  R² permütasyon={sonuc.r2_permutasyon:.3f}  "
                      f"R² baseline={sonuc.r2_baseline:.3f}  MAE gerçek={sonuc.mae_gercek:.5f}  "
                      f"MAE baseline={sonuc.mae_baseline:.5f}")
            else:
                print(f"    -> BAŞARISIZ: {sonuc.hata_mesaji}")
    return sonuclar


def sonuclari_analiz_et(sonuclar: list) -> dict:
    from scipy import stats

    basarili = [s for s in sonuclar if s.basarili]
    if len(basarili) < 3:
        return {'hata': f"Yeterli başarılı test yok ({len(basarili)} adet)."}

    r2_g = np.array([s.r2_gercek for s in basarili])
    r2_p = np.array([s.r2_permutasyon for s in basarili])
    r2_b = np.array([s.r2_baseline for s in basarili])
    mae_g = np.array([s.mae_gercek for s in basarili])
    mae_b = np.array([s.mae_baseline for s in basarili])
    spearman_g = np.array([s.spearman_gercek for s in basarili])
    spearman_b = np.array([s.spearman_baseline for s in basarili])

    fark_vs_permute = r2_g - r2_p
    fark_vs_baseline = r2_g - r2_b
    spearman_fark = spearman_g - spearman_b

    t1, p1 = stats.ttest_1samp(fark_vs_permute, 0.0)
    t2, p2 = stats.ttest_1samp(fark_vs_baseline, 0.0)
    t3, p3 = stats.ttest_1samp(spearman_g, 0.0)

    return {
        'basarili_test_sayisi': len(basarili),
        'toplam_test_sayisi': len(sonuclar),
        'r2_gercek_ortalama': float(np.mean(r2_g)),
        'r2_permutasyon_ortalama': float(np.mean(r2_p)),
        'r2_baseline_ortalama': float(np.mean(r2_b)),
        'mae_gercek_ortalama': float(np.mean(mae_g)),
        'mae_baseline_ortalama': float(np.mean(mae_b)),
        'spearman_gercek_ortalama': float(np.mean(spearman_g)),
        'spearman_baseline_ortalama': float(np.mean(spearman_b)),
        'test_1_model_vs_permutasyon': {
            'aciklama': "Model, RASTGELE etiketle eğitilenden daha mı iyi? (model gerçekten bir şey öğreniyor mu)",
            'p_degeri': float(p1), 'anlamli_mi': bool(p1 < 0.05),
        },
        'test_2_model_vs_baseline': {
            'aciklama': "Model, BASİT 'geçmiş volatilite = gelecek volatilite' baseline'ından daha mı iyi? "
                        "(ML kullanmaya DEĞER mi, yoksa basit kural yeterli mi)",
            'p_degeri': float(p2), 'anlamli_mi': bool(p2 < 0.05),
        },
        'test_3_spearman_vs_sifir': {
            'aciklama': "Model tahmini, gerçek volatiliteyle SIRALAMA/TREND olarak ilişkili mi? "
                        "(R² düşük çıksa bile, model TRENDİ doğru yakalıyor mu — bkz. araştırma notları)",
            'p_degeri': float(p3), 'anlamli_mi': bool(p3 < 0.05),
        },
        'detaylar': [
            {'sembol': s.sembol, 'pazar': s.pazar,
             'r2_gercek': round(s.r2_gercek, 3), 'r2_permutasyon': round(s.r2_permutasyon, 3),
             'r2_baseline': round(s.r2_baseline, 3),
             'spearman_gercek': round(s.spearman_gercek, 3), 'spearman_baseline': round(s.spearman_baseline, 3),
             'fark_vs_permutasyon': round(s.r2_gercek - s.r2_permutasyon, 3),
             'fark_vs_baseline': round(s.r2_gercek - s.r2_baseline, 3)}
            for s in basarili
        ],
    }


def ozet_yazdir(rapor: dict):
    print()
    print("=" * 70)
    print("📊 TUR 14 — VOLATİLİTE TAHMİNİ SONUÇ ÖZETİ")
    print("=" * 70)
    if 'hata' in rapor:
        print("HATA:", rapor['hata'])
        return
    print(f"Başarılı/Toplam: {rapor['basarili_test_sayisi']}/{rapor['toplam_test_sayisi']}")
    print()
    print(f"R² (model, gerçek etiket):        {rapor['r2_gercek_ortalama']:+.4f}")
    print(f"R² (model, permütasyon):           {rapor['r2_permutasyon_ortalama']:+.4f}")
    print(f"R² (basit baseline):               {rapor['r2_baseline_ortalama']:+.4f}")
    print(f"MAE (model, gerçek):                {rapor['mae_gercek_ortalama']:.5f}")
    print(f"MAE (basit baseline):               {rapor['mae_baseline_ortalama']:.5f}")
    print(f"Spearman (model, gerçek):           {rapor['spearman_gercek_ortalama']:+.4f}")
    print(f"Spearman (basit baseline):          {rapor['spearman_baseline_ortalama']:+.4f}")
    print()
    t1 = rapor['test_1_model_vs_permutasyon']
    t2 = rapor['test_2_model_vs_baseline']
    t3 = rapor['test_3_spearman_vs_sifir']
    print(f"Test 1 (model vs permütasyon): p={t1['p_degeri']:.4f}  {'ANLAMLI' if t1['anlamli_mi'] else 'anlamlı değil'}")
    print(f"Test 2 (model vs basit baseline): p={t2['p_degeri']:.4f}  {'ANLAMLI' if t2['anlamli_mi'] else 'anlamlı değil'}")
    print(f"Test 3 (Spearman vs 0, TREND yakalama): p={t3['p_degeri']:.4f}  {'ANLAMLI' if t3['anlamli_mi'] else 'anlamlı değil'}")
    print()
    print("Detaylar:")
    for d in sorted(rapor['detaylar'], key=lambda x: -x['fark_vs_baseline']):
        print(f"  {d['sembol']:<10} R²_gerçek={d['r2_gercek']:+.3f}  R²_baseline={d['r2_baseline']:+.3f}  "
              f"fark_vs_baseline={d['fark_vs_baseline']:+.3f}")
    print("=" * 70)
    print()
    print("🎯 YORUM REHBERİ:")
    print("  - R² genelde DÜŞÜK olabilir bile (volatilite tahmini doğası gereği zordur),")
    print("    önemli olan MUTLAK R² değil, MODEL'in BASELINE'I geçip geçmediğidir.")
    print("  - Eğer R²_baseline ZATEN yüksekse (örn. >0.3), bu volatilite kümelenmesinin")
    print("    GERÇEK ve GÜÇLÜ olduğunu gösterir — ML'siz bile faydalı bir bulgudur.")
    print("  - Eğer Test 2 ANLAMLI ve model R²'si baseline'dan yüksekse, ML gerçekten")
    print("    değer katıyor demektir — bu, yön tahmininde ASLA görmediğimiz bir sonuç olur.")


if __name__ == "__main__":
    print(f"Tur 14 (Volatilite Tahmini) başlıyor — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    sonuclar = tum_kombinasyonlari_test_et()
    rapor = sonuclari_analiz_et(sonuclar)
    ozet_yazdir(rapor)

    cikti_dosyasi = os.path.join(os.path.dirname(__file__), 'tur14_volatilite_tahmini_sonuclar.json')
    with open(cikti_dosyasi, 'w', encoding='utf-8') as f:
        json.dump(rapor, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Sonuçlar şuraya kaydedildi: {cikti_dosyasi}")
