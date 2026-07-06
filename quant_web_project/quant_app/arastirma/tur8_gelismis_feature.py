"""
=====================================================================
🔬 ARAŞTIRMA TUR 8 — GELİŞMİŞ FEATURE MÜHENDİSLİĞİ
=====================================================================
Tur 1-7 sonucu: 19 klasik gösterge (RSI, MACD, Bollinger, ATR vb.) ile,
ne ağaç-tabanlı modellerde (GBM/RF/XGB) ne de LSTM'de istatistiksel
olarak güvenilir bir edge bulunamadı — ikisinde de "büyüyen örneklemde
zayıflayan" aynı desen görüldü.

BU TUR: Model mimarisini DEĞİL, modele verilen BİLGİYİ (feature seti)
değiştiriyoruz. ChatGPT'nin kod değerlendirmesinde önerdiği, OHLCV
verisinden HESAPLANABİLİR olan feature'lar test edildi:

  1. PARKİNSON VOLATİLİTESİ: Sadece Close'a değil, High/Low'a dayanan
     daha verimli bir volatilite tahmincisi (Close-to-Close'dan ~5x
     daha az gürültülü, teorik olarak).

  2. GARMAN-KLASS VOLATİLİTESİ: Open/High/Low/Close'un HEPSİNİ kullanan,
     Parkinson'dan daha da verimli bir tahminci.

  3. SHANNON ENTROPY: Fiyat getirilerinin dağılımının "ne kadar
     öngörülemez/rastgele" olduğunu ölçer — yüksek entropi = düşük
     öngörülebilirlik.

NOT — HURST EXPONENT ÇIKARILDI: İlk denemede eklenmişti ama doğrulama
testlerinde mean-reversion ayrımında güvenilmez çıktığı görüldü (kod
arastirma/tur8_gelismis_feature.py içinde dead-code olarak duruyor,
referans için). Yanlış/güvenilmez bir feature'ı sonuçlara dahil etmek,
"edge bulunamadı" ya da "edge bulundu" sonucunu YANLIŞ bir sebeple
açıklamamıza yol açabilirdi.

NOT: ChatGPT'nin önerdiği Order Flow/CVD/Delta Volume gibi feature'lar
BİLEREK eklenmedi — bunlar tick-level/order-book verisi gerektirir,
bizim OHLCV verimizden hesaplanamaz (gerçekte var olmayan bir veriyi
"varmış gibi" hesaplamak yanıltıcı olurdu).

METODOLOJİK DİSİPLİN: Model mimarisi ESKİ (ağaç-tabanlı GBM, Tur 1-6
ile AYNI) tutuluyor — sadece feature seti değişiyor. Bu, "hangi
değişikliğin etkili olduğunu" karıştırmamak için bilinçli bir karar
(LSTM denemesi ayrı bir tur olarak zaten yapıldı ve kapatıldı).
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
from sklearn.ensemble import GradientBoostingClassifier

import quant_web_project.quant_app.quant_ml_coreBinanceli as core
from arastirma.triple_barrier import triple_barrier_etiketle
from arastirma.purged_cv import PurgedEmbargoCV


# =====================================================================
# 🧮 YENİ FEATURE HESAPLAMA FONKSİYONLARI
# =====================================================================
def hurst_exponent_hesapla(seri: pd.Series, pencere: int = 60) -> pd.Series:
    """
    ⚠️ BU FONKSİYON ŞU AN gelismis_feature_ekle() İÇİNDE KULLANILMIYOR.
    Doğrulama testlerinde momentum/rastgele yürüyüş ayrımında doğru
    çalıştığı (H>0.5 / H≈0.5) ama mean-reversion ayrımında zayıf kaldığı
    (H<0.5 beklenirken H>0.5 çıkması) görüldü — basitleştirilmiş R/S
    yöntemi küçük rolling pencerelerde önyargılı kalabiliyor (literatürde
    bilinen bir sınırlama). Güvenilmez bir feature'ı modele vermemek için
    bilerek devre dışı bırakıldı. Referans/ileride geliştirme için kod
    burada bırakıldı.

    Rolling Hurst Exponent — basitleştirilmiş R/S (rescaled range) yöntemi.
    Her pencere için, farklı alt-aralık boyutlarında R/S istatistiği
    hesaplanır ve log-log eğimi (Hurst) regresyon ile bulunur.

    H ≈ 0.5: rastgele yürüyüş (öngörülemez)
    H > 0.5: trend/momentum (geçmiş yön gelecekte de sürer)
    H < 0.5: mean-reversion (geçmiş yönün tersi beklenir)
    """
    log_getiri = np.log(seri / seri.shift(1)).dropna()

    def _tek_pencere_hurst(pencere_verisi: np.ndarray) -> float:
        if len(pencere_verisi) < 20 or np.std(pencere_verisi) == 0:
            return np.nan
        # 🐛 DÜZELTME: lag boyutları artık pencere uzunluğuna ORANTILI
        # seçiliyor (sabit [5,10,20,40] değil). Sabit değerler, küçük bir
        # rolling pencerede (örn. 60) en büyük lag (40) için sadece 1-2
        # alt-parça bırakıyordu — bu istatistiksel olarak güvenilmez ve
        # Hurst tahminini SİSTEMATİK OLARAK TERS YÖNE çeviriyordu (test:
        # trend seri 0.288, mean-reverting seri 0.77 çıkıyordu — beklenenin
        # TAM TERSİ). En az 4 alt-parça garantisi olan lag'ler seçiliyor.
        maks_lag = len(pencere_verisi) // 4
        lag_boyutlari = [l for l in [5, 8, 12, 18, 26] if l <= maks_lag]
        if len(lag_boyutlari) < 2:
            return np.nan

        rs_degerleri = []
        for lag in lag_boyutlari:
            n_parca = len(pencere_verisi) // lag
            if n_parca < 1:
                continue
            rs_listesi = []
            for i in range(n_parca):
                parca = pencere_verisi[i * lag:(i + 1) * lag]
                ortalama = np.mean(parca)
                sapmalar = np.cumsum(parca - ortalama)
                R = np.max(sapmalar) - np.min(sapmalar)
                S = np.std(parca)
                if S > 0:
                    rs_listesi.append(R / S)
            if rs_listesi:
                rs_degerleri.append((lag, np.mean(rs_listesi)))

        if len(rs_degerleri) < 2:
            return np.nan

        log_lag = np.log([x[0] for x in rs_degerleri])
        log_rs = np.log([x[1] for x in rs_degerleri if x[1] > 0])
        if len(log_rs) < 2:
            return np.nan
        log_lag = log_lag[:len(log_rs)]

        egim, _ = np.polyfit(log_lag, log_rs, 1)
        return float(np.clip(egim, 0, 1))  # Hurst teorik olarak [0,1] aralığında

    sonuc = log_getiri.rolling(window=pencere).apply(
        lambda x: _tek_pencere_hurst(x.values), raw=False
    )
    return sonuc.reindex(seri.index)


def parkinson_volatilite_hesapla(high: pd.Series, low: pd.Series, pencere: int = 20) -> pd.Series:
    """
    Parkinson (1980) volatilite tahmincisi — sadece High/Low kullanır,
    Close-to-Close'dan teorik olarak ~5x daha verimlidir (aynı pencere
    boyutuyla daha az gürültülü bir tahmin verir).
    """
    log_hl_oran = np.log(high / low) ** 2
    katsayi = 1.0 / (4.0 * np.log(2.0))
    gunluk_varyans = katsayi * log_hl_oran
    return np.sqrt(gunluk_varyans.rolling(window=pencere).mean())


def garman_klass_volatilite_hesapla(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, pencere: int = 20
) -> pd.Series:
    """
    Garman-Klass (1980) volatilite tahmincisi — Open/High/Low/Close'un
    HEPSİNİ kullanır, Parkinson'dan daha da verimlidir.
    """
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_) ** 2
    gunluk_varyans = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    gunluk_varyans = gunluk_varyans.clip(lower=0)  # negatif varyans matematiksel artefakt, sıfırla
    return np.sqrt(gunluk_varyans.rolling(window=pencere).mean())


def shannon_entropy_hesapla(seri: pd.Series, pencere: int = 20, kutu_sayisi: int = 10) -> pd.Series:
    """
    Rolling Shannon Entropy — getirilerin dağılımının ne kadar
    "öngörülemez/dağınık" olduğunu ölçer. Pencere içindeki getiriler
    kutu_sayisi adet kutuya (bin) ayrılır, dağılımın entropisi hesaplanır.
    Yüksek entropi = daha rastgele/öngörülemez dağılım.
    """
    getiri = seri.pct_change()

    def _tek_pencere_entropy(pencere_verisi: np.ndarray) -> float:
        if len(pencere_verisi) < 10 or np.all(pencere_verisi == pencere_verisi[0]):
            return np.nan
        hist, _ = np.histogram(pencere_verisi, bins=kutu_sayisi)
        olasiliklar = hist / hist.sum()
        olasiliklar = olasiliklar[olasiliklar > 0]  # log(0) önle
        return float(-np.sum(olasiliklar * np.log2(olasiliklar)))

    sonuc = getiri.rolling(window=pencere).apply(
        lambda x: _tek_pencere_entropy(x.values), raw=False
    )
    return sonuc


# Tur 1-6 ile AYNI 19 feature + 3 YENİ feature = 22 feature toplam.
# NOT: Hurst Exponent BİLEREK ÇIKARILDI — test sırasında momentum/rastgele
# yürüyüş ayrımında doğru çalıştığı görüldü, ama mean-reversion ayrımında
# zayıf kaldı (basitleştirilmiş R/S yöntemi küçük pencerelerde önyargılı
# kalıyor, literatürde bilinen bir sınırlama). Yanlış/güvenilmez bir
# feature'ı modele vermek, sonucu yorumlamayı zorlaştırır — bu yüzden
# güvenilirliği doğrulanan 3 feature ile devam ediliyor.
YENI_FEATURE_KOLONLARI = core.FEATURE_KOLONLARI + [
    'Parkinson_Vol', 'Garman_Klass_Vol', 'Shannon_Entropy',
]


def gelismis_feature_ekle(df_active: pd.DataFrame) -> pd.DataFrame:
    """calculate_metrics() çıktısına 3 yeni feature kolonu ekler."""
    df = df_active.copy()
    df['Parkinson_Vol'] = parkinson_volatilite_hesapla(df['High'], df['Low'], pencere=20)
    df['Garman_Klass_Vol'] = garman_klass_volatilite_hesapla(df['Open'], df['High'], df['Low'], df['Close'], pencere=20)
    df['Shannon_Entropy'] = shannon_entropy_hesapla(df['Close'], pencere=20)
    return df


# =====================================================================
# 🧪 TEK KOMBİNASYON TEST FONKSİYONU (eski ağaç-modeli mimarisiyle)
# =====================================================================
@dataclass
class GelismisFeatureTestSonucu:
    sembol: str
    pazar: str
    interval: str
    basarili: bool
    hata_mesaji: Optional[str] = None
    win_rate_gercek: float = 0.0
    win_rate_permutasyon: float = 0.0
    sharpe_gercek: float = 0.0
    n_test_gozlem: int = 0


def gelismis_feature_tek_kombinasyon_test_et(
    sembol: str, pazar: str, interval: str,
    n_splits: int = 4, rastgele_seed: int = 42,
) -> GelismisFeatureTestSonucu:
    """coklu_test.py'deki tek_kombinasyon_test_et ile AYNI iskelet, ama
    YENİ_FEATURE_KOLONLARI (23 feature) kullanılıyor."""
    try:
        period = core.suggest_period(pazar, interval)
        if pazar == "TR_HISSE" and interval == "1d":
            period = "3y"  # Tur 7'deki düzeltmeyle tutarlı (Hurst pencere=60 için ısınma payı)
        fetch_result = core.get_market_data(sembol, period, interval, pazar)
        df_raw = fetch_result.df

        if df_raw.empty:
            return GelismisFeatureTestSonucu(sembol=sembol, pazar=pazar, interval=interval,
                                              basarili=False, hata_mesaji="Veri çekilemedi.")

        df_active, _ = core.calculate_metrics(df_raw)
        if df_active.empty or len(df_active) < 150:
            return GelismisFeatureTestSonucu(sembol=sembol, pazar=pazar, interval=interval,
                                              basarili=False, hata_mesaji=f"Veri çok az ({len(df_active)} satır).")

        df_active = gelismis_feature_ekle(df_active)

        tb_sonuc = triple_barrier_etiketle(
            close=df_active['Close'], high=df_active['High'], low=df_active['Low'],
            atr=df_active['ATR'], kar_al_katsayisi=2.0, zarar_kes_katsayisi=1.5, max_bar=20,
        )
        df_active['Hedef_TB'] = tb_sonuc.hedef
        df_active['Getiri_TB'] = tb_sonuc.gercek_getiri

        df_ml = df_active.dropna(subset=['Hedef_TB'] + YENI_FEATURE_KOLONLARI).copy()
        if len(df_ml) < 100:
            return GelismisFeatureTestSonucu(sembol=sembol, pazar=pazar, interval=interval,
                                              basarili=False, hata_mesaji=f"Etiketleme sonrası veri çok az ({len(df_ml)}).")

        X_all = df_ml[YENI_FEATURE_KOLONLARI].copy()
        y_all = df_ml['Hedef_TB'].astype(int)
        getiri_all = df_ml['Getiri_TB'].values

        cv = PurgedEmbargoCV(n_splits=n_splits, etiket_ufku=20)

        def _calistir(y_kullanilacak):
            win_rates, sharpes = [], []
            for tr_idx, te_idx in cv.split(X_all):
                if len(tr_idx) < 30 or len(te_idx) < 10:
                    continue
                X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
                y_tr, y_te = y_kullanilacak[tr_idx], y_kullanilacak[te_idx]

                scaler = StandardScaler()
                X_tr_sc = scaler.fit_transform(X_tr)
                X_te_sc = scaler.transform(X_te)

                model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
                model.fit(X_tr_sc, y_tr)
                preds = model.predict(X_te_sc)

                win_rate = float(np.mean(preds == y_te)) * 100
                test_getiri = getiri_all[te_idx]
                strat_returns = np.where(preds == 1, test_getiri, -test_getiri)
                sharpe = (
                    (np.mean(strat_returns) / np.std(strat_returns)) * np.sqrt(252)
                    if np.std(strat_returns) > 0 else 0.0
                )
                win_rates.append(win_rate)
                sharpes.append(sharpe)
            return win_rates, sharpes

        win_rates_gercek, sharpes_gercek = _calistir(y_all.values)
        if not win_rates_gercek:
            return GelismisFeatureTestSonucu(sembol=sembol, pazar=pazar, interval=interval,
                                              basarili=False, hata_mesaji="Hiçbir CV fold'u yeterli veri içermedi.")

        rng = np.random.default_rng(rastgele_seed)
        y_permute = rng.permutation(y_all.values)
        win_rates_permute, _ = _calistir(y_permute)

        return GelismisFeatureTestSonucu(
            sembol=sembol, pazar=pazar, interval=interval, basarili=True,
            win_rate_gercek=float(np.mean(win_rates_gercek)),
            win_rate_permutasyon=float(np.mean(win_rates_permute)) if win_rates_permute else 0.0,
            sharpe_gercek=float(np.mean(sharpes_gercek)),
            n_test_gozlem=len(df_ml),
        )

    except Exception as exc:
        return GelismisFeatureTestSonucu(sembol=sembol, pazar=pazar, interval=interval,
                                          basarili=False, hata_mesaji=f"Beklenmeyen hata: {exc}")


# Tur 7'nin ilk 28 kombinasyonuyla AYNI evren — model/feature etkisini
# doğrudan karşılaştırmak için (Tur 6 ağaç-model + Tur 7 LSTM + Tur 8
# gelişmiş-feature, hepsi AYNI varlık setinde).
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


def tum_kombinasyonlari_test_et(kombinasyonlar=None, ilerleme_yazdir=True):
    if kombinasyonlar is None:
        kombinasyonlar = TEST_KOMBINASYONLARI
    sonuclar = []
    for i, (sembol, pazar, interval) in enumerate(kombinasyonlar):
        if ilerleme_yazdir:
            print(f"[{i+1}/{len(kombinasyonlar)}] {sembol} ({pazar}, {interval}) gelişmiş feature ile test ediliyor...")
        sonuc = gelismis_feature_tek_kombinasyon_test_et(sembol, pazar, interval)
        sonuclar.append(sonuc)
        if ilerleme_yazdir:
            if sonuc.basarili:
                print(f"    -> fark: {sonuc.win_rate_gercek - sonuc.win_rate_permutasyon:+.1f}  "
                      f"sharpe: {sonuc.sharpe_gercek:.2f}")
            else:
                print(f"    -> BAŞARISIZ: {sonuc.hata_mesaji}")
    return sonuclar


def sonuclari_analiz_et(sonuclar: list) -> dict:
    from scipy import stats

    basarili = [s for s in sonuclar if s.basarili]
    if len(basarili) < 3:
        return {'hata': f"Yeterli başarılı test yok ({len(basarili)} adet)."}

    win_gercek = np.array([s.win_rate_gercek for s in basarili])
    win_permute = np.array([s.win_rate_permutasyon for s in basarili])
    farklar = win_gercek - win_permute

    t1, p1 = stats.ttest_1samp(win_gercek, 50.0)
    t2, p2 = stats.ttest_1samp(farklar, 0.0)

    return {
        'basarili_test_sayisi': len(basarili),
        'toplam_test_sayisi': len(sonuclar),
        'win_rate_gercek_ortalama': float(np.mean(win_gercek)),
        'win_rate_permutasyon_ortalama': float(np.mean(win_permute)),
        'ortalama_fark': float(np.mean(farklar)),
        'test_1_vs_50yuzde': {'p_degeri': float(p1), 'anlamli_mi': bool(p1 < 0.05)},
        'test_2_vs_permutasyon': {'p_degeri': float(p2), 'anlamli_mi': bool(p2 < 0.05)},
        'detaylar': [
            {'sembol': s.sembol, 'pazar': s.pazar, 'interval': s.interval,
             'win_rate_gercek': round(s.win_rate_gercek, 1),
             'win_rate_permutasyon': round(s.win_rate_permutasyon, 1),
             'fark': round(s.win_rate_gercek - s.win_rate_permutasyon, 1),
             'sharpe': round(s.sharpe_gercek, 2)}
            for s in basarili
        ],
    }


def ozet_yazdir(rapor: dict):
    print()
    print("=" * 70)
    print("📊 TUR 8 — GELİŞMİŞ FEATURE SONUÇ ÖZETİ")
    print("=" * 70)
    if 'hata' in rapor:
        print("HATA:", rapor['hata'])
        return
    print(f"Başarılı/Toplam: {rapor['basarili_test_sayisi']}/{rapor['toplam_test_sayisi']}")
    print(f"Win Rate gerçek:      %{rapor['win_rate_gercek_ortalama']:.2f}")
    print(f"Win Rate permütasyon: %{rapor['win_rate_permutasyon_ortalama']:.2f}")
    print(f"Ortalama fark:        {rapor['ortalama_fark']:+.2f} puan")
    print()
    print(f"Test 1 (vs %50): p={rapor['test_1_vs_50yuzde']['p_degeri']:.4f}  "
          f"{'ANLAMLI' if rapor['test_1_vs_50yuzde']['anlamli_mi'] else 'anlamlı değil'}")
    print(f"Test 2 (vs permütasyon): p={rapor['test_2_vs_permutasyon']['p_degeri']:.4f}  "
          f"{'ANLAMLI' if rapor['test_2_vs_permutasyon']['anlamli_mi'] else 'anlamlı değil'}")
    print()
    print("Detaylar (en yüksekten en düşüğe):")
    for d in sorted(rapor['detaylar'], key=lambda x: -x['fark']):
        print(f"  {d['sembol']:<10} ({d['pazar']:<10}) fark={d['fark']:+6.1f}  sharpe={d['sharpe']:+.2f}")
    print("=" * 70)
    print()
    print("🎯 KARŞILAŞTIRMA (önceki turlarla, AYNI 28 kombinasyonluk evren):")
    print("  Tur 6 (ağaç-modeli, 19 klasik feature, 47 TR_HISSE özel): p=0.1594")
    print("  Tur 7 (LSTM, 19 klasik feature, 28 kombinasyon):          p=0.0238")
    print(f"  Tur 8 (ağaç-modeli, 23 feature [+Hurst/Vol/Entropy], 28 kombinasyon): "
          f"p={rapor['test_2_vs_permutasyon']['p_degeri']:.4f}")


if __name__ == "__main__":
    print(f"Tur 8 (Gelişmiş Feature Mühendisliği) başlıyor — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    sonuclar = tum_kombinasyonlari_test_et()
    rapor = sonuclari_analiz_et(sonuclar)
    ozet_yazdir(rapor)

    cikti_dosyasi = os.path.join(os.path.dirname(__file__), 'tur8_gelismis_feature_sonuclar.json')
    with open(cikti_dosyasi, 'w', encoding='utf-8') as f:
        json.dump(rapor, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Sonuçlar şuraya kaydedildi: {cikti_dosyasi}")
