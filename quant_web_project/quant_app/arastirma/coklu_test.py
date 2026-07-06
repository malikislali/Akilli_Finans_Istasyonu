"""
=====================================================================
🔬 ARAŞTIRMA — ÇOKLU VARLIK/PERİYOT TESTİ + PERMÜTASYON TESTİ
=====================================================================
Bu modülün amacı, kullanıcının sorduğu asıl soruyu CEVAPLAMAK:
"Bu sistem GERÇEKTEN piyasayı tahmin ediyor mu, yoksa şans eseri
bazı kombinasyonlarda iyi bazılarında kötü mü görünüyor?"

YÖNTEM:
  1. Çok sayıda (30-50) varlık/pazar/periyot kombinasyonunda Triple
     Barrier + Purged CV ile model eğitilir, Win Rate ölçülür.
  2. Bu Win Rate'lerin dağılımına bakılır: ortalama, %50'den istatistiksel
     olarak anlamlı şekilde sapıyor mu (tek örneklem t-testi)?
  3. PERMÜTASYON TESTİ (en kritik adım): gerçek etiketler RASTGELE
     karıştırılıp aynı model aynı veriyle eğitilir. Eğer karıştırılmış
     (anlamsız) etiketlerle de gerçek etiketlerle ELDE EDİLENE YAKIN bir
     performans çıkıyorsa, model HİÇBİR ŞEY ÖĞRENMİYOR demektir — sadece
     gürültüye uyum sağlıyor (overfitting) veya şans eseri iyi görünüyor.

Bu modül quant_ml_core.py'deki fonksiyonları YENİDEN KULLANIR (veri
çekme, gösterge hesaplama) — sadece etiketleme (triple_barrier) ve
CV (purged_cv) kısmı farklıdır.
=====================================================================
"""

from __future__ import annotations

import sys
import os
import json
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import quant_web_project.quant_app.quant_ml_coreBinanceli as core
from arastirma.triple_barrier import triple_barrier_etiketle, etiket_dagilimi_ozet
from arastirma.purged_cv import PurgedEmbargoCV

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


# =====================================================================
# 📋 TEST EVRENİ — 4 pazardan, gerçekçi bir varlık/periyot çeşitliliği
# =====================================================================
TEST_KOMBINASYONLARI = [
    # (sembol, pazar, interval)
    ("BTC-USD", "KRIPTO", "1d"), ("BTC-USD", "KRIPTO", "4h"), ("BTC-USD", "KRIPTO", "1h"),
    ("ETH-USD", "KRIPTO", "1d"), ("ETH-USD", "KRIPTO", "4h"), ("ETH-USD", "KRIPTO", "1h"),
    ("SOL-USD", "KRIPTO", "1d"), ("SOL-USD", "KRIPTO", "4h"),
    ("AVAX-USD", "KRIPTO", "1d"), ("XRP-USD", "KRIPTO", "1d"),
    ("DOGE-USD", "KRIPTO", "1d"), ("LINK-USD", "KRIPTO", "1d"),

    ("THYAO.IS", "TR_HISSE", "1d"), ("THYAO.IS", "TR_HISSE", "4h"),
    ("GARAN.IS", "TR_HISSE", "1d"), ("ASELS.IS", "TR_HISSE", "1d"),
    ("EREGL.IS", "TR_HISSE", "1d"), ("BIMAS.IS", "TR_HISSE", "1d"),
    ("SISE.IS", "TR_HISSE", "1d"), ("TUPRS.IS", "TR_HISSE", "1d"),

    ("AAPL", "ABD_HISSE", "1d"), ("AAPL", "ABD_HISSE", "4h"),
    ("NVDA", "ABD_HISSE", "1d"), ("TSLA", "ABD_HISSE", "1d"),
    ("MSFT", "ABD_HISSE", "1d"), ("AMZN", "ABD_HISSE", "1d"),
    ("META", "ABD_HISSE", "1d"), ("GOOGL", "ABD_HISSE", "1d"),

    ("GC=F", "EMTIA", "1d"), ("SI=F", "EMTIA", "1d"),
    ("CL=F", "EMTIA", "1d"), ("NG=F", "EMTIA", "1d"),
]


@dataclass
class TekTestSonucu:
    sembol: str
    pazar: str
    interval: str
    basarili: bool
    hata_mesaji: Optional[str] = None

    # Gerçek etiketlerle elde edilen sonuçlar
    win_rate_gercek: float = 0.0
    sharpe_gercek: float = 0.0
    n_test_gozlem: int = 0

    # Permütasyon (rastgele etiket) testi sonuçları
    win_rate_permutasyon: float = 0.0
    sharpe_permutasyon: float = 0.0

    # Etiket dağılımı (Triple Barrier'ın ne kadar dengeli etiket ürettiği)
    pozitif_oran: Optional[float] = None
    ortalama_bar_sayisi: Optional[float] = None


FEATURE_KOLONLARI = core.FEATURE_KOLONLARI  # quant_ml_core ile AYNI feature seti kullanılır


def _model_egit_ve_test_et(X_train, y_train, X_test, y_test, scale_pos_weight):
    """Basitleştirilmiş tek-model (GBM) eğitimi — araştırma hızını artırmak
    için 3'lü ensemble yerine tek model kullanılır (kalibrasyon da YOK,
    çünkü burada amaç 'edge var mı' sorusuna hızlı cevap vermektir)."""
    model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds


def tek_kombinasyon_test_et(
    sembol: str, pazar: str, interval: str,
    kar_al_katsayisi: float = 2.0, zarar_kes_katsayisi: float = 1.5, max_bar: int = 10,
    n_splits: int = 4, embargo_bar: int = 5,
    rastgele_seed: int = 42,
) -> TekTestSonucu:
    """
    Tek bir varlık/periyot kombinasyonu için TAM araştırma akışını çalıştırır:
    veri çek -> gösterge hesapla -> Triple Barrier etiketle -> Purged CV ile
    eğit/test et -> AYNI veriyle permütasyon testi yap -> sonuçları karşılaştır.
    """
    try:
        period = core.suggest_period(pazar, interval)
        fetch_result = core.get_market_data(sembol, period, interval, pazar)
        df_raw = fetch_result.df

        if df_raw.empty:
            return TekTestSonucu(sembol=sembol, pazar=pazar, interval=interval,
                                  basarili=False, hata_mesaji="Veri çekilemedi.")

        df_active, _ = core.calculate_metrics(df_raw)
        if df_active.empty or len(df_active) < 100:
            return TekTestSonucu(sembol=sembol, pazar=pazar, interval=interval,
                                  basarili=False, hata_mesaji=f"Veri çok az ({len(df_active)} satır).")

        # ---- Triple Barrier etiketleme ----
        tb_sonuc = triple_barrier_etiketle(
            close=df_active['Close'], high=df_active['High'], low=df_active['Low'],
            atr=df_active['ATR'], kar_al_katsayisi=kar_al_katsayisi,
            zarar_kes_katsayisi=zarar_kes_katsayisi, max_bar=max_bar,
        )
        etiket_ozeti = etiket_dagilimi_ozet(tb_sonuc)

        df_active = df_active.copy()
        df_active['Hedef_TB'] = tb_sonuc.hedef
        df_active['Getiri_TB'] = tb_sonuc.gercek_getiri

        # Etiketi NaN olan (hesaplanamayan) satırları çıkar
        df_ml = df_active.dropna(subset=['Hedef_TB'] + FEATURE_KOLONLARI).copy()
        if len(df_ml) < 100:
            return TekTestSonucu(sembol=sembol, pazar=pazar, interval=interval,
                                  basarili=False, hata_mesaji=f"Etiketleme sonrası veri çok az ({len(df_ml)}).")

        X_all = df_ml[FEATURE_KOLONLARI].copy()
        y_all = df_ml['Hedef_TB'].astype(int)
        getiri_all = df_ml['Getiri_TB'].values

        # ---- Purged + Embargo CV ile gerçek etiketlerle test ----
        cv = PurgedEmbargoCV(n_splits=n_splits, etiket_ufku=max_bar, embargo_bar=embargo_bar)

        win_rates_gercek, sharpes_gercek = [], []
        for train_idx, test_idx in cv.split(X_all):
            if len(train_idx) < 30 or len(test_idx) < 10:
                continue
            X_tr, X_te = X_all.iloc[train_idx], X_all.iloc[test_idx]
            y_tr, y_te = y_all.iloc[train_idx], y_all.iloc[test_idx]

            counts = y_tr.value_counts(normalize=True)
            spw = max(0.1, counts.get(0, 0.5) / (counts.get(1, 0.5) + 1e-10))

            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)

            preds = _model_egit_ve_test_et(X_tr_sc, y_tr, X_te_sc, y_te, spw)

            win_rate = float(np.mean(preds == y_te.values)) * 100
            strat_returns = np.where(preds == 1, getiri_all[test_idx], -getiri_all[test_idx])
            sharpe = (
                (np.mean(strat_returns) / np.std(strat_returns)) * np.sqrt(252)
                if np.std(strat_returns) > 0 else 0.0
            )
            win_rates_gercek.append(win_rate)
            sharpes_gercek.append(sharpe)

        if not win_rates_gercek:
            return TekTestSonucu(sembol=sembol, pazar=pazar, interval=interval,
                                  basarili=False, hata_mesaji="Hiçbir CV fold'u yeterli veri içermedi.")

        # ---- PERMÜTASYON TESTİ: etiketleri rastgele karıştır, AYNI süreci tekrarla ----
        rng = np.random.default_rng(rastgele_seed)
        y_permute = pd.Series(rng.permutation(y_all.values), index=y_all.index)

        win_rates_permute, sharpes_permute = [], []
        for train_idx, test_idx in cv.split(X_all):
            if len(train_idx) < 30 or len(test_idx) < 10:
                continue
            X_tr, X_te = X_all.iloc[train_idx], X_all.iloc[test_idx]
            y_tr, y_te = y_permute.iloc[train_idx], y_permute.iloc[test_idx]

            counts = y_tr.value_counts(normalize=True)
            spw = max(0.1, counts.get(0, 0.5) / (counts.get(1, 0.5) + 1e-10))

            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)

            preds = _model_egit_ve_test_et(X_tr_sc, y_tr, X_te_sc, y_te, spw)

            win_rate = float(np.mean(preds == y_te.values)) * 100
            strat_returns = np.where(preds == 1, getiri_all[test_idx], -getiri_all[test_idx])
            sharpe = (
                (np.mean(strat_returns) / np.std(strat_returns)) * np.sqrt(252)
                if np.std(strat_returns) > 0 else 0.0
            )
            win_rates_permute.append(win_rate)
            sharpes_permute.append(sharpe)

        return TekTestSonucu(
            sembol=sembol, pazar=pazar, interval=interval, basarili=True,
            win_rate_gercek=float(np.mean(win_rates_gercek)),
            sharpe_gercek=float(np.mean(sharpes_gercek)),
            n_test_gozlem=len(df_ml),
            win_rate_permutasyon=float(np.mean(win_rates_permute)) if win_rates_permute else 0.0,
            sharpe_permutasyon=float(np.mean(sharpes_permute)) if sharpes_permute else 0.0,
            pozitif_oran=etiket_ozeti['pozitif_oran'],
            ortalama_bar_sayisi=etiket_ozeti['ortalama_bar_sayisi'],
        )

    except Exception as exc:
        return TekTestSonucu(sembol=sembol, pazar=pazar, interval=interval,
                              basarili=False, hata_mesaji=f"Beklenmeyen hata: {exc}")


def rastgele_yuruyus_baseline_olustur(n_senaryo: int = 10, n_bar: int = 1000, ilerleme_yazdir: bool = True) -> dict:
    """
    🎯 EN KRİTİK FONKSİYON — araştırmanın referans çizgisini üretir.

    GERÇEK EDGE'İN MATEMATİKSEL OLARAK MÜMKÜN OLMADIĞI (rastgele yürüyüş,
    yani fiyatın bir önceki adımdan tamamen bağımsız rastgele hareket
    ettiği) sentetik veri serilerinde, AYNI Triple Barrier + Purged CV +
    permütasyon sürecini çalıştırır. Bu, "gerçek piyasada gördüğümüz
    gerçek-vs-permütasyon farkı, sadece metodolojik bir artefakt mı,
    yoksa gerçekten piyasaya özgü bir sinyal mi?" sorusuna nesnel bir
    karşılaştırma noktası sağlar.
    """
    sonuclar = []
    for i in range(n_senaryo):
        if ilerleme_yazdir:
            print(f"[Baseline {i+1}/{n_senaryo}] Sentetik rastgele yürüyüş test ediliyor...")

        idx = pd.date_range('2022-01-01', periods=n_bar, freq='D')
        rng = np.random.default_rng(10_000 + i)
        close = 100 + np.cumsum(rng.normal(0, 2, n_bar))
        close = np.clip(close, 10, None)
        df_raw = pd.DataFrame({
            'Open': close - 0.5, 'High': close + 1, 'Low': close - 1,
            'Close': close, 'Volume': rng.integers(1000, 5000, n_bar)
        }, index=idx)

        def sahte_fetch(symbol, period, interval, market, prefer_source=None, _df=df_raw):
            return core.FetchResult(df=_df, source="sentetik_rastgele_yuruyus",
                                     requested_interval=interval, actual_native_interval=interval,
                                     is_resampled=False)

        orijinal_fetch = core.get_market_data
        core.get_market_data = sahte_fetch
        try:
            sonuc = tek_kombinasyon_test_et(f"SENTETIK_{i}", "KRIPTO", "1d", rastgele_seed=i)
        finally:
            core.get_market_data = orijinal_fetch

        sonuclar.append(sonuc)
        if ilerleme_yazdir and sonuc.basarili:
            print(f"    -> fark: {sonuc.win_rate_gercek - sonuc.win_rate_permutasyon:+.1f}")

    return sonuclari_analiz_et(sonuclar)


def tum_kombinasyonlari_test_et(kombinasyonlar: Optional[list] = None, ilerleme_yazdir: bool = True) -> list:
    """Tüm test evrenini sırayla çalıştırır, sonuçları liste olarak döner."""
    if kombinasyonlar is None:
        kombinasyonlar = TEST_KOMBINASYONLARI

    sonuclar = []
    for i, (sembol, pazar, interval) in enumerate(kombinasyonlar):
        if ilerleme_yazdir:
            print(f"[{i+1}/{len(kombinasyonlar)}] {sembol} ({pazar}, {interval}) test ediliyor...")
        sonuc = tek_kombinasyon_test_et(sembol, pazar, interval)
        sonuclar.append(sonuc)
        if ilerleme_yazdir:
            if sonuc.basarili:
                print(f"    -> Win Rate (gerçek): %{sonuc.win_rate_gercek:.1f}  |  "
                      f"Win Rate (permütasyon): %{sonuc.win_rate_permutasyon:.1f}  |  "
                      f"Sharpe (gerçek): {sonuc.sharpe_gercek:.2f}")
            else:
                print(f"    -> BAŞARISIZ: {sonuc.hata_mesaji}")
        time.sleep(0.3)  # API rate-limit'e karşı nazik ol
    return sonuclar


def sonuclari_analiz_et(sonuclar: list, rastgele_yuruyus_baseline: Optional[dict] = None) -> dict:
    """
    Asıl araştırma sorusunu cevaplayan istatistiksel özet:
    'Gerçek etiketlerle elde edilen Win Rate, permütasyon (rastgele
    etiket) testininkinden istatistiksel olarak anlamlı şekilde
    farklı mı?' Eğer DEĞİLSE, model gerçek bir edge bulamıyor demektir.

    ⚠️ KRİTİK METODOLOJİK NOT (araştırma sürecinde keşfedildi):
    Teknik göstergeler (RSI, MACD, ATR vb.) ve Triple Barrier etiketi
    İKİSİ DE aynı fiyat serisinden türetildiği için, TAMAMEN RASTGELE
    YÜRÜYÜŞ (gerçek edge'in MATEMATİKSEL OLARAK MÜMKÜN OLMADIĞI) bir
    fiyat serisinde bile "gerçek etiket" Win Rate'i permütasyon
    testininkinden DOĞAL OLARAK biraz yüksek çıkabilir (gözlemlenen:
    ortalama +1.7 puan, 5 denemenin 4'ünde pozitif). Bu, CV/purging
    mantığının hatası değildir (saf rastgele feature'larla fark ~0'a
    döner) — göstergelerin ve etiketin AYNI KAYNAKTAN türetilmiş olması
    nedeniyle kaçınılmaz bir matematiksel ilişkidir.

    BU YÜZDEN: gerçek piyasa sonuçlarını değerlendirirken referans
    noktası "%50" veya "permütasyon=0 fark" DEĞİL, BU FONKSİYONA
    rastgele_yuruyus_baseline parametresiyle verilecek olan, AYNI
    yöntemle sentetik rastgele yürüyüş verisinde ölçülmüş ortalama
    farktır. Gerçek piyasa farkı bu baseline'ı AŞARSA, bu gerçek bir
    sinyal adayıdır; aşmazsa, gözlemlenen fark muhtemelen yalnızca bu
    matematiksel artefakttan kaynaklanıyordur.
    """
    from scipy import stats

    basarili_sonuclar = [s for s in sonuclar if s.basarili]
    if len(basarili_sonuclar) < 3:
        return {'hata': f"Yeterli başarılı test yok ({len(basarili_sonuclar)} adet, en az 3 gerekli)."}

    win_rates_gercek = np.array([s.win_rate_gercek for s in basarili_sonuclar])
    win_rates_permute = np.array([s.win_rate_permutasyon for s in basarili_sonuclar])
    farklar = win_rates_gercek - win_rates_permute

    # Tek örneklem t-testi: gerçek Win Rate'ler %50'den anlamlı şekilde farklı mı?
    # (Bu test TEK BAŞINA yeterli değildir, bkz. yukarıdaki metodolojik not.)
    t_stat_50, p_value_50 = stats.ttest_1samp(win_rates_gercek, 50.0)

    # Eşleştirilmiş t-testi: gerçek vs permütasyon farkı sıfırdan anlamlı şekilde farklı mı?
    t_stat_fark, p_value_fark = stats.ttest_1samp(farklar, 0.0)

    sonuc = {
        'toplam_test_sayisi': len(sonuclar),
        'basarili_test_sayisi': len(basarili_sonuclar),
        'basarisiz_test_sayisi': len(sonuclar) - len(basarili_sonuclar),

        'win_rate_gercek_ortalama': float(np.mean(win_rates_gercek)),
        'win_rate_gercek_std': float(np.std(win_rates_gercek)),
        'win_rate_permutasyon_ortalama': float(np.mean(win_rates_permute)),
        'win_rate_permutasyon_std': float(np.std(win_rates_permute)),
        'ortalama_fark_gercek_eksi_permutasyon': float(np.mean(farklar)),

        'test_1_gercek_vs_50yuzde': {
            'aciklama': "Gerçek Win Rate'ler %50'den (yazı-tura şansından) istatistiksel olarak FARKLI mı? "
                        "(DİKKAT: bu test tek başına YETERSİZ, bkz. test_3_baseline_karsilastirma)",
            't_istatistigi': float(t_stat_50),
            'p_degeri': float(p_value_50),
            'anlamli_mi_0.05': bool(p_value_50 < 0.05),
        },
        'test_2_gercek_vs_permutasyon': {
            'aciklama': "Gerçek etiketlerle elde edilen performans, RASTGELE etiketlerle elde edilenden anlamlı şekilde farklı mı?",
            't_istatistigi': float(t_stat_fark),
            'p_degeri': float(p_value_fark),
            'anlamli_mi_0.05': bool(p_value_fark < 0.05),
        },

        'kombinasyon_detaylari': [
            {
                'sembol': s.sembol, 'pazar': s.pazar, 'interval': s.interval,
                'win_rate_gercek': round(s.win_rate_gercek, 1),
                'win_rate_permutasyon': round(s.win_rate_permutasyon, 1),
                'fark': round(s.win_rate_gercek - s.win_rate_permutasyon, 1),
                'sharpe_gercek': round(s.sharpe_gercek, 2),
            }
            for s in basarili_sonuclar
        ],
    }

    if rastgele_yuruyus_baseline is not None:
        baseline_fark = rastgele_yuruyus_baseline.get('ortalama_fark_gercek_eksi_permutasyon', 0.0)
        gercek_fark = sonuc['ortalama_fark_gercek_eksi_permutasyon']
        sonuc['test_3_baseline_karsilastirma'] = {
            'aciklama': (
                "ASIL CEVAP: gerçek piyasa farkı, rastgele yürüyüş baseline'ını AŞIYOR MU? "
                "Aşmıyorsa gözlemlenen sinyal muhtemelen sahte (matematiksel artefakt)."
            ),
            'rastgele_yuruyus_baseline_farki': round(baseline_fark, 2),
            'gercek_piyasa_farki': round(gercek_fark, 2),
            'baseline_asildi_mi': bool(gercek_fark > baseline_fark),
            'asim_miktari': round(gercek_fark - baseline_fark, 2),
        }

    return sonuc
