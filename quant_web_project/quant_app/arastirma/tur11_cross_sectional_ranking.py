"""
=====================================================================
🔬 ARAŞTIRMA TUR 11 (SON TUR) — CROSS-SECTIONAL RANKING
=====================================================================
Tur 1-10 sonucu: Beş bağımsız yaklaşım (klasik+ağaç, klasik+LSTM,
gelişmiş feature+ağaç, rejim-bazlı, meta-labeling) — HİÇBİRİNDE
istatistiksel olarak güvenilir bir edge bulunamadı.

BU TUR (KAVRAMSAL OLARAK FARKLI BİR PROBLEM TANIMI):
Önceki tüm turlar şu soruyu sordu: "Bu TEK varlık yükselecek mi
düşecek mi?" (mutlak, binary tahmin). Bu yaklaşımın bir zayıflığı:
eğer TÜM piyasa birlikte hareket ediyorsa (ortak piyasa beta'sı), model
"hepsi yükseldi, doğru bildim" diyerek YANILTICI bir başarı görüntüsü
verebilir — bu gerçek bir varlığa-özel bilgi değildir.

CROSS-SECTIONAL RANKING şu soruyu sorar: "Aynı pazardaki N varlık
arasında, hangisi önümüzdeki dönemde EN İYİ performansı gösterecek?"
Bu, ortak piyasa hareketini OTOMATİK OLARAK ÇIKARIR (her gün için
varlıklar birbirine göre sıralanır) — teorik olarak daha varlığa-özel,
daha az piyasa-beta'sı gürültüsü içeren bir sinyal arar. Kurumsal
quant fonlarında (örn. sektör-relatif/market-neutral stratejiler)
yaygın kullanılan bir tekniktir.

YÖNTEM:
  1. Her pazardaki TÜM varlıklar AYNI ZAMAN EKSENİNDE hizalanır
     (ortak tarih indeksi).
  2. Her gün için, her varlığın N bar sonraki GERÇEKLEŞEN getirisi
     hesaplanır (Triple Barrier DEĞİL — burada amaç ranking olduğu
     için basit ileri-getiri kullanılır, bu metodolojik bir farktır
     ve raporda açıkça belirtilir).
  3. Her gün, varlıklar bu gerçekleşen getiriye göre SIRALANIR
     (1=en iyi, N=en kötü).
  4. Model (regresyon), feature'lardan bu GÜNLÜK SIRALAMAYI tahmin
     etmeye çalışır.
  5. TEST: Model tahmininin gerçek sıralamayla Spearman korelasyonu,
     RASTGELE karıştırılmış sıralamayla elde edilen korelasyondan
     (permütasyon testi) anlamlı şekilde yüksek mi?

METODOLOJİK NOT — NEDEN PURGED CV BURADA FARKLI UYGULANIYOR:
Cross-sectional yapıda, "bir günün TÜM varlıkları" tek bir gözlem
birimi gibi davranır (çünkü aynı güne ait sıralama birbirine bağımlıdır).
Bu yüzden CV bölmesi GÜN bazında yapılır (bir günün tüm varlıkları
aynı fold'a girer), tek tek (varlık, gün) çiftleri bazında DEĞİL.
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
from scipy.stats import spearmanr

import quant_web_project.quant_app.quant_ml_coreBinanceli as core

FEATURE_KOLONLARI = core.FEATURE_KOLONLARI
ILERI_BAR_SAYISI = 10  # Sıralama için kullanılan ileri-getiri ufku


def pazar_capraz_kesit_veri_olustur(pazar: str, interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    Bir pazardaki TÜM varlıkları çekip, ORTAK bir tarih eksenine
    hizalayıp, her (tarih, varlık) çifti için feature + ileri-getiri +
    o günkü SIRALAMA içeren tek bir DataFrame döner.
    """
    semboller = core.VARLIK_HAVUZU.get(pazar, [])
    if not semboller:
        return None

    period = core.suggest_period(pazar, interval)
    if pazar == "TR_HISSE" and interval == "1d":
        period = "3y"

    varlik_df_listesi = {}
    for sembol in semboller:
        try:
            fetch_result = core.get_market_data(sembol, period, interval, pazar)
            df_raw = fetch_result.df
            if df_raw.empty or len(df_raw) < 150:
                continue
            df_active, _ = core.calculate_metrics(df_raw)
            if df_active.empty:
                continue

            df_active = df_active.copy()
            df_active['ileri_getiri'] = (
                df_active['Close'].shift(-ILERI_BAR_SAYISI) / df_active['Close'] - 1
            )
            varlik_df_listesi[sembol] = df_active
        except Exception:
            continue

    if len(varlik_df_listesi) < 4:
        return None

    ortak_tarihler = None
    for df in varlik_df_listesi.values():
        if ortak_tarihler is None:
            ortak_tarihler = set(df.index)
        else:
            ortak_tarihler &= set(df.index)
    ortak_tarihler = sorted(ortak_tarihler)

    if len(ortak_tarihler) < 100:
        return None

    satirlar = []
    for sembol, df in varlik_df_listesi.items():
        alt_df = df.loc[df.index.isin(ortak_tarihler)].copy()
        alt_df['tarih'] = alt_df.index
        alt_df['sembol'] = sembol
        satirlar.append(alt_df[['tarih', 'sembol', 'ileri_getiri'] + FEATURE_KOLONLARI])

    capraz_kesit_df = pd.concat(satirlar, ignore_index=True)
    capraz_kesit_df = capraz_kesit_df.dropna(subset=['ileri_getiri'] + FEATURE_KOLONLARI)

    capraz_kesit_df['siralama'] = capraz_kesit_df.groupby('tarih')['ileri_getiri'].rank(
        ascending=False, method='average'
    )
    capraz_kesit_df['n_varlik_o_gun'] = capraz_kesit_df.groupby('tarih')['sembol'].transform('count')
    capraz_kesit_df = capraz_kesit_df[capraz_kesit_df['n_varlik_o_gun'] >= 4]
    capraz_kesit_df['siralama_normalize'] = (
        (capraz_kesit_df['siralama'] - 1) / (capraz_kesit_df['n_varlik_o_gun'] - 1)
    )

    return capraz_kesit_df.sort_values('tarih').reset_index(drop=True)


@dataclass
class RankingSonucu:
    pazar: str
    basarili: bool
    hata_mesaji: Optional[str] = None
    n_varlik: int = 0
    n_gun: int = 0
    n_gozlem: int = 0
    spearman_gercek: float = 0.0
    spearman_permutasyon: float = 0.0


def gun_bazli_purged_split(tarihler: pd.Series, n_splits: int = 4, embargo_gun: int = 10):
    """
    Cross-sectional veri için GÜN BAZLI train/test bölmesi — bir günün
    TÜM varlıkları aynı fold'a girer.
    """
    benzersiz_tarihler = sorted(tarihler.unique())
    n_tarih = len(benzersiz_tarihler)
    fold_sizes = np.full(n_splits, n_tarih // n_splits, dtype=int)
    fold_sizes[: n_tarih % n_splits] += 1

    current = 0
    fold_bounds = []
    for fs in fold_sizes:
        fold_bounds.append((current, current + fs))
        current += fs

    for fold_idx in range(1, n_splits):
        test_start, test_stop = fold_bounds[fold_idx]
        test_tarihleri = set(benzersiz_tarihler[test_start:test_stop])

        purge_baslangic_idx = max(0, test_start - embargo_gun)
        train_tarihleri = set(benzersiz_tarihler[:purge_baslangic_idx])

        train_mask = tarihler.isin(train_tarihleri).values
        test_mask = tarihler.isin(test_tarihleri).values
        yield np.where(train_mask)[0], np.where(test_mask)[0]


def ranking_test_et(pazar: str, n_splits: int = 4, rastgele_seed: int = 42) -> RankingSonucu:
    try:
        capraz_df = pazar_capraz_kesit_veri_olustur(pazar)
        if capraz_df is None or len(capraz_df) < 200:
            return RankingSonucu(pazar=pazar, basarili=False,
                                  hata_mesaji="Çapraz kesit verisi yetersiz veya oluşturulamadı.")

        X_all = capraz_df[FEATURE_KOLONLARI].copy()
        y_siralama = capraz_df['siralama_normalize'].values
        tarihler = capraz_df['tarih']

        spearman_gercek_listesi, spearman_permute_listesi = [], []
        rng = np.random.default_rng(rastgele_seed)

        for tr_idx, te_idx in gun_bazli_purged_split(tarihler, n_splits=n_splits, embargo_gun=ILERI_BAR_SAYISI):
            if len(tr_idx) < 50 or len(te_idx) < 20:
                continue

            X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
            y_tr, y_te = y_siralama[tr_idx], y_siralama[te_idx]

            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)

            model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
            model.fit(X_tr_sc, y_tr)
            tahmin = model.predict(X_te_sc)

            korelasyon, _ = spearmanr(tahmin, y_te)
            if not np.isnan(korelasyon):
                spearman_gercek_listesi.append(korelasyon)

            y_tr_permute = rng.permutation(y_tr)
            model_p = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
            model_p.fit(X_tr_sc, y_tr_permute)
            tahmin_p = model_p.predict(X_te_sc)
            korelasyon_p, _ = spearmanr(tahmin_p, y_te)
            if not np.isnan(korelasyon_p):
                spearman_permute_listesi.append(korelasyon_p)

        if not spearman_gercek_listesi:
            return RankingSonucu(pazar=pazar, basarili=False,
                                  hata_mesaji="Hiçbir CV fold'unda yeterli veri kalmadı.")

        return RankingSonucu(
            pazar=pazar, basarili=True,
            n_varlik=capraz_df['sembol'].nunique(),
            n_gun=capraz_df['tarih'].nunique(),
            n_gozlem=len(capraz_df),
            spearman_gercek=float(np.mean(spearman_gercek_listesi)),
            spearman_permutasyon=float(np.mean(spearman_permute_listesi)) if spearman_permute_listesi else 0.0,
        )

    except Exception as exc:
        return RankingSonucu(pazar=pazar, basarili=False, hata_mesaji=f"Beklenmeyen hata: {exc}")


PAZARLAR = ["KRIPTO", "TR_HISSE", "ABD_HISSE", "EMTIA"]


def tum_pazarlari_test_et(ilerleme_yazdir=True) -> list:
    sonuclar = []
    for pazar in PAZARLAR:
        if ilerleme_yazdir:
            print(f"[{pazar}] Cross-sectional ranking test ediliyor...")
        sonuc = ranking_test_et(pazar)
        sonuclar.append(sonuc)
        if ilerleme_yazdir:
            if sonuc.basarili:
                print(f"    -> n_varlık={sonuc.n_varlik}  n_gün={sonuc.n_gun}  "
                      f"Spearman gerçek={sonuc.spearman_gercek:+.4f}  "
                      f"Spearman permütasyon={sonuc.spearman_permutasyon:+.4f}")
            else:
                print(f"    -> BAŞARISIZ: {sonuc.hata_mesaji}")
    return sonuclar


def sonuclari_analiz_et(sonuclar: list) -> dict:
    from scipy import stats

    basarili = [s for s in sonuclar if s.basarili]
    if len(basarili) < 2:
        return {'hata': f"Yeterli başarılı test yok ({len(basarili)} adet)."}

    spearman_g = np.array([s.spearman_gercek for s in basarili])
    spearman_p = np.array([s.spearman_permutasyon for s in basarili])
    farklar = spearman_g - spearman_p

    t1, p1 = stats.ttest_1samp(spearman_g, 0.0)
    t2, p2 = stats.ttest_1samp(farklar, 0.0)

    return {
        'basarili_pazar_sayisi': len(basarili),
        'toplam_pazar_sayisi': len(sonuclar),
        'spearman_gercek_ortalama': float(np.mean(spearman_g)),
        'spearman_permutasyon_ortalama': float(np.mean(spearman_p)),
        'ortalama_fark': float(np.mean(farklar)),
        'test_1_spearman_vs_sifir': {'p_degeri': float(p1), 'anlamli_mi': bool(p1 < 0.05)},
        'test_2_gercek_vs_permutasyon': {'p_degeri': float(p2), 'anlamli_mi': bool(p2 < 0.05)},
        'detaylar': [
            {'pazar': s.pazar, 'n_varlik': s.n_varlik, 'n_gun': s.n_gun,
             'spearman_gercek': round(s.spearman_gercek, 4),
             'spearman_permutasyon': round(s.spearman_permutasyon, 4),
             'fark': round(s.spearman_gercek - s.spearman_permutasyon, 4)}
            for s in basarili
        ],
    }


def ozet_yazdir(rapor: dict):
    print()
    print("=" * 70)
    print("📊 TUR 11 (SON TUR) — CROSS-SECTIONAL RANKING SONUÇ ÖZETİ")
    print("=" * 70)
    if 'hata' in rapor:
        print("HATA:", rapor['hata'])
        return
    print(f"Başarılı/Toplam pazar: {rapor['basarili_pazar_sayisi']}/{rapor['toplam_pazar_sayisi']}")
    print(f"Spearman korelasyon (gerçek):      {rapor['spearman_gercek_ortalama']:+.4f}")
    print(f"Spearman korelasyon (permütasyon): {rapor['spearman_permutasyon_ortalama']:+.4f}")
    print(f"Ortalama fark: {rapor['ortalama_fark']:+.4f}")
    print()
    print(f"Test 1 (Spearman vs 0): p={rapor['test_1_spearman_vs_sifir']['p_degeri']:.4f}  "
          f"{'ANLAMLI' if rapor['test_1_spearman_vs_sifir']['anlamli_mi'] else 'anlamlı değil'}")
    print(f"Test 2 (gerçek vs permütasyon): p={rapor['test_2_gercek_vs_permutasyon']['p_degeri']:.4f}  "
          f"{'ANLAMLI' if rapor['test_2_gercek_vs_permutasyon']['anlamli_mi'] else 'anlamlı değil'}")
    print()
    print("Pazar bazında detaylar:")
    for d in rapor['detaylar']:
        print(f"  {d['pazar']:<12} n_varlık={d['n_varlik']:<3} n_gün={d['n_gun']:<5} "
              f"Spearman_gerçek={d['spearman_gercek']:+.4f}  fark={d['fark']:+.4f}")
    print("=" * 70)
    print()
    print("🎯 NİHAİ SERİ KARŞILAŞTIRMASI (5 önceki yaklaşımla):")
    print("  Tur 6 (ağaç+klasik):        p=0.159")
    print("  Tur 7 (LSTM+klasik):        p=0.024 (sınırda, büyüyünce zayıfladı)")
    print("  Tur 8 (ağaç+gelişmiş):      p=0.661")
    print("  Tur 9 (rejim-bazlı):        p=0.554/0.023(ters)/0.424")
    print("  Tur 10 (meta-labeling):     p=0.662 / p=0.254")
    print(f"  Tur 11 (cross-sectional ranking): p={rapor['test_2_gercek_vs_permutasyon']['p_degeri']:.4f}")


if __name__ == "__main__":
    print(f"Tur 11 (Cross-Sectional Ranking, SON TUR) başlıyor — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    sonuclar = tum_pazarlari_test_et()
    rapor = sonuclari_analiz_et(sonuclar)
    ozet_yazdir(rapor)

    cikti_dosyasi = os.path.join(os.path.dirname(__file__), 'tur11_cross_sectional_ranking_sonuclar.json')
    with open(cikti_dosyasi, 'w', encoding='utf-8') as f:
        json.dump(rapor, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Sonuçlar şuraya kaydedildi: {cikti_dosyasi}")
