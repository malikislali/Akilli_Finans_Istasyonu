"""
=====================================================================
🔬 ARAŞTIRMA TUR 9 (SON TEKNİK DENEME) — REJİM-BAZLI AYRI MODELLER
=====================================================================
Tur 1-8 sonucu: Klasik feature+ağaç model, klasik feature+LSTM, ve
gelişmiş feature+ağaç model — ÜÇ bağımsız yaklaşım da "edge yok"
sonucuna ulaştı (permütasyon testinden istatistiksel olarak ayrılamadı).

BU TUR (FARKLI BİR HİPOTEZ): Belki tek bir model, TÜM piyasa
durumlarında (Boğa/Ayı/Yatay) AYNI kuralları öğrenmeye çalıştığı için
karışıyor — örneğin Boğa piyasasında işe yarayan bir RSI kalıbı, Ayı
piyasasında tam tersi davranabilir, ve tek model bu çelişkiyi
"ortalayarak" iki rejimde de zayıf kalabilir. ChatGPT'nin önerisi:
her rejim için AYRI bir model eğitmek.

YÖNTEM:
  1. calculate_metrics() çıktısındaki MEVCUT rejim tespiti kullanılır
     (Regime_Bull, Regime_Bear, Regime_Sideways — SMA200 ve Bollinger
     Width tabanlı, üretim kodunda zaten var).
  2. METODOLOJİK İNCELİK: Bir rejimin barları zaman içinde SÜREKSİZ
     olabilir (örn. Boğa: 50-80. günler, sonra 150-200. günler). Bu
     yüzden ÖNCE normal Purged CV ile veri fold'lara bölünür (zaman
     sıralaması ve purging korunur), SONRA her fold içinde "sadece bu
     rejimdeki barlar" filtresi uygulanır. Böylece sızıntı riski
     büyümez, sadece her fold'da daha az (rejime özgü) veri kalır.
  3. Her rejim için AYRI win_rate/permütasyon karşılaştırması yapılır
     — "Boğa modelinde edge var mı, Ayı modelinde var mı, Yatayda var
     mı" sorularına AYRI AYRI cevap aranır.

DÜRÜST BEKLENTİ: Rejime göre bölmek, her alt-kümedeki veri miktarını
ciddi şekilde AZALTIR (3'e bölünen veri, istatistiksel gücü düşürür).
Bu yüzden "anlamlı değil" çıkması, "rejim hipotezi yanlış" değil,
"yeterli veri yok" anlamına da gelebilir — bu ayrımı raporda açıkça
belirteceğiz.
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

FEATURE_KOLONLARI = core.FEATURE_KOLONLARI
REJIM_KOLONLARI = {"BOGA": "Regime_Bull", "AYI": "Regime_Bear", "YATAY": "Regime_Sideways"}


@dataclass
class RejimTestSonucu:
    sembol: str
    pazar: str
    interval: str
    rejim: str
    basarili: bool
    hata_mesaji: Optional[str] = None
    win_rate_gercek: float = 0.0
    win_rate_permutasyon: float = 0.0
    sharpe_gercek: float = 0.0
    n_rejim_gozlem: int = 0


def rejim_bazli_tek_kombinasyon_test_et(
    sembol: str, pazar: str, interval: str,
    n_splits: int = 4, rastgele_seed: int = 42,
) -> dict:
    """
    Tek bir varlık için, her rejim (Boğa/Ayı/Yatay) için AYRI bir
    win_rate/permütasyon sonucu döner. Dönüş: {"BOGA": RejimTestSonucu,
    "AYI": RejimTestSonucu, "YATAY": RejimTestSonucu}
    """
    sonuclar = {}
    try:
        period = core.suggest_period(pazar, interval)
        if pazar == "TR_HISSE" and interval == "1d":
            period = "3y"  # Tur 7-8 ile tutarlı düzeltme
        fetch_result = core.get_market_data(sembol, period, interval, pazar)
        df_raw = fetch_result.df

        if df_raw.empty:
            for rejim_adi in REJIM_KOLONLARI:
                sonuclar[rejim_adi] = RejimTestSonucu(sembol=sembol, pazar=pazar, interval=interval,
                                                       rejim=rejim_adi, basarili=False, hata_mesaji="Veri çekilemedi.")
            return sonuclar

        df_active, _ = core.calculate_metrics(df_raw)
        if df_active.empty or len(df_active) < 150:
            for rejim_adi in REJIM_KOLONLARI:
                sonuclar[rejim_adi] = RejimTestSonucu(sembol=sembol, pazar=pazar, interval=interval,
                                                       rejim=rejim_adi, basarili=False,
                                                       hata_mesaji=f"Veri çok az ({len(df_active)} satır).")
            return sonuclar

        tb_sonuc = triple_barrier_etiketle(
            close=df_active['Close'], high=df_active['High'], low=df_active['Low'],
            atr=df_active['ATR'], kar_al_katsayisi=2.0, zarar_kes_katsayisi=1.5, max_bar=20,
        )
        df_active['Hedef_TB'] = tb_sonuc.hedef
        df_active['Getiri_TB'] = tb_sonuc.gercek_getiri

        df_ml = df_active.dropna(subset=['Hedef_TB'] + FEATURE_KOLONLARI).copy()
        if len(df_ml) < 100:
            for rejim_adi in REJIM_KOLONLARI:
                sonuclar[rejim_adi] = RejimTestSonucu(sembol=sembol, pazar=pazar, interval=interval,
                                                       rejim=rejim_adi, basarili=False,
                                                       hata_mesaji=f"Etiketleme sonrası veri çok az ({len(df_ml)}).")
            return sonuclar

        X_all = df_ml[FEATURE_KOLONLARI].copy()
        y_all = df_ml['Hedef_TB'].astype(int)
        getiri_all = df_ml['Getiri_TB'].values

        # ⚠️ METODOLOJİK İNCELİK: CV bölmesi TÜM veri üzerinde (rejime
        # bakılmadan) yapılır — zaman sıralaması ve purging bu şekilde
        # korunur. Rejim filtresi, her fold'un İÇİNDE ayrıca uygulanır.
        cv = PurgedEmbargoCV(n_splits=n_splits, etiket_ufku=20)

        for rejim_adi, rejim_kolonu in REJIM_KOLONLARI.items():
            rejim_maskesi = df_ml[rejim_kolonu].values == 1.0
            n_rejim_toplam = int(rejim_maskesi.sum())

            if n_rejim_toplam < 80:  # Rejime özgü minimum veri eşiği
                sonuclar[rejim_adi] = RejimTestSonucu(
                    sembol=sembol, pazar=pazar, interval=interval, rejim=rejim_adi,
                    basarili=False, hata_mesaji=f"Bu rejimde yeterli gözlem yok ({n_rejim_toplam}).",
                )
                continue

            def _rejimde_calistir(y_kullanilacak):
                win_rates, sharpes = [], []
                for tr_idx, te_idx in cv.split(X_all):
                    # Fold içindeki indeksleri rejim maskesiyle FİLTRELE
                    tr_idx_rejim = tr_idx[rejim_maskesi[tr_idx]]
                    te_idx_rejim = te_idx[rejim_maskesi[te_idx]]

                    if len(tr_idx_rejim) < 25 or len(te_idx_rejim) < 8:
                        continue

                    X_tr = X_all.iloc[tr_idx_rejim]
                    X_te = X_all.iloc[te_idx_rejim]
                    y_tr = y_kullanilacak[tr_idx_rejim]
                    y_te = y_kullanilacak[te_idx_rejim]

                    if len(np.unique(y_tr)) < 2:  # Tek sınıf kaldıysa eğitim anlamsız
                        continue

                    scaler = StandardScaler()
                    X_tr_sc = scaler.fit_transform(X_tr)
                    X_te_sc = scaler.transform(X_te)

                    model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
                    model.fit(X_tr_sc, y_tr)
                    preds = model.predict(X_te_sc)

                    win_rate = float(np.mean(preds == y_te)) * 100
                    test_getiri = getiri_all[te_idx_rejim]
                    strat_returns = np.where(preds == 1, test_getiri, -test_getiri)
                    sharpe = (
                        (np.mean(strat_returns) / np.std(strat_returns)) * np.sqrt(252)
                        if np.std(strat_returns) > 0 else 0.0
                    )
                    win_rates.append(win_rate)
                    sharpes.append(sharpe)
                return win_rates, sharpes

            win_rates_gercek, sharpes_gercek = _rejimde_calistir(y_all.values)

            if not win_rates_gercek:
                sonuclar[rejim_adi] = RejimTestSonucu(
                    sembol=sembol, pazar=pazar, interval=interval, rejim=rejim_adi,
                    basarili=False, hata_mesaji="Hiçbir CV fold'unda bu rejim için yeterli veri kalmadı.",
                )
                continue

            rng = np.random.default_rng(rastgele_seed)
            y_permute = rng.permutation(y_all.values)
            win_rates_permute, _ = _rejimde_calistir(y_permute)

            sonuclar[rejim_adi] = RejimTestSonucu(
                sembol=sembol, pazar=pazar, interval=interval, rejim=rejim_adi, basarili=True,
                win_rate_gercek=float(np.mean(win_rates_gercek)),
                win_rate_permutasyon=float(np.mean(win_rates_permute)) if win_rates_permute else 0.0,
                sharpe_gercek=float(np.mean(sharpes_gercek)),
                n_rejim_gozlem=n_rejim_toplam,
            )

    except Exception as exc:
        for rejim_adi in REJIM_KOLONLARI:
            if rejim_adi not in sonuclar:
                sonuclar[rejim_adi] = RejimTestSonucu(sembol=sembol, pazar=pazar, interval=interval,
                                                       rejim=rejim_adi, basarili=False,
                                                       hata_mesaji=f"Beklenmeyen hata: {exc}")
    return sonuclar


# Tur 7-8 ile AYNI 28 kombinasyonluk evren (karşılaştırılabilirlik için).
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


def tum_kombinasyonlari_test_et(kombinasyonlar=None, ilerleme_yazdir=True) -> dict:
    """Dönüş: {"BOGA": [RejimTestSonucu, ...], "AYI": [...], "YATAY": [...]}"""
    if kombinasyonlar is None:
        kombinasyonlar = TEST_KOMBINASYONLARI

    rejim_bazinda_sonuclar = {rejim_adi: [] for rejim_adi in REJIM_KOLONLARI}

    for i, (sembol, pazar, interval) in enumerate(kombinasyonlar):
        if ilerleme_yazdir:
            print(f"[{i+1}/{len(kombinasyonlar)}] {sembol} ({pazar}, {interval}) rejim-bazlı test ediliyor...")
        sonuc_dict = rejim_bazli_tek_kombinasyon_test_et(sembol, pazar, interval)
        for rejim_adi, sonuc in sonuc_dict.items():
            rejim_bazinda_sonuclar[rejim_adi].append(sonuc)
            if ilerleme_yazdir:
                if sonuc.basarili:
                    print(f"    [{rejim_adi}] fark: {sonuc.win_rate_gercek - sonuc.win_rate_permutasyon:+.1f}  "
                          f"(n={sonuc.n_rejim_gozlem})")
                else:
                    print(f"    [{rejim_adi}] BAŞARISIZ: {sonuc.hata_mesaji}")

    return rejim_bazinda_sonuclar


def rejim_sonuclarini_analiz_et(rejim_bazinda_sonuclar: dict) -> dict:
    from scipy import stats

    nihai_rapor = {}
    for rejim_adi, sonuclar in rejim_bazinda_sonuclar.items():
        basarili = [s for s in sonuclar if s.basarili]
        if len(basarili) < 3:
            nihai_rapor[rejim_adi] = {'hata': f"Yeterli başarılı test yok ({len(basarili)} adet)."}
            continue

        win_gercek = np.array([s.win_rate_gercek for s in basarili])
        win_permute = np.array([s.win_rate_permutasyon for s in basarili])
        farklar = win_gercek - win_permute

        t1, p1 = stats.ttest_1samp(win_gercek, 50.0)
        t2, p2 = stats.ttest_1samp(farklar, 0.0)

        nihai_rapor[rejim_adi] = {
            'basarili_test_sayisi': len(basarili),
            'toplam_test_sayisi': len(sonuclar),
            'ortalama_rejim_gozlem_sayisi': float(np.mean([s.n_rejim_gozlem for s in basarili])),
            'win_rate_gercek_ortalama': float(np.mean(win_gercek)),
            'win_rate_permutasyon_ortalama': float(np.mean(win_permute)),
            'ortalama_fark': float(np.mean(farklar)),
            'test_1_vs_50yuzde': {'p_degeri': float(p1), 'anlamli_mi': bool(p1 < 0.05)},
            'test_2_vs_permutasyon': {'p_degeri': float(p2), 'anlamli_mi': bool(p2 < 0.05)},
            'detaylar': [
                {'sembol': s.sembol, 'pazar': s.pazar,
                 'win_rate_gercek': round(s.win_rate_gercek, 1),
                 'win_rate_permutasyon': round(s.win_rate_permutasyon, 1),
                 'fark': round(s.win_rate_gercek - s.win_rate_permutasyon, 1),
                 'n_gozlem': s.n_rejim_gozlem}
                for s in basarili
            ],
        }
    return nihai_rapor


def ozet_yazdir(nihai_rapor: dict):
    print()
    print("=" * 70)
    print("📊 TUR 9 — REJİM-BAZLI SONUÇ ÖZETİ")
    print("=" * 70)
    for rejim_adi, rapor in nihai_rapor.items():
        print(f"\n--- {rejim_adi} ---")
        if 'hata' in rapor:
            print("  HATA:", rapor['hata'])
            continue
        print(f"  Başarılı/Toplam: {rapor['basarili_test_sayisi']}/{rapor['toplam_test_sayisi']}")
        print(f"  Ortalama rejim-içi gözlem sayısı: {rapor['ortalama_rejim_gozlem_sayisi']:.0f}")
        print(f"  Win Rate gerçek:      %{rapor['win_rate_gercek_ortalama']:.2f}")
        print(f"  Win Rate permütasyon: %{rapor['win_rate_permutasyon_ortalama']:.2f}")
        print(f"  Ortalama fark:        {rapor['ortalama_fark']:+.2f} puan")
        print(f"  Test 1 (vs %50): p={rapor['test_1_vs_50yuzde']['p_degeri']:.4f}  "
              f"{'ANLAMLI' if rapor['test_1_vs_50yuzde']['anlamli_mi'] else 'anlamlı değil'}")
        print(f"  Test 2 (vs permütasyon): p={rapor['test_2_vs_permutasyon']['p_degeri']:.4f}  "
              f"{'ANLAMLI' if rapor['test_2_vs_permutasyon']['anlamli_mi'] else 'anlamlı değil'}")
    print()
    print("=" * 70)
    print("🎯 SON KARŞILAŞTIRMA (önceki turlarla):")
    print("  Tur 6 (ağaç, klasik feature, TEK model):       p=0.159 (TR_HISSE özel)")
    print("  Tur 7 (LSTM, klasik feature, TEK model):       p=0.024 (28 kombinasyon, n=28)")
    print("  Tur 8 (ağaç, gelişmiş feature, TEK model):     p=0.661 (28 kombinasyon)")
    for rejim_adi, rapor in nihai_rapor.items():
        if 'hata' not in rapor:
            print(f"  Tur 9 ({rejim_adi} rejimi, AYRI model):           p={rapor['test_2_vs_permutasyon']['p_degeri']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    print(f"Tur 9 (Rejim-Bazlı Modeller) başlıyor — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    rejim_sonuclari = tum_kombinasyonlari_test_et()
    nihai_rapor = rejim_sonuclarini_analiz_et(rejim_sonuclari)
    ozet_yazdir(nihai_rapor)

    cikti_dosyasi = os.path.join(os.path.dirname(__file__), 'tur9_rejim_bazli_sonuclar.json')
    with open(cikti_dosyasi, 'w', encoding='utf-8') as f:
        json.dump(nihai_rapor, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Sonuçlar şuraya kaydedildi: {cikti_dosyasi}")
