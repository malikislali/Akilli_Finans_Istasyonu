"""
=====================================================================
🔬 ARAŞTIRMA TUR 10 — META-LABELING (López de Prado yöntemi)
=====================================================================
Tur 1-9 sonucu: Doğrudan yön tahmini (Close üstüne mi düşüşe mi?) hiçbir
model mimarisinde ve feature setinde istatistiksel olarak güvenilir bir
edge üretmedi. ChatGPT'nin raporu inceledikten sonraki önerisi:
META-LABELING (López de Prado, "Advances in Financial Machine
Learning", Bölüm 3.6) denenmesi.

YÖNTEM — İKİ AŞAMALI YAPI:
  AŞAMA 1 — BİRİNCİL MODEL (basit kural, ML DEĞİL):
    Klasik RSI aşırı alım/satım sinyali:
      RSI < 30  -> AL sinyali (birincil_sinyal = 1)
      RSI > 70  -> SAT sinyali (birincil_sinyal = -1)
      Diğer barlar -> SİNYAL YOK (bu barlar meta-modele HİÇ girmez)

  AŞAMA 2 — META-MODEL (ML, burada):
    Birincil modelin SİNYAL VERDİĞİ barlarda, "bu sinyal yönünde işlem
    açılırsa Triple Barrier sonucu KÂR mı ZARAR mı getirir?" sorusuna
    cevap arar. Meta-etiket:
      AL sinyali + Triple Barrier=1 (kâr)  -> meta_etiket=1 (sinyale güven)
      AL sinyali + Triple Barrier=0 (zarar) -> meta_etiket=0 (filtrele)
      SAT sinyali + Triple Barrier=0 (zarar/düşüş gerçekleşti) -> meta_etiket=1
      SAT sinyali + Triple Barrier=1 (yükseliş gerçekleşti, SAT yanlış) -> meta_etiket=0

NEDEN POTANSİYEL OLARAK FARKLI SONUÇ VEREBİLİR:
  Doğrudan yön tahmini, HER barda bir tahmin yapmaya çalışır — bu,
  modelin çoğu zaman gürültüden ayırt edilemeyen, zayıf sinyalli barlarda
  da karar vermeye zorlanması demektir. Meta-Labeling, modelin SADECE
  birincil kuralın "ilginç" bulduğu (aşırı RSI) anlardaki dar bir soruya
  ("bu spesifik sinyal işe yarar mı?") odaklanmasını sağlar — teorik
  olarak daha az gürültülü, daha kolay bir öğrenme görevi olabilir.

ÖNEMLİ — F1 SKORU KULLANILIR, SADECE ACCURACY DEĞİL:
  Meta-Labeling'de pozitif sınıf (sinyale güven) oranı genellikle dengesiz
  olabilir (örn. RSI sinyallerinin çoğu zaten kâr getiriyorsa meta-model
  "her zaman güven" diyerek yüksek accuracy alabilir ama hiçbir filtreleme
  YAPMAMIŞ olur — bu sahte bir başarı görüntüsüdür). Bu yüzden Precision
  ve Recall de raporlanır, accuracy'ye ek olarak.

Aynı disiplin korunuyor: Triple Barrier etiketleme, Purged CV, ve
permütasyon testi YİNE kullanılıyor — sadece PROBLEM TANIMI (ne
öğrenildiği) değişiyor.
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
from sklearn.metrics import precision_score, recall_score, f1_score

import quant_web_project.quant_app.quant_ml_coreBinanceli as core
from arastirma.triple_barrier import triple_barrier_etiketle
from arastirma.purged_cv import PurgedEmbargoCV

FEATURE_KOLONLARI = core.FEATURE_KOLONLARI
RSI_ASIRI_SATIM_ESIGI = 30
RSI_ASIRI_ALIM_ESIGI = 70


def birincil_sinyal_uret(rsi: pd.Series) -> pd.Series:
    """
    Birincil model — BASİT BİR KURAL, ML DEĞİL. RSI aşırı alım/satım
    sinyali üretir. Dönüş: +1 (AL sinyali), -1 (SAT sinyali), 0 (sinyal yok).
    """
    sinyal = pd.Series(0, index=rsi.index)
    sinyal[rsi < RSI_ASIRI_SATIM_ESIGI] = 1
    sinyal[rsi > RSI_ASIRI_ALIM_ESIGI] = -1
    return sinyal


def meta_etiket_uret(birincil_sinyal: pd.Series, triple_barrier_hedef: pd.Series) -> pd.Series:
    """
    Meta-etiket: birincil sinyal yönünde işlem açılırsa, Triple Barrier
    sonucu bu işlemi DOĞRULAR mı (kâr) yoksa YANLIŞLAR mı (zarar)?

    AL sinyali (+1): TB_hedef=1 (yükseliş gerçekleşti) -> meta=1 (doğru sinyal)
                      TB_hedef=0 (düşüş gerçekleşti)    -> meta=0 (yanlış sinyal)
    SAT sinyali (-1): TB_hedef=0 (düşüş gerçekleşti)    -> meta=1 (doğru sinyal)
                       TB_hedef=1 (yükseliş gerçekleşti) -> meta=0 (yanlış sinyal)
    """
    meta = pd.Series(np.nan, index=birincil_sinyal.index)
    al_maskesi = birincil_sinyal == 1
    sat_maskesi = birincil_sinyal == -1

    meta[al_maskesi] = (triple_barrier_hedef[al_maskesi] == 1).astype(float)
    meta[sat_maskesi] = (triple_barrier_hedef[sat_maskesi] == 0).astype(float)
    return meta


@dataclass
class MetaLabelingSonucu:
    sembol: str
    pazar: str
    interval: str
    basarili: bool
    hata_mesaji: Optional[str] = None
    n_birincil_sinyal: int = 0  # Birincil modelin kaç barda sinyal verdiği
    n_test_gozlem: int = 0
    accuracy_gercek: float = 0.0
    precision_gercek: float = 0.0
    recall_gercek: float = 0.0
    f1_gercek: float = 0.0
    accuracy_permutasyon: float = 0.0
    f1_permutasyon: float = 0.0
    # Karşılaştırma: birincil sinyali FİLTRESİZ (her zaman uygula) takip etmenin Win Rate'i
    win_rate_filtresiz_birincil: float = 0.0
    # Meta-model FİLTRESİYLE: sadece meta-model "güven" dediği sinyalleri takip etmenin Win Rate'i
    win_rate_meta_filtreli: float = 0.0


def meta_labeling_tek_kombinasyon_test_et(
    sembol: str, pazar: str, interval: str,
    n_splits: int = 4, rastgele_seed: int = 42,
) -> MetaLabelingSonucu:
    try:
        period = core.suggest_period(pazar, interval)
        if pazar == "TR_HISSE" and interval == "1d":
            period = "3y"
        fetch_result = core.get_market_data(sembol, period, interval, pazar)
        df_raw = fetch_result.df

        if df_raw.empty:
            return MetaLabelingSonucu(sembol=sembol, pazar=pazar, interval=interval,
                                       basarili=False, hata_mesaji="Veri çekilemedi.")

        df_active, _ = core.calculate_metrics(df_raw)
        if df_active.empty or len(df_active) < 150:
            return MetaLabelingSonucu(sembol=sembol, pazar=pazar, interval=interval,
                                       basarili=False, hata_mesaji=f"Veri çok az ({len(df_active)} satır).")

        tb_sonuc = triple_barrier_etiketle(
            close=df_active['Close'], high=df_active['High'], low=df_active['Low'],
            atr=df_active['ATR'], kar_al_katsayisi=2.0, zarar_kes_katsayisi=1.5, max_bar=20,
        )
        df_active['Hedef_TB'] = tb_sonuc.hedef
        df_active['Getiri_TB'] = tb_sonuc.gercek_getiri

        # ---- AŞAMA 1: Birincil sinyal (basit RSI kuralı) ----
        df_active['Birincil_Sinyal'] = birincil_sinyal_uret(df_active['RSI'])
        df_active['Meta_Etiket'] = meta_etiket_uret(df_active['Birincil_Sinyal'], df_active['Hedef_TB'])

        # Sadece BİRİNCİL SİNYALİN VERİLDİĞİ barlar meta-modele girer
        df_ml = df_active.dropna(subset=['Meta_Etiket'] + FEATURE_KOLONLARI).copy()
        n_birincil_sinyal = len(df_ml)

        if n_birincil_sinyal < 100:
            return MetaLabelingSonucu(
                sembol=sembol, pazar=pazar, interval=interval, basarili=False,
                hata_mesaji=f"Birincil sinyal sayısı yetersiz ({n_birincil_sinyal}, en az 100 gerekli).",
                n_birincil_sinyal=n_birincil_sinyal,
            )

        X_all = df_ml[FEATURE_KOLONLARI].copy()
        y_meta = df_ml['Meta_Etiket'].astype(int)
        birincil_yon = df_ml['Birincil_Sinyal'].values  # +1 veya -1
        getiri_all = df_ml['Getiri_TB'].values

        cv = PurgedEmbargoCV(n_splits=n_splits, etiket_ufku=20)

        def _calistir(y_kullanilacak):
            accs, precs, recs, f1s, win_rates_filtreli = [], [], [], [], []
            for tr_idx, te_idx in cv.split(X_all):
                if len(tr_idx) < 30 or len(te_idx) < 10:
                    continue
                X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
                y_tr, y_te = y_kullanilacak[tr_idx], y_kullanilacak[te_idx]

                if len(np.unique(y_tr)) < 2:
                    continue

                scaler = StandardScaler()
                X_tr_sc = scaler.fit_transform(X_tr)
                X_te_sc = scaler.transform(X_te)

                model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
                model.fit(X_tr_sc, y_tr)
                preds = model.predict(X_te_sc)

                accs.append(float(np.mean(preds == y_te)))
                precs.append(precision_score(y_te, preds, zero_division=0))
                recs.append(recall_score(y_te, preds, zero_division=0))
                f1s.append(f1_score(y_te, preds, zero_division=0))

                # Meta-model FİLTRESİYLE Win Rate: sadece meta-model "1" (güven)
                # dediği barlarda birincil sinyali takip et, gerçek sonuca bak.
                guvenilen_maske = preds == 1
                if guvenilen_maske.sum() > 0:
                    te_birincil_yon = birincil_yon[te_idx][guvenilen_maske]
                    te_getiri = getiri_all[te_idx][guvenilen_maske]
                    # Eğer birincil sinyal AL (+1) ise getiri pozitif beklenir, SAT (-1) ise negatif.
                    basarili_islem = np.where(te_birincil_yon == 1, te_getiri > 0, te_getiri < 0)
                    win_rates_filtreli.append(float(np.mean(basarili_islem)) * 100)

            return accs, precs, recs, f1s, win_rates_filtreli

        accs_g, precs_g, recs_g, f1s_g, win_filtreli_g = _calistir(y_meta.values)
        if not accs_g:
            return MetaLabelingSonucu(
                sembol=sembol, pazar=pazar, interval=interval, basarili=False,
                hata_mesaji="Hiçbir CV fold'u yeterli veri içermedi.", n_birincil_sinyal=n_birincil_sinyal,
            )

        # Filtresiz birincil sinyalin Win Rate'i (karşılaştırma için): TÜM
        # birincil sinyalleri (meta-model olmadan) takip etseydik ne olurdu?
        basarili_filtresiz = np.where(birincil_yon == 1, getiri_all > 0, getiri_all < 0)
        win_rate_filtresiz = float(np.mean(basarili_filtresiz)) * 100

        # Permütasyon testi
        rng = np.random.default_rng(rastgele_seed)
        y_permute = rng.permutation(y_meta.values)
        accs_p, _, _, f1s_p, _ = _calistir(y_permute)

        return MetaLabelingSonucu(
            sembol=sembol, pazar=pazar, interval=interval, basarili=True,
            n_birincil_sinyal=n_birincil_sinyal, n_test_gozlem=len(df_ml),
            accuracy_gercek=float(np.mean(accs_g)) * 100,
            precision_gercek=float(np.mean(precs_g)),
            recall_gercek=float(np.mean(recs_g)),
            f1_gercek=float(np.mean(f1s_g)),
            accuracy_permutasyon=float(np.mean(accs_p)) * 100 if accs_p else 0.0,
            f1_permutasyon=float(np.mean(f1s_p)) if f1s_p else 0.0,
            win_rate_filtresiz_birincil=win_rate_filtresiz,
            win_rate_meta_filtreli=float(np.mean(win_filtreli_g)) if win_filtreli_g else 0.0,
        )

    except Exception as exc:
        return MetaLabelingSonucu(sembol=sembol, pazar=pazar, interval=interval,
                                   basarili=False, hata_mesaji=f"Beklenmeyen hata: {exc}")


# Tur 7-9 ile AYNI 28 kombinasyonluk evren (karşılaştırılabilirlik için).
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
            print(f"[{i+1}/{len(kombinasyonlar)}] {sembol} ({pazar}, {interval}) meta-labeling ile test ediliyor...")
        sonuc = meta_labeling_tek_kombinasyon_test_et(sembol, pazar, interval)
        sonuclar.append(sonuc)
        if ilerleme_yazdir:
            if sonuc.basarili:
                print(f"    -> n_sinyal={sonuc.n_birincil_sinyal}  F1 gerçek={sonuc.f1_gercek:.3f}  "
                      f"F1 permütasyon={sonuc.f1_permutasyon:.3f}  "
                      f"WinRate filtresiz={sonuc.win_rate_filtresiz_birincil:.1f}%  "
                      f"WinRate meta-filtreli={sonuc.win_rate_meta_filtreli:.1f}%")
            else:
                print(f"    -> BAŞARISIZ: {sonuc.hata_mesaji}")
    return sonuclar


def sonuclari_analiz_et(sonuclar: list) -> dict:
    from scipy import stats

    basarili = [s for s in sonuclar if s.basarili]
    if len(basarili) < 3:
        return {'hata': f"Yeterli başarılı test yok ({len(basarili)} adet)."}

    f1_gercek = np.array([s.f1_gercek for s in basarili])
    f1_permute = np.array([s.f1_permutasyon for s in basarili])
    f1_farklar = f1_gercek - f1_permute

    win_filtresiz = np.array([s.win_rate_filtresiz_birincil for s in basarili])
    win_meta = np.array([s.win_rate_meta_filtreli for s in basarili if s.win_rate_meta_filtreli > 0])

    t1, p1 = stats.ttest_1samp(f1_farklar, 0.0)
    # Meta-filtreleme, filtresiz birincil sinyalden DAHA İYİ mi? (asıl pratik soru)
    if len(win_meta) >= 3:
        win_filtresiz_eslesen = np.array([s.win_rate_filtresiz_birincil for s in basarili if s.win_rate_meta_filtreli > 0])
        t2, p2 = stats.ttest_rel(win_meta, win_filtresiz_eslesen)
    else:
        t2, p2 = 0.0, 1.0

    return {
        'basarili_test_sayisi': len(basarili),
        'toplam_test_sayisi': len(sonuclar),
        'ortalama_n_birincil_sinyal': float(np.mean([s.n_birincil_sinyal for s in basarili])),
        'f1_gercek_ortalama': float(np.mean(f1_gercek)),
        'f1_permutasyon_ortalama': float(np.mean(f1_permute)),
        'ortalama_f1_fark': float(np.mean(f1_farklar)),
        'win_rate_filtresiz_birincil_ortalama': float(np.mean(win_filtresiz)),
        'win_rate_meta_filtreli_ortalama': float(np.mean(win_meta)) if len(win_meta) > 0 else None,
        'test_1_f1_vs_permutasyon': {'p_degeri': float(p1), 'anlamli_mi': bool(p1 < 0.05)},
        'test_2_meta_filtreli_vs_filtresiz': {
            'aciklama': "Meta-model filtrelemesi, filtresiz (her sinyali takip etmek) yaklaşımdan İYİ mi?",
            'p_degeri': float(p2), 'anlamli_mi': bool(p2 < 0.05),
        },
        'detaylar': [
            {'sembol': s.sembol, 'pazar': s.pazar,
             'n_sinyal': s.n_birincil_sinyal,
             'f1_gercek': round(s.f1_gercek, 3), 'f1_permutasyon': round(s.f1_permutasyon, 3),
             'win_rate_filtresiz': round(s.win_rate_filtresiz_birincil, 1),
             'win_rate_meta_filtreli': round(s.win_rate_meta_filtreli, 1)}
            for s in basarili
        ],
    }


def ozet_yazdir(rapor: dict):
    print()
    print("=" * 70)
    print("📊 TUR 10 — META-LABELING SONUÇ ÖZETİ")
    print("=" * 70)
    if 'hata' in rapor:
        print("HATA:", rapor['hata'])
        return
    print(f"Başarılı/Toplam: {rapor['basarili_test_sayisi']}/{rapor['toplam_test_sayisi']}")
    print(f"Ortalama birincil sinyal sayısı (varlık başına): {rapor['ortalama_n_birincil_sinyal']:.0f}")
    print()
    print(f"F1 gerçek:      {rapor['f1_gercek_ortalama']:.3f}")
    print(f"F1 permütasyon: {rapor['f1_permutasyon_ortalama']:.3f}")
    print(f"Ortalama F1 fark: {rapor['ortalama_f1_fark']:+.3f}")
    print(f"Test 1 (F1 vs permütasyon): p={rapor['test_1_f1_vs_permutasyon']['p_degeri']:.4f}  "
          f"{'ANLAMLI' if rapor['test_1_f1_vs_permutasyon']['anlamli_mi'] else 'anlamlı değil'}")
    print()
    print(f"Win Rate (FİLTRESİZ birincil sinyal, her zaman takip et): %{rapor['win_rate_filtresiz_birincil_ortalama']:.1f}")
    if rapor['win_rate_meta_filtreli_ortalama'] is not None:
        print(f"Win Rate (META-FİLTRELİ, sadece 'güven' dediği sinyaller): %{rapor['win_rate_meta_filtreli_ortalama']:.1f}")
        print(f"Test 2 (meta-filtreli vs filtresiz): p={rapor['test_2_meta_filtreli_vs_filtresiz']['p_degeri']:.4f}  "
              f"{'ANLAMLI' if rapor['test_2_meta_filtreli_vs_filtresiz']['anlamli_mi'] else 'anlamlı değil'}")
    print()
    print("Detaylar:")
    for d in sorted(rapor['detaylar'], key=lambda x: -x['win_rate_meta_filtreli']):
        print(f"  {d['sembol']:<10} n_sinyal={d['n_sinyal']:<5} "
              f"WinRate filtresiz=%{d['win_rate_filtresiz']:<6.1f} "
              f"WinRate meta-filtreli=%{d['win_rate_meta_filtreli']:<6.1f} "
              f"F1_gerçek={d['f1_gercek']:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    print(f"Tur 10 (Meta-Labeling) başlıyor — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    sonuclar = tum_kombinasyonlari_test_et()
    rapor = sonuclari_analiz_et(sonuclar)
    ozet_yazdir(rapor)

    cikti_dosyasi = os.path.join(os.path.dirname(__file__), 'tur10_meta_labeling_sonuclar.json')
    with open(cikti_dosyasi, 'w', encoding='utf-8') as f:
        json.dump(rapor, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Sonuçlar şuraya kaydedildi: {cikti_dosyasi}")
