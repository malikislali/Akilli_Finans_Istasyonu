"""
=====================================================================
🔬 ARAŞTIRMA TUR 7 — LSTM / DERİN ÖĞRENME TESTİ
=====================================================================
ChatGPT'nin kod değerlendirmesi (kullanıcı tarafından paylaşıldı) şunu
önerdi: "LSTM/Temporal Fusion Transformer/N-BEATS gibi derin öğrenme
modelleri eksik." Bu turda bu öneriyi TEST EDİYORUZ.

ÖNEMLİ TASARIM KARARI — "AYNI YAKIT, FARKLI MOTOR":
Bu turda SADECE model mimarisi değişiyor (GBM/RF/XGB -> LSTM). Feature
seti (aynı 19 klasik gösterge) ve etiketleme yöntemi (Triple Barrier,
max_bar=20) AYNI tutuluyor. Bunun nedeni metodolojik temizlik: eğer
hem feature'ları hem modeli aynı anda değiştirirsek, hangi değişikliğin
(varsa) etkili olduğunu ayırt edemeyiz. Feature mühendisliği SONRAKİ
turda (Tur 8) ayrıca test edilecek.

LSTM NEDEN FARKLI DAVRANABİLİR:
Ağaç-tabanlı modeller (GBM/RF/XGB), her satırı BAĞIMSIZ bir gözlem
olarak görür — bir günün RSI'ı ile 5 gün önceki RSI arasındaki SIRALI
ilişkiyi doğrudan modellemez (sadece o anki feature değerlerine bakar).
LSTM, son N barlık bir PENCEREYİ (sequence) tek bir girdi olarak alır
ve zaman içindeki sıralı bağımlılıkları (momentum, kalıp tekrarı)
yakalamaya çalışır. Teorik olarak, eğer piyasada "olay A'dan sonra B
gelir" tarzı bir sıralı kalıp varsa, LSTM bunu ağaç modellerinden daha
iyi görebilir.

DÜRÜST BEKLENTİ YÖNETİMİ (peşin uyarı):
  1. Veri miktarı sınırı: LSTM tipik olarak on binlerce örnek ister;
     bizim en uzun serimiz ~1000-1500 satır. Overfitting riski yüksek.
  2. Aynı feature'lar kullanıldığı için, sinyal zaten feature'larda
     yoksa (Tur 1-6'nın bulgusu), LSTM de onu "icat edemez" — sadece
     var olan ilişkiyi farklı bir şekilde modeller.
  3. Bu nedenle, bu tur "kesin işe yaramaz" ön yargısıyla DEĞİL, ama
     gerçekçi bir beklenti ("muhtemelen ağaç modellerinden çok farklı
     olmayacak") ile yürütülüyor — ki bu, ChatGPT'nin kendi yorumundaki
     "finans piyasasında çoğu zaman XGBoost zaten daha iyi çalışıyor"
     tespitiyle de örtüşüyor.
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
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

import quant_web_project.quant_app.quant_ml_coreBinanceli as core
from arastirma.triple_barrier import triple_barrier_etiketle
from arastirma.purged_cv import PurgedEmbargoCV

torch.manual_seed(42)
np.random.seed(42)

PENCERE_BOYU = 20  # LSTM'e verilecek sıralı pencere uzunluğu (son 20 bar)
FEATURE_KOLONLARI = core.FEATURE_KOLONLARI  # Tur 1-6 ile AYNI feature seti


# =====================================================================
# 🧠 LSTM MODEL MİMARİSİ
# =====================================================================
class BasitLSTM(nn.Module):
    """
    Küçük, sade bir LSTM sınıflandırıcı. Kasıtlı olarak BASİT tutuldu
    (1 katman, küçük hidden boyutu) çünkü veri miktarımız küçük —
    karmaşık bir mimari (çok katmanlı, büyük hidden) overfitting riskini
    daha da artırır.
    """
    def __init__(self, n_features: int, hidden_boyutu: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_boyutu, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_boyutu, 1)

    def forward(self, x):
        # x: (batch, pencere_boyu, n_features)
        _, (h_n, _) = self.lstm(x)
        son_gizli_durum = h_n[-1]  # (batch, hidden_boyutu) — pencerenin SON adımındaki özet
        son_gizli_durum = self.dropout(son_gizli_durum)
        return self.fc(son_gizli_durum).squeeze(-1)  # (batch,) — ham logit (sigmoid UYGULANMADI)


def pencereli_veri_olustur(X: pd.DataFrame, y: pd.Series, pencere_boyu: int):
    """
    Düz (her satır bağımsız) feature tablosunu, LSTM'in beklediği
    3 boyutlu (örnek, zaman_adımı, feature) tensöre çevirir. i'inci
    örnek, [i-pencere_boyu+1, i] aralığındaki barların feature dizisidir
    ve etiketi y[i]'dir (yani pencerenin SON gününün etiketi).
    """
    X_arr = X.values.astype(np.float32)
    y_arr = y.values.astype(np.float32)
    n = len(X_arr)

    pencereler = []
    etiketler = []
    orijinal_indeksler = []

    for i in range(pencere_boyu - 1, n):
        pencereler.append(X_arr[i - pencere_boyu + 1: i + 1])
        etiketler.append(y_arr[i])
        orijinal_indeksler.append(i)

    return np.array(pencereler), np.array(etiketler), np.array(orijinal_indeksler)


def lstm_egit_ve_test_et(
    X_train_pencere, y_train_pencere, X_test_pencere, y_test_pencere,
    epoch_sayisi: int = 30, lr: float = 0.005,
):
    """LSTM'i eğitir, test setinde tahmin üretir. Basit bir eğitim
    döngüsü — production-grade değil, araştırma amaçlı."""
    n_features = X_train_pencere.shape[2]
    model = BasitLSTM(n_features=n_features)

    pos_oran = float(np.mean(y_train_pencere))
    pos_oran = min(max(pos_oran, 0.05), 0.95)
    pos_weight = torch.tensor((1 - pos_oran) / pos_oran, dtype=torch.float32)
    kriter = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_tr_t = torch.tensor(X_train_pencere, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train_pencere, dtype=torch.float32)

    model.train()
    for epoch in range(epoch_sayisi):
        optimizer.zero_grad()
        logits = model(X_tr_t)
        loss = kriter(logits, y_tr_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        X_te_t = torch.tensor(X_test_pencere, dtype=torch.float32)
        test_logits = model(X_te_t)
        test_probs = torch.sigmoid(test_logits).numpy()
        test_preds = (test_probs >= 0.5).astype(int)

    return test_preds, test_probs


@dataclass
class LSTMTestSonucu:
    sembol: str
    pazar: str
    interval: str
    basarili: bool
    hata_mesaji: Optional[str] = None
    win_rate_gercek: float = 0.0
    win_rate_permutasyon: float = 0.0
    sharpe_gercek: float = 0.0
    n_test_gozlem: int = 0


def lstm_tek_kombinasyon_test_et(
    sembol: str, pazar: str, interval: str,
    n_splits: int = 3, rastgele_seed: int = 42,
) -> LSTMTestSonucu:
    """
    Tek bir varlık/periyot kombinasyonu için LSTM tabanlı TAM araştırma
    akışını çalıştırır — coklu_test.py'deki tek_kombinasyon_test_et ile
    AYNI iskelet (veri çek, etiketle, Purged CV, permütasyon testi), ama
    model GBM yerine LSTM, ve veri pencereli (sequence) formata çevriliyor.
    """
    try:
        period = core.suggest_period(pazar, interval)
        # 🐛 DÜZELTME: TR_HISSE'nin varsayılan penceresi (1y, ~252 gün) LSTM
        # için yetersiz kalıyor — calculate_metrics'in ısınma periyotları
        # (SMA200 vb.) 252 satırı ~127'ye düşürüyor, bu da minimum eşiğin
        # (150) altında kalıp TÜM TR_HISSE kombinasyonlarını SESSİZCE
        # araştırma dışı bırakıyordu (Tur 7'nin ilk çalıştırmasında olan
        # tam olarak buydu — 3/3 TR_HISSE kombinasyonu "başarısız" oldu).
        # LSTM'in pencereleme+CV ihtiyacı için TR_HISSE'yeburada özel
        # olarak daha uzun bir pencere veriyoruz (üretim kodundaki
        # suggest_period DEĞİŞTİRİLMİYOR, sadece bu araştırma scriptinde
        # override ediliyor).
        if pazar == "TR_HISSE" and interval == "1d":
            period = "3y"
        fetch_result = core.get_market_data(sembol, period, interval, pazar)
        df_raw = fetch_result.df

        if df_raw.empty:
            return LSTMTestSonucu(sembol=sembol, pazar=pazar, interval=interval,
                                   basarili=False, hata_mesaji="Veri çekilemedi.")

        df_active, _ = core.calculate_metrics(df_raw)
        if df_active.empty or len(df_active) < 150:  # LSTM için minimum daha yüksek (pencere+CV payı)
            return LSTMTestSonucu(sembol=sembol, pazar=pazar, interval=interval,
                                   basarili=False, hata_mesaji=f"Veri çok az ({len(df_active)} satır, LSTM için en az 150 gerekli).")

        tb_sonuc = triple_barrier_etiketle(
            close=df_active['Close'], high=df_active['High'], low=df_active['Low'],
            atr=df_active['ATR'], kar_al_katsayisi=2.0, zarar_kes_katsayisi=1.5, max_bar=20,
        )
        df_active = df_active.copy()
        df_active['Hedef_TB'] = tb_sonuc.hedef
        df_active['Getiri_TB'] = tb_sonuc.gercek_getiri

        df_ml = df_active.dropna(subset=['Hedef_TB'] + FEATURE_KOLONLARI).copy()
        if len(df_ml) < 150:
            return LSTMTestSonucu(sembol=sembol, pazar=pazar, interval=interval,
                                   basarili=False, hata_mesaji=f"Etiketleme sonrası veri çok az ({len(df_ml)}).")

        X_all = df_ml[FEATURE_KOLONLARI].copy()
        y_all = df_ml['Hedef_TB'].astype(int)
        getiri_all = df_ml['Getiri_TB'].values

        # Pencereli veriye çevir (LSTM girdisi)
        X_pencereli, y_pencereli, orj_idx = pencereli_veri_olustur(X_all, y_all, PENCERE_BOYU)
        getiri_pencereli = getiri_all[orj_idx]

        if len(X_pencereli) < 100:
            return LSTMTestSonucu(sembol=sembol, pazar=pazar, interval=interval,
                                   basarili=False, hata_mesaji=f"Pencereleme sonrası veri çok az ({len(X_pencereli)}).")

        cv = PurgedEmbargoCV(n_splits=n_splits, etiket_ufku=20)

        win_rates_gercek, sharpes_gercek = [], []
        for tr_idx, te_idx in cv.split(X_pencereli):
            if len(tr_idx) < 50 or len(te_idx) < 15:
                continue

            X_tr, X_te = X_pencereli[tr_idx], X_pencereli[te_idx]
            y_tr, y_te = y_pencereli[tr_idx], y_pencereli[te_idx]

            # Ölçeklendirme: pencere boyutunu koruyarak feature ekseninde ölçeklendir
            scaler = StandardScaler()
            n_tr, pencere, n_feat = X_tr.shape
            X_tr_sc = scaler.fit_transform(X_tr.reshape(-1, n_feat)).reshape(n_tr, pencere, n_feat)
            n_te = X_te.shape[0]
            X_te_sc = scaler.transform(X_te.reshape(-1, n_feat)).reshape(n_te, pencere, n_feat)

            preds, _ = lstm_egit_ve_test_et(X_tr_sc, y_tr, X_te_sc, y_te)

            win_rate = float(np.mean(preds == y_te)) * 100
            test_getiri = getiri_pencereli[te_idx]
            strat_returns = np.where(preds == 1, test_getiri, -test_getiri)
            sharpe = (
                (np.mean(strat_returns) / np.std(strat_returns)) * np.sqrt(252)
                if np.std(strat_returns) > 0 else 0.0
            )
            win_rates_gercek.append(win_rate)
            sharpes_gercek.append(sharpe)

        if not win_rates_gercek:
            return LSTMTestSonucu(sembol=sembol, pazar=pazar, interval=interval,
                                   basarili=False, hata_mesaji="Hiçbir CV fold'u yeterli veri içermedi.")

        # Permütasyon testi
        rng = np.random.default_rng(rastgele_seed)
        y_permute = rng.permutation(y_pencereli)

        win_rates_permute, sharpes_permute = [], []
        for tr_idx, te_idx in cv.split(X_pencereli):
            if len(tr_idx) < 50 or len(te_idx) < 15:
                continue

            X_tr, X_te = X_pencereli[tr_idx], X_pencereli[te_idx]
            y_tr, y_te = y_permute[tr_idx], y_permute[te_idx]

            scaler = StandardScaler()
            n_tr, pencere, n_feat = X_tr.shape
            X_tr_sc = scaler.fit_transform(X_tr.reshape(-1, n_feat)).reshape(n_tr, pencere, n_feat)
            n_te = X_te.shape[0]
            X_te_sc = scaler.transform(X_te.reshape(-1, n_feat)).reshape(n_te, pencere, n_feat)

            preds, _ = lstm_egit_ve_test_et(X_tr_sc, y_tr, X_te_sc, y_te)

            win_rate = float(np.mean(preds == y_te)) * 100
            test_getiri = getiri_pencereli[te_idx]
            strat_returns = np.where(preds == 1, test_getiri, -test_getiri)
            sharpe = (
                (np.mean(strat_returns) / np.std(strat_returns)) * np.sqrt(252)
                if np.std(strat_returns) > 0 else 0.0
            )
            win_rates_permute.append(win_rate)
            sharpes_permute.append(sharpe)

        return LSTMTestSonucu(
            sembol=sembol, pazar=pazar, interval=interval, basarili=True,
            win_rate_gercek=float(np.mean(win_rates_gercek)),
            win_rate_permutasyon=float(np.mean(win_rates_permute)) if win_rates_permute else 0.0,
            sharpe_gercek=float(np.mean(sharpes_gercek)),
            n_test_gozlem=len(df_ml),
        )

    except Exception as exc:
        return LSTMTestSonucu(sembol=sembol, pazar=pazar, interval=interval,
                               basarili=False, hata_mesaji=f"Beklenmeyen hata: {exc}")


# Tur 1'deki TEST_KOMBINASYONLARI'nın bir alt kümesi — LSTM eğitimi her
# kombinasyon için daha YAVAŞ olduğundan (epoch döngüsü), kapsamı
# makul tutmak için pazar başına temsilci varlıklar seçildi.
#
# 🆕 TUR 7.2 — ÖRNEKLEM BÜYÜTME: İlk çalıştırmada n=11 ile p=0.020 (anlamlı)
# çıktı, ama TR_HISSE araştırmasında (Tur 5->6) n=29'dan n=47'ye çıkınca
# anlamlılık kaybolmuştu (p=0.056 -> p=0.159). Aynı riski LSTM için de
# kontrol etmek için örneklem ~30'a çıkarılıyor (19 YENİ kombinasyon
# eklendi, ilk 11 ile SIFIR çakışma).
LSTM_TEST_KOMBINASYONLARI = [
    # --- İlk turda test edilenler (11) ---
    ("BTC-USD", "KRIPTO", "1d"), ("ETH-USD", "KRIPTO", "1d"), ("SOL-USD", "KRIPTO", "1d"),
    ("THYAO.IS", "TR_HISSE", "1d"), ("GARAN.IS", "TR_HISSE", "1d"), ("BIMAS.IS", "TR_HISSE", "1d"),
    ("AAPL", "ABD_HISSE", "1d"), ("MSFT", "ABD_HISSE", "1d"), ("TSLA", "ABD_HISSE", "1d"),
    ("GC=F", "EMTIA", "1d"), ("CL=F", "EMTIA", "1d"),

    # --- 🆕 19 YENİ kombinasyon (çakışma yok) ---
    ("AVAX-USD", "KRIPTO", "1d"), ("XRP-USD", "KRIPTO", "1d"), ("DOGE-USD", "KRIPTO", "1d"),
    ("LINK-USD", "KRIPTO", "1d"), ("FIL-USD", "KRIPTO", "1d"),

    ("ASELS.IS", "TR_HISSE", "1d"), ("EREGL.IS", "TR_HISSE", "1d"), ("SISE.IS", "TR_HISSE", "1d"),
    ("TUPRS.IS", "TR_HISSE", "1d"), ("KCHOL.IS", "TR_HISSE", "1d"), ("AKBNK.IS", "TR_HISSE", "1d"),

    ("AMZN", "ABD_HISSE", "1d"), ("META", "ABD_HISSE", "1d"), ("GOOGL", "ABD_HISSE", "1d"),
    ("NVDA", "ABD_HISSE", "1d"),

    ("SI=F", "EMTIA", "1d"), ("NG=F", "EMTIA", "1d"),
]


def tum_kombinasyonlari_test_et(kombinasyonlar=None, ilerleme_yazdir=True):
    if kombinasyonlar is None:
        kombinasyonlar = LSTM_TEST_KOMBINASYONLARI
    sonuclar = []
    for i, (sembol, pazar, interval) in enumerate(kombinasyonlar):
        if ilerleme_yazdir:
            print(f"[{i+1}/{len(kombinasyonlar)}] {sembol} ({pazar}, {interval}) LSTM ile test ediliyor...")
        sonuc = lstm_tek_kombinasyon_test_et(sembol, pazar, interval)
        sonuclar.append(sonuc)
        if ilerleme_yazdir:
            if sonuc.basarili:
                print(f"    -> Win Rate gerçek: %{sonuc.win_rate_gercek:.1f}  "
                      f"permütasyon: %{sonuc.win_rate_permutasyon:.1f}  "
                      f"fark: {sonuc.win_rate_gercek - sonuc.win_rate_permutasyon:+.1f}  "
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
    print("📊 TUR 7 — LSTM SONUÇ ÖZETİ")
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
    print("Detaylar:")
    for d in sorted(rapor['detaylar'], key=lambda x: -x['fark']):
        print(f"  {d['sembol']:<10} ({d['pazar']:<10} {d['interval']}) fark={d['fark']:+6.1f}  sharpe={d['sharpe']:+.2f}")
    print("=" * 70)
    print()
    print("🎯 KARŞILAŞTIRMA (Tur 6 ağaç-tabanlı modeller ile):")
    print("  Tur 6 (GBM/RF/XGB, 47 TR_HISSE): p(vs permütasyon)=0.1594 (anlamlı değil)")
    print(f"  Tur 7 (LSTM, {rapor['basarili_test_sayisi']} kombinasyon, çoklu pazar): "
          f"p(vs permütasyon)={rapor['test_2_vs_permutasyon']['p_degeri']:.4f} "
          f"({'ANLAMLI' if rapor['test_2_vs_permutasyon']['anlamli_mi'] else 'anlamlı değil'})")


if __name__ == "__main__":
    print(f"Tur 7 (LSTM) başlıyor — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"PyTorch versiyonu: {torch.__version__}")
    print()

    sonuclar = tum_kombinasyonlari_test_et()
    rapor = sonuclari_analiz_et(sonuclar)
    ozet_yazdir(rapor)

    cikti_dosyasi = os.path.join(os.path.dirname(__file__), 'tur7_lstm_sonuclar.json')
    with open(cikti_dosyasi, 'w', encoding='utf-8') as f:
        json.dump(rapor, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Sonuçlar şuraya kaydedildi: {cikti_dosyasi}")
