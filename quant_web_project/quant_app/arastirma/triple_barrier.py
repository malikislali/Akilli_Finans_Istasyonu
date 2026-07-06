"""
=====================================================================
🔬 ARAŞTIRMA — TRIPLE BARRIER LABELING
=====================================================================
quant_ml_core.py'deki mevcut etiketleme yöntemi şuydu:

    Yuzde_Getiri_3G = Close.pct_change(3).shift(-3)
    Hedef = (Yuzde_Getiri_3G > 0.002).astype(int)

Bu NAİF bir yöntemdir çünkü:
  1. Sabit bir bar sayısı (3) kullanır — ama 1d'de 3 gün, 4h'de 12 saat
     gibi tamamen farklı zaman ufukları anlamına gelir.
  2. Sadece "3 bar sonra nerede" diye bakar — bu süre İÇİNDE fiyat çok
     daha fazla yükselip sonra düşmüş olabilir, ya da tam tersi. Gerçek
     bir trader pozisyonu önceden (stop-loss/take-profit ile) kapatır.
  3. Volatiliteyi hesaba katmaz — %0.2 eşiği, sakin bir günde de
     volatil bir günde de aynıdır; oysa volatil bir piyasada %0.2 anlamsız
     küçük bir hareket, sakin bir piyasada anlamlı bir hareket olabilir.

TRIPLE BARRIER LABELING (Marcos López de Prado, "Advances in Financial
Machine Learning", Bölüm 3) bu sorunları çözer: her gözlem noktasından
başlayarak ÜÇ bariyer çizilir:
  - ÜST bariyer (kâr-al): fiyat +k*ATR kadar yükselirse -> Hedef = 1
  - ALT bariyer (zarar-kes): fiyat -k*ATR kadar düşerse -> Hedef = 0
  - DİKEY bariyer (zaman aşımı): belirli bar sayısı içinde hiçbiri
    tetiklenmezse -> Hedef, o anki getiriye göre belirlenir (veya
    NaN/nötr olarak işaretlenip eğitimden çıkarılabilir)

Bariyerler ATR'ye göre ÖLÇEKLENDİĞİ için artık volatiliteye duyarlıdır
ve hangi bariyerin önce tetiklendiğine bakıldığı için gerçek bir
trader'ın "ilk gerçekleşen olay" mantığını simüle eder.
=====================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class TripleBarrierSonucu:
    hedef: pd.Series          # 1 = üst bariyer (kâr), 0 = alt bariyer (zarar), NaN = belirsiz/zaman aşımı
    gercek_getiri: pd.Series  # Bariyer tetiklendiğinde gerçekleşen getiri (yüzde)
    bariyer_tipi: pd.Series   # 'ust', 'alt', 'zaman_asimi' — hangi bariyer önce tetiklendi
    bar_sayisi: pd.Series     # Bariyer tetiklenene kadar geçen bar sayısı


def triple_barrier_etiketle(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr: pd.Series,
    kar_al_katsayisi: float = 2.0,
    zarar_kes_katsayisi: float = 1.5,
    max_bar: int = 10,
) -> TripleBarrierSonucu:
    """
    Her zaman noktası için, fiyatın ATR'ye göre ölçeklenmiş kâr-al/zarar-kes
    seviyelerine ulaşıp ulaşmadığını (ya da max_bar içinde hiçbirine
    ulaşmayıp zaman aşımına uğrayıp uğramadığını) ileriye bakarak hesaplar.

    NOT: Bu fonksiyon ileriye-dönük bakar (look-ahead) ama bu KASITLIDIR —
    etiketleme aşamasında "gelecekte ne olduğunu" bilmek gerekir (bu,
    modelin EĞİTİM verisini hazırlamak içindir, gerçek zamanlı tahmin
    için DEĞİLDİR). Eğitim/test ayrımı ayrı bir aşamada (purged CV)
    ele alınır.

    Parametreler:
        kar_al_katsayisi: Üst bariyer = giriş fiyatı + katsayı * ATR
        zarar_kes_katsayisi: Alt bariyer = giriş fiyatı - katsayı * ATR
        max_bar: Bu kadar bar içinde hiçbir bariyer tetiklenmezse zaman aşımı

    NEDEN kar_al_katsayisi (2.0) > zarar_kes_katsayisi (1.5)?
    Bu, V61/V60'taki risk yönetimi mantığıyla TUTARLI bir seçim
    (dashboard'daki "Risk/Ödül Oranı: 1:2.0" ile aynı felsefe) — gerçek
    bir trader da genelde kâr hedefini zarar limitinden daha geniş tutar.
    """
    n = len(close)
    hedef = np.full(n, np.nan)
    gercek_getiri = np.full(n, np.nan)
    bariyer_tipi = np.array([None] * n, dtype=object)
    bar_sayisi = np.full(n, np.nan)

    close_arr = close.values
    high_arr = high.values
    low_arr = low.values
    atr_arr = atr.values

    for i in range(n - 1):
        giris_fiyat = close_arr[i]
        atr_degeri = atr_arr[i]
        if np.isnan(atr_degeri) or atr_degeri <= 0:
            continue

        ust_bariyer = giris_fiyat + kar_al_katsayisi * atr_degeri
        alt_bariyer = giris_fiyat - zarar_kes_katsayisi * atr_degeri

        bitis_idx = min(i + 1 + max_bar, n)
        ust_tetiklendi = False
        alt_tetiklendi = False

        for j in range(i + 1, bitis_idx):
            # Aynı barda HEM üst HEM alt bariyer tetiklenebilir (yüksek
            # volatiliteli bir mumda) — bu durumda KONSERVATİF yaklaşımla
            # önce zarar-kes'in tetiklendiğini varsayıyoruz (gerçek
            # hayatta intrabar sıralamayı bilemeyiz, ama bu varsayım
            # modeli optimistik göstermez, aksine daha temkinli yapar).
            if low_arr[j] <= alt_bariyer:
                hedef[i] = 0
                gercek_getiri[i] = (alt_bariyer - giris_fiyat) / giris_fiyat
                bariyer_tipi[i] = 'alt'
                bar_sayisi[i] = j - i
                alt_tetiklendi = True
                break
            if high_arr[j] >= ust_bariyer:
                hedef[i] = 1
                gercek_getiri[i] = (ust_bariyer - giris_fiyat) / giris_fiyat
                bariyer_tipi[i] = 'ust'
                bar_sayisi[i] = j - i
                ust_tetiklendi = True
                break

        if not ust_tetiklendi and not alt_tetiklendi:
            # Zaman aşımı: max_bar içinde hiçbir bariyer tetiklenmedi.
            # Bu durumda, zaman aşımı anındaki GERÇEK getiriye bakıp
            # işareti pozitifse 1, negatifse 0 olarak etiketliyoruz
            # (López de Prado'nun "zaman aşımında mevcut getiriyi kullan"
            # önerisiyle tutarlı).
            son_idx = bitis_idx - 1
            if son_idx > i:
                zaman_asimi_getiri = (close_arr[son_idx] - giris_fiyat) / giris_fiyat
                hedef[i] = 1 if zaman_asimi_getiri > 0 else 0
                gercek_getiri[i] = zaman_asimi_getiri
                bariyer_tipi[i] = 'zaman_asimi'
                bar_sayisi[i] = son_idx - i

    return TripleBarrierSonucu(
        hedef=pd.Series(hedef, index=close.index),
        gercek_getiri=pd.Series(gercek_getiri, index=close.index),
        bariyer_tipi=pd.Series(bariyer_tipi, index=close.index),
        bar_sayisi=pd.Series(bar_sayisi, index=close.index),
    )


def etiket_dagilimi_ozet(sonuc: TripleBarrierSonucu) -> dict:
    """Triple barrier sonucunun hızlı bir özet istatistiğini döner —
    araştırma sırasında 'etiketler dengeli mi, çok mu zaman aşımına
    uğruyor' gibi soruları hemen cevaplamak için kullanışlıdır."""
    gecerli = sonuc.hedef.dropna()
    bariyer_sayim = sonuc.bariyer_tipi.value_counts(dropna=True).to_dict()
    return {
        'toplam_gozlem': len(sonuc.hedef),
        'gecerli_etiket_sayisi': len(gecerli),
        'pozitif_oran': float(gecerli.mean()) if len(gecerli) > 0 else None,
        'bariyer_dagilimi': bariyer_sayim,
        'ortalama_bar_sayisi': float(sonuc.bar_sayisi.dropna().mean()) if sonuc.bar_sayisi.notna().any() else None,
    }
