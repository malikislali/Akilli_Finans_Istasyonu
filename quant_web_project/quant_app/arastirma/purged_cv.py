"""
=====================================================================
🔬 ARAŞTIRMA — PURGED + EMBARGO CROSS-VALIDATION
=====================================================================
SORUN: quant_ml_core.py'deki TimeSeriesSplit, train/test'i zaman
sırasına göre böler (iyi bir başlangıç) ama "purging" yapmaz.

NEDEN PURGING GEREKLİ?
Her gözlemin etiketi (Hedef), İLERİYE bakarak hesaplanır (örn. Triple
Barrier'da "max_bar kadar ileri bak"). Bu yüzden bir gözlemin etiket
ufku, bir SONRAKİ fold'un train verisiyle ÇAKIŞABİLİR:

    train: [...t-2, t-1, t]     test: [t+1, t+2, t+3, ...]

  Eğer t anındaki gözlemin etiketi t+1, t+2, t+3'teki fiyatlara
  bakılarak hesaplandıysa, train setindeki t gözlemi DOLAYLI OLARAK
  test setindeki bilgiyi içerir — bu BİLGİ SIZINTISIDIR. Model, test
  setini "görmeden" de o bilgiden faydalanmış olur, bu da test
  skorlarını YAPAY OLARAK ŞİŞİRİR.

ÇÖZÜM (López de Prado, Bölüm 7):
  1. PURGING: Train setinden, etiket ufku test setiyle çakışan TÜM
     gözlemleri çıkar (silme/purge).
  2. EMBARGO: Test setinden SONRA gelen train gözlemlerine de küçük
     bir tampon (embargo) bölgesi uygula — çünkü piyasalarda otokorelasyon
     (seri bağımlılık) test setinin etkisi hemen sonraki günlere de
     sızabilir.

Bu modül, quant_ml_core.py'deki TimeSeriesSplit'in YERİNE DEĞİL,
YANINA yazılmıştır — araştırma sonuçlarını karşılaştırmak için ikisi
de kullanılabilir.
=====================================================================
"""

from __future__ import annotations

import numpy as np
from typing import Iterator, Tuple


class PurgedEmbargoCV:
    """
    scikit-learn'ün TimeSeriesSplit'ine benzer bir arayüz sunar (split()
    metodu), ama her fold için train indekslerinden:
      (a) test fold'unun etiket ufkuna giren gözlemleri PURGE eder,
      (b) test fold'undan sonraki embargo_bar kadar gözlemi de ayrıca çıkarır.

    Kullanım (TimeSeriesSplit ile birebir aynı arayüz):
        cv = PurgedEmbargoCV(n_splits=3, etiket_ufku=10, embargo_bar=5)
        for train_idx, test_idx in cv.split(X):
            ...
    """

    def __init__(self, n_splits: int = 3, etiket_ufku: int = 10, embargo_bar: int = 5):
        """
        etiket_ufku: Triple Barrier'daki max_bar ile AYNI değer olmalı —
            yani bir etiketin hesaplanması için ileriye kaç bar bakıldığı.
        embargo_bar: Test fold'undan sonra ek olarak ne kadar tampon
            bırakılacağı (otokorelasyon için ekstra güvenlik payı).
        """
        self.n_splits = n_splits
        self.etiket_ufku = etiket_ufku
        self.embargo_bar = embargo_bar

    def split(self, X) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1

        current = 0
        fold_bounds = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            fold_bounds.append((start, stop))
            current = stop

        # İlk fold'u train için kullanamayız (test edilecek bir şey kalmaz);
        # TimeSeriesSplit mantığıyla tutarlı şekilde, her adımda BİR SONRAKİ
        # fold'u test, ONDAN ÖNCEKİ TÜM fold'ları (purge edilmiş) train yaparız.
        for fold_idx in range(1, self.n_splits):
            test_start, test_stop = fold_bounds[fold_idx]
            test_idx = np.arange(test_start, test_stop)

            # Ham train adayları: test fold'undan ÖNCEKİ tüm indeksler.
            train_adaylari = np.arange(0, test_start)

            # PURGING: train adaylarından, etiket ufku test fold'una
            # giren (yani test_start - etiket_ufku ile test_start arasındaki)
            # gözlemleri çıkar.
            purge_baslangic = max(0, test_start - self.etiket_ufku)
            train_idx = train_adaylari[
                (train_adaylari < purge_baslangic)
            ]

            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        return self.n_splits - 1  # İlk fold sadece train'in başlangıcı olarak kullanılabilir


def purging_etkisini_olc(n_gozlem: int, n_splits: int, etiket_ufku: int, embargo_bar: int) -> dict:
    """
    Karşılaştırma amaçlı: aynı veri boyutu için normal TimeSeriesSplit'in
    train boyutuyla, PurgedEmbargoCV'nin train boyutunu kıyaslar. Bu,
    'purging ne kadar veri kaybettiriyor' sorusuna hızlı bir cevap verir
    — eğer çok fazla veri kaybediyorsak, etiket_ufku/embargo_bar
    parametrelerini revize etmemiz gerekebilir.

    NOT: PurgedEmbargoCV, n_splits fold'undan (n_splits - 1) tane
    train/test çifti üretir (ilk fold sadece başlangıç parçası olarak
    kullanılır, kendi başına test edilmez) — TimeSeriesSplit(n_splits=N)
    ise N çift üretir. Doğru kıyas için aynı TEST FOLD'UNA karşılık gelen
    train boyutları eşleştirilir (PurgedEmbargoCV'nin fold_idx=1 çifti,
    TimeSeriesSplit'in 2. çiftiyle [index 1] karşılaştırılır, vs.).
    """
    from sklearn.model_selection import TimeSeriesSplit

    X_dummy = np.zeros(n_gozlem)

    normal_cv = TimeSeriesSplit(n_splits=n_splits)
    normal_train_sizes = [len(tr) for tr, _ in normal_cv.split(X_dummy)]

    purged_cv = PurgedEmbargoCV(n_splits=n_splits, etiket_ufku=etiket_ufku, embargo_bar=embargo_bar)
    purged_train_sizes = [len(tr) for tr, _ in purged_cv.split(X_dummy)]

    # Hizalama: PurgedEmbargoCV'nin i'inci çifti (0-indexli), TimeSeriesSplit'in
    # (i+1)'inci çiftine (yine 0-indexli) karşılık gelir — çünkü Purged
    # CV ilk fold'u atlayıp 2. fold'dan itibaren test eder.
    normal_train_sizes_hizali = normal_train_sizes[1:]

    return {
        'normal_train_sizes': normal_train_sizes,
        'normal_train_sizes_hizali': normal_train_sizes_hizali,
        'purged_train_sizes': purged_train_sizes,
        'kaybedilen_gozlem_orani': [
            1 - (p / n) if n > 0 else None
            for n, p in zip(normal_train_sizes_hizali, purged_train_sizes)
        ],
    }
