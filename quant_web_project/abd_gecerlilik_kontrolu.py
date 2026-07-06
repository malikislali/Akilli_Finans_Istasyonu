"""
ABD Hisse Geçerlilik Kontrolü (S&P 500 + Nasdaq 100)
=======================================================
abd_ham_liste.txt'teki ~417 sembolün Yahoo Finance'te gerçekten veri
döndürüp döndürmediğini kontrol eder. BIST kontrolüyle AYNI mantık,
sadece .IS uzantısı eklenmez (ABD sembolleri uzantısız).

ÇALIŞTIRMA:
    python3 abd_gecerlilik_kontrolu.py

Kendi makinende (gerçek internet erişimi olan) çalıştırılmalı.

ÇIKTI: abd_gecerli_semboller.json
       abd_basarisiz_semboller.txt
"""

import json
import time
import yfinance as yf

GIRDI_DOSYASI = "abd_ham_liste.txt"
CIKTI_GECERLI = "abd_gecerli_semboller.json"
CIKTI_BASARISIZ = "abd_basarisiz_semboller.txt"
BEKLEME_SANIYE = 0.3


def sembol_listesini_oku(dosya_yolu: str) -> list:
    with open(dosya_yolu, encoding="utf-8") as f:
        return [satir.strip() for satir in f if satir.strip()]


def sembol_gecerli_mi(sembol: str) -> bool:
    try:
        df = yf.download(sembol, period="5d", progress=False, threads=False)
        return not df.empty and len(df) > 0
    except Exception:
        return False


def main():
    semboller = sembol_listesini_oku(GIRDI_DOSYASI)
    print(f"Toplam {len(semboller)} sembol kontrol edilecek...\n")

    gecerli, basarisiz = [], []

    for i, sembol in enumerate(semboller):
        if sembol_gecerli_mi(sembol):
            gecerli.append(sembol)
            print(f"[{i+1}/{len(semboller)}] ✅ {sembol}")
        else:
            basarisiz.append(sembol)
            print(f"[{i+1}/{len(semboller)}] ❌ {sembol}")
        time.sleep(BEKLEME_SANIYE)

    print(f"\n{'='*50}")
    print(f"Geçerli: {len(gecerli)} / {len(semboller)}")
    print(f"Başarısız: {len(basarisiz)} / {len(semboller)}")

    with open(CIKTI_GECERLI, "w", encoding="utf-8") as f:
        json.dump(gecerli, f, indent=2, ensure_ascii=False)
    with open(CIKTI_BASARISIZ, "w", encoding="utf-8") as f:
        f.write("\n".join(basarisiz))

    print(f"\nGeçerli semboller: {CIKTI_GECERLI}")
    print(f"Başarısız semboller: {CIKTI_BASARISIZ}")


if __name__ == "__main__":
    main()
