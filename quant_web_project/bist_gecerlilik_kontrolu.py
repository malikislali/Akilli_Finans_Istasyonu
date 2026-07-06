"""
BIST Hisse Geçerlilik Kontrolü
================================
Bu script, bist_ham_liste.txt'teki ~464 hissenin Yahoo Finance'te
(.IS uzantılı sembol olarak) GERÇEKTEN veri döndürüp döndürmediğini
kontrol eder.

ÇALIŞTIRMA:
    python3 bist_gecerlilik_kontrolu.py

Bu, kendi makinende (gerçek internet erişimi olan) çalıştırılmalı —
sunucu/sandbox ortamında Yahoo Finance'e ağ erişimi olmayabilir.

ÇIKTI: bist_gecerli_semboller.json (Django VARLIK_HAVUZU'na eklenecek
       Python listesi formatında, .IS uzantılı, geçerli semboller)
       bist_basarisiz_semboller.txt (veri dönmeyen/hatalı semboller)
"""

import json
import time
import yfinance as yf

GIRDI_DOSYASI = "bist_ham_liste.txt"
CIKTI_GECERLI = "bist_gecerli_semboller.json"
CIKTI_BASARISIZ = "bist_basarisiz_semboller.txt"

# Her sembol arasında küçük bir bekleme — Yahoo'yu rate-limit'e
# düşürmemek için (çok hızlı art arda istek atmak bazen 429 hatası verir).
BEKLEME_SANIYE = 0.3


def kod_listesini_oku(dosya_yolu: str) -> list:
    kodlar = []
    with open(dosya_yolu, encoding="utf-8") as f:
        for satir in f:
            satir = satir.strip()
            if not satir or "," not in satir:
                continue
            kod = satir.split(",")[0].strip()
            kodlar.append(kod)
    return kodlar


def sembol_gecerli_mi(kod: str) -> bool:
    """.IS uzantılı sembolün Yahoo Finance'te veri döndürüp
    döndürmediğini kontrol eder (son 5 günlük veri ister)."""
    sembol = f"{kod}.IS"
    try:
        df = yf.download(sembol, period="5d", progress=False, threads=False)
        return not df.empty and len(df) > 0
    except Exception:
        return False


def main():
    kodlar = kod_listesini_oku(GIRDI_DOSYASI)
    print(f"Toplam {len(kodlar)} hisse kontrol edilecek...\n")

    gecerli = []
    basarisiz = []

    for i, kod in enumerate(kodlar):
        sembol = f"{kod}.IS"
        if sembol_gecerli_mi(kod):
            gecerli.append(sembol)
            print(f"[{i+1}/{len(kodlar)}] ✅ {sembol}")
        else:
            basarisiz.append(sembol)
            print(f"[{i+1}/{len(kodlar)}] ❌ {sembol}")
        time.sleep(BEKLEME_SANIYE)

    print(f"\n{'='*50}")
    print(f"Geçerli: {len(gecerli)} / {len(kodlar)}")
    print(f"Başarısız: {len(basarisiz)} / {len(kodlar)}")

    with open(CIKTI_GECERLI, "w", encoding="utf-8") as f:
        json.dump(gecerli, f, indent=2, ensure_ascii=False)
    with open(CIKTI_BASARISIZ, "w", encoding="utf-8") as f:
        f.write("\n".join(basarisiz))

    print(f"\nGeçerli semboller: {CIKTI_GECERLI}")
    print(f"Başarısız semboller: {CIKTI_BASARISIZ}")


if __name__ == "__main__":
    main()
