import ssl
import requests

# MEB Sertifika kontrolünü tamamen kör ediyoruz
requests.packages.urllib3.disable_warnings()
ssl._create_default_https_context = ssl._create_unverified_context

print("MEB Ağı Üzerinden Kripto Fiyatı Çekiliyor...\n")

# Coingecko'nun basit ve esnek API'sini kullanıyoruz
url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"

try:
    # verify=False diyerek sertifika kontrolünü kesin olarak kapatıyoruz
    cevap = requests.get(url, verify=False)
    veri = cevap.json()
    btc_fiyat = veri["bitcoin"]["usd"]
    print("=========================================")
    print(f" BAŞARILI! Canlı Bitcoin Fiyatı: {btc_fiyat} $")
    print("=========================================")
except Exception as e:
    print("Hata oluştu:", e)