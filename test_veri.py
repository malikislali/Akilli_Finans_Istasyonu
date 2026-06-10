import yfinance as yf
import ssl
import requests

requests.packages.urllib3.disable_warnings()
ssl._create_default_https_context = ssl._create_unverified_context

print("=========================================")
print("  AKILLI BİLGİ İSTASYONU VERİ TESTİ  ")
print("=========================================\n")

# Test için Bitcoin (BTC-USD) verisi çekiyoruz. 
# İleride bunu Altın (GC=F) veya Garanti Bankası (GARAN.IS) yapabiliriz.
veri_kaynagi = yf.Ticker("BTC-USD")

# Son 1 günün, saatlik (1h) verilerini getir diyoruz
guncel_veri = veri_kaynagi.history(period="1d", interval="1h")

# Ekranımıza gelen son veriyi basıyoruz
print(guncel_veri.tail(1))

print("\n=========================================")
print(" Başarılı! Canlı veri musluğu akıyor.")
print("=========================================")