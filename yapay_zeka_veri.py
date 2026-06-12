import yfinance as yf
import pandas as pd
import pandas_ta as ta
import urllib.request
import ssl

print("==================================================")
print("  🤖 QUANT AI - FINANSAL ÖZELLİK MOTORU BAŞLADI 🤖  ")
print("==================================================")

# 1. MEB / Kurumsal Ağ Filtre Güvenliği (Sertifika Bypass)
try:
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib.request.urlopen('https://query1.finance.yahoo.com', timeout=5)
except Exception:
    pass

# 2. Canlı Veri Akışını Çekme (Yapay zeka için son 1 yıllık günlük veri idealdir)
sembol = "BTC-USD"
print(f"-> {sembol} için geçmiş veriler indiriliyor...")
ticker = yf.Ticker(sembol)
df = ticker.history(period="1y", interval="1d")

if df.empty:
    print("[HATA] Veri çekilemedi. İnternet bağlantısını veya sembolü kontrol edin!")
    exit()

print(f"-> {len(df)} günlük ham veri başarıyla alındı. İndikatörler hesaplanıyor...\n")

# =====================================================================
# 📈 BÖLÜM 1: TREND TAKİPÇİLERİ VE HAREKETLİ ORTALAMALAR (FEATURES)
# =====================================================================

# SMA, EMA, WMA ve DEMA Ortalamalar
df['SMA_20'] = ta.sma(df['Close'], length=20)
df['EMA_9'] = ta.ema(df['Close'], length=9)
df['EMA_21'] = ta.ema(df['Close'], length=21)
df['WMA_20'] = ta.wma(df['Close'], length=20)
df['DEMA_20'] = ta.dema(df['Close'], length=20)

# Bollinger Bantları (İsim krizini elle eşitleyerek kökten çözüyoruz hoca)
bbands = ta.bbands(df['Close'], length=20, std=2)
df['Bollinger_Ust'] = bbands.iloc[:, 2] # Kütüphane ne isim verirse versin 3. sütun üst banttır

# Keltner Kanalı
keltner = ta.kc(df['High'], df['Low'], df['Close'], length=20)
df['Keltner_Ust'] = keltner.iloc[:, 2]

# Donchian Kanalı
donchian = ta.donchian(df['High'], df['Low'], lower_length=20, upper_length=20)
df['Donchian_Ust'] = donchian.iloc[:, 1]


# =====================================================================
# 📊 BÖLÜM 2: OSİLATÖRLER (AŞIRI ALIM / SATIM VE MOMENTUM)
# =====================================================================

# RSI (Göreceli Güç Endeksi)
df['RSI_14'] = ta.rsi(df['Close'], length=14)

# MACD (Burada da ilk sütunu doğrudan alıp adını biz koyuyoruz)
macd_veri = ta.macd(df['Close'], fast=12, slow=26, signal=9)
df['MACD_Ana'] = macd_veri.iloc[:, 0]

# Stochastic Osilatör
stoch_veri = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
df['Stoch_K'] = stoch_veri.iloc[:, 0]

# CCI ve ATR
df['CCI_14'] = ta.cci(df['High'], df['Low'], df['Close'], length=14)
df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)


# =====================================================================
# 🧼 BÖLÜM 3: VERİ TEMİZLİĞİ
# =====================================================================
df_temiz = df.dropna().copy()

print("✅ [MÜKEMMEL] Tüm indikatörler ve osilatörler başarıyla hesaplandı!")
print(f"Toplam eğitilebilir gün sayısı: {len(df_temiz)}\n")


# =====================================================================
# 🎯 BÖLÜM 4: YAPAY ZEKA HEDEFİNİ (TARGET) TANIMLAMA VE BÜYÜK ÖNİZLEME
# =====================================================================

# shift(-1) yaparak yarınki kapanış fiyatını bugünün satırına getiriyoruz
df_temiz['Yarin_Kapanis'] = df_temiz['Close'].shift(-1)

# Yarınki fiyat bugünkünden büyükse 1 (AL), değilse 0 (SAT)
df_temiz['Hedef'] = (df_temiz['Yarin_Kapanis'] > df_temiz['Close']).astype(int)

# Son satırı temizliyoruz
df_final = df_temiz.dropna().copy()

print("🎯 [HEDEF KİLİTLENDİ] Yapay zeka yön tahmin sinyalleri (0/1) oluşturuldu.")
print(f"Eğitime girecek nihai veri seti boyutu (Satır, Sütun): {df_final.shape}\n")

# İşte bizim kendi koyduğumuz, asla şaşmayacak gıcır gıcır sütun isimleri listesi:
gosterilecek_sutunlar = ['Close', 'EMA_9', 'Bollinger_Ust', 'RSI_14', 'MACD_Original' if 'MACD_Original' in df_final.columns else 'MACD_Ana', 'ATR_14', 'Hedef']

print("--- SON 3 GÜNÜN NİHAİ YAPAY ZEKA VERİ SETİ (EKSİKSİZ MATRİS) ---")
print(df_final[gosterilecek_sutunlar].tail(3))
print("==================================================")