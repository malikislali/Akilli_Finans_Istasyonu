import yfinance as yf
import pandas as pd
import pandas_ta as ta
import urllib.request
import ssl

# --- 📐 TERMINAL GÖRÜNTÜ AYARLARI (SANSÜRSÜZ GÖRÜNÜM) ---
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("==========================================================")
print("  🤖 QUANT AI - EVRENSEL FİNANSAL ÖZELLİK MOTORU BAŞLADI 🤖  ")
print("==========================================================")

# 1. MEB / Kurumsal Ağ Filtre Güvenliği (Sertifika Bypass)
try:
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib.request.urlopen('https://query1.finance.yahoo.com', timeout=5)
except Exception:
    pass

# =====================================================================
# ⚙️ KULLANICI GİRİŞ ALANI (Sadece Burayı Değiştir Hoca!)
# =====================================================================
# Test etmek istediğin varlığın başındaki '#' işaretini kaldır, diğerlerini kapat!

secilen_pazar = "KRIPTO"        # Seçenekler: "KRIPTO", "ABD_HISSE", "TR_HISSE", "EMTIA"
secilen_sembol = "BTC-USD"      # Örn: BTC-USD, NVDA, THYAO.IS, GC=F

# =====================================================================
# 🧭 OTOMATİK DÖNGÜ VE PERİYOT YÖNETİMİ (Zehirlenme Önleyici Sistem)
# =====================================================================
pazar_ayarlari = {
    "KRIPTO":    {"period": "3y", "aciklama": "Kripto 4 Yıllık Döngü Modu (Dengeli Bağışıklık)"},
    "ABD_HISSE": {"period": "4y", "aciklama": "ABD Kurumsal Bilanço Modu (Derin Algoritma)"},
    "TR_HISSE":  {"period": "1y", "aciklama": "BIST Makro-Enflasyon Hassas Modu (Yakın Hafıza)"},
    "EMTIA":     {"period": "5y", "aciklama": "Kıymetli Maden Makro Trend Modu (Ağır Abiler)"}
}

ayar = pazar_ayarlari.get(secilen_pazar, {"period": "1y", "aciklama": "Standart Mod"})

print(f"🌍 Aktif Pazar: {secilen_pazar} | {ayar['aciklama']}")
print(f"🎛️ Seçilen Sembol: {secilen_sembol} | Analiz Süresi: {ayar['period']}")
print("----------------------------------------------------------")

# Veri Çekme İşlemi
print(f"-> {secilen_sembol} için geçmiş veriler indiriliyor...")
ticker = yf.Ticker(secilen_sembol)
df = ticker.history(period=ayar['period'], interval="1d")

if df.empty:
    print(f"[HATA] {secilen_sembol} için veri çekilemedi! Sembolü kontrol edin.")
    exit()

print(f"-> {len(df)} günlük ham veri başarıyla alındı. İndikatörler hesaplanıyor...\n")


# =====================================================================
# 📈 BÖLÜM 1: TREND TAKİPÇİLERİ VE HAREKETLİ ORTALAMALAR (FEATURES)
# =====================================================================
df['SMA_20'] = ta.sma(df['Close'], length=20)
df['EMA_9'] = ta.ema(df['Close'], length=9)
df['EMA_21'] = ta.ema(df['Close'], length=21)
df['WMA_20'] = ta.wma(df['Close'], length=20)
df['DEMA_20'] = ta.dema(df['Close'], length=20)

# Bollinger Bantları (Pozisyona göre dinamik eşitleme)
bbands = ta.bbands(df['Close'], length=20, std=2)
df['Bollinger_Ust'] = bbands.iloc[:, 2]

# Keltner Kanalı
keltner = ta.kc(df['High'], df['Low'], df['Close'], length=20)
df['Keltner_Ust'] = keltner.iloc[:, 2]

# Donchian Kanalı
donchian = ta.donchian(df['High'], df['Low'], lower_length=20, upper_length=20)
df['Donchian_Ust'] = donchian.iloc[:, 1]


# =====================================================================
# 📊 BÖLÜM 2: OSİLATÖRLER (AŞIRI ALIM / SATIM VE MOMENTUM)
# =====================================================================
df['RSI_14'] = ta.rsi(df['Close'], length=14)

# MACD
macd_veri = ta.macd(df['Close'], fast=12, slow=26, signal=9)
df['MACD_Ana'] = macd_veri.iloc[:, 0]

# Stochastic Osilatör
stoch_veri = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
df['Stoch_K'] = stoch_veri.iloc[:, 0]

df['CCI_14'] = ta.cci(df['High'], df['Low'], df['Close'], length=14)
df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)


# =====================================================================
# 🧼 BÖLÜM 3: VERİ TEMİZLİĞİ
# =====================================================================
df_temiz = df.dropna().copy()
print("✅ [MÜKEMMEL] Tüm indikatörler ve osilatörler başarıyla hesaplandı!")


# =====================================================================
# 🎯 BÖLÜM 4: YAPAY ZEKA HEDEFİNİ (TARGET) TANIMLAMA VE BÜYÜK ÖNİZLEME
# =====================================================================
# Yarınki kapanışı bugüne getirip yön sinyali (0/1) üretiyoruz
df_temiz['Yarin_Kapanis'] = df_temiz['Close'].shift(-1)
df_temiz['Hedef'] = (df_temiz['Yarin_Kapanis'] > df_temiz['Close']).astype(int)

# Son satırı temizleyip final matrisini basıyoruz
df_final = df_temiz.dropna().copy()

print("🎯 [HEDEF KİLİTLENDİ] Yapay zeka yön tahmin sinyalleri (0/1) oluşturuldu.")
print(f"📈 Eğitime girecek nihai veri seti boyutu (Satır, Sütun): {df_final.shape}\n")

# Gösterilecek sansürsüz sütun listemiz
gosterilecek_sutunlar = ['Close', 'EMA_9', 'Bollinger_Ust', 'RSI_14', 'MACD_Ana', 'ATR_14', 'Hedef']

print("--- SON 3 GÜNÜN NİHAİ YAPAY ZEKA VERİ SETİ (EKSİKSİZ MATRİS) ---")
print(df_final[gosterilecek_sutunlar].tail(3))
print("==========================================================")