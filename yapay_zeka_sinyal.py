import pandas as pd
import numpy as np
from yapay_zeka_veri import df_final
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import datetime

print("\n==========================================================")
print("  🛰️ QUANT AI - DERİN AYI FİLTRELİ KOMUT MERKEZİ V2 🛰️  ")
print("==========================================================")

# ESKİ KODUN YERİNE ŞUNU YAZIYORUZ HOCA (TRADINGVIEW SENKRONİZASYONU)
df_model = df_final.copy()

# TradingView'deki gerçek makro SMA_200'ü milimetrik yakalamak için
# Rolling işlemini minimum_periods=1 ile esnetiyoruz. Böylece tüm geçmişi kusursuz süzer!
df_model['SMA_200'] = df_model['Close'].rolling(window=200, min_periods=1).mean()

# =====================================================================
# 📊 1. ÖZELLİK MÜHENDİSLİĞİ VE MUM FİTİL MATEMATİĞİ
# =====================================================================

df_model['SMA_Volume_20'] = df_model['Volume'].rolling(window=20).mean()

# 🎯 Ayı Piyasası Kaçaklarını Yakalayacak Mum Gövde Analizi
df_model['Mum_Boyu'] = df_model['High'] - df_model['Low']
df_model['Alt_Fitil'] = np.minimum(df_model['Open'], df_model['Close']) - df_model['Low']
# Alt fitilin toplam mum boyuna oranı (Dipten dönüş iğnesi mi?)
df_model['Alt_Fitil_Orani'] = np.where(df_model['Mum_Boyu'] > 0, df_model['Alt_Fitil'] / df_model['Mum_Boyu'], 0)

df_model['Getiri_1G'] = df_model['Close'].pct_change(1)
df_model['Getiri_3G'] = df_model['Close'].pct_change(3)
df_model['Getiri_5G'] = df_model['Close'].pct_change(5)
df_model['Volatilite_5G'] = df_model['Getiri_1G'].rolling(window=5).std()

df_model['HA_Close'] = (df_model['Open'] + df_model['High'] + df_model['Low'] + df_model['Close']) / 4
df_model['HA_Open'] = (df_model['Open'].shift(1) + df_model['Close'].shift(1)) / 2

for lag in [1, 2]:
    df_model[f'RSI_Lag_{lag}'] = df_model['RSI_14'].shift(lag)
    df_model[f'MACD_Lag_{lag}'] = df_model['MACD_Ana'].shift(lag)

df_model['Yuzde_Getiri_Yarin'] = (df_model['Yarin_Kapanis'] - df_model['Close']) / df_model['Close']
df_model['Hedef'] = (df_model['Yuzde_Getiri_Yarin'] > 0.005).astype(int)
df_model = df_model.dropna()

# =====================================================================
# 🧼 ESKİ LİSTEDEN 'Dividends' VE 'Stock Splits' ÇIKARILMIŞ GÜNCEL HALİ
# =====================================================================
X = df_model.drop(columns=[
    'Close', 'High', 'Low', 'Open', 'Volume', 'Yarin_Kapanis', 'Hedef', 'Yuzde_Getiri_Yarin',
    'Bollinger_Ust', 'Keltner_Ust', 'Donchian_Ust', 'SMA_200',
    'SMA_Volume_20', 'Mum_Boyu', 'Alt_Fitil', 'Alt_Fitil_Orani'
])
y = df_model['Hedef']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.03, max_depth=3, subsample=0.8, random_state=42)
model.fit(X_scaled, y)

# =====================================================================
# 🚀 2. BUGÜNÜN CANLI VERİLERİ VE DURUM TESPİTİ
# =====================================================================
bugunun_verisi = df_model.iloc[-1]
bugunun_ozellikleri = X.iloc[[-1]] # Sütun isimleri korundu (Warning engellendi hoca)
bugunun_ozellikleri_scaled = scaler.transform(bugunun_ozellikleri)

yükseliş_olasılığı = model.predict_proba(bugunun_ozellikleri_scaled)[0, 1]
fiyat_su_an = bugunun_verisi['Close']
sma_200_degeri = bugunun_verisi['SMA_200']
hacim_su_an = bugunun_verisi['Volume']
hacim_ortalamasi = bugunun_verisi['SMA_Volume_20']
alt_fitil_orani_su_an = bugunun_verisi['Alt_Fitil_Orani']

# =====================================================================
# 📊 3. ESNEK SİNYAL VE KARAR PANELİ
# =====================================================================
print(f"⏱️  Analiz Zamanı : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"💰 Güncel Fiyat  : {fiyat_su_an:,.2f} TL")
print(f"📈 SMA_200 Değeri : {sma_200_degeri:,.2f} TL")
print("----------------------------------------------------------")

mesafe_yuzde = ((fiyat_su_an - sma_200_degeri) / sma_200_degeri) * 100

# Rejim Durum tespiti
if fiyat_su_an > sma_200_degeri:
    trend_durumu = "🟢 BOĞA REJİMİ (Trend Pozitif)"
    rejim = "BOGA"
elif fiyat_su_an <= sma_200_degeri and abs(mesafe_yuzde) <= 6.0:
    trend_durumu = f"🟡 ESNEK AYI REJİMİ (SMA_200'e Yakın Temas: %{mesafe_yuzde:.2f})"
    rejim = "ESNEK_AYI"
else:
    trend_durumu = f"🔴 DERİN AYI REJİMİ (Fiyat Çöküş Bölgesinde: %{mesafe_yuzde:.2f})"
    rejim = "DERIN_AYI"

# İğne ve Hacim Onayı var mı? (Ayıdan kaçış anahtarı hoca!)
igne_onayi = alt_fitil_orani_su_an >= 0.40  # Mumun en az %40'ı fitil olacak
hacim_onayi = hacim_su_an > hacim_ortalamasi

print(f"🌍 Makro Piyasa Rejimi: {trend_durumu}")
print(f"🧠 Yapay Zeka Yükseliş Olasılığı: %{yükseliş_olasılığı * 100:.2f}")
print(f"🧬 Mum Alt Fitil Oranı: %{alt_fitil_orani_su_an * 100:.1f} | Yüksek Hacim: {'EVET' if hacim_onayi else 'HAYIR'}")
print("==========================================================")

# NİHAİ EMİR PANELİ
print("\n🚨 KOMUT MERKEZİ MANUEL İŞLEM TAVSİYESİ 🚨")
print("----------------------------------------------------------")

if rejim == "BOGA" and yükseliş_olasılığı >= 0.55:
    print("🔥 [ SİNYAL: GÜÇLÜ AL (BUY) - BOĞA ONAYLI ] 🔥")
    print("👉 Gerekçe: Ana trend yukarı ve yapay zeka yükseliş kalıbını destekliyor.")

# 🎯 BARAJ %75'TEN %65'E ÇEKİLDİ - KIRILIMI KAÇIRMIYORUZ!
elif rejim == "ESNEK_AYI" and yükseliş_olasılığı >= 0.65:
    print("⚡ [ SİNYAL: KUVVETLİ AL (KIRILIM POTANSİYELİ) ] ⚡")
    print(f"👉 Gerekçe: 200 günlük ortalamaya çok yakınız (%{abs(mesafe_yuzde):.2f}) ve model %{yükseliş_olasılığı*100:.1f} olasılıkla kırılım bekliyor.")
    print("⚠️  Tavsiye: Manuel spot alım denenebilir. 200 günlük ortalamanın üzeri hedefleniyor hoca!")

# 🎯 İŞTE O BÜYÜK AYI KAÇIŞINI YAKALAYAN ÖZEL KOŞUL!
elif rejim == "DERIN_AYI" and yükseliş_olasılığı >= 0.80 and igne_onayi and hacim_onayi:
    print("🚀 [ SİNYAL: SENSASYONEL DİPTEN DÖNÜŞ (AYI RALLİSİ YAKALANDI) ] 🚀")
    print("👉 Gerekçe: Hisse derin ayı piyasasında ama dipten devasa bir hacimle iğne (fitil) toplandı!")
    print(f"⚠️  Tavsiye: Ayı piyasası minimal yükseliş fırsatı. Alt iğnenin ucuna ÇOK SIKI STOP koyarak manuel denenebilir!")

elif rejim == "DERIN_AYI":
    print("🛑 [ SİNYAL: KESİNLİKLE PAS GEÇ (NAKİTTE KAL) ] 🛑")
    print("👉 Gerekçe: Derin ayı tuzak bölgesi. Güçlü bir kurumsal hacim ve dönüş iğnesi olmadan tetiğe basma!")

else:
    print("🟡 [ SİNYAL: NÖTR / İZLEMEDE KAL ] 🟡")
    print("👉 Gerekçe: Belirgin bir kurumsal hacim veya yapay zeka güvencesi yok.")

print("==========================================================\n")