import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

print("📡 QUANT AI - MANUEL STRATEJİ & KASA MUHASEBE MOTORU (V2.1) BAŞLATILIYOR...")

# =====================================================================
# ⚙️ KONFİGÜRASYON VE VERİ ÇEKME
# =====================================================================
SEMBOL = "XRP-USD"  
PERIOD = "3y"       # 3 Yıllık Büyük Döngü
INTERVAL = "1d"     # Günlük Mumlar

# 💰 KASA YÖNETİM AYARLARI
BASLANGIC_SERMAYESI = 10000.0  # Simülasyon başında kasadaki nakit (USD/TL)
KAR_AL_ORANI = 0.015          # %1.5 Kâr Al (Take-Profit)
STOP_LOSS_ORANI = 0.010       # %1.0 Zarar Durdur (Stop-Loss)

print(f"📥 {SEMBOL} için {PERIOD} periyodunda {INTERVAL} verileri indiriliyor...")
df = yf.download(SEMBOL, period=PERIOD, interval=INTERVAL, progress=False)

if df.empty or len(df) < 50:
    print("⚠️ Hata: Backtest için yeterli veri çekilemedi hoca!")
    exit()

# MultiIndex ve Kolon Temizliği
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df.loc[:, ~df.columns.duplicated()]
df.columns = [str(c).strip() for c in df.columns]

# =====================================================================
# 🧮 GÖSTERGE HESAPLAMALARI
# =====================================================================
def hesapla_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def hesapla_macd(series, fast=12, slow=26):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    return exp1 - exp2

def hesapla_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

close_series = df['Close'].squeeze()
high_series = df['High'].squeeze()
low_series = df['Low'].squeeze()
open_series = df['Open'].squeeze()
volume_series = df['Volume'].squeeze()

df['SMA_200'] = close_series.rolling(window=200, min_periods=1).mean()
df['SMA_Volume_20'] = volume_series.rolling(window=20, min_periods=1).mean()
sma_20 = close_series.rolling(window=20).mean()
std_20 = close_series.rolling(window=20).std()
df['Bollinger_Ust'] = sma_20 + (std_20 * 2)

df['RSI_14'] = hesapla_rsi(close_series, 14)
df['MACD_Ana'] = hesapla_macd(close_series, 12, 26)
df['ATR_14'] = hesapla_atr(high_series, low_series, close_series, 14)
df['Hacim_ROC_5'] = volume_series.pct_change(5).replace([np.inf, -np.inf], 0).fillna(0)

# Sıkışma Katsayısı / Trend Gücü
en_yuksek_14 = high_series.rolling(window=14).max()
en_dusuk_14 = low_series.rolling(window=14).min()
df['Trend_Gucu'] = (en_yuksek_14 - en_dusuk_14) / (close_series + 1e-10)
df['Trend_Gucu_Ort_20'] = df['Trend_Gucu'].rolling(window=20).mean()

df['Mum_Boyu'] = high_series - low_series
df['Alt_Fitil'] = np.minimum(open_series, close_series) - low_series
df['Alt_Fitil_Orani'] = np.where(df['Mum_Boyu'] > 0, df['Alt_Fitil'] / df['Mum_Boyu'], 0)

df['Getiri_1G'] = close_series.pct_change(1)
df['Volatilite_5G'] = df['Getiri_1G'].rolling(window=5).std()

# Hedef Etiketleme (Gelecek mum yönü tahmini için)
df['Yuzde_Getiri_Yarin'] = close_series.pct_change(1).shift(-1)
df['Hedef'] = (df['Yuzde_Getiri_Yarin'] > 0.002).astype(int)

df = df.dropna().copy()

# =====================================================================
# 🧠 ZAMAN SİMÜLASYONU VE STRATEJİK MUHASEBE MOTORU
# =====================================================================
ozellikler = ['Close', 'SMA_200', 'Bollinger_Ust', 'RSI_14', 'MACD_Ana', 'Getiri_1G', 'Volatilite_5G', 'ATR_14', 'Hacim_ROC_5', 'Trend_Gucu']

toplam_islem = 0
basarili_islem = 0
zararli_islem = 0
pas_gecilen_yatay = 0
mevcut_kasa = BASLANGIC_SERMAYESI

for i in range(30, len(df) - 1):
    X_train = df[ozellikler].iloc[:i]
    y_train = df['Hedef'].iloc[:i]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = GradientBoostingClassifier(n_estimators=120, learning_rate=0.025, max_depth=3, subsample=0.75, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    bugun = df.iloc[i]
    bugun_scaled = scaler.transform(df[ozellikler].iloc[[i]])
    
    ai_olasilik = model.predict_proba(bugun_scaled)[0, 1]
    
    # Bugünün Teknik Hücreleri
    fiyat = bugun['Close']
    sma200 = bugun['SMA_200']
    trend_gucu = bugun['Trend_Gucu']
    trend_gucu_ort = bugun['Trend_Gucu_Ort_20']
    alt_fitil = bugun['Alt_Fitil_Orani']
    hacim = bugun['Volume']
    hacim_ort = bugun['SMA_Volume_20']
    
    piyasa_yatay_mi = trend_gucu < (trend_gucu_ort * 0.85)
    mesafe_yuzde = ((fiyat - sma200) / sma200) * 100
    
    tetiğe_basildi_mi = False
    
    if piyasa_yatay_mi:
        pas_gecilen_yatay += 1
        continue
        
    # 🔥 YENİ NESİL KURUMSAL FİLTRE EŞİKLERİ hoca
    if fiyat > sma200: # BOĞA
        if ai_olasilik >= 0.58: # Baraj %55 -> %58 yapıldı
            tetiğe_basildi_mi = True
            
    elif fiyat <= sma200 and abs(mesafe_yuzde) <= 6.0: # ESNEK AYI
        if ai_olasilik >= 0.68: # Baraj %65 -> %68 yapıldı
            tetiğe_basildi_mi = True
            
    else: # DERİN AYI
        if ai_olasilik >= 0.80 and alt_fitil >= 0.40 and hacim > hacim_ort:
            tetiğe_basildi_mi = True

    # 💰 GERÇEKÇİ KASA HESAPLAMA MOTORU
    if tetiğe_basildi_mi:
        toplam_islem += 1
        
        # Simülasyon: Sonraki süreçte fiyatın en yüksek/en düşük hareketine göre kâr-zarar kontrolü hoca
        yarin_data = df.iloc[i + 1]
        yarin_en_yuksek_getiri = (yarin_data['High'] - fiyat) / fiyat
        yarin_en_dusuk_getiri = (yarin_data['Low'] - fiyat) / fiyat
        
        # Stop-Loss mu önce tetiklendi, Kâr Al mı? (Muhafazakar yaklaşım: Önce Stop kontrolü)
        if yarin_en_dusuk_getiri <= -STOP_LOSS_ORANI:
            zararli_islem += 1
            mevcut_kasa -= (mevcut_kasa * STOP_LOSS_ORANI) # Kasadan %1 zarar düşüyoruz
        elif yarin_en_yuksek_getiri >= KAR_AL_ORANI:
            basarili_islem += 1
            mevcut_kasa += (mevcut_kasa * KAR_AL_ORANI)  # Kasaya %1.5 kâr ekliyoruz
        else:
            # İkisi de tetiklenmediyse gün kapanışına göre muhasebe yapıyoruz
            yarin_kapanis_getiri = yarin_data['Getiri_1G']
            mevcut_kasa += (mevcut_kasa * yarin_kapanis_getiri)
            if yarin_kapanis_getiri > 0:
                basarili_islem += 1
            else:
                zararli_islem += 1

# =====================================================================
# 📋 GELİŞMİŞ MUHASEBE ÇIKTISI
# =====================================================================
print("\n" + "="*60)
print(f"🎯 QUANT AI FINANSAL BACKTEST KARNESİ: {SEMBOL}")
print("="*60)
print(f"📅 Toplam Simüle Edilen Gün Sayısı           : {len(df) - 31}")
print(f"🛑 Testere Piyasası Engeline Takılan Gün     : {pas_gecilen_yatay} Gün")
print(f"🚀 Kriterlere Uyan Toplam Tetiklenen İşlem   : {toplam_islem}")
print(f"🟢 Hedefe Ulaşan Başarılı Pozisyon           : {basarili_islem}")
print(f"🔴 Stop-Loss Olan Zararlı Pozisyon           : {zararli_islem}")
print("-"*60)

if toplam_islem > 0:
    basari_orani = (basarili_islem / toplam_islem) * 100
    print(f"🏆 STRATEJİ NET BAŞARI ORANI                 : %{basari_orani:.2f}")
else:
    print("🏆 STRATEJİ NET BAŞARI ORANI                 : %0.00")

print("-"*60)
print(f"💰 BAŞLANGIÇ KASASI                          : {BASLANGIC_SERMAYESI:,.2f}")
print(f"💳 NİHAİ KASA BAKİYESİ                       : {mevcut_kasa:,.2f}")

net_kar_zarar = mevcut_kasa - BASLANGIC_SERMAYESI
kar_yuzdesi = (net_kar_zarar / BASLANGIC_SERMAYESI) * 100

if net_kar_zarar > 0:
    print(f"📈 NET KÂR DURUMU                            : +{net_kar_zarar:,.2f} (🚀 %{kar_yuzdesi:.2f} BÜYÜME)")
else:
    print(f"📉 NET ZARAR DURUMU                          : {net_kar_zarar:,.2f} (⚠️ %{kar_yuzdesi:.2f} ERİME)")
print("="*60)