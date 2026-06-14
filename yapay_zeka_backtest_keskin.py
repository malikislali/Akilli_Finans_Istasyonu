import pandas as pd
import numpy as np
from yapay_zeka_veri import df_final
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier

print("\n==========================================================")
print("  🎯 QUANT AI - KESKİN NİŞANCI (PROBABILITY) MOTORU 🎯  ")
print("==========================================================")

df_model = df_final.copy()

# Özellik Mühendisliği
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

X = df_model.drop(columns=[
    'Close', 'High', 'Low', 'Open', 'Volume', 'Yarin_Kapanis', 'Hedef', 'Yuzde_Getiri_Yarin',
    'Bollinger_Ust', 'Keltner_Ust', 'Donchian_Ust', 'Dividends', 'Stock Splits'
])
y = df_model['Hedef']

# Model Eğitimi
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.03, max_depth=3, subsample=0.8, random_state=42)
tscv = TimeSeriesSplit(n_splits=5)

# Olasılık değerlerini toplamak için boş dizi oluşturuyoruz
olasilik_sinyalleri = np.zeros(len(y))
test_indeksleri = []

scaler = StandardScaler()
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model.fit(X_train_scaled, y_train)
    
    # [:, 1] -> Bize doğrudan 1 (yükseliş) olma olasılığını % olarak verir hoca!
    olasilik_sinyalleri[test_idx] = model.predict_proba(X_test_scaled)[:, 1]
    test_indeksleri.extend(test_idx)

df_backtest = df_model.iloc[test_indeksleri].copy()
df_backtest['Yükseliş_Olasılığı'] = olasilik_sinyalleri[test_indeksleri]

# =====================================================================
# 💸 3. KESKİN NİŞANCI SİMÜLASYONU ( GÜVEN EŞİĞİ FİLTRESİ )
# =====================================================================
baslangic_bakiyesi = 1000.0
bakiye = baslangic_bakiyesi
komisyon_orani = 0.001 

# 🚨 KRİTİK FİLTRE: Yapay zeka %56 ve üzerinde emin değilse işleme girmeyeceğiz!
guven_esigi = 0.58

islem_sayisi = 0
basarili_islem = 0

print(f"💰 Başlangıç Bakiyesi: {baslangic_bakiyesi}$")
print(f"🎯 Yapay Zeka Güven Eşiği Filtresi: %{guven_esigi * 100:.1f}")
print("-> Sadece en yüksek olasılıklı sinyaller süzülüyor...\n")

for idx, row in df_backtest.iterrows():
    # Yapay zeka kararsız değilse, tahmini güven eşiğinin üzerindeyse tetikle!
    if row['Yükseliş_Olasılığı'] >= guven_esigi:
        islem_sayisi += 1
        
        # Alış komisyonu
        bakiye = bakiye * (1 - komisyon_orani)
        
        # Pazar getirisi yansır
        getiri = row['Yuzde_Getiri_Yarin']
        bakiye = bakiye * (1 + getiri)
        
        # Satış komisyonu
        bakiye = bakiye * (1 - komisyon_orani)
        
        if getiri > 0:
            basarili_islem += 1

toplam_net_kar_yuzde = ((bakiye - baslangic_bakiyesi) / baslangic_bakiyesi) * 100
win_rate = (basarili_islem / islem_sayisi) * 100 if islem_sayisi > 0 else 0

print("=================== 🏆 KESKİN NİŞANCI RAPORU 🏆 ===================")
print(f"💰 BAŞLANGIÇ BAKİYESİ : {baslangic_bakiyesi}$")
print(f"💵 FİNAL CÜZDAN BAKİYESİ: {bakiye:.2f}$")
print(f"📈 TOPLAM NET KÂR/ZARAR : %{toplam_net_kar_yuzde:.2f}")
print("----------------------------------------------------------")
print(f"🔄 TOPLAM YAPILAN İŞLEM : {islem_sayisi}")
print(f"🎯 BAŞARILI (KÂRLI) İŞLEM: {basarili_islem}")
print(f"📊 İŞLEM BAŞARI ORANI  : %{win_rate:.2f}")
print("==========================================================")