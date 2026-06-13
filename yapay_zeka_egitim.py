import pandas as pd
import numpy as np
from yapay_zeka_veri import df_final
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

print("\n==========================================================")
print("  🤖 QUANT AI - MOMENTUM ODAKLI ÇAPRAZ DOĞRULAMA 🤖  ")
print("==========================================================")

df_model = df_final.copy()

# Gürültü Filtresi (%0.5 net getiri hedefi)
esik_deger = 0.005
df_model['Yuzde_Getiri'] = (df_model['Yarin_Kapanis'] - df_model['Close']) / df_model['Close']
df_model['Hedef'] = (df_model['Yuzde_Getiri'] > esik_deger).astype(int)

# =====================================================================
# ⚡ DİNAMİK MOMENTUM ÖZELLİKLERİ (LAG FEATURES) EKLEME
# =====================================================================
# RSI ve MACD'nin son 1 ve 2 gün önceki değerlerini ekleyerek yön değişimini modele öğretiyoruz.
for lag in [1, 2]:
    df_model[f'RSI_Lag_{lag}'] = df_model['RSI_14'].shift(lag)
    df_model[f'MACD_Lag_{lag}'] = df_model['MACD_Ana'].shift(lag)

# Yeni eklenen shift işlemlerinden dolayı oluşan boşlukları temizliyoruz
df_model = df_model.dropna()

# =====================================================================
# 🧼 SABOTAJCI VE GEREKSİZ SÜTUNLARI TAMAMEN SİLME
# =====================================================================
cop_sutunlar = [
    'Close', 'High', 'Low', 'Open', 'Volume', 'Yarin_Kapanis', 'Hedef', 'Yuzde_Getiri',
    'Bollinger_Ust', 'Keltner_Ust', 'Donchian_Ust', 'Dividends', 'Stock Splits'
]
X = df_model.drop(columns=cop_sutunlar)
y = df_model['Hedef']

print(f"🎯 Kalan saf ve nitelikli gösterge sayısı: {X.shape[1]}")
print(f"📋 Eğitime Giren Yeni Akıllı Sütunlar: {list(X.columns)}\n")

# Veriyi standartlaştırma
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Kronolojik Çapraz Doğrulama
tscv = TimeSeriesSplit(n_splits=5)

model = RandomForestClassifier(
    n_estimators=150, 
    max_depth=3, # Ezberlemeyi önleyen sığ derinlik
    class_weight="balanced", 
    random_state=42
)

print("🏋️‍♂️ Model momentum ve yön verileriyle yeniden eğitiliyor...")
skorlar = cross_val_score(model, X_scaled, y, cv=tscv, scoring='accuracy')

print("\n=================== SINAV SONUÇLARI ===================")
for i, skor in enumerate(skorlar, 1):
    print(f"📂 {i}. Zaman Dilimi Sınav Notu: %{skor * 100:.2f}")

print("----------------------------------------------------------")
print(f"🏆 YENİ GERÇEK ORTALAMA BAŞARI: %{skorlar.mean() * 100:.2f}")
print("==========================================================")