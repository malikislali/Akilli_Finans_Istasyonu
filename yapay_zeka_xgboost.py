import pandas as pd
import numpy as np
from yapay_zeka_veri import df_final
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier  # <-- ASLA ÇÖKMEYEN ESNEK CANAVAR!

print("\n==========================================================")
print("  🚀 QUANT AI - GRADIENT BOOSTING & MOMENTUM MOTORU 🚀  ")
print("==========================================================")

df_model = df_final.copy()

# =====================================================================
# 📈 ÖZELLİK: FİYAT MOMENTUMU VE GETİRİ ORANLARI (PRICE LAGS)
# =====================================================================
df_model['Getiri_1G'] = df_model['Close'].pct_change(1)
df_model['Getiri_3G'] = df_model['Close'].pct_change(3)
df_model['Getiri_5G'] = df_model['Close'].pct_change(5)

# İndikatör momentumları (Yön ivmesi)
for lag in [1, 2]:
    df_model[f'RSI_Lag_{lag}'] = df_model['RSI_14'].shift(lag)
    df_model[f'MACD_Lag_{lag}'] = df_model['MACD_Ana'].shift(lag)

# %0.5 Gürültü Filtreli Hedef Mekanizması
df_model['Yuzde_Getiri_Yarin'] = (df_model['Yarin_Kapanis'] - df_model['Close']) / df_model['Close']
df_model['Hedef'] = (df_model['Yuzde_Getiri_Yarin'] > 0.005).astype(int)

df_model = df_model.dropna()

# Sütunları Temizleme
cop_sutunlar = [
    'Close', 'High', 'Low', 'Open', 'Volume', 'Yarin_Kapanis', 'Hedef', 'Yuzde_Getiri_Yarin',
    'Bollinger_Ust', 'Keltner_Ust', 'Donchian_Ust', 'Dividends', 'Stock Splits'
]
X = df_model.drop(columns=cop_sutunlar)
y = df_model['Hedef']

print(f"🎯 Toplam Saf Nitelikli Gösterge Sayısı: {X.shape[1]}")
print(f"📋 Aktif Özellikler: {list(X.columns)}\n")

# Verileri standartlaştırma
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Kronolojik Çapraz Doğrulama
tscv = TimeSeriesSplit(n_splits=5)

# =====================================================================
# 🏋️‍♂️ GRADIENT BOOSTING MODELİNİN YAPILANDIRILMASI
# =====================================================================
# XGBoost'un scikit-learn içindeki ikiz kardeşidir hoca. Mac bağımlılığı yoktur, saf Python'dır.
# subsample=0.8 -> Verilerin her adımda %80'ini rastgele seçerek ezberlemeyi (overfitting) engeller.
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=42
)

print("🏋️‍♂️ Gradient Boosting motoru finansal hafıza katmanıyla eğitime alınıyor...")
skorlar = cross_val_score(model, X_scaled, y, cv=tscv, scoring='accuracy')

print("\n=================== GBC SINAV SONUÇLARI ===================")
for i, skor in enumerate(skorlar, 1):
    print(f"📂 {i}. Zaman Dilimi Başarı Notu: %{skor * 100:.2f}")

print("----------------------------------------------------------")
print(f"🏆 GRADIENT BOOSTING GERÇEK ORTALAMA BAŞARISI: %{skorlar.mean() * 100:.2f}")
print("==========================================================")