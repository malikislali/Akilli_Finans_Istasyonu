import pandas as pd
import numpy as np
from yapay_zeka_veri import df_final
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

print("\n==========================================================")
print("  🎛️ QUANT AI - SEÇENEK A + B BİRLEŞİK KUANTUM MOTORU 🎛️  ")
print("==========================================================")

df_model = df_final.copy()

# =====================================================================
# 📊 HAMLE 1: VOLATİLİTE (STANDART SAPMA) ÖZELLİKLERİ
# =====================================================================
df_model['Getiri_1G'] = df_model['Close'].pct_change(1)
df_model['Getiri_3G'] = df_model['Close'].pct_change(3)
df_model['Getiri_5G'] = df_model['Close'].pct_change(5)

# Fiyatın 5 günlük standart sapması (Piyasanın Gerginlik Derecesi)
df_model['Volatilite_5G'] = df_model['Getiri_1G'].rolling(window=5).std()

# =====================================================================
# ⏳ HAMLE 2: HEIKIN-ASHI (GÜRÜLTÜ FİLTRELİ MUM) MATEMATİĞİ
# =====================================================================
# Heikin-Ashi kapanış ve açılış formüllerini bizzat elle hesaplayıp gömüyoruz hoca.
df_model['HA_Close'] = (df_model['Open'] + df_model['High'] + df_model['Low'] + df_model['Close']) / 4
df_model['HA_Open'] = (df_model['Open'].shift(1) + df_model['Close'].shift(1)) / 2

# İndikatör momentumları
for lag in [1, 2]:
    df_model[f'RSI_Lag_{lag}'] = df_model['RSI_14'].shift(lag)
    df_model[f'MACD_Lag_{lag}'] = df_model['MACD_Ana'].shift(lag)

# %0.5 Gürültü Filtreli Hedef Mekanizması
df_model['Yuzde_Getiri_Yarin'] = (df_model['Yarin_Kapanis'] - df_model['Close']) / df_model['Close']
df_model['Hedef'] = (df_model['Yuzde_Getiri_Yarin'] > 0.005).astype(int)

# Shift ve rolling işlemlerinden kalan boşlukları temizleyelim
df_model = df_model.dropna()

# Sütunları Ayıklama
cop_sutunlar = [
    'Close', 'High', 'Low', 'Open', 'Volume', 'Yarin_Kapanis', 'Hedef', 'Yuzde_Getiri_Yarin',
    'Bollinger_Ust', 'Keltner_Ust', 'Donchian_Ust', 'Dividends', 'Stock Splits'
]
X = df_model.drop(columns=cop_sutunlar)
y = df_model['Hedef']

print(f"🎯 Toplam Zenginleştirilmiş Gösterge Sayısı: {X.shape[1]}")
print(f"📋 Yeni Eklenen Kombinasyonlar: ['Volatilite_5G', 'HA_Close', 'HA_Open']\n")

# Verileri standartlaştırma
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Kronolojik Çapraz Doğrulama Ayarı
tscv = TimeSeriesSplit(n_splits=5)

# =====================================================================
# 🧠 HAMLE 3: KONTROLLÜ HİPERPARAMETRE OPTMİZASYONU (GRIDSEARCH)
# =====================================================================
# Motor yanmasın diye arama uzayını çok net ve stratejik parametrelerle sınırlıyoruz.
param_gridi = {
    'n_estimators': [100, 150],
    'learning_rate': [0.03, 0.05],
    'max_depth': [3, 4],
    'subsample': [0.8]
}

gbc_model = GradientBoostingClassifier(random_state=42)

print("🏋️‍♂️ M4 İşlemci Devrede: GridSearch 5 farklı zaman diliminde en iyi DNA'yı arıyor...")
grid_arama = GridSearchCV(estimator=gbc_model, param_grid=param_gridi, cv=tscv, scoring='accuracy', n_jobs=-1)
grid_arama.fit(X_scaled, y)

print("\n=================== 🏆 BİRLEŞİK MOTOR SONUÇLARI 🏆 ===================")
print(f"🥇 En İyi Kombinasyon Parametreleri: {grid_arama.best_params_}")
print("----------------------------------------------------------")
print(f"🚀 YENİ HEDEF HİPER-BAŞARI ORANI: %{grid_arama.best_score_ * 100:.2f}")
print("==========================================================")