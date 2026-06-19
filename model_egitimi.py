import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

print("📡 NIHAI VE KARARLI MODEL EĞİTİM MOTORU BAŞLATILIYOR...")

# =====================================================================
# ⚙️ VERİ HAZIRLAMA (3 YILLIK KRİPTO HAVUZU)
# =====================================================================
SEMBOL = "XRP-USD"
df = yf.download(SEMBOL, period="3y", interval="1d", progress=False)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df.loc[:, ~df.columns.duplicated()]
df.columns = [str(c).strip() for c in df.columns]

close_series = df['Close'].squeeze()
high_series = df['High'].squeeze()
low_series = df['Low'].squeeze()
open_series = df['Open'].squeeze()
volume_series = df['Volume'].squeeze()

# Göstergeler
df['SMA_200'] = close_series.rolling(window=200, min_periods=1).mean()
df['SMA_Volume_20'] = volume_series.rolling(window=20, min_periods=1).mean()
sma_20 = close_series.rolling(window=20).mean()
std_20 = close_series.rolling(window=20).std()
df['Bollinger_Ust'] = sma_20 + (std_20 * 2)

# Saf Matematiksel İndikatörler
delta = close_series.diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
df['RSI_14'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))

exp1 = close_series.ewm(span=12, adjust=False).mean()
exp2 = close_series.ewm(span=26, adjust=False).mean()
df['MACD_Ana'] = exp1 - exp2

tr1 = high_series - low_series
tr2 = (high_series - close_series.shift(1)).abs()
tr3 = (low_series - close_series.shift(1)).abs()
df['ATR_14'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(window=14).mean()
df['Hacim_ROC_5'] = volume_series.pct_change(5).fillna(0)

en_yuksek_14 = high_series.rolling(window=14).max()
en_dusuk_14 = low_series.rolling(window=14).min()
df['Trend_Gucu'] = (en_yuksek_14 - en_dusuk_14) / (close_series + 1e-10)

df['Getiri_1G'] = close_series.pct_change(1)
df['Volatilite_5G'] = df['Getiri_1G'].rolling(window=5).std()

# Hedef: Yarınki yön yukarı mı?
df['Yuzde_Getiri_Yarin'] = close_series.pct_change(1).shift(-1)
df['Hedef'] = (df['Yuzde_Getiri_Yarin'] > 0.002).astype(int)

df = df.dropna().copy()

# =====================================================================
# 🧠 REYALİST MODEL EĞİTİMİ (TRAIN / TEST SPLIT)
# =====================================================================
ozellikler = ['Close', 'SMA_200', 'Bollinger_Ust', 'RSI_14', 'MACD_Ana', 'Getiri_1G', 'Volatilite_5G', 'ATR_14', 'Hacim_ROC_5', 'Trend_Gucu']

# Verinin %80'ini eğitime, son %20'sini (yaklaşık son 6-7 ayı) teste ayırıyoruz hoca
sinir_indeksi = int(len(df) * 0.80)

X_train = df[ozellikler].iloc[:sinir_indeksi]
y_train = df['Hedef'].iloc[:sinir_indeksi]

X_test = df[ozellikler].iloc[sinir_indeksi:]
y_test = df['Hedef'].iloc[sinir_indeksi:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dengeli ve stabil model parametreleri (Aşırı karmaşa uçuruldu)
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=3, subsample=0.8, random_state=42)
model.fit(X_train_scaled, y_train)

# Test seti üzerinde gerçek başarıyı ölçüyoruz
dogruluk_skoru = model.score(X_test_scaled, y_test)

print("\n" + "="*50)
print("--- Out-of-Sample (0.5 Yıl) Kararlı Model Değerlendirmesi ---")
print("="*50)
print(f"• Eğitim Seti Gün Sayısı (Train) : {len(X_train)}")
print(f"• Temiz Test Seti Gün Sayısı (Test): {len(X_test)}")
print("-"*50)
print(f"🏆 MODELİN GERÇEK YÖN TAHMİN BAŞARISI: %{dogruluk_skoru * 100:.2f}")
print("*" * 50)
print("Model, kararlı bir şekilde verimsiz sinyalleri filtrelemektedir hoca.")
print("="*50)