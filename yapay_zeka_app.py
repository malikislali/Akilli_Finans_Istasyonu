import streamlit as str_ui
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import datetime

# --- 📐 SAYFA AYARLARI ---
str_ui.set_page_config(page_title="Quant AI - Komut Merkezi", page_icon="📡", layout="wide")

str_ui.markdown("""
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 0rem;}
    h1 {color: #1E88E5;}
    h3 {color: #424242;}
    </style>
""", unsafe_allow_html=True)

str_ui.title("📡 QUANT AI - EVRENSEL FİNANSAL KOMUT MERKEZİ V3")
str_ui.subheader("Çoklu Zaman Dilimi & Akıllı Manuel İşlem Paneli (Sıfır Bağımlılık Modu)")
str_ui.write("---")

# =====================================================================
# 🎛️ SOL MENÜ (KONTROL PANELİ)
# =====================================================================
str_ui.sidebar.header("🎯 Analiz Konfigürasyonu")

pazar = str_ui.sidebar.selectbox("1. Pazar Seçimi", ["TR_HISSE", "KRIPTO", "ABD_HISSE", "EMTIA"])

varlik_sozlugu = {
    "TR_HISSE": ["THYAO.IS", "SOKM.IS", "ASELS.IS", "EREGL.IS"],
    "KRIPTO": ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD"],
    "ABD_HISSE": ["AAPL", "NVDA", "TSLA", "MSFT"],
    "EMTIA": ["GC=F", "SI=F", "CL=F"]
}

sembol = str_ui.sidebar.selectbox("2. Varlık Sembolü", varlik_sozlugu[pazar])

zaman_dilimi = str_ui.sidebar.selectbox(
    "3. Zaman Dilimi (Timeframe)", 
    ["1 Saat (1h)", "2 Saat (2h)", "4 Saat (4h)", "1 Gün (1d)", "1 Hafta (1wk)"]
)

tf_map = {"1 Saat (1h)": "1h", "2 Saat (2h)": "2h", "4 Saat (4h)": "4h", "1 Gün (1d)": "1d", "1 Hafta (1wk)": "1wk"}
period_map = {"1h": "2mo", "2h": "2mo", "4h": "3mo", "1d": "1y", "1wk": "3y"}

tf_param = tf_map[zaman_dilimi]
period_param = period_map[tf_param]

# =====================================================================
# 🧮 SAF MATEMATİKSEL İNDİKATÖR FONKSİYONLARI (PANDAS_TA BYPASS)
# =====================================================================
def hesapla_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def hesapla_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    return macd

# =====================================================================
# ⚙️ ARKA PLAN VERİ MOTORU
# =====================================================================
@str_ui.cache_data(ttl=300)
def verileri_hazirla(sembol, period, interval):
    df = yf.download(sembol, period=period, interval=interval)
    if df.empty:
        return df
    df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Saf Pandas ve Matematik ile Özellik Mühendisliği
    df['SMA_200'] = df['Close'].rolling(window=200, min_periods=1).mean()
    df['SMA_Volume_20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
    
    # Bollinger Üst Bant
    sma_20 = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    df['Bollinger_Ust'] = sma_20 + (std_20 * 2)
    
    # Osilatörler
    df['RSI_14'] = hesapla_rsi(df['Close'], 14)
    df['MACD_Ana'] = hesapla_macd(df['Close'], 12, 26, 9)
    
    # Mum Analizleri
    df['Mum_Boyu'] = df['High'] - df['Low']
    df['Alt_Fitil'] = np.minimum(df['Open'], df['Close']) - df['Low']
    df['Alt_Fitil_Orani'] = np.where(df['Mum_Boyu'] > 0, df['Alt_Fitil'] / df['Mum_Boyu'], 0)
    
    df['Getiri_1G'] = df['Close'].pct_change(1)
    df['Volatilite_5G'] = df['Getiri_1G'].rolling(window=5).std()
    
    df['Yuzde_Getiri_Yarin'] = df['Close'].pct_change(1).shift(-1)
    df['Hedef'] = (df['Yuzde_Getiri_Yarin'] > 0.002).astype(int)
    
    return df.dropna()

df_model = verileri_hazirla(sembol, period_param, tf_param)

if df_model.empty:
    str_ui.error("⚠️ Seçilen zaman diliminde yeterli veri yok veya veri çekilemedi. Lütfen başka bir kombinasyon deneyin.")
else:
    # Model Özellik Seti
    ozellikler = ['Close', 'SMA_200', 'Bollinger_Ust', 'RSI_14', 'MACD_Ana', 'Getiri_1G', 'Volatilite_5G']
    
    X = df_model[ozellikler].copy()
    y = df_model['Hedef']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.03, max_depth=3, subsample=0.8, random_state=42)
    model.fit(X_scaled, y)
    
    # Canlı Tahmin Verisi
    bugunun_verisi = df_model.iloc[-1]
    bugunun_ozellikleri = X.iloc[[-1]]
    bugunun_ozellikleri_scaled = scaler.transform(bugunun_ozellikleri)
    
    yukselis_olasiligi = model.predict_proba(bugunun_ozellikleri_scaled)[0, 1]
    fiyat_su_an = bugunun_verisi['Close']
    sma_200_degeri = bugunun_verisi['SMA_200']
    alt_fitil_orani_su_an = bugunun_verisi['Alt_Fitil_Orani']
    hacim_su_an = bugunun_verisi['Volume']
    hacim_ortalamasi = bugunun_verisi['SMA_Volume_20']
    
    mesafe_yuzde = ((fiyat_su_an - sma_200_degeri) / sma_200_degeri) * 100
    
    # Rejim Tespiti
    if fiyat_su_an > sma_200_degeri:
        rejim, rejim_renk, rejim_text = "BOGA", "🟢", "BOĞA REJİMİ (Trend Pozitif)"
    elif fiyat_su_an <= sma_200_degeri and abs(mesafe_yuzde) <= 6.0:
        rejim, rejim_renk, rejim_text = "ESNEK_AYI", "🟡", f"ESNEK AYI REJİMİ (SMA_200'e Yakın: %{mesafe_yuzde:.2f})"
    else:
        rejim, rejim_renk, rejim_text = "DERIN_AYI", "🔴", f"DERİN AYI REJİMİ (Mesafe: %{mesafe_yuzde:.2f})"

    # =====================================================================
    # 📊 ARAYÜZ METRİKLERİ VE KARTLAR
    # =====================================================================
    col1, col2, col3 = str_ui.columns(3)
    col1.metric("💰 Güncel Fiyat", f"{fiyat_su_an:,.2f} USD" if pazar != "TR_HISSE" else f"{fiyat_su_an:,.2f} TL")
    col2.metric("📈 SMA_200 Değeri", f"{sma_200_degeri:,.2f} USD" if pazar != "TR_HISSE" else f"{sma_200_degeri:,.2f} TL")
    col3.metric("🧠 AI Yükseliş Olasılığı", f"%{yukselis_olasiligi * 100:.2f}")
    
    str_ui.info(f"{rejim_renk} **Mevcut Piyasa Durumu:** {rejim_text}")
    
    # =====================================================================
    # 🚨 KARAR VE EMİR PANELİ
    # =====================================================================
    str_ui.markdown("### 🚨 KOMUT MERKEZİ MANUEL İŞLEM TAVSİYESİ")
    
    igne_onayi = alt_fitil_orani_su_an >= 0.40
    hacim_onayi = hacim_su_an > hacim_ortalamasi
    
    if rejim == "BOGA" and yukselis_olasiligi >= 0.55:
        str_ui.success(f"🔥 **[ SİNYAL: GÜÇLÜ AL ]** \n\nGerekçe: Seçilen **{zaman_dilimi}** grafiğinde ana trend yukarı ve yapay zeka yükseliş formasyonunu destekliyor.")
    elif rejim == "ESNEK_AYI" and yukselis_olasiligi >= 0.65:
        str_ui.warning(f"⚡ **[ SİNYAL: KUVVETLİ AL (KIRILIM POTANSİYELİ) ]** \n\nGerekçe: Fiyat {zaman_dilimi} grafiğinde ortalamaya çok yakın ve model yukarı patlama bekliyor.")
    elif rejim == "DERIN_AYI" and yukselis_olasiligi >= 0.80 and igne_onayi and hacim_onayi:
        str_ui.info(f"🚀 **[ SİNYAL: SENSASYONEL DİPTEN DÖNÜŞ (AYI RALLİSİ) ]** \n\nGerekçe: Derin ayı bölgesinde mum altında uzun bir iğne (%{alt_fitil_orani_su_an*100:.1f}) ve kurumsal hacim toplandı!")
    elif rejim == "DERIN_AYI":
        str_ui.error("🛑 **[ SİNYAL: KESİNLİKLE PAS GEÇ (NAKİTTE KAL) ]** \n\nGerekçe: Bu zaman diliminde derin ayı tuzağı var. Güçlü bir dönüş iğnesi veya hacim desteği yok, sakın tetiğe basma hoca!")
    else:
        str_ui.info("🟡 **[ SİNYAL: NÖTR / İZLEMEDE KAL ]** \n\nGerekçe: Bu zaman diliminde yapay zeka güven barajının altında kaldı veya net bir kurumsal iz yok.")

    str_ui.write("---")
    str_ui.caption(f"⚙️ Son Veri Güncelleme Zamanı: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Veri Sağlayıcı: Yahoo Finance")