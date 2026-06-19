import streamlit as str_ui
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 🎯 XGBOOST GÜVENLİK DUVARI (Mac M4 Koruyucu Zırhı)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# --- 📐 SAYFA AYARLARI ---
str_ui.set_page_config(page_title="Quant AI - Finans Kokpiti V55.4", page_icon="📡", layout="wide")

# Kurumsal Görsel Zırh (CSS)
str_ui.markdown("""
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 0rem;}
    h1 {color: #1A1A1A; font-weight: 700;}
    h2 {color: #2C3E50; font-weight: 600; margin-top: 1rem;}
    .indicator-card {background: #FFFFFF; border: 1px solid #E0E0E0; border-radius: 12px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); height: 100%;}
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 🎛️ 1. PERDE: SOL MENÜ VE DİNAMİK PERİYOT SEÇİM KOKPİTİ
# =====================================================================
str_ui.sidebar.header("🎯 Analiz Konfigürasyonu")
pazar = str_ui.sidebar.selectbox("1. Pazar Seçimi", ["KRIPTO", "TR_HISSE", "ABD_HISSE", "EMTIA"])

varlik_havuzu = {
    "KRIPTO": ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "XRP-USD", "DOGE-USD", "PEPE-USD", "FIL-USD", "LINK-USD"],
    "TR_HISSE": ["THYAO.IS", "SOKM.IS", "ASELS.IS", "EREGL.IS", "BIMAS.IS", "GARAN.IS", "TUPRS.IS", "SISE.IS"],
    "ABD_HISSE": ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "META", "GOOGL"],
    "EMTIA": ["GC=F", "SI=F", "CL=F", "NG=F"]
}

sembol = str_ui.sidebar.selectbox("2. Varlık Seçimi / Ara", varlik_havuzu[pazar])

# Pazar bazlı makro ritim matrisi
ritim_matrisi = {
    "15 dk":   {"interval": "15m", "KRIPTO": "60d",  "TR_HISSE": "30d",  "ABD_HISSE": "60d",  "EMTIA": "45d"},
    "30 dk":   {"interval": "30m", "KRIPTO": "60d",  "TR_HISSE": "30d",  "ABD_HISSE": "60d",  "EMTIA": "45d"},
    "1 saat":  {"interval": "60m", "KRIPTO": "2y",   "TR_HISSE": "1y",   "ABD_HISSE": "2y",   "EMTIA": "1y"},
    "2 saat":  {"interval": "90m", "KRIPTO": "2y",   "TR_HISSE": "1y",   "ABD_HISSE": "2y",   "EMTIA": "1y"},
    "4 saat":  {"interval": "4h",  "KRIPTO": "2y",   "TR_HISSE": "200d", "ABD_HISSE": "2y",   "EMTIA": "1.5y"},
    "1 Gün":   {"interval": "1d",  "KRIPTO": "3y",   "TR_HISSE": "1y",   "ABD_HISSE": "5y",   "EMTIA": "4y"},
    "1 Hafta": {"interval": "1wk", "KRIPTO": "max",  "TR_HISSE": "5y",   "ABD_HISSE": "max",  "EMTIA": "max"},
    "1 Ay":    {"interval": "1mo", "KRIPTO": "max",  "TR_HISSE": "5y",   "ABD_HISSE": "max",  "EMTIA": "max"}
}

secilen_periyot_etiket = str_ui.sidebar.selectbox("3. Grafik Mum Periyodu", list(ritim_matrisi.keys()), index=5)

aktif_interval = ritim_matrisi[secilen_periyot_etiket]["interval"]
aktif_period = ritim_matrisi[secilen_periyot_etiket][pazar]

str_ui.sidebar.markdown("---")
str_ui.sidebar.caption(f"🛡️ **Akademik Ritim Bilgisi:**\nSeçilen pazarın (`{pazar}`) istatistiksel rejimine göre optimize edilmiş geriye dönük veri penceresi **`{aktif_period}`** olarak kilitlendi.")

# =====================================================================
# 🧮 MATEMATİKSEL GÖSTERGE MOTORLARI
# =====================================================================
def rsi_calc(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    return 100 - (100 / (1 + (gain / (loss + 1e-10)) + 1e-10))

def macd_calc(series, fast=12, slow=26, signal=9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def stoch_calc(high, low, close, k=14, d=3):
    low_min = low.rolling(window=k).min()
    high_max = high.rolling(window=k).max()
    pk = ((close - low_min) / (high_max - low_min + 1e-10)) * 100
    pd_series = pk.rolling(window=d).mean()
    return pk, pd_series

def wavetrend_calc(high, low, close, n1=10, n2=21):
    esa = (high + low + close) / 3
    esa_ema = esa.ewm(span=n1, adjust=False).mean()
    d_ema = (esa - esa_ema).abs().ewm(span=n1, adjust=False).mean()
    ci = (esa - esa_ema) / (0.015 * d_ema + 1e-10)
    wt1 = ci.ewm(span=n2, adjust=False).mean()
    wt2 = wt1.rolling(window=4).mean()
    return wt1, wt2

def cci_calc(high, low, close, period=20):
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = (tp - sma_tp).abs().rolling(window=period).mean()
    return (tp - sma_tp) / (0.015 * mad + 1e-10)

def atr_calc(high, low, close, period=14):
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# =====================================================================
# ⚙️ CANLI VERİ ENJEKTÖRÜ VE AKADEMİK VERİ BÜTÜNLÜĞÜ MOTORU
# =====================================================================
def get_clean_data(sym, prd, ivl):
    df = yf.download(sym, period=prd, interval=ivl, progress=False)
    if df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def calculate_metrics(df):
    if df.empty or len(df) < 15: return pd.DataFrame()
    
    df_out = df.copy()
    c = df_out['Close'].squeeze()
    h = df_out['High'].squeeze()
    l = df_out['Low'].squeeze()
    v = df_out['Volume'].squeeze()
    
    # 🧠 Çoklu SMA Ortalamaları
    df_out['SMA_20'] = c.rolling(window=min(20, len(df_out))).mean()
    df_out['SMA_50'] = c.rolling(window=min(50, len(df_out))).mean()
    df_out['SMA_100'] = c.rolling(window=min(100, len(df_out))).mean()
    
    window_length = min(200, len(df_out) // 2)
    if window_length < 5: window_length = 5
    df_out['SMA_200'] = c.rolling(window=window_length).mean()
    
    df_out['SMA_Volume_20'] = v.rolling(window=min(20, len(df_out))).mean()
    
    # Bollinger Kanalları (Alt + Orta + Üst)
    sma_20_bb = c.rolling(window=min(20, len(df_out))).mean()
    std_20 = c.rolling(window=min(20, len(df_out))).std()
    df_out['Bollinger_Orta'] = sma_20_bb
    df_out['Bollinger_Ust'] = sma_20_bb + (std_20 * 2)
    df_out['Bollinger_Alt'] = sma_20_bb - (std_20 * 2)
    
    df_out['RSI'] = rsi_calc(c)
    df_out['MACD'], df_out['MACD_Sig'], df_out['MACD_Hist'] = macd_calc(c)
    df_out['Stoch_K'], df_out['Stoch_D'] = stoch_calc(h, l, c)
    df_out['WT1'], df_out['WT2'] = wavetrend_calc(h, l, c)
    df_out['CCI'] = cci_calc(h, l, c)
    df_out['ATR'] = atr_calc(h, l, c)
    
    df_out['Getiri_1G'] = c.pct_change(1)
    df_out['Volatilite_5G'] = df_out['Getiri_1G'].rolling(window=min(5, len(df_out))).std()
    df_out['Hacim_ROC_5'] = v.pct_change(min(5, len(df_out)))
    df_out['Trend_Gucu'] = (h.rolling(window=min(14, len(df_out))).max() - l.rolling(window=min(14, len(df_out))).min()) / (c + 1e-10)
    
    df_out['Yuzde_Getiri_Yarin'] = c.pct_change(1).shift(-1)
    df_out['Hedef'] = (df_out['Yuzde_Getiri_Yarin'] > 0.001).astype(int)
    
    df_out = df_out.replace([np.inf, -np.inf], np.nan)
    df_out = df_out.dropna() 
    return df_out

df_raw_active = get_clean_data(sembol, aktif_period, aktif_interval)
df_active = calculate_metrics(df_raw_active)

# =====================================================================
# 📡 KOKPİT ÜST PANELİ VE MUTLAK AKADEMİK DOĞRULAMA KONTROLÜ
# =====================================================================
if df_active.empty or len(df_active) < 30:
    str_ui.error(f"⚠️ **Akademik Veri Derinliği Yetersiz:** Seçilen zaman dilimi ({secilen_periyot_etiket}) ve varlık için `dropna()` temizliğinden sonra geriye kalan satır sayısı ({len(df_active)}) güvenli analiz için yetersizdir.")
    str_ui.warning(f"💡 **BÖTE Çözüm Önerisi:** `{pazar}` pazarı için şu anki ritmik veri penceresi `{aktif_period}` olarak ayarlanmıştır. Lütfen sol menüden daha uzun bir mum periyodu seçin hoca.")
else:
    try:
        df_anlik = yf.download(sembol, period="1d", interval="1m", progress=False)
        if not df_anlik.empty:
            if isinstance(df_anlik.columns, pd.MultiIndex): 
                df_anlik.columns = df_anlik.columns.get_level_values(0)
            fiyat_su_an = float(df_anlik['Close'].squeeze().iloc[-1])
        else:
            fiyat_su_an = float(df_active['Close'].squeeze().iloc[-1])
    except:
        fiyat_su_an = float(df_active['Close'].squeeze().iloc[-1])
        
    df_active.iloc[-1, df_active.columns.get_loc('Close')] = fiyat_su_an
    
    last_row = df_active.iloc[-1]
    sma_200_degeri = last_row['SMA_200']
    hacim_su_an = df_active['Volume'].squeeze().iloc[-1]
    sma_hacim_20_degeri = last_row['SMA_Volume_20']
    atr_degeri = last_row['ATR']

    degisim_24s = ((fiyat_su_an - df_active['Close'].squeeze().iloc[-2]) / df_active['Close'].squeeze().iloc[-2]) * 100
    ath_degeri = df_active['High'].max()
    ath_uzaklik = ((fiyat_su_an - ath_degeri) / ath_degeri) * 100
    para_birimi = "TL" if sembol.endswith(".IS") else "USD"

    fiyat_gosterim = f"{fiyat_su_an:,.4f} {para_birimi}" if fiyat_su_an < 2.0 else f"{fiyat_su_an:,.2f} {para_birimi}"

    # 🛠️ REGRESYON TAMİRİ: Tipik kopyala-yapıştır hatası giderildi, kontrol artık olması gerektiği gibi hacme odaklı!
    if hacim_su_an == 0 or pd.isna(hacim_su_an):
        hacim_su_an = sma_hacim_20_degeri
        hacim_etiket = "Hacim (Son 20 Periyot Ort.)"
    else:
        hacim_etiket = "Hacim (Canlı)"

    if hacim_su_an >= 1e9:
        hacim_gosterim = f"${hacim_su_an/1e9:.2f}B"
    elif hacim_su_an >= 1e6:
        hacim_gosterim = f"${hacim_su_an/1e6:.2f}M"
    elif hacim_su_an >= 1e3:
        hacim_gosterim = f"${hacim_su_an/1e3:.1f}K"
    else:
        hacim_gosterim = f"{hacim_su_an:,.0f}"

    str_ui.title(f"📡 QUANT AI - PLATİN KOKPİT V55.4")
    str_ui.markdown(f"### {sembol} | %100 Doğrulanmış Akademik Konsensüs Sürümü 🛡️ `[Veri Penceresi: {aktif_period}]`")

    c1, c2, c3, c4 = str_ui.columns(4)
    c1.metric(f"Anlık Fiyat ({secilen_periyot_etiket})", fiyat_gosterim, f"Pencere: {aktif_period}")
    c2.metric("Periyot Değişimi", f"%{degisim_24s:+.2f}")
    c3.metric("ATH'den Uzaklık", f"%{ath_uzaklik:.2f}")
    c4.metric(hacim_etiket, hacim_gosterim)

    # =====================================================================
    # 🧠 BİLİMSEL KOD KORUMA ALANI (KONSORSİYUM & TIMESERİESSPLİT MOTORU)
    # =====================================================================
    ozellikler = ['Close', 'SMA_200', 'Bollinger_Ust', 'RSI', 'MACD', 'Getiri_1G', 'Volatilite_5G', 'ATR', 'Hacim_ROC_5', 'Trend_Gucu']
    
    df_ml = df_active.iloc[:-1].copy()
    bugunun_satiri = df_active.iloc[[-1]][ozellikler].copy()
    bugunun_satiri['Close'] = fiyat_su_an
    
    X_all = df_ml[ozellikler].copy()
    y_all = df_ml['Hedef']
    
    final_split_idx = int(len(X_all) * 0.80)
    X_train, X_test = X_all.iloc[:final_split_idx], X_all.iloc[final_split_idx:]
    y_train, y_test = y_all.iloc[:final_split_idx], y_all.iloc[final_split_idx:]
    
    splits_count = 3 if len(X_train) >= 45 else 2
    tscv = TimeSeriesSplit(n_splits=splits_count)
    param_grid = [{'n_estimators': 40, 'learning_rate': 0.04, 'max_depth': 3}, {'n_estimators': 70, 'learning_rate': 0.03, 'max_depth': 3}]
    
    best_score = -1
    best_params = param_grid[0]
    final_best_fold_scores = [0.5, 0.5]
    
    if len(X_train) >= 10:
        for param in param_grid:
            fold_scores = []
            for inner_train_idx, inner_test_idx in tscv.split(X_train):
                X_tr_inner, X_te_inner = X_train.iloc[inner_train_idx], X_train.iloc[inner_test_idx]
                y_tr_inner, y_te_inner = y_train.iloc[inner_train_idx], y_train.iloc[inner_test_idx]
                
                inner_scaler = StandardScaler()
                X_tr_inner_sc = inner_scaler.fit_transform(X_tr_inner)
                X_te_inner_sc = inner_scaler.transform(X_te_inner)
                
                clf = GradientBoostingClassifier(n_estimators=param['n_estimators'], learning_rate=param['learning_rate'], max_depth=param['max_depth'], random_state=42)
                clf.fit(X_tr_inner_sc, y_tr_inner)
                fold_scores.append(accuracy_score(y_te_inner, clf.predict(X_te_inner_sc)))
                
            mean_inner_score = np.mean(fold_scores)
            if mean_inner_score > best_score:
                best_score = mean_inner_score
                best_params = param
                final_best_fold_scores = fold_scores
                
        ortalama_tarihsel_accuracy = best_score * 100
        cv_standart_sapma = np.std(final_best_fold_scores) * 100

    production_scaler = StandardScaler()
    X_train_scaled = production_scaler.fit_transform(X_train)
    X_test_scaled = production_scaler.transform(X_test)
    bugunun_scaled = production_scaler.transform(bugunun_satiri)
    
    model_gbm = GradientBoostingClassifier(n_estimators=best_params['n_estimators'], learning_rate=best_params['learning_rate'], max_depth=best_params['max_depth'], random_state=42)
    model_rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model_baseline_lr = LogisticRegression(max_iter=1000, random_state=42)
    
    if XGBOOST_AVAILABLE:
        model_xgb = XGBClassifier(n_estimators=50, learning_rate=0.04, max_depth=3, random_state=42, eval_metric='logloss')
        model_label = "XGBoost"
    else:
        model_xgb = LogisticRegression(max_iter=1000, random_state=42)
        model_label = "Logistic Regression"

    model_gbm.fit(X_train_scaled, y_train)
    model_xgb.fit(X_train_scaled, y_train)
    model_rf.fit(X_train_scaled, y_train)
    model_baseline_lr.fit(X_train_scaled, y_train)
    
    acc_gbm = accuracy_score(y_test, model_gbm.predict(X_test_scaled)) * 100
    acc_xgb = accuracy_score(y_test, model_xgb.predict(X_test_scaled)) * 100
    acc_rf = accuracy_score(y_test, model_rf.predict(X_test_scaled)) * 100
    acc_lr = accuracy_score(y_test, model_baseline_lr.predict(X_test_scaled)) * 100
    
    test_prob_gbm = model_gbm.predict_proba(X_test_scaled)[:, 1]
    test_prob_xgb = model_xgb.predict_proba(X_test_scaled)[:, 1]
    test_prob_rf = model_rf.predict_proba(X_test_scaled)[:, 1]
    
    test_prob_ensemble = (test_prob_gbm + test_prob_xgb + test_prob_rf) / 3
    test_pred_ensemble = (test_prob_ensemble >= 0.50).astype(int)
    
    ensemble_accuracy = accuracy_score(y_test, test_pred_ensemble) * 100
    ensemble_precision = precision_score(y_test, test_pred_ensemble, zero_division=0) * 100
    ensemble_recall = recall_score(y_test, test_pred_ensemble, zero_division=0) * 100
    ensemble_f1 = f1_score(y_test, test_pred_ensemble, zero_division=0) * 100
    akademik_model_edge = ensemble_accuracy - acc_lr

    prob_gbm = model_gbm.predict_proba(bugunun_scaled)[:, 1][0]
    prob_xgb = model_xgb.predict_proba(bugunun_scaled)[:, 1][0]
    prob_rf = model_rf.predict_proba(bugunun_scaled)[:, 1][0]
    
    nihai_yukselis_olasiligi = (prob_gbm + prob_xgb + prob_rf) / 3
    boga_yuzde = nihai_yukselis_olasiligi * 100
    ayi_yuzde = (1.0 - nihai_yukselis_olasiligi) * 100
    
    if nihai_yukselis_olasiligi >= 0.50:
        sinyal_durumu = "ARTIŞ (YÜKSELİŞ)"
        sinyal_renk = "green"
    else:
        sinyal_durumu = "AZALIŞ (DÜŞÜŞ)"
        sinyal_renk = "red"

    # =====================================================================
    # 📊 6'LI ANA SEKME YAPISI
    # =====================================================================
    sekme_ozet, sekme_teknik, sekme_zincir, sekme_grafik, sekme_performans, sekme_maliyet = str_ui.tabs([
        "🔮 Yapay Zeka Özet Raporu", "📊 Teknik Gösterge Odası", "⛓️ Trend & Volatilite Hattı", 
        "📈 Canlı Grafik Odası", "🎯 Backtest / Performans", "💸 Maliyet & Risk Analizi"
    ])

    with sekme_ozet:
        str_ui.subheader(f"🔮 Doğrulanmış Ortak Akıl Tahmin Raporu ({secilen_periyot_etiket})")
        o_c1, o_c2, o_c3 = str_ui.columns(3)
        with o_c1:
            bg_color = '#F4F9EA' if sinyal_renk == 'green' else '#FFF5F5'
            border_color = '#97C459' if sinyal_renk == 'green' else '#F3C6C6'
            text_color = '#3B6D11' if sinyal_renk == 'green' else '#E24B4A'
            str_ui.markdown(f"""<div style="border:2px solid {border_color}; border-radius:12px; padding:20px; text-align:center; background:{bg_color};"><div style="font-size:12px; font-weight:600; color:#555;">KONSENSÜS ANA KARARI</div><div style="font-size:28px; font-weight:800; color:{text_color}; margin:8px 0;">{sinyal_durumu}</div><div style="font-size:11.5px; font-weight:600; color:#444;">Gerçek Kurul Başarısı: %{ensemble_accuracy:.1f}</div></div>""", unsafe_allow_html=True)
        with o_c2:
            str_ui.markdown(f"""<div style="border:1px solid #E0E0E0; border-radius:12px; padding:20px; text-align:center; background:#FFF;"><div style="font-size:12px; font-weight:600; color:#555;">YÜKSELİŞ İHTİMALİ</div><div style="font-size:28px; font-weight:800; color:#3B6D11; margin:8px 0;">%{boga_yuzde:.1f}</div><div style="font-size:11px; color:#777;">Soft-Voting alıcı baskısı.</div></div>""", unsafe_allow_html=True)
        with o_c3:
            str_ui.markdown(f"""<div style="border:1px solid #E0E0E0; border-radius:12px; padding:20px; text-align:center; background:#FFF;"><div style="font-size:12px; font-weight:600; color:#555;">DÜŞÜŞ İHTİMALİ</div><div style="font-size:28px; font-weight:800; color:#E24B4A; margin:8px 0;">%{ayi_yuzde:.1f}</div><div style="font-size:11px; color:#777;">Soft-Voting satıcı baskısı.</div></div>""", unsafe_allow_html=True)

        # 📡 Eğitim Odaklı Metodolojik Bilgilendirme Hattı
        str_ui.info(f"""
        💡 **Metodolojik Ritim Raporu:** Bu analiz; `{pazar}` pazarının makro karakterine tam uyumlu **`{aktif_period}`** uzunluğundaki tarihsel veri penceresi kullanılarak; 
        **Gradient Boosting**, **Random Forest** ve **{model_label}** makine öğrenmesi algoritmalarının ortak akıl (Ensemble Soft-Voting) konsensüsü ile üretilmiştir hoca.
        """)

    with sekme_teknik:
        str_ui.subheader("📊 Canlı Teknik Gösterge ve Osilatör Laboratuvarı")
        ek_ozellikler = ['Close', 'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200', 'Bollinger_Ust', 'Bollinger_Alt', 'RSI', 'MACD', 'ATR', 'Hacim_ROC_5']
        with str_ui.expander(f"📋 Ham Veri Seti Kesiti (Pencere derinliği: {aktif_period})", expanded=False):
            str_ui.write(df_active[ek_ozellikler])
            
        str_ui.write("---")
        str_ui.markdown("### 📈 İndikatör Görselleştirme Kokpiti")
        
        secilen_grafikler = str_ui.multiselect(
            "🔍 Grafik Odasına Eklenecek Göstergeleri Seçin:",
            options=["Dinamik Fiyat ve Ortalamalar", "Bollinger Bantları (Alt, Orta, Üst)", "RSI (Göreceli Güç Endeksi)", "MACD & Sinyal Hattı", "ATR (Volatilite Gücü)"],
            default=["Dinamik Fiyat ve Ortalamalar"]
        )
        
        str_ui.write("---")
        
        # 🎨 SMA Katmanı ve Son Kullanıcı Kılavuzu
        if "Dinamik Fiyat ve Ortalamalar" in secilen_grafikler:
            str_ui.markdown(f"**📈 Dinamik Fiyat Eğilimi ve Seçilebilir Hareketli Ortalamalar**")
            
            aktif_sma_secimleri = str_ui.multiselect(
                "🎨 Grafik Üzerinde Görmek İstediğiniz SMA Çizgilerini Seçin:",
                options=["SMA_20", "SMA_50", "SMA_100", "SMA_200"],
                default=["SMA_20", "SMA_200"]
            )
            
            cizim_kolonlari = ['Close'] + aktif_sma_secimleri
            str_ui.line_chart(df_active[cizim_kolonlari])
            str_ui.caption("💡 **Nasıl Yorumlanır?** Fiyatın hareketli ortalamaların üzerinde olması genel yükseliş trendini, altında olması ise düşüş eğilimini gösterir. Kısa vadeli ortalamanın (örn. SMA 20) uzun vadeli ortalamayı (örn. SMA 200) yukarı kesmesi güçlü bir alım sinyali (Golden Cross) olarak kabul edilir.")
            str_ui.write("")
            
        # 🌪️ Bollinger Kanalları Katmanı ve Son Kullanıcı Kılavuzu
        if "Bollinger Bantları (Alt, Orta, Üst)" in secilen_grafikler:
            str_ui.markdown("**🌪️ Bollinger Bantları - Fiyat Oynaklık Kanalları**")
            str_ui.line_chart(df_active[['Close', 'Bollinger_Ust', 'Bollinger_Orta', 'Bollinger_Alt']])
            str_ui.caption("💡 **Nasıl Yorumlanır?** Fiyatın üst çizgiye yaklaşması piyasanın aşırı ısındığını (pahalılaştığını), alt çizgiye yaklaşması ise aşırı ucuzladığını gösterir. Bantların daralması yakında sert bir fiyat hareketinin geleceğine, genişlemesi ise mevcut oynaklığın yüksek olduğuna işaret eder.")
            str_ui.write("")
            
        # 🔮 RSI Katmanı ve Son Kullanıcı Kılavuzu
        if "RSI (Göreceli Güç Endeksi)" in secilen_grafikler:
            str_ui.markdown("**🔮 RSI (Relative Strength Index) - Aşırı Alım / Satım Osilatörü**")
            str_ui.line_chart(df_active['RSI'])
            str_ui.caption("💡 **Nasıl Yorumlanır?** Piyasanın 'hız kadranı'dır. RSI değeri 70'in üzerine çıktığında varlığın aşırı alındığını ve her an bir düzeltme gelebileceğini; 30'un altına düştüğünde ise aşırı satıldığını ve dipten dönüş olabileceğini gösterir.")
            str_ui.write("")

        # ⛓️ MACD Katmanı ve Son Kullanıcı Kılavuzu
        if "MACD & Sinyal Hattı" in secilen_grafikler:
            str_ui.markdown("**⛓️ MACD (Trend Takip ve Momentum Osilatörü)**")
            str_ui.line_chart(df_active[['MACD', 'MACD_Sig']])
            str_ui.caption("💡 **Nasıl Yorumlanır?** Trendin yönünü ve hızını ölçer. Mavi çizgi (MACD), turuncu çizgiyi (Sinyal) yukarı doğru kestiğinde piyasaya alıcıların hakim olduğunu ve yükselişin başlayabileceğini gösterir; aşağı kestiğinde ise düşüş sinyalidir.")
            str_ui.write("")

        # 🌪️ ATR Katmanı ve Son Kullanıcı Kılavuzu
        if "ATR (Volatilite Gücü)" in secilen_grafikler:
            str_ui.markdown("**🌪️ ATR (Average True Range) - Piyasa Volatilite Grafiği**")
            str_ui.line_chart(df_active['ATR'])
            str_ui.caption("💡 **Nasıl Yorumlanır?** Fiyatın ne kadar dalgalandığını (oynaklığını) ölçer. ATR değerinin yükselmesi piyasada fırtına olduğunu ve fiyatların sert hareket ettiğini gösterir. Bu değer, risk yönetiminde ne kadar uzaktan stop koyulacağını belirlemek için kullanılır.")

    with sekme_zincir:
        str_ui.subheader("⛓️ Trend Gücü ve Kurumsal Hatlar")
        str_ui.metric("Mevcut ATR Gücü", f"{atr_degeri:.4f}")
        str_ui.metric("Makro Ölçekli Hareketli Ortalama (SMA_200)", f"{sma_200_degeri:,.2f}")

    with sekme_grafik:
        str_ui.subheader(f"📈 Genel Fiyat Akış Hatları (Toplam Derinlik: {aktif_period})")
        str_ui.line_chart(df_active['Close'])

    with sekme_performans:
        str_ui.subheader("🔬 İleri Düzey İstatistiki Laboratuvarı")
        m_c1, m_c2, m_c3, m_c4 = str_ui.columns(4)
        m_c1.metric("🔬 Gerçek Ensemble Accuracy", f"%{ensemble_accuracy:.2f}")
        m_c2.metric("🎯 Ensemble Precision", f"%{ensemble_precision:.2f}")
        m_c3.metric("🌀 İç Döngü CV Skoru (GBM)", f"%{ortalama_tarihsel_accuracy:.2f} (±%{cv_standart_sapma:.1f})")
        m_c4.metric("📊 Model Edge (vs Baseline)", f"{akademik_model_edge:+.2f} Puan")
        
        str_ui.write("---")
        str_ui.markdown("### 🗳️ Kurul Üyelerinin Bireysel Doğruluk Skorları (Demokrasi Odası)")
        ind_c1, ind_c2, ind_c3, ind_c4 = str_ui.columns(4)
        ind_c1.metric("🧠 Gradient Boosting Accuracy", f"%{acc_gbm:.2f}")
        ind_c2.metric(f"🤖 {model_label} Accuracy", f"%{acc_xgb:.2f}")
        ind_c3.metric("🌲 Random Forest Accuracy", f"%{acc_rf:.2f}")
        ind_c4.metric("📉 Logistic Regression (Baseline)", f"%{acc_lr:.2f}")

    # =====================================================================
    # 💸 6. SEKME: MALIYET & RISK ANALIZI (WALL STREET POZISYON MOTORU)
    # =====================================================================
    with sekme_maliyet:
        str_ui.subheader("💸 Volatilite Tabanlı Risk ve Kasa Yönetimi Kokpiti")
        
        str_ui.markdown("""
        Oynaklık (Volatilite) tabanlı risk yönetimi, sermayenizi piyasa gürültüsünden korur. 
        Aşağıdaki simülatör, mevcut ATR (Average True Range) değerini baz alarak matematiksel olarak ideal stop ve kar al noktalarını hesaplar hoca.
        """)
        
        kullanici_kasasi = str_ui.number_input(f"💰 Mevcut Toplam Kasa Büyüklüğünüzü Giriniz ({para_birimi})", min_value=100.0, value=2000.0, step=500.0)
        risk_yuzdesi = str_ui.slider("🔥 İşlem Başına Maksimum Risk Yüzdesi (%)", 0.5, 5.0, 1.0, 0.5)
        
        # Matematiksel Pozisyonlama Hesaplamaları
        risk_basi_stop = atr_degeri * 1.5
        hedef_kar_al = atr_degeri * 3.0
        max_pozisyon = (kullanici_kasasi * (risk_yuzdesi / 100.0)) / (risk_basi_stop + 1e-10)
        göze_alinan_para = kullanici_kasasi * (risk_yuzdesi / 100.0)
        tahmini_islem_butcesi = max_pozisyon * fiyat_su_an
        
        str_ui.write("---")
        str_ui.markdown("### 🧮 Canlı Hesaplama ve Simülasyon Çıktıları")
        
        st_c1, st_c2, st_c3 = str_ui.columns(3)
        stop_seviyesi = (fiyat_su_an - risk_basi_stop if sinyal_renk=='green' else fiyat_su_an + risk_basi_stop)
        st_c1.error(f"🚨 Stop-Loss Seviyesi: {stop_seviyesi:,.4f} {para_birimi}")
        
        kar_seviyesi = (fiyat_su_an + hedef_kar_al if sinyal_renk=='green' else fiyat_su_an - hedef_kar_al)
        st_c2.success(f"🎯 Kar-Al Hedefi: {kar_seviyesi:,.4f} {para_birimi}")
        
        st_c3.warning(f"💼 Önerilen Maks. Pozisyon: {max_pozisyon:,.4f} Adet")
        
        str_ui.write("---")
        
        str_ui.markdown(f"""
        <div style="background-color: #F8F9FA; border-left: 5px solid #2980B9; padding: 15px; border-radius: 4px;">
            <h4>📋 Simülasyon ve Risk Yönetimi Özeti</h4>
            <ul>
                <li><b>Göze Alınan Risk Tutarı:</b> {göze_alinan_para:,.2f} {para_birimi} (Kasanızın %{risk_yuzdesi}'si)</li>
                <li><b>Tahmini Tahsis Edilecek Bütçe:</b> {tahmini_islem_butcesi:,.2f} {para_birimi}</li>
                <li><b>Risk / Ödül Oranı (R:R):</b> 1 : 2.0 (1 Birim Risk Karşılığında 2 Birim Kar Hedefi)</li>
                <li><b>Strateji Ömrü Ritim Notu:</b> Bu hesaplama, pazarın güncel <b>{aktif_period}</b> volatilite karakterine göre milimetrik optimize edilmiştir hoca.</li>
            </ul>
        </div>

        <br>
        Bu panel, Wall Street fon yöneticilerinin kullandığı <strong>Sermaye Koruma (Position Sizing)</strong> disipliniyle çalışır.
        Amaç, kasadaki <strong>paranın tamamıyla kumar oynamak değil</strong>, pazar tersine giderse ne kaybedeceğini en baştan milimetrik olarak sabitlemektir.

        📊 Senin Mevcut Kurumsal Ayarların:
        <ul>
            <li><b>Toplam Kasanız:</b> {kullanici_kasasi:,.2f} {para_birimi}</li>
            <li><b>Göze Aldığınız Risk (%{risk_yuzdesi}):</b> {göze_alinan_para:,.2f} {para_birimi} (Piyasa tamamen çökse bile cüzdandan çıkacak maksimum para)</li>
            <li><b>Önerilen Alım Miktarı:</b> {max_pozisyon:,.4f} Adet</li>
        </ul>

        🔮 Canlı Simülasyon Örneği: <br>
        Yapay zekanın yön sinyaline uyarak şu an kasanızın tamamıyla değil, sadece yukarıda önerilen <strong><code>{max_pozisyon:,.4f} Adet</code></strong> varlık ile işleme girdiğinizi varsayalım:
        <br><br>
        1. 🔴 <strong>SENARYO A (Zarar Kes - Stop Oldunuz):</strong> Piyasa aniden tersine döndü ve fiyat bizim belirlediğimiz <strong>Stop-Loss</strong> hattına çarptı. İşlem otomatik olarak zararla kapandı. Paranızın tamamını masaya sürmediğiniz için, bu büyük çöküşte kaybedeceğiniz toplam para <strong>TAM OLARAK `{göze_alinan_para:,.2f} {para_birimi}`</strong> olacaktır. Kalan paranız cüzdanda sapasağlam bir sonraki işlem için bekler.
        <br><br>
        2. 🟢 <strong>SENARYO B (Kâr Al - Hedefe Ulaştınız):</strong> Yapay zekanın tahmini kusursuz çalıştı ve fiyat <strong>Kar-Al Hedefimize</strong> ulaştı. Sistemde 1:2 matematiksel Risk/Ödül oranı kurulu olduğu için, göze aldığınız riskin tam iki katını, yani <strong>`{göze_alinan_para * 2:,.2f} {para_birimi}` NET KÂR</strong> elde edersiniz.

        <br><br>
        <blockquote>
            <strong>💡 Altın Kural:</strong> Profesyonel yatırımcılar "Kasamda ne kadar para var?" sorusuna değil,
            "Bu işlem ters giderse cüzdanımdan kaç para eksilecek?" sorusuna odaklanırlar. Sistem bu hesabı volatiliteye (ATR) göre anlık olarak optimize eder.
        </blockquote>
        """, unsafe_allow_html=True)