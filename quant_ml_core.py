import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# XGBoost Güvenlik Duvarı
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# --- 🧮 İNDİKATÖR MOTORLARI ---
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

def atr_calc(high, low, close, period=14):
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# --- ⚙️ METRİK VE VERİ TEMİZLEME MOTORU ---
def calculate_metrics(df):
    if df.empty or len(df) < 15: return pd.DataFrame()
    df_out = df.copy()
    c = df_out['Close'].squeeze()
    h = df_out['High'].squeeze()
    l = df_out['Low'].squeeze()
    v = df_out['Volume'].squeeze()
    
    # Çoklu SMA Katmanları
    df_out['SMA_20'] = c.rolling(window=min(20, len(df_out))).mean()
    df_out['SMA_50'] = c.rolling(window=min(50, len(df_out))).mean()
    df_out['SMA_100'] = c.rolling(window=min(100, len(df_out))).mean()
    
    window_length = min(200, len(df_out) // 2)
    if window_length < 5: window_length = 5
    df_out['SMA_200'] = c.rolling(window=window_length).mean()
    df_out['SMA_Volume_20'] = v.rolling(window=min(20, len(df_out))).mean()
    
    # Bollinger Kanalları
    sma_20_bb = c.rolling(window=min(20, len(df_out))).mean()
    std_20 = c.rolling(window=min(20, len(df_out))).std()
    df_out['Bollinger_Orta'] = sma_20_bb
    df_out['Bollinger_Ust'] = sma_20_bb + (std_20 * 2)
    df_out['Bollinger_Alt'] = sma_20_bb - (std_20 * 2)
    
    df_out['RSI'] = rsi_calc(c)
    df_out['MACD'], df_out['MACD_Sig'], _ = macd_calc(c)
    df_out['ATR'] = atr_calc(h, l, c)
    
    df_out['Getiri_1G'] = c.pct_change(1)
    df_out['Volatilite_5G'] = df_out['Getiri_1G'].rolling(window=min(5, len(df_out))).std()
    df_out['Hacim_ROC_5'] = v.pct_change(min(5, len(df_out)))
    df_out['Trend_Gucu'] = (h.rolling(window=min(14, len(df_out))).max() - l.rolling(window=min(14, len(df_out))).min()) / (c + 1e-10)
    
    df_out['Yuzde_Getiri_Yarin'] = c.pct_change(1).shift(-1)
    df_out['Hedef'] = (df_out['Yuzde_Getiri_Yarin'] > 0.001).astype(int)
    
    # 🛡️ Sızıntısız dropna() koruması
    df_out = df_out.replace([np.inf, -np.inf], np.nan).dropna()
    return df_out

# =====================================================================
# 🚀 ANA ANAHTAR: KANTİTATİF ANALİZ VE ENSEMBLE TAHMİN FONKSİYONU
# =====================================================================
def calistir_quant_analiz(sembol, pazar, aktif_period, aktif_interval):
    """
    Dışarıdan gelen parametrelere göre ML Ensemble pipeline'ını çalıştırır 
     ve Django/FastAPI arayüzünün tüketeceği temiz bir Python sözlüğü (dict) döner hoca.
    """
    # 1. Veri Çekme
    df_raw = yf.download(sembol, period=aktif_period, interval=aktif_interval, progress=False)
    if df_raw.empty:
        return {"durum": "hata", "mesaj": "Veri seti boş döndü."}
        
    if isinstance(df_raw.columns, pd.MultiIndex): 
        df_raw.columns = df_raw.columns.get_level_values(0)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]
    
    # 2. İndikatörleri Hesaplama
    df_active = calculate_metrics(df_raw)
    if df_active.empty or len(df_active) < 30:
        return {"durum": "yetersiz_veri", "satir_sayisi": len(df_active)}
        
    # 3. Anlık Fiyat ve Hacim Eşitleme Zırhı
    try:
        df_anlik = yf.download(sembol, period="1d", interval="1m", progress=False)
        if not df_anlik.empty:
            if isinstance(df_anlik.columns, pd.MultiIndex): df_anlik.columns = df_anlik.columns.get_level_values(0)
            fiyat_su_an = float(df_anlik['Close'].squeeze().iloc[-1])
        else:
            fiyat_su_an = float(df_active['Close'].squeeze().iloc[-1])
    except:
        fiyat_su_an = float(df_active['Close'].squeeze().iloc[-1])
        
    df_active.iloc[-1, df_active.columns.get_loc('Close')] = fiyat_su_an
    
    last_row = df_active.iloc[-1]
    hacim_su_an = float(last_row['Volume'])
    sma_hacim_20_degeri = float(last_row['SMA_Volume_20'])
    atr_degeri = float(last_row['ATR'])
    
    # 🛡️ V54.0 Akıllı Hacim Regresyon Tamiri
    if hacim_su_an == 0 or pd.isna(hacim_su_an):
        hacim_su_an = sma_hacim_20_degeri
        hacim_turu = "ortalama"
    else:
        hacim_turu = "canli"

    degisim_24s = ((fiyat_su_an - df_active['Close'].squeeze().iloc[-2]) / df_active['Close'].squeeze().iloc[-2]) * 100
    ath_degeri = float(df_active['High'].max())
    ath_uzaklik = ((fiyat_su_an - ath_degeri) / ath_degeri) * 100

    # 4. ML Pipeline & Zaman Serisi İzolasyonu
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
    
    if len(X_train) >= 10:
        for param in param_grid:
            fold_scores = []
            for inner_train_idx, inner_test_idx in tscv.split(X_train):
                y_tr_inner, y_te_inner = y_train.iloc[inner_train_idx], y_train.iloc[inner_test_idx]
                inner_scaler = StandardScaler()
                X_tr_inner_sc = inner_scaler.fit_transform(X_train.iloc[inner_train_idx])
                X_te_inner_sc = inner_scaler.transform(X_train.iloc[inner_test_idx])
                
                clf = GradientBoostingClassifier(n_estimators=param['n_estimators'], learning_rate=param['learning_rate'], max_depth=param['max_depth'], random_state=42)
                clf.fit(X_tr_inner_sc, y_tr_inner)
                fold_scores.append(accuracy_score(y_te_inner, clf.predict(X_te_inner_sc)))
            if np.mean(fold_scores) > best_score:
                best_score = np.mean(fold_scores)
                best_params = param

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
    
    # Skor Hesaplamaları
    acc_gbm = accuracy_score(y_test, model_gbm.predict(X_test_scaled)) * 100
    acc_xgb = accuracy_score(y_test, model_xgb.predict(X_test_scaled)) * 100
    acc_rf = accuracy_score(y_test, model_rf.predict(X_test_scaled)) * 100
    acc_lr = accuracy_score(y_test, model_baseline_lr.predict(X_test_scaled)) * 100
    
    test_prob_ensemble = (model_gbm.predict_proba(X_test_scaled)[:, 1] + model_xgb.predict_proba(X_test_scaled)[:, 1] + model_rf.predict_proba(X_test_scaled)[:, 1]) / 3
    test_pred_ensemble = (test_prob_ensemble >= 0.50).astype(int)
    
    ensemble_accuracy = accuracy_score(y_test, test_pred_ensemble) * 100
    ensemble_precision = precision_score(y_test, test_pred_ensemble, zero_division=0) * 100
    ensemble_recall = recall_score(y_test, test_pred_ensemble, zero_division=0) * 100
    ensemble_f1 = f1_score(y_test, test_pred_ensemble, zero_division=0) * 100
    
    prob_ensemble = (model_gbm.predict_proba(bugunun_scaled)[:, 1][0] + model_xgb.predict_proba(bugunun_scaled)[:, 1][0] + model_rf.predict_proba(bugunun_scaled)[:, 1][0]) / 3
    
    # 5. Paketleme (Django/FastAPI'ye fırlatılacak nihai mühimmat)
    return {
        "durum": "basarili",
        "meta": {"sembol": sembol, "pazar": pazar, "period": aktif_period, "interval": aktif_interval},
        "anlik_veri": {
            "fiyat": fiyat_su_an,
            "degisim_24s": degisim_24s,
            "ath_uzaklik": ath_uzaklik,
            "hacim": hacim_su_an,
            "hacim_turu": hacim_turu,
            "atr": atr_degeri
        },
        "tahmin_raporu": {
            "karar": "ARTIŞ (YÜKSELİŞ)" if prob_ensemble >= 0.50 else "AZALIŞ (DÜŞÜŞ)",
            "boga_ihtimali": prob_ensemble * 100,
            "ayi_ihtimali": (1.0 - prob_ensemble) * 100,
            "model_label": model_label
        },
        "performans_metrikleri": {
            "ensemble_accuracy": ensemble_accuracy,
            "ensemble_precision": ensemble_precision,
            "ensemble_recall": ensemble_recall,
            "ensemble_f1": ensemble_f1,
            "model_edge": ensemble_accuracy - acc_lr,
            "bireysel_skorlar": {"gbm": acc_gbm, "xgb_lr": acc_xgb, "rf": acc_rf, "baseline_lr": acc_lr}
        },
        "grafik_verisi": {
            "tarihler": df_active.index.strftime('%Y-%m-%d %H:%M').tolist(),
            "close": df_active['Close'].squeeze().tolist(),
            "sma_20": df_active['SMA_20'].squeeze().tolist(),
            "sma_50": df_active['SMA_50'].squeeze().tolist(),
            "sma_100": df_active['SMA_100'].squeeze().tolist(),
            "sma_200": df_active['SMA_200'].squeeze().tolist(),
            "bollinger_ust": df_active['Bollinger_Ust'].squeeze().tolist(),
            "bollinger_alt": df_active['Bollinger_Alt'].squeeze().tolist(),
            "rsi": df_active['RSI'].squeeze().tolist(),
            "macd": df_active['MACD'].squeeze().tolist()
        }
    }