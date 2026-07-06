import streamlit as str_ui
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

np.random.seed(42)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

str_ui.set_page_config(page_title="Quant AI - Sovereign Pro V58.0", page_icon="🏛️", layout="wide")

str_ui.markdown("""
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
    h1 {color: #1A1A1A; font-weight: 700;}
    h2 {color: #2C3E50; font-weight: 600; margin-top: 1rem;}
    .indicator-card {background: #FFFFFF; border: 1px solid #E0E0E0; border-radius: 12px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); height: 100%; text-align: center;}
    .ind-title {font-size: 14px; font-weight: 700; color: #333333; margin-bottom: 2px;}
    .ind-value {font-size: 26px; font-weight: 800; color: #111111; margin: 10px 0;}
    .ind-desc {font-size: 11px; color: #555555; text-align: left; line-height: 1.4;}
    </style>
""", unsafe_allow_html=True)

str_ui.sidebar.header("🏛️ Sovereign V58.0 Pro")
pazar = str_ui.sidebar.selectbox("1. Pazar Seçimi", ["KRIPTO", "TR_HISSE", "ABD_HISSE", "EMTIA"])

varlik_havuzu = {
    "KRIPTO": ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "XRP-USD", "DOGE-USD", "PEPE-USD", "FIL-USD", "LINK-USD"],
    "TR_HISSE": ["THYAO.IS", "SOKM.IS", "ASELS.IS", "EREGL.IS", "BIMAS.IS", "GARAN.IS", "TUPRS.IS", "SISE.IS"],
    "ABD_HISSE": ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "META", "GOOGL"],
    "EMTIA": ["GC=F", "SI=F", "CL=F", "NG=F"]
}
sembol = str_ui.sidebar.selectbox("2. Varlık Seçimi / Ara", varlik_havuzu[pazar])

ritim_matrisi = {
    "15 dk":   {"interval": "15m", "annual_factor": 252 * 6.5 * 4, "KRIPTO": "60d",  "TR_HISSE": "30d",  "ABD_HISSE": "60d", "EMTIA": "45d"},
    "30 dk":   {"interval": "30m", "annual_factor": 252 * 6.5 * 2, "KRIPTO": "60d",  "TR_HISSE": "30d",  "ABD_HISSE": "60d", "EMTIA": "45d"},
    "1 saat":  {"interval": "60m", "annual_factor": 252 * 6.5,     "KRIPTO": "2y",   "TR_HISSE": "1y",   "ABD_HISSE": "2y",  "EMTIA": "1y"},
    "4 saat":  {"interval": "1h",  "annual_factor": 252 * 6.5,     "KRIPTO": "2y",   "TR_HISSE": "1y",   "ABD_HISSE": "2y",  "EMTIA": "1.5y"},
    "1 Gün":   {"interval": "1d",  "annual_factor": 252,           "KRIPTO": "3y",   "TR_HISSE": "1y",   "ABD_HISSE": "5y",  "EMTIA": "4y"}
}
secilen_periyot_etiket = str_ui.sidebar.selectbox("3. Grafik Mum Periyodu", list(ritim_matrisi.keys()), index=4)

aktif_interval = ritim_matrisi[secilen_periyot_etiket]["interval"]
annual_factor = ritim_matrisi[secilen_periyot_etiket]["annual_factor"]
aktif_period = ritim_matrisi[secilen_periyot_etiket][pazar]

str_ui.sidebar.markdown("---")
str_ui.sidebar.caption(f"🛡️ **Akademik Ritim Bilgisi:**\nVeri penceresi **`{aktif_period}`** olarak kilitlendi.")

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
    return pk, pk.rolling(window=d).mean()

def wavetrend_calc(high, low, close, n1=10, n2=21):
    esa = (high + low + close) / 3
    esa_ema = esa.ewm(span=n1, adjust=False).mean()
    d_ema = (esa - esa_ema).abs().ewm(span=n1, adjust=False).mean()
    ci = (esa - esa_ema) / (0.015 * d_ema + 1e-10)
    wt1 = ci.ewm(span=n2, adjust=False).mean()
    return wt1, wt1.rolling(window=4).mean()

def cci_calc(high, low, close, period=20):
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = (tp - sma_tp).abs().rolling(window=period).mean()
    return (tp - sma_tp) / (0.015 * mad + 1e-10)

def atr_calc(high, low, close, period=14):
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_metrics(df):
    if df.empty or len(df) < 35: return pd.DataFrame()
    df_out = df.copy()
    c = df_out['Close'].squeeze()
    h = df_out['High'].squeeze()
    l = df_out['Low'].squeeze()
    v = df_out['Volume'].squeeze()

    df_out['EMA_20'] = c.ewm(span=20, adjust=False).mean()
    df_out['EMA_50'] = c.ewm(span=50, adjust=False).mean()
    df_out['SMA_200'] = c.rolling(window=min(200, len(df_out) // 2)).mean()
    df_out['SMA_Volume_20'] = v.rolling(window=min(20, len(df_out))).mean()

    sma_20 = c.rolling(20).mean()
    std_20 = c.rolling(20).std()
    df_out['Bollinger_Orta'] = sma_20
    df_out['Bollinger_Ust'] = sma_20 + (std_20 * 2)
    df_out['Bollinger_Alt'] = sma_20 - (std_20 * 2)
    df_out['Bollinger_Width'] = (df_out['Bollinger_Ust'] - df_out['Bollinger_Alt']) / (sma_20 + 1e-10)

    df_out['RSI'] = rsi_calc(c)
    df_out['MACD'], df_out['MACD_Sig'], df_out['MACD_Hist'] = macd_calc(c)
    df_out['Stoch_K'], df_out['Stoch_D'] = stoch_calc(h, l, c)
    df_out['WT1'], df_out['WT2'] = wavetrend_calc(h, l, c)
    df_out['CCI'] = cci_calc(h, l, c)
    df_out['ATR'] = atr_calc(h, l, c)
    df_out['Getiri_1G'] = c.pct_change(1)
    df_out['Volatilite_5G'] = df_out['Getiri_1G'].rolling(5).std()
    df_out['Hacim_ROC_5'] = v.pct_change(5)
    df_out['Fiyat_SMA200_Orani'] = c / (df_out['SMA_200'] + 1e-10)
    df_out['Trend_Gucu'] = (h.rolling(14).max() - l.rolling(14).min()) / (c + 1e-10)

    width_mean = df_out['Bollinger_Width'].rolling(20).mean()
    df_out['Regime_Sideways'] = np.where(df_out['Bollinger_Width'] < width_mean * 0.8, 1.0, 0.0)
    df_out['Regime_Bull'] = np.where((df_out['Regime_Sideways'] == 0) & (c > df_out['SMA_200']), 1.0, 0.0)
    df_out['Regime_Bear'] = np.where((df_out['Regime_Sideways'] == 0) & (c <= df_out['SMA_200']), 1.0, 0.0)

    df_out['Yuzde_Getiri_3G'] = c.pct_change(3).shift(-3)
    df_out['Hedef'] = (df_out['Yuzde_Getiri_3G'] > 0.002).astype(int)
    return df_out.replace([np.inf, -np.inf], np.nan).dropna()

# =====================================================================
# 🧠 ÖNBELLEKLİ MODEL FABRİKASI — Sadece klasik modeller (LSTM kaldırıldı)
# =====================================================================
@str_ui.cache_resource(show_spinner=False)
def egit_klasik_modelleri(_X_tr_sc, _y_tr, scale_pos_weight_value, cache_anahtari):
    m_gbm = GradientBoostingClassifier(n_estimators=40, max_depth=3, random_state=42).fit(_X_tr_sc, _y_tr)
    m_rf = RandomForestClassifier(n_estimators=40, max_depth=5, class_weight='balanced', random_state=42).fit(_X_tr_sc, _y_tr)
    m_xgb = XGBClassifier(n_estimators=40, max_depth=3, learning_rate=0.03, subsample=0.8, scale_pos_weight=scale_pos_weight_value, random_state=42, eval_metric='logloss') if XGB_AVAILABLE else LogisticRegression(class_weight='balanced')
    m_xgb.fit(_X_tr_sc, _y_tr)
    return m_gbm, m_rf, m_xgb

@str_ui.cache_resource(show_spinner=False)
def egit_uretim_modellerini(_X_train_sc, _y_train, scale_pos_weight_value, cache_anahtari):
    model_gbm = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42).fit(_X_train_sc, _y_train)
    model_rf = RandomForestClassifier(n_estimators=50, max_depth=5, class_weight='balanced', random_state=42).fit(_X_train_sc, _y_train)
    model_xgb = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.03, subsample=0.8, scale_pos_weight=scale_pos_weight_value, random_state=42, eval_metric='logloss') if XGB_AVAILABLE else LogisticRegression(class_weight='balanced')
    model_xgb.fit(_X_train_sc, _y_train)
    return model_gbm, model_rf, model_xgb

str_ui.title("🏛️ QUANT AI - SOVEREIGN COCKPIT V58.0")
str_ui.markdown(f"### {sembol} | %100 Güvenli Katman Kontrollü Altyapı Sürümü 👑 — 3'lü Konsensüs Çekirdeği ⚡")

df_raw = yf.download(sembol, period=aktif_period, interval=aktif_interval, progress=False)
if isinstance(df_raw.columns, pd.MultiIndex):
    df_raw.columns = df_raw.columns.get_level_values(0)
df_raw.columns = [str(c).strip() for c in df_raw.columns]

df_active = calculate_metrics(df_raw)
veri_yetersiz = df_active.empty or len(df_active) < 30

if not veri_yetersiz:
    fiyat_su_an = float(df_active['Close'].squeeze().iloc[-1])
    degisim_24s = ((fiyat_su_an - df_active['Close'].squeeze().iloc[-2]) / df_active['Close'].squeeze().iloc[-2]) * 100
    atr_gucu = float(df_active['ATR'].iloc[-1])
    sma_200_degeri = float(df_active['SMA_200'].iloc[-1])
    para_birimi = "TL" if sembol.endswith(".IS") else "USD"

    ozellikler = ['Close', 'EMA_20', 'EMA_50', 'SMA_200', 'Bollinger_Width', 'RSI', 'MACD', 'ATR', 'Getiri_1G', 'Volatilite_5G', 'Fiyat_SMA200_Orani', 'Trend_Gucu', 'Regime_Sideways', 'Regime_Bull', 'Regime_Bear']
    df_ml = df_active.iloc[:-3].copy()
    X_all = df_ml[ozellikler].copy()
    y_all = df_ml['Hedef']

    split_idx = int(len(X_all) * 0.80)
    X_train, X_test = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
    y_train, y_test = y_all.iloc[:split_idx], y_all.iloc[split_idx:]

    counts_tr = y_train.value_counts(normalize=True)
    scale_pos_weight_value = max(0.1, counts_tr.get(0, 0.5) / (counts_tr.get(1, 0.5) + 1e-10))

    cache_anahtari = f"{sembol}_{secilen_periyot_etiket}_{pazar}_{len(X_all)}"

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    bugunun_sc = scaler.transform(df_active.iloc[[-1]][ozellikler])

    tscv = TimeSeriesSplit(n_splits=3)
    final_scores = {"gbm": [], "rf": [], "xgb": []}
    acc_pure_scores = {"gbm": [], "rf": [], "xgb": []}

    def ic_dongu_karma_skor_hesapla(preds, forward_returns_sliced, y_true_sliced):
        s_ret = np.where(preds == 1, forward_returns_sliced, -forward_returns_sliced)
        g = np.sum(s_ret[s_ret > 0])
        l = np.abs(np.sum(s_ret[s_ret < 0]))
        pf = np.clip(g / (l + 1e-10), 0.1, 5.0)
        pf_norm = pf / 5.0
        acc = accuracy_score(y_true_sliced, preds)
        return (0.5 * pf_norm) + (0.5 * acc), acc

    for fold_no, (tr_idx, te_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_te = X_train.iloc[tr_idx], X_train.iloc[te_idx]
        y_tr, y_te = y_train.iloc[tr_idx], y_train.iloc[te_idx]

        sc_inner = StandardScaler()
        X_tr_sc = sc_inner.fit_transform(X_tr)
        X_te_sc = sc_inner.transform(X_te)
        te_forward_returns = df_ml['Yuzde_Getiri_3G'].iloc[:split_idx].values[te_idx]

        fold_cache_anahtari = f"{cache_anahtari}_fold{fold_no}"
        m_gbm, m_rf, m_xgb = egit_klasik_modelleri(X_tr_sc, y_tr, scale_pos_weight_value, fold_cache_anahtari)

        for name, model in [("gbm", m_gbm), ("rf", m_rf), ("xgb", m_xgb)]:
            scr, ac = ic_dongu_karma_skor_hesapla(model.predict(X_te_sc), te_forward_returns, y_te)
            final_scores[name].append(scr)
            acc_pure_scores[name].append(ac)

    cv_gbm = max(0.1, np.mean(final_scores["gbm"])) if final_scores["gbm"] else 1.0
    cv_rf = max(0.1, np.mean(final_scores["rf"])) if final_scores["rf"] else 1.0
    cv_xgb = max(0.1, np.mean(final_scores["xgb"])) if final_scores["xgb"] else 1.0

    log_gbm = np.log1p(cv_gbm)
    log_rf = np.log1p(cv_rf)
    log_xgb = np.log1p(cv_xgb)

    toplam_log_pf = log_gbm + log_rf + log_xgb
    w_gbm = log_gbm / toplam_log_pf
    w_rf = log_rf / toplam_log_pf
    w_xgb = log_xgb / toplam_log_pf

    model_gbm, model_rf, model_xgb = egit_uretim_modellerini(X_train_sc, y_train, scale_pos_weight_value, cache_anahtari)

    prob_gbm = model_gbm.predict_proba(bugunun_sc)[0][1]
    prob_rf = model_rf.predict_proba(bugunun_sc)[0][1]
    prob_xgb = model_xgb.predict_proba(bugunun_sc)[0][1]

    boga_ihtimali = ((prob_gbm * w_gbm) + (prob_rf * w_rf) + (prob_xgb * w_xgb)) * 100
    ayi_ihtimali = 100 - boga_ihtimali
    karar = "ARTIŞ (YÜKSELİŞ)" if boga_ihtimali >= 50 else "AZALIŞ (DÜŞÜŞ)"

    t_p_gbm_sliced = model_gbm.predict_proba(X_test_sc)[:, 1]
    t_p_rf_sliced = model_rf.predict_proba(X_test_sc)[:, 1]
    t_p_xgb_sliced = model_xgb.predict_proba(X_test_sc)[:, 1]
    ens_probs_sliced = (t_p_gbm_sliced * w_gbm) + (t_p_rf_sliced * w_rf) + (t_p_xgb_sliced * w_xgb)

    test_signals_sliced = (ens_probs_sliced >= 0.50).astype(int)
    test_returns_sliced = df_ml['Yuzde_Getiri_3G'].iloc[split_idx:].values

    profit_factor, max_dd, sharpe = 1.0, 0.0, 0.0

    if len(test_signals_sliced) > 0 and len(test_returns_sliced) == len(test_signals_sliced):
        raw_strat_returns = np.where(test_signals_sliced == 1, test_returns_sliced, -test_returns_sliced)
        signal_changes = np.diff(test_signals_sliced, prepend=test_signals_sliced[0])
        strategy_returns = raw_strat_returns - np.where(signal_changes != 0, 0.0005, 0.0)

        gains = np.sum(strategy_returns[strategy_returns > 0])
        losses = np.abs(np.sum(strategy_returns[strategy_returns < 0]))
        if losses > 0:
            profit_factor = gains / losses

        cum_r = np.cumsum(strategy_returns)
        if len(cum_r) > 0:
            max_dd = np.max(np.maximum.accumulate(cum_r) - cum_r) * 100
        if len(strategy_returns) > 1 and np.std(strategy_returns) > 0:
            sharpe = (np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(annual_factor)

sekme_ozet, sekme_teknik, sekme_zincir, sekme_grafik, sekme_performans, sekme_maliyet = str_ui.tabs([
    "🔮 Yapay Zeka Özet Raporu", "📊 Teknik Gösterge Odası", "⛓️ Trend & Volatilite Hattı",
    "📈 Canlı Grafik Odası", "🎯 Backtest / Performans", "💸 Maliyet & Risk Analizi"
])

if veri_yetersiz:
    for tab in [sekme_ozet, sekme_teknik, sekme_zincir, sekme_grafik, sekme_performans, sekme_maliyet]:
        with tab:
            str_ui.error(f"⚠️ **Veri Derinliği Yetersiz:** Süzgeçten sonra kalan mum sayısı ({len(df_active)}) analize elvermiyor hoca.")
            str_ui.warning("💡 Lütfen sol menüden daha uzun bir mum periyodu (Örn: 1 Gün) seçerek havuzu genişletin.")
else:
    with sekme_ozet:
        str_ui.subheader(f"🔮 Doğrulanmış Ortak Akıl Tahmin Raporu ({secilen_periyot_etiket})")
        str_ui.markdown(f"""
        <div style="background-color: #0F172A; border: 1px solid #334155; border-radius: 12px; padding: 12px 20px; margin-bottom: 20px;">
            <span style="font-size: 11px; font-weight: 700; color: #94A3B8; letter-spacing: 0.05em;">📡 KONSENSÜS ÇEKİRDEĞİ MONİTÖRÜ</span>
            <div style="display: flex; gap: 40px; margin-top: 6px;">
                <div><span style="font-size: 11px; color: #64748B;">Mevcut Train Verisi:</span> <strong style="font-size: 14px; color: #F1F5F9;">{len(X_train)} Satır</strong></div>
                <div><span style="font-size: 11px; color: #64748B;">Model Mimarisi:</span> <strong style="font-size: 14px; color: #38BDF8;">3'lü Konsensüs (GBM + RF + XGBoost)</strong></div>
                <div><span style="font-size: 11px; color: #64748B;">Önbellek Durumu:</span> <strong style="font-size: 14px; color: #10B981;">AKTİF ⚡</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        o_c1, o_c2, o_c3 = str_ui.columns(3)
        with o_c1:
            bg_color = '#F4F9EA' if boga_ihtimali >= 50 else '#FFF5F5'
            border_color = '#97C459' if boga_ihtimali >= 50 else '#F3C6C6'
            text_color = '#3B6D11' if boga_ihtimali >= 50 else '#E24B4A'
            str_ui.markdown(f"""<div style="border:2px solid {border_color}; border-radius:12px; padding:20px; text-align:center; background:{bg_color};"><div style="font-size:12px; font-weight:600; color:#555;">KONSENSÜS ANA KARARI</div><div style="font-size:28px; font-weight:800; color:{text_color}; margin:8px 0;">{karar}</div></div>""", unsafe_allow_html=True)
        with o_c2:
            str_ui.markdown(f"""<div style="border:1px solid #E0E0E0; border-radius:12px; padding:20px; text-align:center; background:#FFF;"><div style="font-size:12px; font-weight:600; color:#555;">YÜKSELİŞ İHTİMALİ</div><div style="font-size:28px; font-weight:800; color:#3B6D11; margin:8px 0;">%{boga_ihtimali:.1f}</div></div>""", unsafe_allow_html=True)
        with o_c3:
            str_ui.markdown(f"""<div style="border:1px solid #E0E0E0; border-radius:12px; padding:20px; text-align:center; background:#FFF;"><div style="font-size:12px; font-weight:600; color:#555;">DÜŞÜŞ İHTİMALİ</div><div style="font-size:28px; font-weight:800; color:#E24B4A; margin:8px 0;">%{ayi_ihtimali:.1f}</div></div>""", unsafe_allow_html=True)

        str_ui.write("---")
        last_row_data = df_active.iloc[-1]
        card_c1, card_c2, card_c3, card_c4, card_c5, card_c6 = str_ui.columns(6)
        with card_c1:
            rsi_val = float(last_row_data['RSI'])
            str_ui.markdown(f"""<div class="indicator-card"><div class="ind-title">RSI (14)</div><div class="ind-value">{rsi_val:.1f}</div></div>""", unsafe_allow_html=True)
        with card_c2:
            macd_val = float(last_row_data['MACD'])
            str_ui.markdown(f"""<div class="indicator-card"><div class="ind-title">MACD</div><div class="ind-value">{macd_val:.2f}</div></div>""", unsafe_allow_html=True)
        with card_c3:
            stoch_k = float(last_row_data['Stoch_K'])
            str_ui.markdown(f"""<div class="indicator-card"><div class="ind-title">Stochastic</div><div class="ind-value">{stoch_k:.1f}</div></div>""", unsafe_allow_html=True)
        with card_c4:
            wt1_val = float(last_row_data['WT1'])
            str_ui.markdown(f"""<div class="indicator-card"><div class="ind-title">WaveTrend</div><div class="ind-value">{wt1_val:.1f}</div></div>""", unsafe_allow_html=True)
        with card_c5:
            cci_val = float(last_row_data['CCI'])
            str_ui.markdown(f"""<div class="indicator-card"><div class="ind-title">CCI</div><div class="ind-value">{cci_val:.1f}</div></div>""", unsafe_allow_html=True)
        with card_c6:
            atr_val = float(last_row_data['ATR'])
            str_ui.markdown(f"""<div class="indicator-card"><div class="ind-title">ATR (14)</div><div class="ind-value">{atr_val:.4f}</div></div>""", unsafe_allow_html=True)

    with sekme_teknik:
        str_ui.subheader("📊 Canlı Teknik Gösterge ve Osilatör Laboratuvarı")
        str_ui.line_chart(df_active[['Close', 'EMA_20', 'SMA_200']])

    with sekme_zincir:
        str_ui.subheader("⛓️ Makro Trend Gücü ve Rejim İzleme Hattı")
        son_satir = df_active.iloc[-1]
        rejim_str = "YATAY PİYASA 🔒" if son_satir['Regime_Sideways'] == 1 else ("BOĞA REJİMİ 🐂" if son_satir['Regime_Bull'] == 1 else "AYI REJİMİ 🐻")
        str_ui.warning(f"Piyasa Yapısı: {rejim_str}")

    with sekme_grafik:
        str_ui.subheader("📈 Canlı Fiyat Akış Hatları")
        str_ui.line_chart(df_active['Close'])

    with sekme_performans:
        str_ui.subheader("🎯 Kurul Üyelerinin Oy Güçleri")
        ind_c1, ind_c2, ind_c3 = str_ui.columns(3)
        ind_c1.metric("GBM Ağırlığı", f"%{w_gbm*100:.1f}", f"CV Skor: {cv_gbm:.2f}")
        ind_c2.metric("RF Ağırlığı", f"%{w_rf*100:.1f}", f"CV Skor: {cv_rf:.2f}")
        ind_c3.metric("XGBoost Ağırlığı", f"%{w_xgb*100:.1f}", f"CV Skor: {cv_xgb:.2f}")

        str_ui.write("---")
        str_ui.markdown("### 🗳️ İç Döngü Saf Doğruluk (Accuracy) İstatistikleri")
        a_c1, a_c2, a_c3 = str_ui.columns(3)
        a_c1.metric("GBM Fold Ortalaması", f"%{np.mean(acc_pure_scores['gbm'])*100:.1f}" if acc_pure_scores['gbm'] else "%0.0")
        a_c2.metric("RF Fold Ortalaması", f"%{np.mean(acc_pure_scores['rf'])*100:.1f}" if acc_pure_scores['rf'] else "%0.0")
        a_c3.metric("XGB Fold Ortalaması", f"%{np.mean(acc_pure_scores['xgb'])*100:.1f}" if acc_pure_scores['xgb'] else "%0.0")

    with sekme_maliyet:
        str_ui.subheader("💸 Volatilite Tabanlı Risk ve Kasa Yönetimi Kokpiti")
        kullanici_kasasi = str_ui.number_input(f"💰 Kasa Büyüklüğü ({para_birimi})", min_value=100.0, value=2000.0)
        risk_yuzdesi = str_ui.slider("🔥 Risk Yüzdesi (%)", 0.5, 5.0, 1.0)

        risk_basi_stop = atr_gucu * 1.5
        hedef_kar_al = atr_gucu * 3.0
        max_pozisyon = (kullanici_kasasi * (risk_yuzdesi / 100.0)) / (risk_basi_stop + 1e-10)

        stop_fiyat = fiyat_su_an - risk_basi_stop if karar == "ARTIŞ (YÜKSELİŞ)" else fiyat_su_an + risk_basi_stop
        kar_fiyat = fiyat_su_an + hedef_kar_al if karar == "ARTIŞ (YÜKSELİŞ)" else fiyat_su_an - hedef_kar_al

        st_c1, st_c2, st_c3 = str_ui.columns(3)
        st_c1.error(f"🚨 Stop-Loss: {stop_fiyat:,.2f}")
        st_c2.success(f"🎯 Kâr-Al Hedefi: {kar_fiyat:,.2f}")
        st_c3.warning(f"💼 Maks. Pozisyon: {max_pozisyon:,.4f} Adet")

        str_ui.markdown(f"""<div style="background-color: #F8F9FA; border-left: 5px solid #2980B9; padding: 15px; border-radius: 4px;"><ul><li><b>Realist Kârlılık Faktörü (3G PF):</b> {profit_factor:.2f}</li><li><b>Maksimum Çöküş (Max DD):</b> %{max_dd:.1f}</li><li><b>Sharpe Oranı:</b> {sharpe:.2f}</li></ul></div>""", unsafe_allow_html=True)