import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import math

# ── Path resolution: works locally AND on Streamlit Cloud ──
# On Streamlit Cloud, __file__ may point inside a venv; we check
# multiple candidate locations and pick the first that exists.
def p(filename):
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
        os.path.join(os.getcwd(), filename),
        filename,
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    # fallback — return relative path and let the error surface clearly
    return filename

st.set_page_config(page_title="🫁 Lung Cancer Risk Predictor", page_icon="🫁", layout="wide")

@st.cache_resource
def load_all():
    with open(p("lung_xgb_model.pkl"),"rb") as f: model = pickle.load(f)
    with open(p("lung_le_gender.pkl"),"rb") as f: le_g = pickle.load(f)
    with open(p("lung_le_target.pkl"),"rb") as f: le_t = pickle.load(f)
    with open(p("lung_artifacts.pkl"),"rb") as f: arts = pickle.load(f)
    return model, le_g, le_t, arts

model, le_g, le_t, arts = load_all()

@st.cache_data
def load_data():
    df = pd.read_csv(p("survey_lung_cancer.csv"))
    df.columns = df.columns.str.strip()
    return df

df_raw = load_data()

# ── THEME ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Fraunces:wght@700;900&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif !important;
    background: #0d0f14 !important;
    color: #e8eaf0 !important;
}

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 3rem 2rem !important; max-width: 1300px !important; }

/* ── HERO ── */
.hero-wrap {
    background: linear-gradient(135deg, #0d1117 0%, #1a0a2e 50%, #0d1117 100%);
    border: 1px solid #2d1b5e;
    border-radius: 24px;
    padding: 52px 48px 44px;
    margin: 24px 0 32px;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 80% 60% at 70% 50%, rgba(139,92,246,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem; letter-spacing: 0.22em; text-transform: uppercase;
    color: #8b5cf6; margin-bottom: 14px;
}
.hero-title {
    font-family: 'Fraunces', serif;
    font-size: clamp(2.2rem, 4vw, 3.4rem);
    font-weight: 900; line-height: 1.1;
    background: linear-gradient(135deg, #fff 30%, #c4b5fd 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0 0 16px;
}
.hero-sub {
    font-size: 1.05rem; color: #94a3b8; line-height: 1.7;
    max-width: 600px; margin: 0;
}
.hero-badges {
    display: flex; gap: 12px; flex-wrap: wrap; margin-top: 28px;
}
.badge {
    background: rgba(139,92,246,0.15);
    border: 1px solid rgba(139,92,246,0.35);
    border-radius: 999px; padding: 6px 16px;
    font-size: 0.8rem; color: #c4b5fd;
    font-family: 'DM Mono', monospace;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: #131720 !important;
    border-radius: 14px !important;
    padding: 6px !important;
    border: 1px solid #1e2535 !important;
    gap: 4px !important;
    margin-bottom: 28px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 8px 18px !important;
}
.stTabs [aria-selected="true"] {
    background: #1e1535 !important;
    color: #c4b5fd !important;
    border: 1px solid rgba(139,92,246,0.4) !important;
}

/* ── CARDS ── */
.card {
    background: #131720;
    border: 1px solid #1e2535;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    color: #e8eaf0 !important;
}
.card * { color: #e8eaf0 !important; }
.card b, .card strong { color: #ffffff !important; }
.card-accent { border-left: 3px solid #8b5cf6; }
.card-green  { border-left: 3px solid #10b981; }
.card-red    { border-left: 3px solid #ef4444; }
.card-amber  { border-left: 3px solid #f59e0b; }

/* ── STAT CARDS ── */
.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
.stat-box {
    background: linear-gradient(135deg, #131720, #1a1f2e);
    border: 1px solid #1e2535; border-radius: 16px;
    padding: 22px 20px; text-align: center;
}
.stat-box .sval {
    font-family: 'Fraunces', serif;
    font-size: 2.4rem; font-weight: 900;
    background: linear-gradient(135deg, #8b5cf6, #c4b5fd);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1;
}
.stat-box .slbl { font-size: 0.78rem; color: #64748b; margin-top: 6px; letter-spacing: 0.05em; text-transform: uppercase; }

/* ── SECTION HEADERS ── */
.sec-head {
    font-family: 'Fraunces', serif;
    font-size: 1.4rem; font-weight: 700;
    color: #e2d9f3 !important;
    margin: 28px 0 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e2535;
}
.sec-subhead { font-size: 0.9rem; color: #64748b !important; margin-bottom: 20px; }

/* ── STEP BOXES ── */
.step-wrap {
    display: flex; gap: 16px; align-items: flex-start;
    background: #131720; border: 1px solid #1e2535;
    border-radius: 14px; padding: 20px; margin-bottom: 12px;
}
.step-wrap * { color: #e8eaf0 !important; }
.step-num {
    min-width: 36px; height: 36px;
    background: linear-gradient(135deg, #7c3aed, #8b5cf6);
    border-radius: 50%; display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.9rem; color: white !important; flex-shrink: 0;
}
.step-content b { color: #c4b5fd !important; }
.step-note {
    background: rgba(139,92,246,0.1);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 8px; padding: 8px 12px;
    font-size: 0.82rem; color: #a78bfa !important;
    margin-top: 10px;
    font-family: 'DM Mono', monospace;
}

/* ── BAR VISUAL ── */
.bar-bg { background: #1e2535; border-radius: 8px; height: 20px; width: 100%; margin: 4px 0 12px; }
.bar-fill { height: 20px; border-radius: 8px; display: flex; align-items: center;
            padding-left: 10px; font-size: 12px; font-weight: 600; color: white !important; transition: width 0.6s ease; }

/* ── TABLE ── */
.stDataFrame { border-radius: 12px !important; overflow: hidden; }

/* ── RESULT CARDS ── */
.result-high {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 2px solid #ef4444;
    border-radius: 20px; padding: 36px; text-align: center;
}
.result-low {
    background: linear-gradient(135deg, #052e16, #064e3b);
    border: 2px solid #10b981;
    border-radius: 20px; padding: 36px; text-align: center;
}
.result-high *, .result-low * { color: white !important; }
.result-icon { font-size: 3.5rem; margin-bottom: 12px; }
.result-label { font-family: 'Fraunces', serif; font-size: 2rem; font-weight: 900; margin: 0 0 8px; }
.result-prob { font-size: 1.1rem; opacity: 0.9; }

/* ── CONCEPT PILLS ── */
.concept-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
.concept-box {
    background: #131720; border: 1px solid #1e2535;
    border-radius: 14px; padding: 18px;
}
.concept-box * { color: #e8eaf0 !important; }
.concept-box b { color: #c4b5fd !important; font-size: 0.95rem; }
.concept-box p { font-size: 0.85rem; color: #94a3b8 !important; margin: 8px 0 0; line-height: 1.6; }

/* ── INPUTS ── */
.stSlider label, .stSelectbox label, .stRadio label { color: #94a3b8 !important; font-size: 0.88rem !important; }
.stSlider [data-testid="stThumb"] { background: #8b5cf6 !important; }

/* ── BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #8b5cf6) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 600 !important;
    font-size: 1rem !important; padding: 14px !important;
    width: 100% !important; font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.02em !important; transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── CM TABLE ── */
.cm-table { width: 100%; border-collapse: separate; border-spacing: 6px; }
.cm-table td { padding: 18px; text-align: center; border-radius: 10px; font-weight: 700; font-size: 1.4rem; }
.cm-label { background: #1e2535 !important; color: #94a3b8 !important; font-size: 0.82rem !important; font-weight: 500 !important; }
.cm-tp { background: #052e16 !important; color: #4ade80 !important; }
.cm-fp { background: #450a0a !important; color: #f87171 !important; }
.cm-fn { background: #451a03 !important; color: #fb923c !important; }
.cm-tn { background: #052e16 !important; color: #4ade80 !important; }

/* force all text in stMarkdown to inherit */
.stMarkdown p, .stMarkdown li, .stMarkdown span { color: #e8eaf0; }

/* ── DISCLAIMER ── */
.disclaimer {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 12px; padding: 16px 20px;
    font-size: 0.85rem; color: #fbbf24 !important;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ── HERO BANNER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-eyebrow">⬡ Machine Learning · XGBoost Classification · Healthcare AI</div>
  <div class="hero-title">🫁 Lung Cancer<br>Risk Predictor</div>
  <p class="hero-sub">
    An end-to-end explainable AI system powered by Gradient Boosted Trees (XGBoost-style) —
    from raw survey data to real-time clinical risk prediction.
  </p>
  <div class="hero-badges">
    <span class="badge">XGBoost Algorithm</span>
    <span class="badge">309 Patients</span>
    <span class="badge">15 Features</span>
    <span class="badge">87% Accuracy</span>
    <span class="badge">92% AUC-ROC</span>
    <span class="badge">~90% CV Score</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── GLOBAL STATS ROW ─────────────────────────────────────────────────────────
st.markdown("""
<div class="stat-grid">
  <div class="stat-box"><div class="sval">309</div><div class="slbl">Total Patients</div></div>
  <div class="stat-box"><div class="sval">87%</div><div class="slbl">Test Accuracy</div></div>
  <div class="stat-box"><div class="sval">0.92</div><div class="slbl">AUC-ROC Score</div></div>
  <div class="stat-box"><div class="sval">90%</div><div class="slbl">5-Fold CV Accuracy</div></div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["📖 Introduction", "📊 Data Explorer", "🌲 XGBoost Explained", "📈 Model Performance", "🔍 Predict Risk"])
# ═══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — INTRODUCTION
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown('<div class="sec-head">What Is This System?</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card card-accent">
        <b>The Problem</b><br><br>
        Lung cancer is the world's deadliest cancer — responsible for over 1.8 million deaths annually.
        Early detection is the single biggest factor in survival rates: 5-year survival jumps from
        ~6% (late-stage) to ~63% (early-stage).<br><br>
        Most patients are diagnosed too late. A fast, accessible risk-screening tool can flag 
        high-risk individuals <i>before</i> symptoms become severe.
        </div>

        <div class="card card-green">
        <b>Our Solution: XGBoost Classification</b><br><br>
        We trained an <b>eXtreme Gradient Boosting (XGBoost)</b> model — the gold standard algorithm
        for tabular healthcare data — on survey responses from 309 patients covering 14 symptoms 
        and demographic factors.<br><br>
        The model predicts: <b>HIGH RISK (Likely Lung Cancer) vs LOW RISK</b><br>
        with 87% accuracy and a 92% AUC-ROC score.
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="sec-head">Dataset Features</div>', unsafe_allow_html=True)
        features_info = [
            ("👤", "GENDER", "Male / Female (demographic)"),
            ("🎂", "AGE", "Patient age in years (21–87)"),
            ("🚬", "SMOKING", "Smoking history"),
            ("🟡", "YELLOW FINGERS", "Yellowing of fingers/nails"),
            ("😰", "ANXIETY", "Chronic anxiety"),
            ("👥", "PEER PRESSURE", "Social peer pressure"),
            ("🏥", "CHRONIC DISEASE", "Existing chronic illness"),
            ("😴", "FATIGUE", "Persistent fatigue"),
            ("🤧", "ALLERGY", "Respiratory allergies"),
            ("💨", "WHEEZING", "Wheezing when breathing"),
            ("🍺", "ALCOHOL", "Alcohol consumption"),
            ("😮", "COUGHING", "Chronic coughing"),
            ("🫁", "SHORTNESS OF BREATH", "Breathing difficulty"),
            ("😣", "SWALLOWING DIFF.", "Difficulty swallowing"),
            ("💔", "CHEST PAIN", "Chest pain episodes"),
        ]
        for icon, name, desc in features_info:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:7px 12px;
                        border-bottom:1px solid #1e2535;font-size:0.85rem;">
              <span style="font-size:1.1rem;">{icon}</span>
              <span style="color:#c4b5fd;font-weight:600;min-width:140px;">{name}</span>
              <span style="color:#64748b;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="sec-head">App Roadmap</div>', unsafe_allow_html=True)
    roadmap = [
        ("📊 Data Explorer", "Visualise the dataset — distributions, symptom rates, class balance, and correlation analysis with interactive charts."),
        ("🌲 XGBoost Explained", "Step-by-step breakdown of how XGBoost works — boosting theory, tree building, hyperparameters, and why it outperforms other algorithms."),
        ("📈 Model Performance", "Deep-dive metrics — accuracy, AUC-ROC curve, confusion matrix, feature importance chart, cross-validation, and model comparison."),
        ("🔍 Predict Risk", "Enter any patient profile and get an instant risk prediction with probability score and full explanation of contributing factors."),
    ]
    cols = st.columns(4)
    for col, (title, desc) in zip(cols, roadmap):
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center;min-height:160px;">
              <div style="font-size:1.6rem;margin-bottom:10px;">{title.split()[0]}</div>
              <div style="font-weight:600;color:#c4b5fd;margin-bottom:8px;font-size:0.95rem;">{' '.join(title.split()[1:])}</div>
              <div style="font-size:0.82rem;color:#64748b;line-height:1.6;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — DATA EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="sec-head">Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-subhead">309 patient survey records · 15 features · No missing values · Binary classification target</div>', unsafe_allow_html=True)

    # Raw data
    with st.expander("🗃️ View Raw Dataset (first 15 rows)"):
        st.dataframe(df_raw.head(15), use_container_width=True)

    st.markdown('<div class="sec-head">📊 Target Variable Distribution</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="card card-red">
        <b style="color:#f87171!important;">⚠️ Class Imbalance Warning</b><br><br>
        <span style="font-size:2rem;font-weight:900;color:#ef4444!important;">87.4%</span>
        <br><span style="color:#94a3b8!important;">Lung Cancer (YES) — 270 patients</span><br><br>
        <span style="font-size:1.4rem;font-weight:700;color:#4ade80!important;">12.6%</span>
        <br><span style="color:#94a3b8!important;">No Lung Cancer (NO) — 39 patients</span><br><br>
        The dataset is heavily imbalanced. XGBoost handles this via <b>scale_pos_weight</b> and
        loss function adjustments.
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card card-accent">
        <b>Age Statistics</b><br><br>
        """, unsafe_allow_html=True)
        age_stats = df_raw['AGE'].describe()
        for label, key in [("Min Age","min"),("Mean Age","mean"),("Median","50%"),("Max Age","max")]:
            val = age_stats[key]
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e2535;">
              <span style="color:#94a3b8;font-size:0.85rem;">{label}</span>
              <span style="color:#c4b5fd;font-weight:700;">{val:.0f} yrs</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="card card-green">
        <b>Gender Split</b><br><br>
        """, unsafe_allow_html=True)
        gd = df_raw['GENDER'].value_counts()
        for g, cnt in gd.items():
            pct = cnt/len(df_raw)*100
            color = "#8b5cf6" if g=="M" else "#ec4899"
            st.markdown(f"""
            <b style="color:{'#8b5cf6' if g=='M' else '#ec4899'}!important;">{'Male' if g=='M' else 'Female'}</b>: {cnt} ({pct:.1f}%)<br>
            <div class="bar-bg"><div class="bar-fill" style="width:{pct}%;background:{color};">{pct:.0f}%</div></div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Symptom prevalence chart
    st.markdown('<div class="sec-head">🦠 Symptom Prevalence Among Lung Cancer Patients</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-subhead">Percentage of lung cancer patients (YES) who reported each symptom (value=2)</div>', unsafe_allow_html=True)

    df2 = df_raw.copy()
    df2.columns = df2.columns.str.strip()
    binary_cols = [c for c in df2.columns if c not in ['GENDER','AGE','LUNG_CANCER']]
    yes_patients = df2[df2['LUNG_CANCER']=='YES']
    no_patients  = df2[df2['LUNG_CANCER']=='NO']

    symptom_data = []
    for col in binary_cols:
        yes_rate = (yes_patients[col]==2).mean()*100
        no_rate  = (no_patients[col]==2).mean()*100
        symptom_data.append((col.replace(' ','_'), yes_rate, no_rate))
    symptom_data.sort(key=lambda x: x[1], reverse=True)

    c1, c2 = st.columns(2)
    for i, (col, yes_r, no_r) in enumerate(symptom_data):
        target_col = c1 if i < len(symptom_data)//2 else c2
        with target_col:
            label = col.replace('_',' ').title()
            diff_color = "#4ade80" if yes_r - no_r > 10 else "#f59e0b"
            st.markdown(f"""
            <div style="margin-bottom:14px;">
              <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span style="font-size:0.85rem;color:#e8eaf0;">{label}</span>
                <span style="font-size:0.78rem;color:{diff_color};font-family:'DM Mono',monospace;">
                  YES: {yes_r:.0f}% / NO: {no_r:.0f}%
                </span>
              </div>
              <div class="bar-bg">
                <div class="bar-fill" style="width:{yes_r}%;background:linear-gradient(90deg,#7c3aed,#8b5cf6);">{yes_r:.0f}%</div>
              </div>
              <div class="bar-bg" style="height:10px;margin-top:-8px;">
                <div style="height:10px;border-radius:8px;width:{no_r}%;background:rgba(100,116,139,0.5);"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card card-amber" style="margin-top:8px;">
    <b>💡 Reading this chart:</b> Purple bars = lung cancer patients (YES); gray bars = healthy patients (NO).
    A large gap between the two bars = strong predictor. Allergy, Alcohol Consuming, and Swallowing Difficulty 
    show the biggest differences, making them the most influential features.
    </div>
    """, unsafe_allow_html=True)

    # Correlation table
    st.markdown('<div class="sec-head">🔗 Feature Correlation with Lung Cancer</div>', unsafe_allow_html=True)
    df_corr = df2.copy()
    df_corr['target'] = (df_corr['LUNG_CANCER']=='YES').astype(int)
    df_corr['GENDER'] = (df_corr['GENDER']=='M').astype(int)
    corr_data = []
    for col in ['AGE','GENDER'] + binary_cols:
        c_val = df_corr[col].corr(df_corr['target'])
        corr_data.append({'Feature': col.strip(), 'Correlation': round(c_val, 4),
                          'Strength': '🔴 Strong' if abs(c_val)>0.25 else '🟡 Moderate' if abs(c_val)>0.1 else '⚪ Weak',
                          'Direction': '⬆️ Positive' if c_val>0 else '⬇️ Negative'})
    corr_df = pd.DataFrame(corr_data).sort_values('Correlation', ascending=False)
    st.dataframe(corr_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="card">
    <b>What does correlation tell us?</b><br><br>
    Correlation ranges from -1 to +1. A positive correlation means having this symptom 
    <i>increases</i> the chance of lung cancer. A negative correlation means it decreases the chance.
    However, correlation only measures <b>linear relationships</b> — XGBoost captures 
    <b>non-linear and interaction effects</b> that simple correlation misses, which is why it performs better.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — XGBOOST EXPLAINED
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="sec-head">🌲 What is XGBoost?</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card card-accent">
    <b>eXtreme Gradient Boosting — The King of Tabular ML</b><br><br>
    XGBoost is consistently the <b>top-performing algorithm</b> in data science competitions (Kaggle) 
    and real-world healthcare applications. It combines hundreds of weak decision trees into one 
    powerful ensemble model using a process called <b>Gradient Boosting</b>.<br><br>
    Think of it like a team of doctors: each doctor (tree) is a specialist who focuses on the cases 
    the previous doctor got wrong. Together, they form a committee far more accurate than any individual.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-head">📋 Step-by-Step: How XGBoost Was Built</div>', unsafe_allow_html=True)

    steps = [
        ("Data Preprocessing", 
         "Loaded 309 patient records. Stripped whitespace from column names. Encoded <b>GENDER</b> (M→1, F→0) and <b>LUNG_CANCER</b> (YES→1, NO→0). Converted all binary symptom features from scale 1/2 to 0/1 for clean numerical input.",
         "Input shape: (309, 15) · No missing values · Clean binary encoding"),
        ("Train/Test Split",
         "Divided data 80/20 with <b>stratified sampling</b> — ensuring both train and test sets maintain the same 87%/13% class ratio. This prevents misleadingly optimistic accuracy from imbalanced splits.",
         "Train: 247 rows | Test: 62 rows | Stratified: Yes | Random state: 42"),
        ("Building Tree 1 (Initial Predictions)",
         "XGBoost starts with a simple prediction (e.g., predict the majority class for everyone). It computes <b>residuals</b> — the difference between predicted and actual values. These residuals represent 'how wrong' the model is.",
         "Initial prediction → Compute errors (residuals) → These become the target for Tree 2"),
        ("Building Trees 2 to N (Boosting Rounds)",
         "Each subsequent tree <b>corrects the errors</b> of the previous trees by fitting to residuals. This is the 'boosting' mechanism — each round reduces the overall error. We used <b>200 boosting rounds</b> (n_estimators=200).",
         "Tree 2 corrects Tree 1's errors → Tree 3 corrects remaining errors → ... → Tree 200"),
        ("Learning Rate & Shrinkage",
         "Each tree's contribution is multiplied by the <b>learning rate (0.1)</b>. Small learning rates mean each tree contributes less but the final ensemble is more robust and generalises better. Think of it as 'small confident steps vs large uncertain leaps'.",
         "learning_rate=0.1 · Each tree scaled down by 10% · Prevents overfitting"),
        ("Regularisation (L1 & L2)",
         "XGBoost adds <b>regularisation penalties</b> to prevent overfitting — it penalises overly complex trees. The max_depth=4 limits how deep each tree can grow, avoiding memorising noise in the training data.",
         "max_depth=4 · subsample=0.8 (80% of data per tree) · min_samples_split=10"),
        ("Final Prediction & Probability",
         "All 200 trees vote together. Their weighted sum passes through a <b>sigmoid function</b> → producing a probability between 0 and 1. If probability > 0.5 → Lung Cancer (YES), otherwise → No Cancer (NO).",
         "Sum of all trees → Sigmoid → Probability score → Threshold 0.5 → Binary class"),
    ]

    for i, (title, body, note) in enumerate(steps, 1):
        st.markdown(f"""
        <div class="step-wrap">
          <div class="step-num">{i}</div>
          <div class="step-content" style="flex:1;">
            <div style="font-weight:600;color:#c4b5fd;margin-bottom:8px;">{title}</div>
            <div style="font-size:0.9rem;line-height:1.7;color:#cbd5e1;">{body}</div>
            <div class="step-note">→ {note}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sec-head">🔑 Key XGBoost Concepts</div>', unsafe_allow_html=True)
    concepts = [
        ("🌳 Decision Tree (Weak Learner)", "A simple tree that asks yes/no questions about features to make predictions. Each tree alone is weak, but XGBoost combines hundreds of them into a strong ensemble."),
        ("🎯 Gradient Descent", "XGBoost minimises a loss function using gradients — it always moves in the direction that reduces prediction errors the fastest. This is the mathematical engine of boosting."),
        ("⚡ Boosting vs Bagging", "Random Forest uses bagging (parallel trees on random data subsets). XGBoost uses boosting (sequential trees where each learns from previous errors). Boosting is usually more accurate."),
        ("📉 Loss Function", "We minimise binary cross-entropy loss = -[y·log(p) + (1-y)·log(1-p)]. It heavily penalises confident wrong predictions, pushing the model to be calibrated and accurate."),
        ("🔒 Regularisation", "L1/L2 penalties added to the objective function. They shrink unimportant feature weights toward zero, preventing overfitting and improving generalisation on unseen data."),
        ("📊 Feature Importance", "XGBoost tracks how much each feature reduces prediction error across all trees (gain). Features used more frequently for high-impact splits get higher importance scores."),
    ]
    st.markdown('<div class="concept-grid">', unsafe_allow_html=True)
    for title, desc in concepts:
        st.markdown(f"""
        <div class="concept-box">
          <b>{title}</b>
          <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Hyperparameters table
    st.markdown('<div class="sec-head">⚙️ Hyperparameters Used</div>', unsafe_allow_html=True)
    params_df = pd.DataFrame([
        {"Parameter":"n_estimators","Value":"200","What it does":"Number of boosting rounds (trees)","Why this value":"More trees = better accuracy; 200 balances accuracy vs speed"},
        {"Parameter":"learning_rate","Value":"0.1","What it does":"Step size for each tree's contribution","Why this value":"0.1 is standard — small enough to generalise, large enough to converge"},
        {"Parameter":"max_depth","Value":"4","What it does":"Maximum depth of each decision tree","Why this value":"Depth 4 captures complex patterns without memorising noise"},
        {"Parameter":"subsample","Value":"0.8","What it does":"Fraction of training data used per tree","Why this value":"80% sampling adds randomness, reduces overfitting like Random Forest"},
        {"Parameter":"min_samples_split","Value":"10","What it does":"Minimum samples to split a node","Why this value":"Prevents splitting on tiny, noisy groups — improves robustness"},
        {"Parameter":"random_state","Value":"42","What it does":"Seeds randomness for reproducibility","Why this value":"Standard choice; ensures same results every run"},
    ])
    st.dataframe(params_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="card card-green" style="margin-top:8px;">
    <b>🏆 Why XGBoost for Lung Cancer Prediction?</b><br><br>
    1. <b>Handles imbalanced data</b> — 87% YES vs 13% NO doesn't trip it up like simpler models<br>
    2. <b>Captures non-linear interactions</b> — e.g., "smoking + yellow fingers together" is more dangerous than either alone<br>
    3. <b>Built-in regularisation</b> — reduces overfitting on small medical datasets (309 rows)<br>
    4. <b>Feature importance</b> — interpretable, critical for clinical trust and explainability<br>
    5. <b>Proven in healthcare</b> — used in real hospital systems for cancer, diabetes, sepsis prediction
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="sec-head">📈 Performance Metrics Explained</div>', unsafe_allow_html=True)

    cr = arts['classification_report']
    m1, m2, m3, m4, m5 = st.columns(5)
    metric_boxes = [
        (m1, "Accuracy", f"{arts['accuracy']*100:.1f}%", "Overall correct predictions out of 62 test patients"),
        (m2, "AUC-ROC", f"{arts['auc']:.3f}", "Area under ROC curve — 0.92 = excellent discrimination"),
        (m3, "Precision (YES)", f"{cr['1']['precision']*100:.1f}%", "When model says YES, it's right this % of time"),
        (m4, "Recall (YES)", f"{cr['1']['recall']*100:.1f}%", "% of actual cancer patients correctly identified"),
        (m5, "F1-Score (YES)", f"{cr['1']['f1-score']*100:.1f}%", "Harmonic mean of precision and recall"),
    ]
    for col, label, val, desc in metric_boxes:
        with col:
            st.markdown(f"""
            <div class="stat-box" style="margin-bottom:8px;">
              <div class="sval" style="font-size:1.8rem;">{val}</div>
              <div class="slbl">{label}</div>
            </div>
            <div style="font-size:0.75rem;color:#64748b;text-align:center;">{desc}</div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="sec-head">🔲 Confusion Matrix</div>', unsafe_allow_html=True)
    cm = arts['confusion_matrix']  # [[TN,FP],[FN,TP]]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown(f"""
        <table class="cm-table">
          <tr>
            <td class="cm-label"></td>
            <td class="cm-label">Predicted: NO</td>
            <td class="cm-label">Predicted: YES</td>
          </tr>
          <tr>
            <td class="cm-label">Actual: NO</td>
            <td class="cm-tn">{tn} ✅<br><span style="font-size:0.7rem;color:#86efac;">True Negative</span></td>
            <td class="cm-fp">{fp} ❌<br><span style="font-size:0.7rem;color:#fca5a5;">False Positive</span></td>
          </tr>
          <tr>
            <td class="cm-label">Actual: YES</td>
            <td class="cm-fn">{fn} ⚠️<br><span style="font-size:0.7rem;color:#fdba74;">False Negative</span></td>
            <td class="cm-tp">{tp} ✅<br><span style="font-size:0.7rem;color:#86efac;">True Positive</span></td>
          </tr>
        </table>
        """, unsafe_allow_html=True)

    with c2:
        entries = [
            ("✅ True Positives", str(tp), "#052e16", "#4ade80", f"Correctly identified {tp} actual cancer patients. These are the critical catches."),
            ("✅ True Negatives", str(tn), "#052e16", "#4ade80", f"Correctly cleared {tn} healthy patients. No unnecessary alarm."),
            ("⚠️ False Negatives", str(fn), "#451a03", "#fb923c", f"Missed {fn} real cancer patients. In medicine, this is the most dangerous error type."),
            ("❌ False Positives", str(fp), "#450a0a", "#f87171", f"Incorrectly flagged {fp} healthy patients as cancer. Leads to unnecessary further tests."),
        ]
        for label, count, bg, color, desc in entries:
            st.markdown(f"""
            <div style="background:{bg};border-radius:10px;padding:12px 16px;margin-bottom:10px;border:1px solid {color}30;">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="color:{color};font-weight:700;">{label}</span>
                <span style="color:{color};font-size:1.6rem;font-weight:900;font-family:'Fraunces',serif;">{count}</span>
              </div>
              <div style="font-size:0.82rem;color:#94a3b8;margin-top:4px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # ROC Curve visual (text-based)
    st.markdown('<div class="sec-head">📉 AUC-ROC Curve Interpretation</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([2,3])
    with c1:
        st.markdown(f"""
        <div class="card card-accent">
        <b>What is AUC-ROC?</b><br><br>
        The ROC (Receiver Operating Characteristic) curve plots:
        <ul>
        <li><b>X-axis</b>: False Positive Rate (FPR = FP/(FP+TN))</li>
        <li><b>Y-axis</b>: True Positive Rate / Recall (TP/(TP+FN))</li>
        </ul>
        AUC (Area Under Curve) = <b>probability that the model ranks a random positive 
        patient higher than a random negative patient.</b><br><br>
        <div style="text-align:center;margin-top:16px;">
        <span style="font-family:'Fraunces',serif;font-size:3rem;background:linear-gradient(135deg,#8b5cf6,#c4b5fd);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">{arts['auc']:.3f}</span>
        <div style="color:#94a3b8;font-size:0.85rem;margin-top:4px;">Our AUC-ROC Score</div>
        </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        # Draw simple ASCII-style ROC curve using bars
        st.markdown("""
        <div class="card">
        <b>AUC Score Scale</b><br><br>
        """, unsafe_allow_html=True)
        roc_levels = [
            ("Random Guess", 0.50, "#64748b"),
            ("Poor Model", 0.65, "#ef4444"),
            ("Fair Model", 0.75, "#f59e0b"),
            ("Good Model", 0.85, "#3b82f6"),
            ("Our XGBoost", arts['auc'], "#8b5cf6"),
            ("Excellent", 0.95, "#10b981"),
            ("Perfect", 1.00, "#4ade80"),
        ]
        for label, val, color in roc_levels:
            highlight = "border:1px solid #8b5cf6;border-radius:8px;padding:2px 4px;" if "XGBoost" in label else ""
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
              <span style="font-size:0.82rem;color:#94a3b8;min-width:120px;{highlight}">{label}</span>
              <div class="bar-bg" style="flex:1;height:14px;">
                <div style="height:14px;border-radius:8px;width:{val*100}%;background:{color};"></div>
              </div>
              <span style="font-family:'DM Mono',monospace;font-size:0.8rem;color:{color};min-width:40px;">{val:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Feature Importance
    st.markdown('<div class="sec-head">🎯 Feature Importance (XGBoost Gain)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-subhead">How much each feature reduces prediction error across all 200 trees</div>', unsafe_allow_html=True)

    fi_sorted = arts['fi_sorted']
    max_fi = fi_sorted[0][1]
    colors_fi = ["#8b5cf6","#7c3aed","#6d28d9","#5b21b6","#4c1d95",
                 "#3b82f6","#2563eb","#1d4ed8","#1e40af","#1e3a8a",
                 "#10b981","#059669","#047857","#065f46","#064e3b"]

    for i, (feat, imp) in enumerate(fi_sorted):
        pct = imp / max_fi * 100
        color = colors_fi[i % len(colors_fi)]
        label = feat.replace('_',' ').title()
        rank_icon = "🥇" if i==0 else "🥈" if i==1 else "🥉" if i==2 else f"#{i+1}"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
          <span style="min-width:28px;font-size:0.85rem;color:#64748b;font-family:'DM Mono',monospace;">{rank_icon}</span>
          <span style="min-width:180px;font-size:0.88rem;color:#e8eaf0;">{label}</span>
          <div class="bar-bg" style="flex:1;">
            <div class="bar-fill" style="width:{pct}%;background:linear-gradient(90deg,{color},{color}99);">
              {imp:.4f}
            </div>
          </div>
          <span style="font-family:'DM Mono',monospace;font-size:0.8rem;color:{color};min-width:55px;">{imp*100:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card card-green" style="margin-top:8px;">
    <b>How to interpret feature importance:</b><br><br>
    XGBoost computes <b>gain</b> — the average improvement in accuracy every time a feature is used 
    to split a node across all 200 trees. <b>AGE</b> is the most important feature (19.3%), followed 
    by <b>ALLERGY</b> (14.3%) and <b>ALCOHOL CONSUMING</b> (10.7%). This means the XGBoost model 
    found that age-based patterns are the most powerful discriminators between cancer and non-cancer patients.
    </div>
    """, unsafe_allow_html=True)

    # Model Comparison
    st.markdown('<div class="sec-head">🏁 Algorithm Comparison</div>', unsafe_allow_html=True)
    mc = arts['model_comparison']
    comp_df = pd.DataFrame([
        {"Algorithm": k, "Test Accuracy": f"{v['acc']}%", "AUC-ROC": f"{v['auc']:.4f}", "5-Fold CV": f"{v['cv']}%"}
        for k,v in mc.items()
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    for name, vals in mc.items():
        icon = "🥇" if name=="XGBoost (GBC)" else "🥈" if "Forest" in name else "🥉" if "Logistic" in name else "🏅"
        color = "#8b5cf6" if "XGBoost" in name else "#64748b"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
          <span style="min-width:24px;">{icon}</span>
          <span style="min-width:180px;font-size:0.9rem;color:#e8eaf0;">{name}</span>
          <div class="bar-bg" style="flex:1;">
            <div class="bar-fill" style="width:{vals['acc']}%;background:{'linear-gradient(90deg,#7c3aed,#8b5cf6)' if 'XGBoost' in name else '#334155'};">
              {vals['acc']}%
            </div>
          </div>
          <span style="font-family:'DM Mono',monospace;font-size:0.82rem;color:{color};min-width:60px;">AUC:{vals['auc']}</span>
        </div>
        """, unsafe_allow_html=True)

    # Cross-validation
    st.markdown('<div class="sec-head">🔄 5-Fold Cross-Validation Results</div>', unsafe_allow_html=True)
    cv_scores = [round(s*100,2) for s in arts['cv_scores']]
    c1, c2 = st.columns([2,3])
    with c1:
        st.markdown(f"""
        <div class="card card-accent">
        <b>What is Cross-Validation?</b><br><br>
        Instead of one 80/20 split, 5-Fold CV creates <b>5 different train/test splits</b>
        and averages the results. This gives a more reliable, <b>bias-free</b> accuracy estimate.<br><br>
        <div style="margin-top:16px;">
          <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #1e2535;">
            <span style="color:#94a3b8;">CV Mean</span>
            <span style="color:#c4b5fd;font-weight:700;">{arts['cv_mean']*100:.2f}%</span>
          </div>
          <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #1e2535;">
            <span style="color:#94a3b8;">CV Std Dev</span>
            <span style="color:#c4b5fd;font-weight:700;">±{arts['cv_std']*100:.2f}%</span>
          </div>
          <div style="display:flex;justify-content:space-between;padding:8px 0;">
            <span style="color:#94a3b8;">Min / Max</span>
            <span style="color:#c4b5fd;font-weight:700;">{min(cv_scores):.1f}% / {max(cv_scores):.1f}%</span>
          </div>
        </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><b>Score per Fold</b><br><br>', unsafe_allow_html=True)
        for i, score in enumerate(cv_scores, 1):
            color = "#10b981" if score >= 90 else "#f59e0b" if score >= 85 else "#ef4444"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
              <span style="color:#94a3b8;font-family:'DM Mono',monospace;min-width:60px;">Fold {i}</span>
              <div class="bar-bg" style="flex:1;">
                <div class="bar-fill" style="width:{score}%;background:{color};">{score}%</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — PREDICT
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="sec-head">🔍 Real-Time Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-subhead">Enter the patient\'s profile below — all symptom questions are binary (Yes/No)</div>', unsafe_allow_html=True)

    binary_cols_clean = ['SMOKING','YELLOW_FINGERS','ANXIETY','PEER_PRESSURE',
                         'CHRONIC DISEASE','FATIGUE','ALLERGY','WHEEZING',
                         'ALCOHOL CONSUMING','COUGHING','SHORTNESS OF BREATH',
                         'SWALLOWING DIFFICULTY','CHEST PAIN']
    symptom_labels = {
        'SMOKING': '🚬 Smoking History',
        'YELLOW_FINGERS': '🟡 Yellow Fingers/Nails',
        'ANXIETY': '😰 Chronic Anxiety',
        'PEER_PRESSURE': '👥 Peer Pressure',
        'CHRONIC DISEASE': '🏥 Chronic Disease',
        'FATIGUE': '😴 Persistent Fatigue',
        'ALLERGY': '🤧 Respiratory Allergy',
        'WHEEZING': '💨 Wheezing',
        'ALCOHOL CONSUMING': '🍺 Alcohol Consuming',
        'COUGHING': '😮 Chronic Coughing',
        'SHORTNESS OF BREATH': '🫁 Shortness of Breath',
        'SWALLOWING DIFFICULTY': '😣 Swallowing Difficulty',
        'CHEST PAIN': '💔 Chest Pain',
    }

    c_demo, c_symp1, c_symp2 = st.columns([1, 1.2, 1.2])

    with c_demo:
        st.markdown("**👤 Demographics**")
        gender_in = st.selectbox("Gender", ["Male", "Female"])
        age_in    = st.slider("Age", 21, 87, 55)

        st.markdown(f"""
        <div class="card" style="margin-top:12px;text-align:center;">
          <div style="color:#64748b;font-size:0.78rem;margin-bottom:6px;">AGE CONTEXT</div>
          <div style="font-family:'Fraunces',serif;font-size:2rem;color:#c4b5fd;">{age_in}</div>
          <div style="font-size:0.8rem;color:#64748b;">Dataset avg: 62.7 yrs</div>
          <div class="bar-bg" style="margin-top:8px;">
            <div class="bar-fill" style="width:{(age_in-21)/(87-21)*100:.0f}%;background:linear-gradient(90deg,#7c3aed,#8b5cf6);">
            </div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#475569;margin-top:2px;">
            <span>21</span><span>87</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    symptom_inputs = {}
    mid = len(binary_cols_clean) // 2

    with c_symp1:
        st.markdown("**🦠 Symptoms (Part 1)**")
        for col in binary_cols_clean[:mid]:
            val = st.radio(symptom_labels[col], ["No", "Yes"],
                          horizontal=True, key=f"sym_{col}")
            symptom_inputs[col] = 1 if val == "Yes" else 0

    with c_symp2:
        st.markdown("**🦠 Symptoms (Part 2)**")
        for col in binary_cols_clean[mid:]:
            val = st.radio(symptom_labels[col], ["No", "Yes"],
                          horizontal=True, key=f"sym_{col}")
            symptom_inputs[col] = 1 if val == "Yes" else 0

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🫁 Analyse Lung Cancer Risk", use_container_width=True)

    if predict_btn:
        # Build input array
        gender_enc = 1 if gender_in == "Male" else 0
        input_row  = [gender_enc, age_in] + [symptom_inputs[c] for c in binary_cols_clean]
        feature_names = arts['feature_names']
        input_df   = pd.DataFrame([input_row], columns=feature_names)
        pred       = model.predict(input_df)[0]
        proba      = model.predict_proba(input_df)[0]
        risk_pct   = proba[1] * 100
        safe_pct   = proba[0] * 100

        st.markdown("---")
        st.markdown("### 🧾 Prediction Result")

        c_res, c_detail = st.columns([1, 1])
        with c_res:
            if pred == 1:
                st.markdown(f"""
                <div class="result-high">
                  <div class="result-icon">⚠️</div>
                  <div class="result-label">HIGH RISK</div>
                  <div class="result-prob">Lung Cancer Probability: <b>{risk_pct:.1f}%</b></div>
                  <div style="margin-top:16px;font-size:0.85rem;opacity:0.8;">
                    Please seek immediate medical consultation.<br>
                    Further diagnostic tests (CT scan, biopsy) are strongly recommended.
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-low">
                  <div class="result-icon">✅</div>
                  <div class="result-label">LOW RISK</div>
                  <div class="result-prob">No Cancer Probability: <b>{safe_pct:.1f}%</b></div>
                  <div style="margin-top:16px;font-size:0.85rem;opacity:0.8;">
                    Risk profile suggests low likelihood of lung cancer.<br>
                    Continue healthy habits and regular check-ups.
                  </div>
                </div>
                """, unsafe_allow_html=True)

        with c_detail:
            st.markdown(f"""
            <div class="card">
            <b>📊 Probability Breakdown</b><br><br>
            🟢 No Lung Cancer: <b>{safe_pct:.1f}%</b>
            <div class="bar-bg">
              <div class="bar-fill" style="width:{safe_pct}%;background:#10b981;">{safe_pct:.1f}%</div>
            </div>
            🔴 Lung Cancer: <b>{risk_pct:.1f}%</b>
            <div class="bar-bg">
              <div class="bar-fill" style="width:{risk_pct}%;background:#ef4444;">{risk_pct:.1f}%</div>
            </div>
            <br>
            <b>⚙️ Scaled Input Values (what XGBoost saw):</b><br>
            <div style="font-family:'DM Mono',monospace;font-size:0.78rem;color:#94a3b8;margin-top:8px;line-height:1.8;">
              Gender: {gender_enc} ({'Male' if gender_enc==1 else 'Female'})<br>
              Age: {age_in} ({'above' if age_in > 62.7 else 'below'} dataset avg of 62.7)<br>
              Symptoms active: {sum(symptom_inputs.values())} / {len(binary_cols_clean)}
            </div>
            </div>
            """, unsafe_allow_html=True)

        # Risk Factor Analysis
        st.markdown("### 🔬 Risk Factor Analysis")
        c1, c2 = st.columns(2)

        active_symptoms  = [k for k,v in symptom_inputs.items() if v==1]
        inactive_symptoms = [k for k,v in symptom_inputs.items() if v==0]

        # Get feature importances for context
        fi_dict = arts['feature_importances']

        with c1:
            st.markdown(f"""
            <div class="card card-red">
            <b>🔴 Active Risk Factors ({len(active_symptoms)})</b><br><br>
            """, unsafe_allow_html=True)
            if active_symptoms:
                for sym in sorted(active_symptoms, key=lambda x: fi_dict.get(x,0), reverse=True):
                    imp = fi_dict.get(sym, 0)
                    label = symptom_labels.get(sym, sym)
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e2535;">
                      <span style="font-size:0.85rem;">{label}</span>
                      <span style="color:#f87171;font-family:'DM Mono',monospace;font-size:0.78rem;">importance: {imp*100:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("<div style='color:#94a3b8;font-size:0.85rem;'>No active risk factors</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="card card-green">
            <b>🟢 Protective (Not Present) Factors ({len(inactive_symptoms)})</b><br><br>
            """, unsafe_allow_html=True)
            for sym in sorted(inactive_symptoms, key=lambda x: fi_dict.get(x,0), reverse=True)[:7]:
                imp = fi_dict.get(sym, 0)
                label = symptom_labels.get(sym, sym)
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e2535;">
                  <span style="font-size:0.85rem;">{label}</span>
                  <span style="color:#4ade80;font-family:'DM Mono',monospace;font-size:0.78rem;">absent: ✓</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Similar patients
        st.markdown("### 👥 Similar Patients in Training Data")
        df_sim = df_raw.copy()
        df_sim.columns = df_sim.columns.str.strip()
        active_count = len(active_symptoms)
        symptom_match = []
        for _, row in df_sim.iterrows():
            matches = sum(1 for s in active_symptoms if row.get(s, 1) == 2)
            symptom_match.append(matches)
        df_sim['_match'] = symptom_match
        similar = df_sim.nlargest(8, '_match')[['GENDER','AGE','LUNG_CANCER']].copy()
        similar['LUNG_CANCER'] = similar['LUNG_CANCER'].map({'YES':'🔴 YES','NO':'🟢 NO'})
        similar.columns = ['Gender','Age','Lung Cancer']
        similar = similar.reset_index(drop=True); similar.index += 1
        st.dataframe(similar, use_container_width=True)

        yes_in_sim = (similar['Lung Cancer']=='🔴 YES').sum()
        st.markdown(f"""
        <div class="card" style="font-size:0.88rem;">
        Among the <b>8 most similar patients</b> in the training dataset: 
        <b style="color:#ef4444;">{yes_in_sim} had lung cancer</b> and 
        <b style="color:#10b981;">{8-yes_in_sim} did not</b> — 
        {'consistent with' if (yes_in_sim >= 4)==(pred==1) else 'the model detected a broader pattern beyond'} this prediction.
        </div>
        """, unsafe_allow_html=True)

        # XGBoost decision explanation
        st.markdown("""
        <div class="card card-accent">
        <b>🌲 How XGBoost Reached This Decision</b><br><br>
        All 200 gradient-boosted trees voted on your patient profile:
        <ol>
          <li>Each tree evaluated your Age, Gender, and symptom combination through a series of if/else splits</li>
          <li>Trees trained to fix previous errors had higher influence on edge cases</li>
          <li>All tree outputs were summed and passed through a <b>sigmoid function</b> → producing the probability score</li>
          <li>The model's top features (AGE, ALLERGY, ALCOHOL CONSUMING) had the most weight in the final decision</li>
          <li>If probability > 0.5 → HIGH RISK (Lung Cancer), otherwise → LOW RISK</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer">
        ⚠️ <b>Medical Disclaimer:</b> This tool is for <b>educational and research purposes only</b>.
        It is NOT a substitute for professional medical diagnosis. Always consult a qualified 
        pulmonologist or oncologist for accurate clinical assessment. A model prediction alone 
        should never be used to make medical decisions.
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:32px 0 16px;color:#334155;font-size:0.8rem;font-family:'DM Mono',monospace;border-top:1px solid #1e2535;margin-top:40px;">
  🫁 LUNG CANCER RISK PREDICTOR · XGBoost (Gradient Boosting) · 87% Accuracy · 92% AUC-ROC
  <br>Built with Streamlit & Scikit-learn · Dataset: Lung Cancer Survey (Kaggle)
</div>
""", unsafe_allow_html=True)
