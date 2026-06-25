import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd

# ── PAGE CONFIGURATION ──
st.set_page_config(
    page_title="CardioSense Ai · Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── FULLY RESPONSIVE PREMIUM STYLING (CSS) ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* Global Reset & Base Layout */
html, body, [class*="css"] { 
    font-family: 'Inter', sans-serif; 
}
.stApp { 
    background: #060d1f; 
    color: #e2e8f0; 
}
#MainMenu, footer, header { visibility: hidden; }

/* ── RESPONSIVE CONTAINER PAD (Desktop Default) ── */
.block-container {
    padding: 2.5rem 5rem !important;
}

/* Premium Dashboard Header */
.dashboard-hero {
    background: linear-gradient(135deg, #060d1f 0%, #0d1b3e 60%, #0a1628 100%);
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 20px;
    padding: 40px 44px;
    margin-bottom: 32px;
    position: relative;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
}
.badge-clinical {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: 100px; padding: 6px 16px; margin-bottom: 16px;
    font-size: 11px; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; color: #ef4444;
}
.hero-title-main {
    font-family: 'Space Grotesk', sans-serif; font-size: 42px; font-weight: 700; color: #ffffff; line-height: 1.1; margin: 0 0 12px;
    letter-spacing: -0.5px;
}
.hero-title-main span { color: #ef4444; }
.hero-subtitle-main { font-size: 15px; color: #64748b; max-width: 600px; line-height: 1.6; margin: 0 0 28px; }

/* Responsive Stat Grid Layout */
.stat-panel-row { 
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px; 
}
.stat-metric-card {
    background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 12px; padding: 16px 20px;
}
.stat-metric-val { font-family: 'Space Grotesk', sans-serif; font-size: 24px; font-weight: 700; color: #ffffff; }
.stat-metric-lbl { font-size: 11px; color: #475569; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }

/* Pulse Animation */
.pulse-indicator {
    width: 8px; height: 8px; background: #22c55e; border-radius: 50%;
    display: inline-block; animation: nativePulse 2s infinite; margin-right: 4px;
}
@keyframes nativePulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.4); }
    50% { opacity: .8; box-shadow: 0 0 0 6px rgba(34, 197, 94, 0); }
}

/* Structuring Elements */
.step-header-pill {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(59, 130, 246, 0.08); border: 1px solid rgba(59, 130, 246, 0.18);
    border-radius: 100px; padding: 6px 14px; margin-bottom: 12px;
    font-size: 11px; font-weight: 600; color: #3b82f6;
}
.step-number-node {
    background: #3b82f6; color: #ffffff; border-radius: 50%;
    width: 18px; height: 18px; display: inline-flex; align-items: center; justify-content: center;
    font-size: 10px; font-weight: 700;
}
.section-headline { font-family: 'Space Grotesk', sans-serif; font-size: 22px; font-weight: 700; color: #ffffff; margin-bottom: 24px; }

/* Premium Input Cards */
.clinical-card {
    background: linear-gradient(145deg, #0d1525, #0a1020);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px; padding: 24px; margin-bottom: 20px;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
}
.card-header-block { display: flex; align-items: center; gap: 12px; margin-bottom: 20px; }
.card-header-icon {
    width: 36px; height: 36px; border-radius: 10px;
    background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.18);
    display: flex; align-items: center; justify-content: center; font-size: 16px;
}
.card-header-title { font-family: 'Space Grotesk', sans-serif; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 1.2px; color: #64748b; }

/* Streamlit Input Overrides */
div[data-testid="stNumberInput"] input {
    background: #0d1525 !important; border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important; color: #f1f5f9 !important; padding: 10px 14px !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: #0d1525 !important; border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important; color: #f1f5f9 !important;
}
label[data-testid="stWidgetLabel"] p {
    color: #475569 !important; font-size: 11px !important; font-weight: 600 !important;
    text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px !important;
}

/* Patient Case Summary Card */
.overview-panel-card {
    background: #0d1525; border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 16px;
    overflow: hidden; margin-bottom: 20px;
}
.overview-panel-header {
    background: rgba(255, 255, 255, 0.02); padding: 16px 20px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    font-family: 'Space Grotesk', sans-serif; font-size: 12px; font-weight: 600; text-transform: uppercase; color: #3b82f6;
}
.overview-panel-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 20px; border-bottom: 1px solid rgba(255, 255, 255, 0.04);
}
.overview-panel-row:last-child { border-bottom: none; }
.overview-panel-key { font-size: 11px; color: #475569; font-weight: 500; text-transform: uppercase; }
.overview-panel-val { font-size: 13px; color: #cbd5e1; font-weight: 600; }

/* Premium Button Architecture */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #ef4444, #dc2626) !important;
    color: #ffffff !important; border: none !important; border-radius: 12px !important;
    font-family: 'Space Grotesk', sans-serif !important; font-size: 15px !important; font-weight: 700 !important; padding: 14px 24px !important;
    width: 100% !important; box-shadow: 0 4px 24px rgba(239, 68, 68, 0.3) !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(239, 68, 68, 0.45) !important;
}

/* Callouts & Notifications */
.clinical-disclaimer {
    background: rgba(59, 130, 246, 0.05); border: 1px solid rgba(59, 130, 246, 0.12);
    border-radius: 12px; padding: 14px 18px; margin-top: 16px;
    font-size: 12px; color: #475569; line-height: 1.5; text-align: center;
}
.clinical-info-note {
    background: rgba(239, 68, 68, 0.06); border-left: 3px solid rgba(239, 68, 68, 0.35);
    border-radius: 0 10px 10px 0; padding: 10px 14px; margin-top: 14px;
    font-size: 12px; color: #94a3b8; line-height: 1.5;
}

/* Diagnostic Output Blocks */
.diagnostic-container-positive {
    background: linear-gradient(145deg, rgba(239, 68, 68, 0.08), rgba(127, 29, 29, 0.04));
    border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 20px;
    padding: 40px 24px; text-align: center;
}
.diagnostic-container-negative {
    background: linear-gradient(145deg, rgba(34, 197, 94, 0.07), rgba(20, 83, 45, 0.04));
    border: 1px solid rgba(34, 197, 94, 0.2); border-radius: 20px;
    padding: 40px 24px; text-align: center;
}
.diagnostic-badge-pos { font-size: 11px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: #ef4444; margin-bottom: 6px; }
.diagnostic-badge-neg { font-size: 11px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: #22c55e; margin-bottom: 6px; }
.diagnostic-title-pos { font-family: 'Space Grotesk', sans-serif; font-size: 26px; font-weight: 700; color: #fca5a5; margin-bottom: 12px; }
.diagnostic-title-neg { font-family: 'Space Grotesk', sans-serif; font-size: 26px; font-weight: 700; color: #86efac; margin-bottom: 12px; }
.diagnostic-summary-text { font-size: 14px; color: #64748b; line-height: 1.6; max-width: 440px; margin: 0 auto 24px; }

/* Progress / Probability Tracking System */
.metric-track-wrapper { max-width: 340px; margin: 0 auto 24px; }
.metric-track-label-row { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 12px; }
.track-percentage-pos { color: #ef4444; font-weight: 700; }
.track-percentage-neg { color: #22c55e; font-weight: 700; }
.track-base-line { background: rgba(255, 255, 255, 0.06); border-radius: 100px; height: 8px; overflow: hidden; }
.track-fill-pos { height: 100%; border-radius: 100px; background: linear-gradient(90deg, #ef4444, #f97316); }
.track-fill-neg { height: 100%; border-radius: 100px; background: linear-gradient(90deg, #22c55e, #16a34a); }

/* Output Badges Group */
.pill-metadata-group { display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; }
.pill-metadata-node {
    background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 100px; padding: 6px 14px; font-size: 11px; color: #64748b;
}
.pill-metadata-node strong { color: #cbd5e1; font-weight: 600; }
.clinical-divider-line { border: none; border-top: 1px solid rgba(255, 255, 255, 0.05); margin: 36px 0; }

/* Premium Dashboard Custom Footer */
.dashboard-footer {
    text-align: center;
    padding: 24px 0 10px;
    margin-top: 60px;
    font-size: 12px;
    color: #475569;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    letter-spacing: 0.8px;
    text-transform: uppercase;
    font-weight: 500;
}

/* ── MEDIA QUERIES FOR PERFECT RESPONSIVENESS ── */

/* For Tablets (Small Desktops & iPads) */
@media screen and (max-width: 1024px) {
    .block-container { padding: 2rem 2.5rem !important; }
    .hero-title-main { font-size: 36px; }
}

/* For Mobile Phones (iPhone, Android Devices) */
@media screen and (max-width: 640px) {
    .block-container { padding: 1.2rem 1rem !important; }
    .dashboard-hero { padding: 28px 20px; margin-bottom: 24px; }
    .hero-title-main { font-size: 28px; }
    .hero-subtitle-main { font-size: 13px; line-height: 1.5; }
    .section-headline { font-size: 18px; margin-bottom: 18px; }
    .clinical-card { padding: 18px 16px; }
    .diagnostic-container-positive, .diagnostic-container-negative { padding: 32px 16px; }
    .diagnostic-title-pos, .diagnostic-title-neg { font-size: 22px; }
    .pill-metadata-group { flex-direction: column; align-items: center; gap: 6px; width: 100%; }
    .pill-metadata-node { width: 100%; text-align: center; padding: 8px; }
    .dashboard-footer { font-size: 11px; margin-top: 40px; }
}
</style>
""", unsafe_allow_html=True)


# ── MACHINE LEARNING ARTIFACT MANAGEMENT ──
@st.cache_resource
def load_artifacts():
    try:
        col_names = joblib.load("columns.pkl")
    except Exception:
        with open("columns.pkl", "rb") as f:
            col_names = pickle.load(f)
    scaler = joblib.load("scaler.pkl")
    model  = joblib.load("LogisticRegression_heart.pkl")
    return col_names, scaler, model

try:
    col_names, scaler, model = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)


# ── DASHBOARD APPLICATION HEADER ──
st.markdown("""
<div class="dashboard-hero">
    <div class="badge-clinical"><span class="pulse-indicator"></span>Clinical Decision Support</div>
    <div class="hero-title-main">Cardio<span>Sense</span></div>
    <p class="hero-subtitle-main">AI-powered cardiovascular risk assessment using 11 validated clinical markers. Get an instant prediction in seconds.</p>
    <div class="stat-panel-row">
        <div class="stat-metric-card"><div class="stat-metric-val">86.4%</div><div class="stat-metric-lbl">Model Accuracy</div></div>
        <div class="stat-metric-card"><div class="stat-metric-val">918</div><div class="stat-metric-lbl">Training Samples</div></div>
        <div class="stat-metric-card"><div class="stat-metric-val">11</div><div class="stat-metric-lbl">Clinical Markers</div></div>
        <div class="stat-metric-card"><div class="stat-metric-val">88%</div><div class="stat-metric-lbl">F1 Score</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"⚠️ Could not load model files.\n\nError: {load_error}")
    st.stop()


# ── MAIN ANALYSIS SYSTEM LAYOUT ──
layout_left, layout_right = st.columns([1.6, 1], gap="large")

with layout_left:
    st.markdown('<div class="step-header-pill"><span class="step-number-node">1</span> Patient Clinical Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-headline">Enter Patient Information</div>', unsafe_allow_html=True)

    # Card 1 — Demographics & Baseline Vitals
    st.markdown('<div class="clinical-card"><div class="card-header-block"><div class="card-header-icon">👤</div><div class="card-header-title">Demographics & Vitals</div></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=50, step=1)
    with c2:
        sex = st.selectbox("Sex", ["Male", "Female"])
    with c3:
        resting_bp = st.number_input("Resting BP (mmHg)", min_value=80, max_value=220, value=130, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

    # Card 2 — Cardiac Markers
    st.markdown('<div class="clinical-card"><div class="card-header-block"><div class="card-header-icon">💉</div><div class="card-header-title">Cardiac Markers</div></div>', unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    with c4:
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200, step=1)
    with c5:
        fasting_bs = st.selectbox("Fasting BS > 120 mg/dL", ["No (≤120)", "Yes (>120)"])
    with c6:
        max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

    # Card 3 — ECG & Stress Test
    st.markdown('<div class="clinical-card"><div class="card-header-block"><div class="card-header-icon">📈</div><div class="card-header-title">ECG & Stress Test</div></div>', unsafe_allow_html=True)
    c7, c8, c9 = st.columns(3)
    with c7:
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    with c8:
        exercise_angina = st.selectbox("Exercise Angina", ["No", "Yes"])
    with c9:
        oldpeak = st.number_input("ST Depression", min_value=-3.0, max_value=7.0, value=0.0, step=0.1, format="%.1f")
    st.markdown('</div>', unsafe_allow_html=True)

    # Card 4 — Chest Pain & ST Slope
    st.markdown('<div class="clinical-card"><div class="card-header-block"><div class="card-header-icon">🫁</div><div class="card-header-title">Chest Pain & ST Slope</div></div>', unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        chest_pain = st.selectbox("Chest Pain Type", [
            "ATA — Atypical Angina",
            "NAP — Non-Anginal Pain",
            "ASY — Asymptomatic",
            "TA — Typical Angina"
        ])
    with cb:
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
    st.markdown('<div class="clinical-info-note">💡 <b>ASY (Asymptomatic)</b> is the highest-risk type — disease may be present without symptoms.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


with layout_right:
    st.markdown('<div class="step-header-pill"><span class="step-number-node">2</span> Run Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-headline">Patient Summary</div>', unsafe_allow_html=True)

    # Live Compilation Matrix Panel
    st.markdown(f"""
    <div class="overview-panel-card">
        <div class="overview-panel-header">📋 Overview</div>
        <div class="overview-panel-row"><span class="overview-panel-key">Age</span><span class="overview-panel-val">{age} years</span></div>
        <div class="overview-panel-row"><span class="overview-panel-key">Sex</span><span class="overview-panel-val">{sex}</span></div>
        <div class="overview-panel-row"><span class="overview-panel-key">Resting BP</span><span class="overview-panel-val">{resting_bp} mmHg</span></div>
        <div class="overview-panel-row"><span class="overview-panel-key">Cholesterol</span><span class="overview-panel-val">{cholesterol} mg/dL</span></div>
        <div class="overview-panel-row"><span class="overview-panel-key">Max Heart Rate</span><span class="overview-panel-val">{max_hr} bpm</span></div>
        <div class="overview-panel-row"><span class="overview-panel-key">ST Depression</span><span class="overview-panel-val">{oldpeak}</span></div>
        <div class="overview-panel-row"><span class="overview-panel-key">Resting ECG</span><span class="overview-panel-val">{resting_ecg}</span></div>
        <div class="overview-panel-row"><span class="overview-panel-key">ST Slope</span><span class="overview-panel-val">{st_slope}</span></div>
        <div class="overview-panel-row"><span class="overview-panel-key">Exercise Angina</span><span class="overview-panel-val">{exercise_angina}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Action Trigger Button
    predict_btn = st.button("🫀  Analyze Heart Disease Risk", use_container_width=True)

    st.markdown("""
    <div class="clinical-disclaimer">
        ⚕️ This tool assists clinical decision-making only. Always consult a qualified cardiologist for diagnosis and treatment.
    </div>
    """, unsafe_allow_html=True)


# ── PREDICTION PIPELINE EXECUTION ──
if predict_btn:
    st.markdown('<hr class="clinical-divider-line">', unsafe_allow_html=True)
    st.markdown('<div class="step-header-pill"><span class="step-number-node">3</span> Result</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-headline">Risk Assessment</div>', unsafe_allow_html=True)

    # Original pipeline mappings
    sex_m    = 1 if sex == "Male" else 0
    fbs      = 1 if "Yes" in fasting_bs else 0
    angina_y = 1 if exercise_angina == "Yes" else 0
    cp_code  = chest_pain.split(" — ")[0]

    row = {
        "Age": age, "RestingBP": resting_bp, "Cholesterol": cholesterol,
        "FastingBS": fbs, "MaxHR": max_hr, "Oldpeak": oldpeak, "Sex_M": sex_m,
        "ChestPainType_ATA": 1 if cp_code == "ATA" else 0,
        "ChestPainType_NAP": 1 if cp_code == "NAP" else 0,
        "ChestPainType_TA":  1 if cp_code == "TA"  else 0,
        "RestingECG_Normal": 1 if resting_ecg == "Normal" else 0,
        "RestingECG_ST":     1 if resting_ecg == "ST" else 0,
        "ExerciseAngina_Y":  angina_y,
        "ST_Slope_Flat":     1 if st_slope == "Flat" else 0,
        "ST_Slope_Up":       1 if st_slope == "Up"   else 0,
    }

    # Pipeline transforms
    input_df     = pd.DataFrame([row])[col_names]
    input_scaled = scaler.transform(input_df)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0]

    risk_pct = int(probability[1] * 100)
    safe_pct = int(probability[0] * 100)

    # Clean Output grid
    result_col_wrap, _ = st.columns([1.5, 1])
    
    with result_col_wrap:
        if prediction == 1:
            st.markdown(f"""
            <div class="diagnostic-container-positive">
                <span style="font-size: 54px; margin-bottom: 16px; display: block;">⚠️</span>
                <div class="diagnostic-badge-pos">Positive Detection</div>
                <div class="diagnostic-title-pos">Heart Disease Detected</div>
                <p class="diagnostic-summary-text">The model predicts elevated cardiovascular risk. Immediate cardiology consultation is strongly recommended.</p>
                <div class="metric-track-wrapper">
                    <div class="metric-track-label-row">
                        <span style="color: #475569;">Risk Probability</span>
                        <span class="track-percentage-pos">{risk_pct}%</span>
                    </div>
                    <div class="track-base-line">
                        <div class="track-fill-pos" style="width:{risk_pct}%"></div>
                    </div>
                </div>
                <div class="pill-metadata-group">
                    <div class="pill-metadata-node">Confidence <strong>{risk_pct}%</strong></div>
                    <div class="pill-metadata-node">Model <strong>Logistic Reg.</strong></div>
                    <div class="pill-metadata-node">Accuracy <strong>86.4%</strong></div>
                    <div class="pill-metadata-node">F1 Score <strong>88.0%</strong></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="diagnostic-container-negative">
                <span style="font-size: 54px; margin-bottom: 16px; display: block;">✅</span>
                <div class="diagnostic-badge-neg">Negative Detection</div>
                <div class="diagnostic-title-neg">No Heart Disease Detected</div>
                <p class="diagnostic-summary-text">The model predicts low cardiovascular risk. Continue regular health monitoring and a heart-healthy lifestyle.</p>
                <div class="metric-track-wrapper">
                    <div class="metric-track-label-row">
                        <span style="color: #475569;">Healthy Probability</span>
                        <span class="track-percentage-neg">{safe_pct}%</span>
                    </div>
                    <div class="track-base-line">
                        <div class="track-fill-neg" style="width:{safe_pct}%"></div>
                    </div>
                </div>
                <div class="pill-metadata-group">
                    <div class="pill-metadata-node">Confidence <strong>{safe_pct}%</strong></div>
                    <div class="pill-metadata-node">Model <strong>Logistic Reg.</strong></div>
                    <div class="pill-metadata-node">Accuracy <strong>86.4%</strong></div>
                    <div class="pill-metadata-node">F1 Score <strong>88.0%</strong></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── DOM FOOTER INJECTION ──
st.markdown("""
<div class="dashboard-footer">
    CardioSense Ai - Developed by Amit Bind
</div>
""", unsafe_allow_html=True)