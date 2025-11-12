import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

# ---------------------------
# Load Model
# ---------------------------
with open("final_logreg_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------------------
# Page Setup
# ---------------------------
st.set_page_config(
    page_title="Bankruptcy Prediction App",
    page_icon="🏦",
    layout="wide",
)

# ---------------------------
# Force LIGHT MODE (white background, blue theme)
# ---------------------------
st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        font-family: 'Inter', sans-serif;
    }
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #f5f7fa !important;
        color: #1a1a1a !important;
    }
    /* General text */
    h1,h2,h3,h4,h5,h6,p,div,span,label {
        color: #1a1a1a !important;
    }
    /* Headings */
    .title {
        color: #003366 !important;
        font-weight: 800;
        text-align: center;
        font-size: 2rem;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        text-align: center;
        color: #406080 !important;
        margin-bottom: 2rem;
        font-size: 1rem;
    }
    /* Card */
    .card {
        background: #ffffff;
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 12px;
        padding: 1.3rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }
    /* Section header */
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #003366 !important;
        margin-bottom: 1rem;
    }
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg,#007bff,#339cff);
        color: #ffffff !important;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        font-size: 1rem;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg,#339cff,#007bff);
        transform: scale(1.03);
    }
    /* Dropdowns / Inputs */
    div[data-baseweb="select"] > div {
        background-color: #f9fbff !important;
        border: 1px solid rgba(0,0,0,0.1) !important;
        color: #1a1a1a !important;
        border-radius: 8px !important;
    }
    /* Metrics */
    .metric-card {
        background-color: #f7faff;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(0,0,0,0.05);
    }
    /* Suggestions */
    .suggestion {
        border-radius: 10px;
        padding: 1rem;
        font-weight: 500;
        margin-top: 1rem;
    }
    .good {
        background: #e9f9ef;
        border-left: 6px solid #00b37a;
    }
    .bad {
        background: #fff3f3;
        border-left: 6px solid #e63946;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/money.png", width=70)
    st.markdown("### 🏦 Bankruptcy Predictor")
    st.caption("Light Theme • Logistic Regression Model")
    st.markdown("---")

    uploaded = st.file_uploader("📂 Upload CSV or Excel for batch predictions", type=["csv", "xlsx"])
    show_metrics = st.checkbox("Show evaluation if labels provided", value=True)

    st.markdown("---")
    st.markdown("**Developed by:** Your Name  \n📧 your@email.com")
    if st.button("🔄 Reset"):
        st.experimental_rerun()

# ---------------------------
# Header
# ---------------------------
st.markdown("<div class='title'>🏦 Bankruptcy Prediction App</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict company bankruptcy risk using Logistic Regression — accurate, clear, and user-friendly.</div>", unsafe_allow_html=True)

# ---------------------------
# Input Section
# ---------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>📊 Enter Company Financial Indicators</div>", unsafe_allow_html=True)

options = [0.0, 0.5, 1.0]
c1, c2 = st.columns(2)
with c1:
    industrial_risk = st.selectbox("Industrial Risk", options, index=1)
    management_risk = st.selectbox("Management Risk", options, index=1)
    financial_flexibility = st.selectbox("Financial Flexibility", options, index=1)
with c2:
    credibility = st.selectbox("Credibility", options, index=1)
    competitiveness = st.selectbox("Competitiveness", options, index=1)
    operating_risk = st.selectbox("Operating Risk", options, index=1)

data = pd.DataFrame({
    "industrial_risk": [industrial_risk],
    "management_risk": [management_risk],
    "financial_flexibility": [financial_flexibility],
    "credibility": [credibility],
    "competitiveness": [competitiveness],
    "operating_risk": [operating_risk],
})

# ---------------------------
# Prediction Button
# ---------------------------
st.markdown("</div>", unsafe_allow_html=True)
if st.button("🔍 Predict Bankruptcy"):
    pred = model.predict(data)[0]
    prob_bank = model.predict_proba(data)[0][0]
    prob_non = 1 - prob_bank

    if pred == 0:
        st.markdown("<div class='suggestion bad'>⚠️ The company may be at risk of <b>Bankruptcy</b>. Improve liquidity and reduce operational risk.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='suggestion good'>✅ The company appears <b>Financially Healthy</b>. Maintain your financial flexibility and management quality.</div>", unsafe_allow_html=True)

    # Probability metrics
    colA, colB = st.columns(2)
    with colA:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**Bankruptcy Probability**")
        st.markdown(f"<h2 style='color:#d62828'>{prob_bank*100:.1f}%</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with colB:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**Non-Bankruptcy Probability**")
        st.markdown(f"<h2 style='color:#0077b6'>{prob_non*100:.1f}%</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Donut chart
    fig = go.Figure(data=[go.Pie(
        labels=["Bankruptcy", "Non-Bankruptcy"],
        values=[prob_bank, prob_non],
        hole=0.5,
        marker=dict(colors=["#ff6b6b", "#0077b6"]),
        textinfo="percent+label",
    )])
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Batch Predictions
# ---------------------------
if uploaded is not None:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📂 Batch Predictions</div>", unsafe_allow_html=True)

    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        st.dataframe(df.head())

        preds = model.predict(df)
        df["Prediction"] = preds
        df["Bankruptcy Probability"] = model.predict_proba(df)[:, 0]
        st.success("✅ Predictions complete.")
        st.dataframe(df.head())

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")

        label_col = next((c for c in ["class","target","label","y"] if c in df.columns), None)
        if show_metrics and label_col:
            y_true = df[label_col]
            y_pred = df["Prediction"]
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Accuracy", f"{acc:.3f}")
            c2.metric("Precision", f"{prec:.3f}")
            c3.metric("Recall", f"{rec:.3f}")
            c4.metric("F1", f"{f1:.3f}")
    except Exception as e:
        st.error(f"Error: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

