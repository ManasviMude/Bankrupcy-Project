import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Load trained Logistic Regression model
# ---------------------------
with open('final_logreg_model.pkl', 'rb') as f:
    model = pickle.load(f)

# ---------------------------
# Streamlit Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Bankruptcy Prediction App",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# Custom CSS for Light Theme + Proper Contrast
# ---------------------------
st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
            color: #2c2c2c;
            font-family: 'Segoe UI', sans-serif;
        }

        .stApp {
            background-color: #ffffff;
            padding: 2.5rem;
            border-radius: 1.5rem;
            box-shadow: 0px 6px 18px rgba(0,0,0,0.05);
        }

        .main-title {
            text-align: center;
            color: #002855;
            font-weight: 800;
            font-size: 2.2rem;
            background: linear-gradient(90deg, #003366, #0077b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.3rem;
        }

        .subtitle {
            text-align: center;
            color: #444;
            font-size: 1.05rem;
            margin-bottom: 2rem;
        }

        .section-box {
            background-color: #f0f4f8;
            border-radius: 12px;
            padding: 1.5rem 1.5rem 0.5rem 1.5rem;
            margin-bottom: 2rem;
            box-shadow: inset 0 0 8px rgba(0,0,0,0.03);
        }

        .section-header {
            color: #003366;
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }

        .stSelectbox label {
            font-weight: 600;
            color: #2c3e50;
        }

        .stButton button {
            background: linear-gradient(to right, #4A90E2, #50E3C2);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.3rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease-in-out;
        }

        .stButton button:hover {
            transform: scale(1.05);
            background: linear-gradient(to right, #50E3C2, #4A90E2);
        }

        /* Suggestion box styling */
        .suggestion-box {
            padding: 14px 16px;
            border-radius: 10px;
            margin-top: 10px;
            font-weight: 500;
            font-size: 0.95rem;
        }

        .bad-suggestion {
            background-color: #ffe6e6;  /* soft red */
            border-left: 6px solid #ff4d4d;
            color: #333333;  /* darker text for readability */
        }

        .good-suggestion {
            background-color: #e6ffed;  /* soft green */
            border-left: 6px solid #00cc66;
            color: #333333;  /* dark text for readability */
        }

        .footer {
            text-align: center;
            color: #777;
            margin-top: 2.5rem;
            font-size: 0.9rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Header Section
# ---------------------------
st.markdown("<h1 class='main-title'>🏦 Bankruptcy Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict a company's financial health using Logistic Regression</p>", unsafe_allow_html=True)

# ---------------------------
# Input Section
# ---------------------------
st.markdown("<div class='section-box'>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>📊 Enter Company Financial Indicators</div>", unsafe_allow_html=True)

options = [0.0, 0.5, 1.0]

col1, col2 = st.columns(2)
with col1:
    industrial_risk = st.selectbox("Industrial Risk", options, index=1)
    management_risk = st.selectbox("Management Risk", options, index=1)
    financial_flexibility = st.selectbox("Financial Flexibility", options, index=1)

with col2:
    credibility = st.selectbox("Credibility", options, index=1)
    competitiveness = st.selectbox("Competitiveness", options, index=1)
    operating_risk = st.selectbox("Operating Risk", options, index=1)

st.markdown("</div>", unsafe_allow_html=True)  # close section box

# ---------------------------
# Data Preparation
# ---------------------------
data = pd.DataFrame({
    'industrial_risk': [industrial_risk],
    'management_risk': [management_risk],
    'financial_flexibility': [financial_flexibility],
    'credibility': [credibility],
    'competitiveness': [competitiveness],
    'operating_risk': [operating_risk]
})

# ---------------------------
# Prediction Section + Probability Chart
# ---------------------------
st.markdown("---")
if st.button("🔍 Predict Bankruptcy"):
    # Prediction and (if available) probabilities
    prediction = model.predict(data)[0]

    # Try to obtain probability for each class
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(data)[0]  # array in order of model.classes_
        # Ensure mapping: find index of class 0 (bankruptcy) and class 1 (non-bankruptcy)
        classes = list(model.classes_)
        # Default mapping
        try:
            idx_bankruptcy = classes.index(0)
            idx_nonbank = classes.index(1)
        except ValueError:
            # if classes are strings or reversed, try common labels
            # map smallest prob to bankruptcy if unsure (fallback)
            idx_bankruptcy = 0
            idx_nonbank = 1 if len(classes) > 1 else 0

        prob_bankruptcy = probs[idx_bankruptcy]
        prob_nonbank = probs[idx_nonbank] if idx_nonbank < len(probs) else (1 - prob_bankruptcy)
    else:
        # fallback: no predict_proba available
        prob_bankruptcy = 1.0 if prediction == 0 else 0.0
        prob_nonbank = 1.0 - prob_bankruptcy

    # Show textual result and suggestion
    st.markdown("---")
    if prediction == 0:
        st.error("⚠️ **Result:** The company is predicted to be at risk of **Bankruptcy**.")
        st.markdown(
            "<div class='suggestion-box bad-suggestion'>"
            "💡 <b>Suggestion:</b> Improve liquidity, reduce operational risk, and enhance management efficiency."
            "</div>",
            unsafe_allow_html=True
        )
    else:
        st.success("✅ **Result:** The company is predicted to be **Financially Healthy**.")
        st.markdown(
            "<div class='suggestion-box good-suggestion'>"
            "💡 <b>Suggestion:</b> Maintain strong financial flexibility and competitiveness to ensure long-term stability."
            "</div>",
            unsafe_allow_html=True
        )

    # ---------------------------
    # Show probabilities (percentages)
    # ---------------------------
    st.markdown("### 🔢 Predicted Probabilities")
    perc_bank = round(prob_bankruptcy * 100, 1)
    perc_non = round(prob_nonbank * 100, 1)

    # numeric display side-by-side
    c1, c2 = st.columns(2)
    c1.metric("Bankruptcy", f"{perc_bank} %", delta=None)
    c2.metric("Non-Bankruptcy", f"{perc_non} %", delta=None)

    # ---------------------------
    # Create a pie chart and horizontal bar chart using matplotlib
    # ---------------------------
    labels = ['Bankruptcy', 'Non-Bankruptcy']
    values = [prob_bankruptcy, prob_nonbank]
    colors = ['#ff6b6b', '#4cd964']

    # Pie chart (donut)
    fig1, ax1 = plt.subplots(figsize=(3.5, 3.5))
    wedges, texts, autotexts = ax1.pie(
        values,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.45, edgecolor='w')
    )
    ax1.axis('equal')  # equal aspect ratio ensures pie is drawn as a circle
    plt.setp(autotexts, size=10, weight="bold", color="#333333")
    plt.setp(texts, size=10)
    st.pyplot(fig1, clear_figure=True)

    # Horizontal bar chart
    fig2, ax2 = plt.subplots(figsize=(6, 1.2))
    y_pos = np.arange(len(labels))
    ax2.barh(y_pos, [v * 100 for v in values], color=colors, height=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.set_xlim(0, 100)
    ax2.invert_yaxis()  # highest on top
    ax2.set_xlabel('Probability (%)')
    for i, v in enumerate([v * 100 for v in values]):
        ax2.text(v + 1.5, i, f"{v:.1f}%", va='center', color='#333333', fontweight='600')
    st.pyplot(fig2, clear_figure=True)


