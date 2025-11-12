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
# Sidebar
# ---------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/money.png", width=70)
    st.title("Bankruptcy Predictor")
    st.caption("Logistic Regression Model")

    st.markdown("---")
    uploaded = st.file_uploader("📂 Upload CSV or Excel for batch predictions", type=["csv", "xlsx"])
    show_metrics = st.checkbox("Show evaluation if labels are provided", value=True)

    st.markdown("---")
    st.markdown("**Developed by:** Your Name")
    st.markdown("📧 your@email.com")

    if st.button("🔄 Reset"):
        st.experimental_rerun()

# ---------------------------
# Header
# ---------------------------
st.title("🏦 Bankruptcy Prediction App")
st.write("Predict company bankruptcy risk using Logistic Regression — accurate, clear, and easy to use.")

# ---------------------------
# Input Section
# ---------------------------
st.header("📊 Enter Company Financial Indicators")

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
if st.button("🔍 Predict Bankruptcy"):
    pred = model.predict(data)[0]
    prob_bank = model.predict_proba(data)[0][0]
    prob_non = 1 - prob_bank

    if pred == 0:
        st.error("⚠️ The company may be at risk of **Bankruptcy**. Consider improving liquidity and reducing operational risk.")
    else:
        st.success("✅ The company appears **Financially Healthy**. Maintain good management and financial flexibility.")

    # Display probabilities
    st.subheader("Predicted Probabilities")
    colA, colB = st.columns(2)
    colA.metric("Bankruptcy Probability", f"{prob_bank*100:.1f}%")
    colB.metric("Non-Bankruptcy Probability", f"{prob_non*100:.1f}%")

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
    st.header("📂 Batch Predictions")

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

        # Evaluation (optional)
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

