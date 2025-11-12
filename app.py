import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import BytesIO
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ---------------------------
# Load model (must exist in same folder)
# ---------------------------
with open("final_logreg_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Bankruptcy Prediction",
    page_icon="🏦",
    layout="wide",
)

# ---------------------------
# CSS / Theme: light, high-contrast, modern
# ---------------------------
st.markdown(
    """
    <style>
    /* Page background */
    html, body, .main {
        background-color: #f7fafc;
        color: #0b2545;
        font-family: Inter, 'Segoe UI', Roboto, sans-serif;
    }

    /* Main container card */
    .app-card {
        background: #fff;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 6px 20px rgba(16,24,40,0.06);
        margin-bottom: 18px;
    }

    /* Header */
    .title {
        font-size: 26px;
        font-weight: 800;
        color: #072244;
    }
    .subtitle {
        color: #475569;
        margin-top: 6px;
        margin-bottom: 18px;
    }

    /* Section header */
    .section-header {
        color: #05264c;
        font-weight: 700;
        margin-bottom: 10px;
        font-size: 16px;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg,#2b8ff4,#2ad5b7);
        color: white;
        border: none;
        padding: 8px 14px;
        border-radius: 10px;
        font-weight: 700;
    }

    /* Suggestions */
    .suggest {
        border-radius: 10px;
        padding: 12px;
        font-weight: 600;
        color: #0b2545;
    }

    /* Metrics card tweaks */
    .metric-card {
        background: linear-gradient(180deg, rgba(247,250,255,1), rgba(255,255,255,1));
        border-radius: 10px;
        padding: 14px;
        border: 1px solid rgba(6, 78, 59, 0.04);
    }

    /* Ensure text always dark */
    h1,h2,h3,p,div,span,label {
        color: #0b2545 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Sidebar: app info, upload/eval controls
# ---------------------------
with st.sidebar:
    st.markdown("<div style='text-align:center'><img src='https://img.icons8.com/fluency/48/000000/bankruptcy.png' /></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;margin-bottom:4px;'>Bankruptcy Predictor</h3>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;color:#475569;font-size:13px;'>Logistic Regression — Light UI</div>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 👩‍💼 App actions")
    uploaded = st.file_uploader("Upload CSV for batch prediction (optional)", type=["csv","xlsx","xls"])
    st.write("Tip: CSV should contain feature columns matching the model features.")
    st.markdown("---")

    st.markdown("### ⚙️ Advanced / Admin")
    show_metrics = st.checkbox("Show model evaluation when CSV has true labels", value=True)
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("• Developer: Your Name  \n• Contact: your@email.com")
    st.markdown("---")
    if st.button("🔄 Reset selections"):
        st.experimental_rerun()

# ---------------------------
# Top header
# ---------------------------
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("<div class='title'>🏦 Bankruptcy Prediction App</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Simple, reliable bankruptcy risk predictions with probability visualization and evaluation.</div>", unsafe_allow_html=True)

with col_h2:
    # small model info card
    classes = getattr(model, "classes_", None)
    model_name = model.__class__.__name__
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown(f"**Model:** {model_name}  \n**Supports proba:** {'Yes' if hasattr(model,'predict_proba') else 'No'}")
    if classes is not None:
        st.markdown(f"**Classes:** {classes}")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Main UI card: inputs & predict
# ---------------------------
st.markdown("<div class='app-card'>", unsafe_allow_html=True)

st.markdown("<div class='section-header'>📊 Enter Company Financial Indicators</div>", unsafe_allow_html=True)
# use selectboxes for discrete options
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

# pack into dataframe
input_df = pd.DataFrame({
    "industrial_risk":[industrial_risk],
    "management_risk":[management_risk],
    "financial_flexibility":[financial_flexibility],
    "credibility":[credibility],
    "competitiveness":[competitiveness],
    "operating_risk":[operating_risk]
})

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# predict
if st.button("🔍 Predict Bankruptcy"):
    pred = model.predict(input_df)[0]

    # probabilities
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        # robust mapping of class index for label 0 (bankruptcy) and 1 (non-bankruptcy)
        try:
            idx0 = list(model.classes_).index(0)
        except Exception:
            # fallback to first index being bankruptcy (if model was trained that way)
            idx0 = 0
        idx1 = 1 if idx0 == 0 else 0
        prob_bank = float(proba[idx0])
        prob_non = float(proba[idx1])
    else:
        prob_bank = 1.0 if pred == 0 else 0.0
        prob_non = 1.0 - prob_bank

    # result card
    if pred == 0:
        st.markdown("<div style='background:#fff6f6;border-left:6px solid #ff4d4d;border-radius:10px;padding:12px;margin-bottom:10px'>"
                    "<b style='color:#7a1f1f;'>⚠️ Result: Bankruptcy risk</b><div style='color:#243b55;margin-top:6px;'>The model predicts the company may be at risk. Consider reviewing liquidity and management.</div>"
                    "</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='background:#f6fffb;border-left:6px solid #00b37a;border-radius:10px;padding:12px;margin-bottom:10px'>"
                    "<b style='color:#066a3b;'>✅ Result: Financially Healthy</b><div style='color:#243b55;margin-top:6px;'>Keep monitoring financial flexibility and competitiveness.</div>"
                    "</div>", unsafe_allow_html=True)

    # probability metrics (two high-contrast cards)
    colA, colB = st.columns(2)
    with colA:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-weight:700;color:#6b1f1f'>Bankruptcy</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:28px;font-weight:800;color:#081426'>{prob_bank*100:.1f} %</div>", unsafe_allow_html=True)
        st.markdown("<div style='color:#536878;font-size:12px;margin-top:4px;'>Probability (model)</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with colB:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-weight:700;color:#0b6a34'>Non-Bankruptcy</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:28px;font-weight:800;color:#081426'>{prob_non*100:.1f} %</div>", unsafe_allow_html=True)
        st.markdown("<div style='color:#536878;font-size:12px;margin-top:4px;'>Probability (model)</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # interactive donut using plotly
    fig = go.Figure(data=[go.Pie(
        labels=["Bankruptcy", "Non-Bankruptcy"],
        values=[prob_bank, prob_non],
        hole=0.45,
        marker=dict(colors=["#ff6b6b","#2ad5b7"]),
        sort=False
    )])
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="center", x=0.5),
                      annotations=[dict(text="Probabilities", x=0.5, y=0.5, font_size=14, showarrow=False)])
    st.plotly_chart(fig, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)  # close main card

# ---------------------------
# Batch predictions & evaluation (if user uploaded CSV)
# ---------------------------
if uploaded is not None:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📁 Batch prediction & evaluation</div>", unsafe_allow_html=True)

    # read CSV or Excel
    try:
        if uploaded.name.endswith((".xls",".xlsx")):
            batch = pd.read_excel(uploaded)
        else:
            batch = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
        batch = None

    if batch is not None:
        st.markdown("**Preview:**")
        st.dataframe(batch.head())

        # Check that model features exist; if not, warn user
        model_features = getattr(model, "feature_names_in_", None)
        if model_features is None:
            # attempt to use batch columns minus 'class' if present
            model_features = [c for c in batch.columns if c.lower() != "class" and c.lower() != "target"]
        missing = [c for c in model_features if c not in batch.columns]
        if missing:
            st.warning(f"Uploaded data is missing model features: {missing}. App will attempt to use columns that match.")
            X_batch = batch.reindex(columns=[c for c in model_features if c in batch.columns]).fillna(0)
        else:
            X_batch = batch[model_features].fillna(0)

        # Predict
        preds = model.predict(X_batch)
        batch["prediction"] = preds
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_batch)
            # find bankruptcy index
            try:
                idx_bank = list(model.classes_).index(0)
            except Exception:
                idx_bank = 0
            batch["prob_bankruptcy"] = proba[:, idx_bank]
            batch["prob_nonbank"] = 1 - proba[:, idx_bank]
        st.success("✅ Batch predictions complete.")
        st.dataframe(batch.head())

        # allow download
        csv_bytes = batch.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download predictions CSV", data=csv_bytes, file_name="batch_predictions.csv", mime="text/csv")

        # Evaluation if user provided label column name (class/target)
        label_col = None
        for cand in ["class", "Class", "target", "y", "label"]:
            if cand in batch.columns:
                label_col = cand
                break

        if show_metrics and label_col:
            st.markdown("### 📈 Evaluation metrics (using uploaded labels)")
            y_true = batch[label_col]
            y_pred = batch["prediction"]

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Accuracy", f"{acc:.3f}")
            mcol2.metric("Precision", f"{prec:.3f}")
            mcol3.metric("Recall", f"{rec:.3f}")
            mcol4.metric("F1", f"{f1:.3f}")

            # confusion matrix (plotly heatmap)
            cm = confusion_matrix(y_true, y_pred)
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=["Pred: 0", "Pred: 1"][:cm.shape[1]],
                y=["True: 0", "True: 1"][:cm.shape[0]],
                colorscale="Blues",
                showscale=True,
            ))
            fig_cm.update_layout(title="Confusion Matrix", margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)

            # ROC / AUC if probabilities available
            if "prob_bankruptcy" in batch.columns:
                try:
                    y_true_bin = (y_true == 0).astype(int)  # bankruptcy positive class as 1
                    auc = roc_auc_score(y_true_bin, batch["prob_bankruptcy"])
                    fpr, tpr, _ = roc_curve(y_true_bin, batch["prob_bankruptcy"])
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
                    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
                    fig_roc.update_layout(title=f"ROC Curve (AUC = {auc:.3f})", xaxis_title="FPR", yaxis_title="TPR", margin=dict(l=10,r=10,t=30,b=10))
                    st.plotly_chart(fig_roc, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not compute ROC: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

