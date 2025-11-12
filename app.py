# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
)

# ---------------------------
# MUST call set_page_config before any other Streamlit calls
# ---------------------------
st.set_page_config(page_title="Bankruptcy Prediction App", page_icon="🏦", layout="wide")

# ---------------------------
# Configuration & constants
# ---------------------------
MODEL_PATH = "final_logreg_model.pkl"
LABEL_CANDIDATES = ["class", "Class", "target", "Target", "y", "label", "Label"]

# ---------------------------
# Load model (cached)
# ---------------------------
@st.cache_resource
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file '{MODEL_PATH}' not found in the app folder. Upload or place the model file and restart.")
    st.stop()

# ---------------------------
# Utility helpers
# ---------------------------
def prepare_features_from_df(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    - Drop a label column if present (common names).
    - Align columns to model.feature_names_in_ (if available) by adding missing columns with 0.
    - Otherwise use numeric columns.
    """
    X = df.copy()
    for c in LABEL_CANDIDATES:
        if c in X.columns:
            X = X.drop(columns=[c])
            break

    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        for col in expected:
            if col not in X.columns:
                X[col] = 0.0
        X = X.reindex(columns=expected)
    else:
        X = X.select_dtypes(include=[np.number]).copy()
        if X.shape[1] == 0:
            raise ValueError("No numeric columns found and model.feature_names_in_ is not available.")
    return X.fillna(0)

def get_class_indices(model):
    """Return (idx_bankruptcy, idx_nonbank) mapping for model.classes_ order."""
    classes = list(getattr(model, "classes_", []))
    idx_bank = None
    # try numeric 0 label
    try:
        idx_bank = classes.index(0)
    except ValueError:
        # try common string forms
        for i, c in enumerate(classes):
            s = str(c).lower()
            if s.startswith("bank") or "bankrupt" in s:
                idx_bank = i
                break
    if idx_bank is None:
        idx_bank = 0
    idx_non = 1 if idx_bank == 0 and len(classes) > 1 else 0
    return idx_bank, idx_non

def plot_horizontal_prob_bar(prob_bank: float, prob_non: float):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[prob_bank*100],
        y=["Bankruptcy"],
        orientation="h",
        name="Bankruptcy",
        marker_color="#ef553b",
        hovertemplate="%{x:.1f}%<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        x=[prob_non*100],
        y=["Non-Bankruptcy"],
        orientation="h",
        name="Non-Bankruptcy",
        marker_color="#00cc96",
        hovertemplate="%{x:.1f}%<extra></extra>"
    ))
    fig.update_layout(
        barmode="group",
        xaxis=dict(range=[0,100], title="Probability (%)"),
        yaxis=dict(title=""),
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f"Pred {i}" for i in range(cm.shape[1])],
        y=[f"True {i}" for i in range(cm.shape[0])],
        colorscale="Blues",
        showscale=True,
        text=cm,
        texttemplate="%{text}"
    ))
    fig.update_layout(title="Confusion Matrix", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

def plot_roc_curve(y_true_binary, y_scores):
    fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        margin=dict(l=10, r=10, t=40, b=10),
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_pr_curve(y_true_binary, y_scores):
    precision, recall, _ = precision_recall_curve(y_true_binary, y_scores)
    pr_auc = auc(recall, precision)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"PR (AUC={pr_auc:.3f})"))
    fig.update_layout(
        title="Precision–Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        margin=dict(l=10, r=10, t=40, b=10),
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Sidebar & Project Info
# ---------------------------
with st.sidebar:
    st.title("About / Upload")
    st.write("- Predict bankruptcy risk using a trained Logistic Regression model.")
    st.write("- Provide inputs manually or upload CSV/XLSX with same feature columns.")
    st.markdown("---")
    uploaded = st.file_uploader("Upload CSV/XLSX for batch predictions (optional)", type=["csv", "xlsx", "xls"])
    st.caption("If your file has a label column (e.g., `class`), the app will evaluate and show ROC/PR/CM.")
    st.markdown("---")
    st.write("Model info:")
    st.write(f"- Type: `{model.__class__.__name__}`")
    st.write(f"- predict_proba: {'Yes' if hasattr(model,'predict_proba') else 'No'}")
    if hasattr(model, "classes_"):
        st.write(f"- Classes: {list(model.classes_)}")
    st.markdown("---")
    st.write("Developer: Your Name")

# ---------------------------
# Header & Overview
# ---------------------------
st.title("🏦 Bankruptcy Prediction App")
st.write("Enter the six features (0, 0.5, or 1) for a single prediction, or upload a file for batch predictions and evaluation.")

with st.expander("Features expected (click to expand)"):
    st.markdown(
        """
        - `industrial_risk`, `management_risk`, `financial_flexibility`,  
          `credibility`, `competitiveness`, `operating_risk`
        - Labels in uploaded file can be numeric (0/1) or text ('bankruptcy'/'non-bankruptcy').
        """
    )

# ---------------------------
# Manual input (single)
# ---------------------------
st.header("Single prediction (manual)")
opts = [0.0, 0.5, 1.0]
c1, c2 = st.columns(2)
with c1:
    industrial_risk = st.selectbox("Industrial Risk", opts, index=1)
    management_risk = st.selectbox("Management Risk", opts, index=1)
    financial_flexibility = st.selectbox("Financial Flexibility", opts, index=1)
with c2:
    credibility = st.selectbox("Credibility", opts, index=1)
    competitiveness = st.selectbox("Competitiveness", opts, index=1)
    operating_risk = st.selectbox("Operating Risk", opts, index=1)

single_df = pd.DataFrame([{
    "industrial_risk": industrial_risk,
    "management_risk": management_risk,
    "financial_flexibility": financial_flexibility,
    "credibility": credibility,
    "competitiveness": competitiveness,
    "operating_risk": operating_risk
}])

try:
    X_single = prepare_features_from_df(single_df, model)
except Exception as e:
    st.error(f"Input preparation failed: {e}")
    X_single = None

if st.button("Predict (single)"):
    if X_single is None:
        st.error("Input not ready for prediction.")
    else:
        try:
            pred = model.predict(X_single)[0]
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_single)[0]
                idx_bank, idx_non = get_class_indices(model)
                p_bank = float(probs[idx_bank])
                p_non = float(probs[idx_non]) if idx_non < len(probs) else 1.0 - p_bank
            else:
                p_bank = 1.0 if pred == 0 else 0.0
                p_non = 1.0 - p_bank

            if pred == 0:
                st.error("Prediction: RISK OF BANKRUPTCY")
            else:
                st.success("Prediction: NON-BANKRUPT (Financially Healthy)")

            st.subheader("Predicted probabilities")
            sa, sb = st.columns(2)
            sa.metric("Bankruptcy", f"{p_bank*100:.1f}%")
            sb.metric("Non-Bankruptcy", f"{p_non*100:.1f}%")

            plot_horizontal_prob_bar(p_bank, p_non)

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ---------------------------
# Batch upload & evaluation
# ---------------------------
if uploaded is not None:
    st.header("Batch predictions & evaluation")
    try:
        if uploaded.name.endswith((".xls", ".xlsx")):
            df_in_
