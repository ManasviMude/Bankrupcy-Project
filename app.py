# app.py  — Streamlit Bankruptcy Predictor (detailed UI, upload enabled, no pie charts; uses ROC/CM/etc.)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
)

# =========================
# Configuration
# =========================
st.set_page_config(page_title="Bankruptcy Prediction App", page_icon="🏦", layout="wide")

MODEL_PATH = "final_logreg_model.pkl"
LABEL_CANDIDATES = ["class", "Class", "target", "Target", "y", "label", "Label"]

# =========================
# Load trained model
# =========================
@st.cache_resource
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file '{MODEL_PATH}' not found in the app folder.")
    st.stop()

# =========================
# Utility functions
# =========================
def prepare_features_from_df(df: pd.DataFrame, model) -> pd.DataFrame:
    """Drop label column if present; align columns to model.feature_names_in_ (if available)."""
    X = df.copy()
    for c in LABEL_CANDIDATES:
        if c in X.columns:
            X = X.drop(columns=[c])
            break
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        # add missing expected columns as zeros
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
    """Return indices (idx_bankruptcy, idx_nonbankruptcy) for model.predict_proba columns."""
    classes = list(getattr(model, "classes_", []))
    idx_bank = None
    try:
        idx_bank = classes.index(0)
    except ValueError:
        # if string labels, try to guess bankruptcy
        for i, c in enumerate(classes):
            if str(c).lower().startswith("bank"):
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
        height=250,
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

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.title("About this Project")
    st.write(
        "- **Goal**: Predict bankruptcy risk using Logistic Regression.\n"
        "- **Inputs**: Six risk/score indicators coded as `0`, `0.5`, or `1`.\n"
        "- **Outputs**: Class prediction + probabilities.\n"
        "- **Batch Mode**: Upload CSV/XLSX to predict many rows and (if labels present) evaluate with ROC/PR curves."
    )
    st.markdown("---")
    uploaded = st.file_uploader("📂 Upload CSV/XLSX for batch predictions", type=["csv", "xlsx", "xls"])
    st.caption("Tip: If your file includes a label column (e.g., `class`), the app will show evaluation with ROC, PR, and Confusion Matrix.")
    st.markdown("---")
    st.write("**Model Info**")
    st.write(f"- Type: `{model.__class__.__name__}`")
    st.write(f"- Probabilities: `{'Yes' if hasattr(model, 'predict_proba') else 'No'}`")
    if hasattr(model, "classes_"):
        st.write(f"- Classes: `{list(model.classes_)}`")
    st.markdown("---")
    st.write("Developer: Your Name")
    st.write("Contact: you@example.com")

# =========================
# Header & Overview
# =========================
st.title("🏦 Bankruptcy Prediction App")
st.write(
    "Provide the company's financial indicators below or upload a dataset in the sidebar. "
    "Predictions are made with a trained Logistic Regression model. "
    "If your uploaded data has true labels, you'll also see evaluation charts (ROC, PR) and metrics."
)

with st.expander("Project Details", expanded=False):
    st.markdown(
        """
        **Inputs (features expected by the model):**
        - `industrial_risk` — overall industry-related risk  
        - `management_risk` — risk linked to management quality  
        - `financial_flexibility` — ability to adapt financially  
        - `credibility` — reliability/perceived creditworthiness  
        - `competitiveness` — market competitiveness  
        - `operating_risk` — operational/process risks  

        **Targets / Labels (if present in uploaded data):**
        - `0` → Bankruptcy
        - `1` → Non-Bankruptcy
        """
    )

# =========================
# Manual Input (Single Prediction)
# =========================
st.header("Single Prediction (Manual Input)")
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

X_single = prepare_features_from_df(single_df, model)

if st.button("🔍 Predict (Manual)"):
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
            st.error("Result: The company is predicted to be at **RISK OF BANKRUPTCY**.")
        else:
            st.success("Result: The company is predicted to be **NON-BANKRUPT (Financially Healthy)**.")

        st.subheader("Predicted Probabilities")
        cpa, cpb = st.columns(2)
        cpa.metric("Bankruptcy", f"{p_bank*100:.1f}%")
        cpb.metric("Non-Bankruptcy", f"{p_non*100:.1f}%")

        # Horizontal probability bar (no pie chart)
        plot_horizontal_prob_bar(p_bank, p_non)

    except Exception as e:
        st.error(f"Prediction error: {e}")

# =========================
# Batch Prediction (Upload)
# =========================
if uploaded is not None:
    st.header("Batch Predictions & Evaluation")

    # Read uploaded file
    try:
        if uploaded.name.endswith((".xls", ".xlsx")):
            df_in = pd.read_excel(uploaded)
        else:
            df_in = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
        df_in = None

    if df_in is not None:
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df_in.head())

        try:
            X_batch = prepare_features_from_df(df_in, model)
        except Exception as e:
            st.error(f"Error preparing features: {e}")
            X_batch = None

        if X_batch is not None:
            try:
                preds = model.predict(X_batch)
                out = df_in.copy()
                out["prediction"] = preds

                # Attach probabilities if available
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_batch)
                    idx_bank, idx_non = get_class_indices(model)
                    out["prob_bankruptcy"] = proba[:, idx_bank]
                    if proba.shape[1] > 1:
                        out["prob_nonbankruptcy"] = proba[:, idx_non]

                st.success("✅ Batch predictions complete.")
                st.subheader("Predictions (head)")
                st.dataframe(out.head())

                # Download predictions
                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

                # If label present, evaluate with ROC/PR/CM
                label_col = next((c for c in LABEL_CANDIDATES if c in df_in.columns), None)
                if label_col is not None:
                    st.subheader("Evaluation (using uploaded true labels)")
                    y_true = df_in[label_col]
                    y_pred = out["prediction"]

                    acc = accuracy_score(y_true, y_pred)
                    prec = precision_score(y_true, y_pred, zero_division=0)
                    rec = recall_score(y_true, y_pred, zero_division=0)
                    f1 = f1_score(y_true, y_pred, zero_division=0)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Accuracy", f"{acc:.3f}")
                    m2.metric("Precision", f"{prec:.3f}")
                    m3.metric("Recall", f"{rec:.3f}")
                    m4.metric("F1-Score", f"{f1:.3f}")

                    # Confusion Matrix
                    plot_confusion_matrix(y_true, y_pred)

                    # ROC & PR (need probabilities for the positive class = bankruptcy)
                    if "prob_bankruptcy" in out.columns:
                        # create binary labels: 1 for bankruptcy, else 0
                        # if your dataset encodes bankruptcy as 0, map to 1 for ROC/PR where '1' means positive class
                        # here: y_true_bank = (y_true == 0).astype(int)
                        y_true_bank = (y_true == 0).astype(int)
                        scores = out["prob_bankruptcy"].values

                        # ROC curve
                        plot_roc_curve(y_true_bank, scores)

                        # Precision–Recall curve
                        plot_pr_curve(y_true_bank, scores)
                    else:
                        st.info("Model probabilities not available; ROC/PR curves require predict_proba().")

                else:
                    st.info("No label column detected in uploaded data. Add a label column (e.g., 'class') to see evaluation metrics and ROC/PR curves.")

            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

# =========================
# Footer
# =========================
st.markdown("---")
st.caption("Built with Streamlit • Logistic Regression • No pie charts — using probabilities, ROC, PR, and confusion matrix for insights.")
