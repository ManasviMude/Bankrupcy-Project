# app.py  —  Detailed UI + Manual & Batch + ROC/PR/CM (no pie chart)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from typing import Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
)

# =========================
# Config
# =========================
st.set_page_config(
    page_title="Bankruptcy Prediction App",
    page_icon="🏦",
    layout="wide",
)

MODEL_PATH = "final_logreg_model.pkl"
LABEL_CANDIDATES = ["class", "Class", "target", "Target", "y", "label", "Label"]

# =========================
# Load model
# =========================
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(f"❌ Model file '{MODEL_PATH}' not found. Put your trained model in the app folder.")
    st.stop()

# =========================
# Helpers
# =========================
def get_class_indices(model) -> Tuple[int, int]:
    """Return (idx_bankruptcy, idx_nonbankruptcy) inside model.classes_."""
    classes = list(getattr(model, "classes_", []))
    idx_bank = None
    idx_non = None
    # Prefer numeric labels 0/1 if they exist
    try:
        idx_bank = classes.index(0)
    except ValueError:
        # fallback: look for string class like "bankruptcy"
        for i, c in enumerate(classes):
            if str(c).lower().startswith("bank"):
                idx_bank = i
                break
    if idx_bank is None:
        idx_bank = 0 if classes else 0
    if len(classes) > 1:
        idx_non = 1 if idx_bank == 0 else 0
    else:
        idx_non = idx_bank
    return idx_bank, idx_non

def prepare_features_from_df(df: pd.DataFrame, model) -> pd.DataFrame:
    """Drop label column if present, reorder/select to model.feature_names_in_, fill missing with 0."""
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
            raise ValueError("No numeric feature columns found and model.feature_names_in_ is unavailable.")
    return X.fillna(0)

def find_label_column(df: pd.DataFrame) -> Optional[str]:
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    return None

# =========================
# UI: Tabs
# =========================
st.title("🏦 Bankruptcy Prediction App")
st.write(
    "Use a trained **Logistic Regression** model to predict whether a company is likely to be "
    "**Bankrupt** (positive class) or **Non-Bankrupt**. Enter values manually or upload a file for batch predictions. "
    "Evaluation metrics (Accuracy, Precision, Recall, F1), **ROC Curve**, and **Precision–Recall Curve** are provided when labels are present."
)

tab_overview, tab_manual, tab_batch, tab_model = st.tabs(
    ["ℹ️ Overview", "🧮 Manual Prediction", "📂 Batch Upload & Evaluation", "🧠 Model Info"]
)

# =========================
# Overview
# =========================
with tab_overview:
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.subheader("Project Summary")
        st.markdown(
            """
- **Goal:** Predict corporate bankruptcy risk from six key indicators.
- **Model:** Logistic Regression trained on labeled data (binary: 0 = Bankruptcy, 1 = Non-Bankruptcy).
- **Inputs (each as 0, 0.5, 1):**
  - `industrial_risk`, `management_risk`, `financial_flexibility`,
  - `credibility`, `competitiveness`, `operating_risk`
- **Outputs:**
  - Predicted class (Bankrupt / Non-Bankrupt)
  - Class probabilities
  - Evaluation charts for labeled batch files:
    - ROC Curve (AUC)
    - Precision–Recall Curve
    - Confusion Matrix
            """
        )
        st.subheader("How to Use")
        st.markdown(
            """
1. **Manual Prediction:** Go to the *Manual Prediction* tab, select feature values, and click **Predict**.  
2. **Batch Upload:** Go to *Batch Upload & Evaluation*, upload a CSV/XLSX with the six feature columns.  
   - Optionally include a label column named one of: `class`, `target`, `y`, or `label`.  
3. **Results & Evaluation:** View predictions, download a CSV, and (if labels provided) see **metrics**, **ROC**, **PR**, and **Confusion Matrix**.
            """
        )
    with c2:
        st.subheader("Tip")
        st.info(
            "If you get a feature-name error, ensure your file has exactly the same feature names "
            "the model was trained on. This app will **automatically drop** any label column and **reorder** columns to match the model."
        )

# =========================
# Manual Prediction
# =========================
with tab_manual:
    st.subheader("Enter Financial Indicators")
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

    single = pd.DataFrame([{
        "industrial_risk": industrial_risk,
        "management_risk": management_risk,
        "financial_flexibility": financial_flexibility,
        "credibility": credibility,
        "competitiveness": competitiveness,
        "operating_risk": operating_risk
    }])

    if st.button("🔍 Predict (Manual)", type="primary"):
        try:
            X_single = prepare_features_from_df(single, model)
            pred = model.predict(X_single)[0]

            # Probabilities
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_single)[0]
                idx_bank, idx_non = get_class_indices(model)
                p_bank = float(probs[idx_bank])
                p_non = float(probs[idx_non]) if idx_non is not None and idx_non < len(probs) else (1 - p_bank)
            else:
                p_bank = 1.0 if pred == 0 else 0.0
                p_non = 1.0 - p_bank

            # Result
            if pred == 0:
                st.error("⚠️ Prediction: **Bankruptcy Risk (Class 0)**")
            else:
                st.success("✅ Prediction: **Non-Bankrupt (Class 1)**")

            # Metrics display
            st.subheader("Predicted Probabilities")
            cA, cB = st.columns(2)
            cA.metric("Bankruptcy (0)", f"{p_bank*100:.1f}%")
            cB.metric("Non-Bankruptcy (1)", f"{p_non*100:.1f}%")

            # Show a simple probability bar (no pie chart)
            bar_fig = go.Figure(go.Bar(
                x=["Bankruptcy (0)", "Non-Bankruptcy (1)"],
                y=[p_bank*100, p_non*100],
                text=[f"{p_bank*100:.1f}%", f"{p_non*100:.1f}%"],
                textposition="auto"
            ))
            bar_fig.update_yaxes(title="Probability (%)", range=[0, 100])
            bar_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(bar_fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# =========================
# Batch Upload & Evaluation
# =========================
with tab_batch:
    st.subheader("Upload File for Batch Prediction")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

    if uploaded is not None:
        try:
            if uploaded.name.endswith((".xls", ".xlsx")):
                df = pd.read_excel(uploaded)
            else:
                df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            df = None

        if df is not None:
            st.write("**Preview:**")
            st.dataframe(df.head())

            # Save original for final output
            out_df = df.copy()

            # Prepare features (drops label if present, reorders)
            try:
                X = prepare_features_from_df(df, model)
            except Exception as e:
                st.error(f"Feature preparation error: {e}")
                X = None

            if X is not None:
                try:
                    preds = model.predict(X)
                    out_df["prediction"] = preds

                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X)
                        idx_bank, idx_non = get_class_indices(model)
                        out_df["prob_bankruptcy"] = proba[:, idx_bank]
                        if proba.shape[1] > 1:
                            out_df["prob_nonbankruptcy"] = proba[:, idx_non] if idx_non is not None else (1 - out_df["prob_bankruptcy"])

                    st.success("✅ Predictions complete.")
                    st.dataframe(out_df.head())

                    # Download predictions
                    st.download_button(
                        label="⬇️ Download Predictions (CSV)",
                        data=out_df.to_csv(index=False).encode("utf-8"),
                        file_name="predictions.csv",
                        mime="text/csv"
                    )

                    # Evaluation if label present
                    label_col = find_label_column(df)
                    if label_col:
                        st.subheader("Evaluation (labels detected)")
                        y_true = df[label_col]
                        y_pred = out_df["prediction"]

                        acc = accuracy_score(y_true, y_pred)
                        prec = precision_score(y_true, y_pred, zero_division=0)
                        rec = recall_score(y_true, y_pred, zero_division=0)
                        f1 = f1_score(y_true, y_pred, zero_division=0)

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Accuracy", f"{acc:.3f}")
                        m2.metric("Precision", f"{prec:.3f}")
                        m3.metric("Recall", f"{rec:.3f}")
                        m4.metric("F1-score", f"{f1:.3f}")

                        # Confusion Matrix
                        st.markdown("#### Confusion Matrix")
                        cm = confusion_matrix(y_true, y_pred)
                        cm_df = pd.DataFrame(
                            cm,
                            index=[f"True 0", f"True 1"][:cm.shape[0]],
                            columns=[f"Pred 0", f"Pred 1"][:cm.shape[1]],
                        )
                        st.dataframe(cm_df)

                        # ROC & PR curves only if we have probs for bankruptcy
                        if "prob_bankruptcy" in out_df.columns:
                            st.markdown("#### ROC Curve & AUC (Positive class = Bankruptcy)")
                            # define positive as bankruptcy (class label 0)
                            if set(np.unique(y_true)).issubset({0, 1}):
                                y_true_bank = (y_true == 0).astype(int)
                                try:
                                    auc = roc_auc_score(y_true_bank, out_df["prob_bankruptcy"])
                                    fpr, tpr, _ = roc_curve(y_true_bank, out_df["prob_bankruptcy"])
                                    roc_fig = go.Figure()
                                    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
                                    roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                                                 name="Chance", line=dict(dash="dash")))
                                    roc_fig.update_layout(
                                        title=f"ROC Curve (AUC = {auc:.3f})",
                                        xaxis_title="False Positive Rate",
                                        yaxis_title="True Positive Rate",
                                        margin=dict(l=10, r=10, t=40, b=10),
                                    )
                                    st.plotly_chart(roc_fig, use_container_width=True)
                                except Exception as e:
                                    st.warning(f"Could not compute ROC/AUC: {e}")

                                st.markdown("#### Precision–Recall Curve (Positive class = Bankruptcy)")
                                try:
                                    precision, recall, _ = precision_recall_curve(y_true_bank, out_df["prob_bankruptcy"])
                                    pr_fig = go.Figure()
                                    pr_fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR"))
                                    pr_fig.update_layout(
                                        title="Precision–Recall Curve",
                                        xaxis_title="Recall",
                                        yaxis_title="Precision",
                                        margin=dict(l=10, r=10, t=40, b=10),
                                    )
                                    st.plotly_chart(pr_fig, use_container_width=True)
                                except Exception as e:
                                    st.warning(f"Could not compute Precision–Recall curve: {e}")
                            else:
                                st.info("Labels are not strictly 0/1; ROC/PR skipped.")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# =========================
# Model Info
# =========================
with tab_model:
    st.subheader("Model Details")
    st.write(f"**Type:** {model.__class__.__name__}")
    if hasattr(model, "classes_"):
        st.write(f"**Classes:** {list(model.classes_)}")
    if hasattr(model, "feature_names_in_"):
        st.write("**Feature names used at training:**")
        st.code(", ".join(model.feature_names_in_))

    # Coefficients for Logistic Regression (if accessible)
    if hasattr(model, "coef_"):
        try:
            coeffs = model.coef_[0]
            if hasattr(model, "feature_names_in_"):
                coef_df = pd.DataFrame({
                    "feature": model.feature_names_in_,
                    "coefficient": coeffs
                }).sort_values("coefficient", ascending=False)
                st.write("**Feature Coefficients (higher magnitude = stronger impact):**")
                st.dataframe(coef_df, use_container_width=True)

                coef_bar = px.bar(coef_df, x="coefficient", y="feature", orientation="h",
                                  title="Logistic Regression Coefficients")
                coef_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(coef_bar, use_container_width=True)
            else:
                st.write("Coefficients found, but `feature_names_in_` not available.")
        except Exception as e:
            st.info(f"Could not display coefficients: {e}")

st.markdown("---")
st.caption("Built with Streamlit • Logistic Regression • Manual & Batch Predictions • ROC/PR/Confusion Matrix")
