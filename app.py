# app.py
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
# Load model (expects final_logreg_model.pkl present)
# ---------------------------
MODEL_PATH = "final_logreg_model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(f"Model file '{MODEL_PATH}' not found. Place your trained model in the app folder.")
    st.stop()

# ---------------------------
# Page config (Streamlit default theme)
# ---------------------------
st.set_page_config(page_title="Bankruptcy Prediction App", page_icon="🏦", layout="wide")

# ---------------------------
# Sidebar: upload, options, info
# ---------------------------
with st.sidebar:
    st.title("Bankruptcy Predictor")
    st.write("Logistic Regression — Demo")
    st.markdown("---")
    uploaded = st.file_uploader("Upload CSV / Excel for batch prediction (optional)", type=["csv", "xlsx", "xls"])
    show_metrics = st.checkbox("Show evaluation if uploaded file contains true labels", value=True)
    st.markdown("---")
    st.write("Developed by: Your Name")
    st.write("Contact: your@email.com")
    if st.button("Reset / Refresh"):
        st.experimental_rerun()

# ---------------------------
# Main header
# ---------------------------
st.title("🏦 Bankruptcy Prediction App")

# ---------------------------
# Helper utilities
# ---------------------------
LABEL_CANDIDATES = ["class", "Class", "target", "Target", "y", "label", "Label"]

def prepare_features_from_df(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Prepare a DataFrame for prediction:
    - Drop label column if present
    - If model has feature_names_in_, reindex (add missing cols with 0)
    - Else, select numeric columns
    Returns X (DataFrame) ready for model.predict
    """
    df2 = df.copy()
    # Drop label if present
    for c in LABEL_CANDIDATES:
        if c in df2.columns:
            df2 = df2.drop(columns=[c])
            break

    # If model has feature_names_in_, reorder/select those
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        # Add missing expected columns as zeros
        for col in expected:
            if col not in df2.columns:
                df2[col] = 0.0
        # Reindex to expected order (keeps only expected columns)
        X = df2.reindex(columns=expected)
    else:
        # fallback: use numeric columns only
        X = df2.select_dtypes(include=[np.number]).copy()
        if X.shape[1] == 0:
            raise ValueError("No numeric feature columns found and model.feature_names_in_ is not available.")
    # Fill NaNs
    X = X.fillna(0)
    return X

def get_class_indices(model):
    """
    Return indices for bankruptcy class (0) and non-bankruptcy (1) in model.classes_.
    If exact labels not found, assume index 0 is bankruptcy and 1 is non-bankruptcy when available.
    """
    classes = list(getattr(model, "classes_", []))
    idx_bank = None
    idx_non = None
    # try numeric labels first
    try:
        idx_bank = classes.index(0)
    except ValueError:
        # try string matches
        for i, c in enumerate(classes):
            s = str(c).lower()
            if s.startswith("bank"):
                idx_bank = i
                break
    if idx_bank is None and len(classes) >= 1:
        idx_bank = 0
    if len(classes) > 1:
        idx_non = 1 if idx_bank == 0 else 0
    else:
        idx_non = idx_bank
    return idx_bank, idx_non

# ---------------------------
# Manual input (single record)
# ---------------------------
st.header("Manual input")
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

single_df = pd.DataFrame([{
    "industrial_risk": industrial_risk,
    "management_risk": management_risk,
    "financial_flexibility": financial_flexibility,
    "credibility": credibility,
    "competitiveness": competitiveness,
    "operating_risk": operating_risk
}])

# Prepare features using model expectations if available
try:
    X_single = prepare_features_from_df(single_df, model)
except Exception as e:
    st.error(f"Error preparing input features: {e}")
    X_single = None

if st.button("Predict (Manual)"):
    if X_single is None:
        st.error("Input features not prepared correctly.")
    else:
        try:
            pred = model.predict(X_single)[0]
            # probabilities
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_single)[0]
                idx_bank, idx_non = get_class_indices(model)
                prob_bank = float(probs[idx_bank]) if idx_bank is not None and idx_bank < len(probs) else 0.0
                prob_non = float(probs[idx_non]) if idx_non is not None and idx_non < len(probs) else 1.0 - prob_bank
            else:
                prob_bank = 1.0 if pred == 0 else 0.0
                prob_non = 1.0 - prob_bank

            # show result
            if pred == 0:
                st.error("⚠️ Prediction: The company is predicted to be at RISK OF BANKRUPTCY.")
            else:
                st.success("✅ Prediction: The company is predicted to be NON-BANKRUPT (financially healthy).")

            # show probabilities as metrics
            st.subheader("Predicted probabilities")
            c1, c2 = st.columns(2)
            c1.metric("Bankruptcy", f"{prob_bank*100:.1f} %")
            c2.metric("Non-Bankruptcy", f"{prob_non*100:.1f} %")

            # pie / donut chart (plotly)
            fig = go.Figure(data=[go.Pie(
                labels=["Bankruptcy", "Non-Bankruptcy"],
                values=[prob_bank, prob_non],
                hole=0.45,
                marker=dict(colors=["#ef553b", "#00cc96"])
            )])
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ---------------------------
# Batch upload & prediction
# ---------------------------
if uploaded is not None:
    st.header("Batch prediction")
    # read file
    try:
        if uploaded.name.endswith((".xls", ".xlsx")):
            batch = pd.read_excel(uploaded)
        else:
            batch = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
        batch = None

    if batch is not None:
        st.subheader("Uploaded data preview")
        st.dataframe(batch.head())

        # prepare features: drop label if present and reindex to model features
        try:
            X_batch = prepare_features_from_df(batch, model)
        except Exception as e:
            st.error(f"Error preparing features from uploaded file: {e}")
            X_batch = None

        if X_batch is not None:
            try:
                preds = model.predict(X_batch)
                batch_out = batch.copy()
                batch_out["prediction"] = preds

                # probabilities if available
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_batch)
                    idx_bank, idx_non = get_class_indices(model)
                    # attach bankruptcy prob and non-bank prob if possible
                    batch_out["prob_bankruptcy"] = proba[:, idx_bank] if idx_bank is not None else np.nan
                    # if two-class, add other prob
                    if proba.shape[1] > 1:
                        other_idx = idx_non
                        batch_out["prob_nonbankruptcy"] = proba[:, other_idx] if other_idx is not None else (1 - batch_out["prob_bankruptcy"])
                st.success("✅ Predictions complete.")
                st.subheader("Predictions preview")
                st.dataframe(batch_out.head())

                # download button
                csv_bytes = batch_out.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions (CSV)", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

                # evaluation if label present and user requested
                label_col = next((c for c in LABEL_CANDIDATES if c in batch.columns), None)
                if show_metrics and label_col:
                    st.subheader("Evaluation (using uploaded true labels)")
                    y_true = batch[label_col]
                    y_pred = batch_out["prediction"]
                    acc = accuracy_score(y_true, y_pred)
                    prec = precision_score(y_true, y_pred, zero_division=0)
                    rec = recall_score(y_true, y_pred, zero_division=0)
                    f1 = f1_score(y_true, y_pred, zero_division=0)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Accuracy", f"{acc:.3f}")
                    m2.metric("Precision", f"{prec:.3f}")
                    m3.metric("Recall", f"{rec:.3f}")
                    m4.metric("F1-score", f"{f1:.3f}")

                    # confusion matrix
                    cm = confusion_matrix(y_true, y_pred)
                    st.write("Confusion matrix")
                    st.write(pd.DataFrame(cm, index=[f"True_{i}" for i in range(cm.shape[0])],
                                         columns=[f"Pred_{i}" for i in range(cm.shape[1])]))

                    # ROC-AUC if probabilities for bankruptcy exist
                    if "prob_bankruptcy" in batch_out.columns:
                        try:
                            # define positive class as bankruptcy (1) for roc_auc calculation
                            # if y_true is 0/1 and bankruptcy labeled 0, convert accordingly:
                            # create y_true_bank = 1 when bankruptcy, else 0
                            # detect bankruptcy label in uploaded y_true: assume value 0 means bankruptcy if model.classes_ contain 0
                            if set(np.unique(y_true)).issubset({0,1}):
                                # create y_true_bank where 1 indicates bankruptcy label (model's bankruptcy label assumed to be 0)
                                bankruptcy_label = 0
                                y_true_bank = (y_true == bankruptcy_label).astype(int)
                                auc = roc_auc_score(y_true_bank, batch_out["prob_bankruptcy"])
                                fpr, tpr, _ = roc_curve(y_true_bank, batch_out["prob_bankruptcy"])
                                fig_roc = go.Figure()
                                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
                                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
                                fig_roc.update_layout(title=f"ROC Curve (AUC={auc:.3f})", xaxis_title="FPR", yaxis_title="TPR")
                                st.plotly_chart(fig_roc, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not compute ROC-AUC: {e}")

            except Exception as e:
                st.error(f"Prediction failed: {e}")


