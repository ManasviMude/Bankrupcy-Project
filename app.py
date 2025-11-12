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
            df_in = pd.read_excel(uploaded)
        else:
            df_in = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
        df_in = None

    if df_in is not None:
        st.subheader("Uploaded data preview")
        st.dataframe(df_in.head())

        # Prepare X for prediction (drop label if present)
        try:
            X_batch = prepare_features_from_df(df_in, model)
        except Exception as e:
            st.error(f"Feature preparation failed: {e}")
            X_batch = None

        if X_batch is not None:
            try:
                preds = model.predict(X_batch)
                out = df_in.copy()
                out["prediction"] = preds

                # probabilities if available
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_batch)
                    idx_bank, idx_non = get_class_indices(model)
                    out["prob_bankruptcy"] = proba[:, idx_bank]
                    if proba.shape[1] > 1:
                        out["prob_nonbankruptcy"] = proba[:, idx_non]

                st.success("Batch predictions complete.")
                st.subheader("Predictions preview")
                st.dataframe(out.head())

                # allow download
                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions (CSV)", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

                # If label present — evaluate
                label_col = next((c for c in LABEL_CANDIDATES if c in df_in.columns), None)
                if label_col is not None:
                    st.subheader("Evaluation using uploaded true labels")

                    # Copy and normalize labels to numeric 0/1
                    y_true_raw = df_in[label_col].copy()

                    # If string labels, map common forms to numeric
                    if y_true_raw.dtype == object:
                        y_norm = y_true_raw.str.strip().str.lower().map({
                            "bankruptcy": 0,
                            "bankrupt": 0,
                            "yes": 0,
                            "y": 0,
                            "1": 1,  # in case stored as string '1'
                            "non-bankruptcy": 1,
                            "non bankruptcy": 1,
                            "nonbankruptcy": 1,
                            "nonbankrupt": 1,
                            "non bankrupt": 1,
                            "non": 1,
                            "no": 1,
                            "n": 1,
                            "non-bankrupt": 1,
                            "non-bank": 1
                        })
                        # If mapping produced NaN (unmapped), try to infer by unique values
                        if y_norm.isna().any():
                            uniques = list(y_true_raw.unique())
                            # simple inference: if two uniques and one contains 'bank' text
                            try:
                                if len(uniques) == 2:
                                    u0 = str(uniques[0]).lower()
                                    u1 = str(uniques[1]).lower()
                                    if "bank" in u0 and "non" in u1 or "non" in u1:
                                        y_norm = y_true_raw.str.strip().str.lower().map({uniques[0]:0, uniques[1]:1})
                                    elif "bank" in u1 and "non" in u0 or "non" in u0:
                                        y_norm = y_true_raw.str.strip().str.lower().map({uniques[1]:0, uniques[0]:1})
                            except Exception:
                                pass
                        # fallback: if still NaN, attempt to convert to numeric
                        if y_norm.isna().any():
                            try:
                                y_norm = pd.to_numeric(y_true_raw, errors='coerce')
                            except Exception:
                                pass
                    else:
                        # numeric dtype: assume 0/1 already
                        y_norm = pd.to_numeric(y_true_raw, errors='coerce')

                    # Drop rows where we couldn't interpret label
                    valid_mask = ~y_norm.isna()
                    if valid_mask.sum() < len(y_norm):
                        st.warning(f"{len(y_norm) - valid_mask.sum()} rows have labels that couldn't be interpreted; they will be ignored for evaluation.")
                    y_true = y_norm[valid_mask].astype(int)
                    y_pred_full = pd.Series(out["prediction"]).iloc[valid_mask.index][valid_mask].astype(int).values
                    # Ensure y_pred aligns indices with y_true
                    # y_pred_full derived above should match

                    # Now check types and shapes
                    if len(y_true) != len(y_pred_full):
                        # try aligning by index (safer)
                        y_pred_series = pd.Series(out["prediction"])
                        y_pred_aligned = y_pred_series.loc[y_true.index].astype(int)
                        y_pred_final = y_pred_aligned.values
                    else:
                        y_pred_final = y_pred_full

                    # Compute metrics
                    try:
                        acc = accuracy_score(y_true, y_pred_final)
                        prec = precision_score(y_true, y_pred_final, zero_division=0)
                        rec = recall_score(y_true, y_pred_final, zero_division=0)
                        f1s = f1_score(y_true, y_pred_final, zero_division=0)

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Accuracy", f"{acc:.3f}")
                        m2.metric("Precision", f"{prec:.3f}")
                        m3.metric("Recall", f"{rec:.3f}")
                        m4.metric("F1-Score", f"{f1s:.3f}")

                        # Confusion matrix
                        plot_confusion_matrix(y_true, y_pred_final)

                        # ROC & PR require probabilities for the positive class (we treat bankruptcy as positive here)
                        if "prob_bankruptcy" in out.columns:
                            # create y_true_bank where 1 indicates bankruptcy
                            # Our y_true currently uses 0 for bankruptcy per mapping above; invert to 1=bankruptcy
                            y_true_bank = (y_true == 0).astype(int)
                            scores = out.loc[y_true.index, "prob_bankruptcy"].values
                            plot_roc_curve(y_true_bank, scores)
                            plot_pr_curve(y_true_bank, scores)
                        else:
                            st.info("Model probabilities not available; ROC/PR curves require predict_proba().")
                    except Exception as e:
                        st.error(f"Evaluation failed: {e}")

                else:
                    st.info("No label column detected. Include a label column (e.g., 'class') for evaluation.")

            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Built with Streamlit • Logistic Regression • Evaluation supports text labels (mapped) and numeric labels.")
