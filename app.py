# app.py — Fancy visual upgrade for Bankruptcy Prediction App
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from time import sleep

# ---------------------------
# Page config (must be first)
# ---------------------------
st.set_page_config(page_title="Bankruptcy Predictor ✨", page_icon="🏦", layout="wide")

# ---------------------------
# Minimal JS-free CSS: gradient header, glass cards, animated button
# ---------------------------
st.markdown(
    """
    <style>
    /* Page base */
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #f7fbff 0%, #ffffff 40%) !important;
        color: #07213a !important;
        font-family: Inter, 'Segoe UI', Roboto, sans-serif;
    }

    /* Hero header */
    .hero {
        background: linear-gradient(90deg, rgba(3,102,214,0.12), rgba(2,184,170,0.06));
        border-radius: 14px;
        padding: 26px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 20px;
        box-shadow: 0 8px 30px rgba(3,102,214,0.06);
        border: 1px solid rgba(3,102,214,0.06);
    }
    .hero .logo {
        width:72px;
        height:72px;
        border-radius:14px;
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:34px;
        background: linear-gradient(135deg,#0077ff,#00c2a8);
        color: #fff;
        box-shadow: 0 6px 18px rgba(3,102,214,0.12);
    }
    .hero h1 {margin:0; color:#032a4a; font-size:28px; font-weight:800;}
    .hero p {margin:0;color:#345c78;font-size:14px;}

    /* Glass cards */
    .glass {
        background: rgba(255,255,255,0.85);
        border-radius:12px;
        padding:16px;
        border: 1px solid rgba(11, 44, 78, 0.06);
        box-shadow: 0 10px 30px rgba(11,44,78,0.04);
    }

    /* Cool button */
    .glow-button .stButton>button {
        background: linear-gradient(90deg,#0077ff,#00c2a8) !important;
        color: #fff !important;
        padding: 10px 16px !important;
        border-radius: 12px !important;
        font-weight:700 !important;
        box-shadow: 0 8px 24px rgba(3,102,214,0.18) !important;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .glow-button .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 18px 40px rgba(3,102,214,0.22) !important;
    }

    /* Metric bigger */
    .big-metric { font-size:28px; font-weight:800; color:#032a4a; margin-bottom:4px; }
    .micro { color:#547892; font-size:13px; }

    /* Suggestion pill */
    .suggest {
        padding:10px 12px; border-radius:10px; font-weight:600;
        background: linear-gradient(90deg, rgba(0,194,168,0.06), rgba(3,102,214,0.02));
        color:#063a3a;
        border-left:6px solid #00c2a8;
    }

    /* Small muted */
    .muted { color:#6b7d90; font-size:13px; }

    /* Footer small */
    .footer { text-align:center; color:#6b7d90; margin-top:18px; font-size:13px; }

    /* Responsive tweaks */
    @media (max-width: 800px) {
        .hero { flex-direction: column; text-align:center; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Constants and model
# ---------------------------
MODEL_PATH = "final_logreg_model.pkl"
LABEL_CANDIDATES = ["class", "Class", "target", "Target", "y", "label", "Label"]

@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file '{MODEL_PATH}' not found. Put the pickle in app folder.")
    st.stop()

# ---------------------------
# Helper functions
# ---------------------------
def prepare_features_from_df(df: pd.DataFrame, model) -> pd.DataFrame:
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
        X = X.select_dtypes(include=[np.number])
    return X.fillna(0)

def get_class_indices(model):
    classes = list(getattr(model, "classes_", []))
    idx_bank = None
    try:
        idx_bank = classes.index(0)
    except Exception:
        for i, c in enumerate(classes):
            if "bank" in str(c).lower():
                idx_bank = i
                break
    if idx_bank is None:
        idx_bank = 0
    idx_non = 1 if idx_bank == 0 and len(classes) > 1 else 0
    return idx_bank, idx_non

def fancy_donut(prob_bank, prob_non):
    labels = ["Bankruptcy", "Non-Bankruptcy"]
    values = [prob_bank, prob_non]
    colors = ["#ff6b6b", "#20c997"]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.62,
                                 marker=dict(colors=colors), sort=False,
                                 textinfo='none', hoverinfo='label+percent')])
    # center annotation shows both percentages stacked
    fig.update_layout(
        showlegend=True,
        annotations=[dict(text=f"<b>{prob_bank*100:.1f}%</b><br><span style='font-size:12px;color:#6b7d90'>Bankrupt</span><br><b>{prob_non*100:.1f}%</b><br><span style='font-size:12px;color:#6b7d90'>Healthy</span>",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=14))],
        margin=dict(t=10,b=10,l=10,r=10)
    )
    return fig

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.markdown("<div style='text-align:center'><div style='width:64px;height:64px;border-radius:12px;background:linear-gradient(135deg,#0077ff,#00c2a8);display:inline-flex;align-items:center;justify-content:center;color:#fff;font-size:28px'>💰</div></div>", unsafe_allow_html=True)
    st.title("Bankruptcy Predictor")
    st.caption("Stylish UI • Logistic Regression")
    st.markdown("---")
    uploaded = st.file_uploader("Upload CSV / Excel for batch prediction (optional)", type=["csv", "xlsx"])
    st.markdown("---")
    st.write("Model info")
    st.write(f"- Type: `{model.__class__.__name__}`")
    st.write(f"- Probabilities: {'Yes' if hasattr(model,'predict_proba') else 'No'}")
    if hasattr(model, "classes_"):
        st.write(f"- Classes: `{list(model.classes_)}`")
    st.markdown("---")
    st.write("Developed by: Your Name")

# ---------------------------
# Hero header
# ---------------------------
st.markdown(
    f"""
    <div class="hero">
      <div class="logo">🏦</div>
      <div>
        <h1>Bankruptcy Prediction</h1>
        <p>Interactive, modern UI for quick bankruptcy risk checks. Single prediction donut + batch summary.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Single Prediction (fancy)
# ---------------------------
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown("### Single Company Prediction  — Enter indicators (0 / 0.5 / 1)", unsafe_allow_html=True)
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

# fancy button wrapper
st.markdown('<div class="glow-button">', unsafe_allow_html=True)
predict_single = st.button("✨ Predict (Single)")
st.markdown('</div>', unsafe_allow_html=True)

if predict_single:
    with st.spinner("Analyzing the company..."):
        sleep(0.7)  # short UX pause to let animation show
        try:
            X_single = prepare_features_from_df(single_df, model)
            pred = model.predict(X_single)[0]
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_single)[0]
                idx_bank, idx_non = get_class_indices(model)
                prob_bank = float(probs[idx_bank]) if idx_bank is not None else 0.0
                prob_non = float(probs[idx_non]) if idx_non is not None else (1.0 - prob_bank)
            else:
                prob_bank = 1.0 if pred == 0 else 0.0
                prob_non = 1.0 - prob_bank

            # Layout: left = result + metrics, right = donut
            left, right = st.columns([1.1, 1])
            with left:
                if pred == 0:
                    st.markdown("<div class='result-card bad'>", unsafe_allow_html=True)
                    st.error("⚠️ Prediction: RISK OF BANKRUPTCY")
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("<div class='suggest'>💡 Suggestion: Improve liquidity and management efficiency.</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                    st.success("✅ Prediction: FINANCIALLY HEALTHY")
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("<div class='suggest'>💡 Suggestion: Maintain competitiveness and cash flexibility.</div>", unsafe_allow_html=True)

                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                st.write("<div class='micro'>Probabilities</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='big-metric'>{prob_bank*100:.1f}%</div>", unsafe_allow_html=True)
                st.markdown("<div class='muted'>Bankruptcy probability</div>", unsafe_allow_html=True)
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='big-metric'>{prob_non*100:.1f}%</div>", unsafe_allow_html=True)
                st.markdown("<div class='muted'>Non-Bankruptcy probability</div>", unsafe_allow_html=True)

            with right:
                fig = fancy_donut(prob_bank, prob_non)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown('</div>', unsafe_allow_html=True)  # close glass

# ---------------------------
# Batch upload & summary (keeps simple pie)
# ---------------------------
if uploaded is not None:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### Batch Dataset Prediction (Uploaded)", unsafe_allow_html=True)
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        df = None

    if df is not None:
        st.subheader("Preview")
        st.dataframe(df.head())

        try:
            X = prepare_features_from_df(df, model)
            preds = model.predict(X)
            df["Prediction"] = np.where(preds == 0, "Bankruptcy", "Non-Bankruptcy")
            st.success("Batch predictions complete.")
            st.dataframe(df.head())

            bankrupt_count = (df["Prediction"] == "Bankruptcy").sum()
            non_bankrupt_count = (df["Prediction"] == "Non-Bankruptcy").sum()

            colA, colB = st.columns(2)
            colA.metric("Bankruptcy cases", int(bankrupt_count))
            colB.metric("Non-Bankruptcy cases", int(non_bankrupt_count))

            # Pie chart
            fig2 = go.Figure(data=[go.Pie(
                labels=["Bankruptcy", "Non-Bankruptcy"],
                values=[bankrupt_count, non_bankrupt_count],
                hole=0.45,
                marker=dict(colors=["#ff6b6b", "#20c997"]),
                textinfo="label+percent"
            )])
            fig2.update_layout(title_text="Bankruptcy distribution in uploaded dataset", title_x=0.5, margin=dict(t=30, b=10))
            st.plotly_chart(fig2, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Predictions CSV", csv, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
