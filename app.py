import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# =========================
# Page Configuration
# =========================
st.set_page_config(page_title="Bankruptcy Prediction App", page_icon="🏦", layout="wide")

MODEL_PATH = "final_logreg_model.pkl"

# =========================
# Load model
# =========================
@st.cache_resource
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file '{MODEL_PATH}' not found. Upload it to the same folder and restart.")
    st.stop()

# =========================
# Helper Functions
# =========================
def prepare_features_from_df(df: pd.DataFrame, model) -> pd.DataFrame:
    """Ensure uploaded data columns match model input features"""
    X = df.copy()
    if "class" in X.columns:
        X = X.drop(columns=["class"])
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        for col in expected:
            if col not in X.columns:
                X[col] = 0.0
        X = X.reindex(columns=expected)
    else:
        X = X.select_dtypes(include=[np.number])
    return X.fillna(0)

def plot_pie_chart(bankrupt_count, non_bankrupt_count):
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Bankruptcy", "Non-Bankruptcy"],
                values=[bankrupt_count, non_bankrupt_count],
                hole=0.4,
                marker=dict(colors=["#ef553b", "#00cc96"]),
                textinfo="label+percent",
                textfont_size=14,
            )
        ]
    )
    fig.update_layout(
        title_text="📊 Bankruptcy Distribution in Dataset",
        title_x=0.5,
        height=400,
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# Sidebar Information
# =========================
with st.sidebar:
    st.title("ℹ️ About the App")
    st.write(
        """
        This app predicts company **bankruptcy risk** using a trained Logistic Regression model.  
        You can:
        - Enter values manually (0, 0.5, 1)
        - Upload a dataset (CSV/XLSX)  
        """
    )
    uploaded = st.file_uploader("📂 Upload dataset for batch prediction", type=["csv", "xlsx"])

# =========================
# App Title
# =========================
st.title("🏦 Bankruptcy Prediction App")
st.write(
    "Predict whether a company is likely to face **Bankruptcy** or remain **Financially Healthy** based on financial indicators."
)

# =========================
# Manual Input Section
# =========================
st.header("📊 Single Company Prediction")

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

input_data = pd.DataFrame([{
    "industrial_risk": industrial_risk,
    "management_risk": management_risk,
    "financial_flexibility": financial_flexibility,
    "credibility": credibility,
    "competitiveness": competitiveness,
    "operating_risk": operating_risk
}])

# =========================
# Manual Prediction
# =========================
if st.button("🔍 Predict Bankruptcy"):
    try:
        X = prepare_features_from_df(input_data, model)
        prediction = model.predict(X)[0]
        probs = model.predict_proba(X)[0]

        st.subheader("Prediction Result")
        if prediction == 0:
            st.error("⚠️ The company is predicted to be at **RISK OF BANKRUPTCY**.")
        else:
            st.success("✅ The company is predicted to be **FINANCIALLY HEALTHY**.")

        st.write(f"**Bankruptcy Probability:** {probs[0]*100:.2f}%")
        st.write(f"**Non-Bankruptcy Probability:** {probs[1]*100:.2f}%")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# =========================
# Batch Prediction (Uploaded File)
# =========================
if uploaded is not None:
    st.header("📁 Batch Dataset Prediction")

    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        st.write("### Uploaded Data Preview")
        st.dataframe(df.head())

        X = prepare_features_from_df(df, model)
        preds = model.predict(X)

        df["Prediction"] = np.where(preds == 0, "Bankruptcy", "Non-Bankruptcy")

        st.success("✅ Predictions generated successfully.")
        st.dataframe(df.head())

        # Count bankrupt vs non-bankrupt
        bankrupt_count = (df["Prediction"] == "Bankruptcy").sum()
        non_bankrupt_count = (df["Prediction"] == "Non-Bankruptcy").sum()

        st.subheader("📈 Dataset Bankruptcy Summary")
        st.write(f"**Bankruptcy Cases:** {bankrupt_count}")
        st.write(f"**Non-Bankruptcy Cases:** {non_bankrupt_count}")

        # Plot Pie Chart
        plot_pie_chart(bankrupt_count, non_bankrupt_count)

        # Option to download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions CSV", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Error processing file: {e}")

# =========================
# Footer
# =========================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit • Logistic Regression Model")
