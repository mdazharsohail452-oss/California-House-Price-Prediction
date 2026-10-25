import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="California HomeValue | AI Estimator",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "house_model.joblib"
FEATURES_PATH = Path(__file__).parent / "house_features.joblib"

REQUIRED_COLUMNS = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]

# ------------------------------ Styling ------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
  --ink: #15233b;
  --muted: #6b7890;
  --blue: #315efb;
  --teal: #17b890;
  --line: #e6ebf3;
  --panel: #ffffff;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
  background:
    radial-gradient(ellipse at 12% 0%, rgba(49,94,251,.10), transparent 34%),
    radial-gradient(ellipse at 95% 8%, rgba(23,184,144,.09), transparent 28%),
    #f5f7fb;
  color: var(--ink);
}
.block-container { max-width: 1280px; padding-top: 1.5rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #111d35 0%, #172949 100%);
  border-right: 0;
}
[data-testid="stSidebar"] * { color: #edf3ff !important; }
[data-testid="stSidebar"] .stRadio label { padding: 7px 0; }
.hero {
  position: relative; overflow: hidden; border-radius: 26px;
  padding: 32px 34px; color: white; margin-bottom: 22px;
  background: linear-gradient(120deg, #142342 0%, #2447a8 58%, #267e9a 100%);
  box-shadow: 0 18px 45px rgba(24,49,100,.18);
}
.hero:after {
  content: "⌂"; position: absolute; right: 36px; top: -38px;
  font-family: 'Manrope', sans-serif; font-size: 230px; font-weight: 800;
  color: rgba(255,255,255,.07); line-height: 1;
}
.eyebrow {
  text-transform: uppercase; letter-spacing: 2px; font-size: 11px;
  font-weight: 800; color: #b9d8ff; margin-bottom: 10px;
}
.hero h1 {
  font-family: 'Manrope', sans-serif; font-size: clamp(28px, 3vw, 42px);
  font-weight: 800; letter-spacing: -1.5px; line-height: 1.1;
  color: #fff; margin: 0 0 12px 0;
}
.hero p { color: #e0eaff; font-size: 15px; max-width: 650px; margin: 0; }
.pill-row { display:flex; flex-wrap:wrap; gap:8px; margin-top:20px; }
.pill {
  display:inline-block; padding:7px 11px; border:1px solid rgba(255,255,255,.22);
  background:rgba(255,255,255,.10); border-radius:999px; font-size:12px;
  color:white; font-weight:600;
}
.section-title {
  font-family: 'Manrope', sans-serif; font-size: 22px; font-weight: 800;
  color:#172743; letter-spacing:-.5px; margin: 4px 0 3px 0;
}
.section-subtitle { color:#748198; font-size:13px; margin-bottom:18px; }
.metric-card {
  background:rgba(255,255,255,.92); border:1px solid #e7ecf4;
  border-radius:18px; padding:18px 20px; min-height:112px;
  box-shadow:0 6px 20px rgba(24,43,80,.045);
}
.metric-label { color:#7b879b; font-size:12px; font-weight:700; letter-spacing:.5px; text-transform:uppercase; }
.metric-value { color:#172743; font-family:'Manrope',sans-serif; font-size:25px; font-weight:800; margin-top:7px; }
.metric-note { color:#8792a5; font-size:12px; margin-top:4px; }
.surface {
  background:rgba(255,255,255,.94); border:1px solid #e7ecf4;
  border-radius:22px; padding:24px; box-shadow:0 8px 25px rgba(24,43,80,.045);
  margin-bottom:16px;
}
.info-strip {
  background:#edf4ff; border:1px solid #dce8ff; border-radius:14px;
  padding:13px 15px; color:#3c5684; font-size:13px; margin:12px 0 18px;
}
.stButton > button, .stDownloadButton > button {
  border-radius:12px; border:0; min-height:45px; font-weight:700;
  background:linear-gradient(110deg,#315efb,#4d7cff); color:#fff;
  box-shadow:0 7px 16px rgba(49,94,251,.19); transition:all .18s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  transform:translateY(-1px); box-shadow:0 10px 22px rgba(49,94,251,.25);
  color:white; border:0;
}
div[data-testid="stNumberInput"] input, div[data-testid="stTextInput"] input {
  border-radius:10px;
}
div[data-testid="stTabs"] button { font-weight:700; }
[data-testid="stMetric"] {
  background:#f7f9ff; padding:18px; border-radius:16px; border:1px solid #e6ebf5;
}
[data-testid="stDataFrame"] { border:1px solid #e7ecf4; border-radius:12px; overflow:hidden; }
.footer {
  text-align:center; color:#8792a5; font-size:12px; padding:24px 0 4px;
}
.small-note { color:#7c879a; font-size:12px; }
@media (max-width: 700px) {
  .hero { padding:25px 22px; border-radius:20px; }
  .hero:after { right:0; font-size:150px; }
  .surface { padding:17px; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "Model files are missing. Put house_model.joblib and "
            "house_features.joblib next to app.py."
        )
    loaded_model = joblib.load(MODEL_PATH)
    loaded_features = joblib.load(FEATURES_PATH)
    if set(loaded_features) != set(REQUIRED_COLUMNS):
        raise ValueError(
            "Saved feature names do not match the eight expected columns."
        )
    return loaded_model, loaded_features


try:
    model, features = load_model()
except Exception as exc:
    st.error(str(exc))
    st.stop()


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    data = df[features].copy()
    for column in features:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data.isna().any().any():
        bad_columns = data.columns[data.isna().any()].tolist()
        raise ValueError("Blank or non-numeric values in: " + ", ".join(bad_columns))

    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("All input values must be finite numbers.")

    positive_columns = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"
    ]
    for column in positive_columns:
        if (data[column] <= 0).any():
            raise ValueError(f"{column} must be greater than zero.")

    if (data["HouseAge"] > 52).any():
        raise ValueError("HouseAge must be 52 or less.")

    if not data["Latitude"].between(32, 42).all():
        raise ValueError("Latitude must be between 32 and 42.")

    if not data["Longitude"].between(-125, -114).all():
        raise ValueError("Longitude must be between -125 and -114.")

    return data


def predict_usd(data: pd.DataFrame) -> np.ndarray:
    values = np.asarray(model.predict(data), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("The model returned invalid predictions.")
    return values * 100_000


def money(value):
    return f"${value:,.0f}"


with st.sidebar:
    st.markdown("## 🏡 HomeValue AI")
    st.caption("CALIFORNIA • PROPERTY ESTIMATOR")
    st.markdown("---")
    page = st.radio(
        "WORKSPACE",
        ["Single prediction", "Batch CSV prediction", "Model overview"],
        label_visibility="visible"
    )
    st.markdown("---")
    st.markdown("**MODEL STATUS**")
    st.success("Model loaded and ready")
    st.markdown(
        "<div class='small-note'>Random Forest Regressor<br>8 input features<br>USD estimate</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.caption("Built with Streamlit + scikit-learn")

st.markdown("""
<div class="hero">
  <div class="eyebrow">SMART PROPERTY ANALYTICS</div>
  <h1>Find your home's<br>estimated value.</h1>
  <p>Get a data-driven  house price estimater .Enter property details or analyze a full CSV in seconds.</p>
  <div class="pill-row">
    <span class="pill">✦ Machine learning</span>
    <span class="pill">↗ Instant estimates</span>
    <span class="pill">▦ Batch analysis</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Quick stat cards
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="metric-card">
      <div class="metric-label">Prediction engine</div>
      <div class="metric-value">Random Forest</div>
      <div class="metric-note">100 decision trees</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="metric-card">
      <div class="metric-label">Input variables</div>
      <div class="metric-value">08 features</div>
      <div class="metric-note">Location + neighborhood data</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="metric-card">
      <div class="metric-label">Output currency</div>
      <div class="metric-value">USD ($)</div>
      <div class="metric-note">Converted from dataset target units</div>
    </div>""", unsafe_allow_html=True)

st.write("")

if page == "Single prediction":
    st.markdown('<div class="section-title">Property details</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Add the property and neighborhood information below to generate an estimate.</div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="info-strip">ⓘ Use values in the same format as the original California Housing dataset. Income is represented in tens of thousands of dollars.</div>', unsafe_allow_html=True)

    with st.form("house_prediction_form"):
        st.markdown("## 🏠 Home characteristics")
        col1, col2 = st.columns(2)
        with col1:
            med_inc = st.number_input("Median income · MedInc", min_value=0.01, value=5.0, step=0.1, help="Dataset income value, in tens of thousands of dollars.")
            house_age = st.number_input("Median house age · HouseAge", min_value=0.1, max_value=52.0, value=30.0, step=1.0)
            ave_rooms = st.number_input("Average rooms · AveRooms", min_value=0.01, value=5.5, step=0.1)
            ave_bedrms = st.number_input("Average bedrooms · AveBedrms", min_value=0.01, value=1.1, step=0.1)
        with col2:
            population = st.number_input("Neighborhood population · Population", min_value=0.01, value=1000.0, step=100.0)
            ave_occup = st.number_input("Average occupancy · AveOccup", min_value=0.01, value=3.0, step=0.1)
            latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=34.05, step=0.01, format="%.4f")
            longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-118.25, step=0.01, format="%.4f")

        submitted = st.form_submit_button("✦  Estimate house value", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([{
            "MedInc": med_inc,
            "HouseAge": house_age,
            "AveRooms": ave_rooms,
            "AveBedrms": ave_bedrms,
            "Population": population,
            "AveOccup": ave_occup,
            "Latitude": latitude,
            "Longitude": longitude,
        }])
        try:
            validated = validate_input(input_df)
            price = float(predict_usd(validated)[0])
            st.markdown('<div class="surface">', unsafe_allow_html=True)
            st.markdown("#### Your estimated value")
            st.metric("Predicted house price", money(price))
            m1, m2 = st.columns(2)
            m1.metric("Model output", f"{price / 100000:.3f}", "× $100,000")
            m2.metric("Currency", "USD", "US dollars")
            st.caption("This is a model estimate based on historical data, not a formal appraisal or a live market quote.")
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as exc:
            st.error(str(exc))

elif page == "Batch CSV prediction":
    st.markdown('<div class="section-title">Batch prediction studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Upload multiple properties, predict every row, and export the enriched dataset.</div>',
        unsafe_allow_html=True
    )

    template = pd.DataFrame([{
        "MedInc": 5.0,
        "HouseAge": 30.0,
        "AveRooms": 5.5,
        "AveBedrms": 1.1,
        "Population": 1000.0,
        "AveOccup": 3.0,
        "Latitude": 34.05,
        "Longitude": -118.25,
    }])
    st.download_button(
        "↓  Download sample CSV template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="house_input_sample.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload your properties CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            if uploaded_df.empty:
                st.warning("The uploaded CSV contains no data rows.")
            else:
                st.markdown("##File preview")
                st.caption(f"{len(uploaded_df):,} rows · {len(uploaded_df.columns)} columns")
                st.dataframe(uploaded_df.head(10), use_container_width=True)

                if st.button("Run batch predictions", use_container_width=True):
                    validated = validate_input(uploaded_df)
                    prices = predict_usd(validated)
                    result_df = uploaded_df.copy()
                    result_df["predicted_price_usd"] = np.round(prices, 2)
                    result_df["predicted_price"] = [money(price) for price in prices]

                    st.success(f"Successfully predicted {len(result_df):,} properties.")
                    left, right = st.columns(2)
                    left.metric("Properties analyzed", f"{len(result_df):,}")
                    right.metric("Average predicted value", money(float(np.mean(prices))))
                    st.dataframe(result_df, use_container_width=True)
                    st.download_button(
                        "↓  Download prediction results",
                        data=result_df.to_csv(index=False).encode("utf-8"),
                        file_name="house_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        except pd.errors.EmptyDataError:
            st.error("The CSV file is empty or unreadable.")
        except (UnicodeDecodeError, pd.errors.ParserError):
            st.error("Could not read this CSV. Check its format and encoding.")
        except Exception as exc:
            st.error(f"Could not process the file: {exc}")

else:
    st.markdown('<div class="section-title">Under the hood</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">A quick look at the model and the data it expects.</div>',
        unsafe_allow_html=True
    )
    a, b = st.columns([1, 1])
    with a:
        st.markdown('<div class="surface">', unsafe_allow_html=True)
        st.markdown("#### Model configuration")
        st.write(f"**Algorithm:** {type(model).__name__}")
        st.write("**Trees:** 100 (configured during training)")
        st.write("**Dataset:** scikit-learn California Housing")
        st.write("**Split:** 80% training / 20% testing")
        st.write("**Random state:** 42")
        st.markdown('</div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="surface">', unsafe_allow_html=True)
        st.markdown("#### Required input features")
        feature_labels = {
            "MedInc": "Median income",
            "HouseAge": "Median house age",
            "AveRooms": "Average rooms",
            "AveBedrms": "Average bedrooms",
            "Population": "Neighborhood population",
            "AveOccup": "Average occupancy",
            "Latitude": "Latitude",
            "Longitude": "Longitude",
        }
        for feature in features:
            st.write(f"• **{feature}** — {feature_labels.get(feature, feature)}")
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer">HOMEVALUE AI · California House Price Prediction · Built with Streamlit & scikit-learn</div>',
    unsafe_allow_html=True
)
