import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Weather2Go", layout="centered")
st.title("Weather2Go Risk Predictor (Demo)")

@st.cache_resource
def load_model():
    return joblib.load("rf_weather_risk_model_2.joblib")

model = load_model()

# --- Helpers ---
def prob_to_bucket(p: float) -> str:
    if p < 0.2:
        return "Low"
    elif p < 0.4:
        return "Moderate-Low"
    elif p < 0.6:
        return "Moderate"
    elif p < 0.8:
        return "Heavy"
    else:
        return "Severe"

# --- Inputs ---
st.subheader("Manual test inputs")

weather_category = st.selectbox(
    "Weather_Category",
    ["clear","cloudy","fog","rain_light","rain_heavy","freezing_rain",
     "snow_light","snow_heavy","thunder","hail","other"]
)

temp_f = st.number_input("Temperature_F", value=32.0)
humidity = st.number_input("Humidity_Pct", value=70.0, min_value=0.0, max_value=100.0)
wind = st.number_input("Wind_Speed_mph", value=10.0, min_value=0.0, max_value=50.0)
precip = st.number_input("Precipitation_in", value=0.0, min_value=0.0, max_value=5.0)

day = st.selectbox(
    "Day_of_Week",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)

month = st.number_input("Month (1-12)", value=2, min_value=1, max_value=12)
hour = st.number_input("Hour (0-23)", value=12, min_value=0, max_value=23)

X = pd.DataFrame([{
    "Weather_Category": weather_category,
    "Temperature_F": temp_f,
    "Humidity_Pct": humidity,
    "Wind_Speed_mph": wind,
    "Precipitation_in": precip,
    "Day_of_Week": day,
    "Month": int(month),
    "Hour": int(hour),
}])

# --- Predict ---
if st.button("Predict"):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = model.classes_

        # Choose which probability to bucket:
        # If your classes are ordered low->high (common), use the last class as "highest risk"
        high_idx = len(classes) - 1
        p_high = float(proba[high_idx])

        risk_bucket = prob_to_bucket(p_high)

        st.subheader(f"Risk Level: {risk_bucket}")
        st.caption(f"High-risk probability: {p_high*100:.0f}%")

    else:
        st.warning("This model does not support predict_proba().")
