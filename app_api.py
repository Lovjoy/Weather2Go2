import streamlit as st
import pandas as pd
import joblib
import requests
from datetime import datetime

col_logo, col_text = st.columns([2, 3])
with col_logo:
    st.image("logo.png", use_container_width=True)
with col_text:
    st.markdown("### Know the road before you go!")
    st.caption("Michigan-only demo")



# Hide timezone dropdown by hard-coding it
timezone = "auto"

@st.cache_resource
def load_model():
    return joblib.load("rf_weather_risk_model_2.joblib")

model = load_model()

# ---------- Your mapping ----------
def wmo_to_category(code: int) -> str:
    if code in [0]:
        return "clear"
    if code in [1, 2, 3]:
        return "cloudy"
    if code in [45, 48]:
        return "fog"
    if code in [51, 53, 55]:
        return "rain_light"
    if code in [56, 57]:
        return "freezing_rain"
    if code in [61, 63]:
        return "rain_light"
    if code in [65]:
        return "rain_heavy"
    if code in [66, 67]:
        return "freezing_rain"
    if code in [71, 73, 77]:
        return "snow_light"
    if code in [75]:
        return "snow_heavy"
    if code in [80, 81]:
        return "rain_light"
    if code in [82]:
        return "rain_heavy"
    if code in [85]:
        return "snow_light"
    if code in [86]:
        return "snow_heavy"
    if code in [95]:
        return "thunder"
    if code in [96, 99]:
        return "hail"
    return "other"

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

def c_to_f(c: float) -> float:
    return (c * 9.0 / 5.0) + 32.0

def mm_to_in(mm: float) -> float:
    return mm / 25.4

def kmh_to_mph(kmh: float) -> float:
    return kmh * 0.621371

# ---------- API helpers ----------
def geocode_michigan(query: str):
    """
    Michigan-only geocoding.
    - Ignores state in the input (Detroit, MI -> Detroit)
    - Filters results to admin1 == Michigan
    """
    base = (query or "").strip()
    if "," in base:
        base = base.split(",")[0].strip()

    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": base,
        "count": 20,
        "language": "en",
        "format": "json",
        "country_code": "US",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    payload = r.json()
    results = payload.get("results", []) or []

    mi = [x for x in results if (x.get("admin1") == "Michigan")]
    return mi

def fetch_hourly_weather(lat: float, lon: float, timezone: str = "auto"):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
        "timezone": timezone,
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

# ---------- UI ----------

# Keep results across reruns
if "geo_results" not in st.session_state:
    st.session_state.geo_results = []
if "last_output" not in st.session_state:
    st.session_state.last_output = None

st.subheader("Search Michigan city and predict")

with st.form("search_form"):
    city_query = st.text_input("Michigan city", value="Detroit")
    use_now = st.checkbox("Use next available hour", value=True)

    selected_dt = None
    if not use_now:
        selected_date = st.date_input("Forecast date")
        selected_hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=12)
        selected_dt = datetime.combine(selected_date, datetime.min.time()).replace(hour=int(selected_hour))

    search_clicked = st.form_submit_button("Search Michigan locations")

if search_clicked:
    st.session_state.geo_results = geocode_michigan(city_query)
    st.session_state.last_output = None

results = st.session_state.geo_results

if results:
    labels = [
        f"{r.get('name')}, {r.get('admin2','')} MI (lat={r.get('latitude'):.3f}, lon={r.get('longitude'):.3f})"
        for r in results
    ]
    choice = st.selectbox("Pick the best match", labels, index=0)
    chosen = results[labels.index(choice)]

    if st.button("Get weather + Predict"):
        lat, lon = float(chosen["latitude"]), float(chosen["longitude"])
        wx = fetch_hourly_weather(lat, lon, timezone=timezone)
        hourly = wx.get("hourly", {})
        times = hourly.get("time", [])

        if not times:
            st.error("Weather API returned no hourly data for this location.")
            st.stop()

        if use_now:
            idx = 0
            picked_time_str = times[idx]
        else:
            target = selected_dt.strftime("%Y-%m-%dT%H:00")
            if target not in times:
                st.error("That hour isn't available in the forecast range. Pick a nearer time.")
                st.stop()
            idx = times.index(target)
            picked_time_str = target

        temp_c = float(hourly["temperature_2m"][idx])
        rh = float(hourly["relative_humidity_2m"][idx])
        precip_mm = float(hourly["precipitation"][idx])
        wind_kmh = float(hourly["wind_speed_10m"][idx])
        wmo = int(hourly["weather_code"][idx])

        weather_category = wmo_to_category(wmo)
        temp_f = c_to_f(temp_c)
        precip_in = mm_to_in(precip_mm)
        wind_mph = kmh_to_mph(wind_kmh)

        dt = datetime.fromisoformat(picked_time_str)
        day_str = dt.strftime("%A")
        month = int(dt.strftime("%m"))
        hour = int(dt.strftime("%H"))

        X = pd.DataFrame([{
            "Weather_Category": weather_category,
            "Temperature_F": temp_f,
            "Humidity_Pct": rh,
            "Wind_Speed_mph": wind_mph,
            "Precipitation_in": precip_in,
            "Day_of_Week": day_str,
            "Month": month,
            "Hour": hour,
        }])

        proba = model.predict_proba(X)[0]
        classes = model.classes_
        high_idx = len(classes) - 1
        p_high = float(proba[high_idx])
        risk_bucket = prob_to_bucket(p_high)

        st.session_state.last_output = {
            "picked_time": picked_time_str,
            "X": X,
            "risk_bucket": risk_bucket,
            "p_high": p_high,
            "classes": classes,
            "proba": proba,
        }
else:
    st.info("Search a Michigan city to get started.")


# Persist output so changing dropdown doesn't wipe it
out = st.session_state.last_output
if out:
    st.write("Using weather at:", out["picked_time"])

    # Pretty summary first
    st.subheader("🌦️ Weather Summary")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Temperature", f"{out['X']['Temperature_F'].iloc[0]:.0f} °F")
    col2.metric("Humidity", f"{out['X']['Humidity_Pct'].iloc[0]:.0f} %")
    col3.metric("Wind", f"{out['X']['Wind_Speed_mph'].iloc[0]:.0f} mph")
    col4.metric("Precipitation", f"{out['X']['Precipitation_in'].iloc[0]:.2f} in")

    st.caption(
        f"{out['X']['Weather_Category'].iloc[0].replace('_',' ').title()} · "
    )

    # Risk headline
    st.subheader(f"Risk Level: {out['risk_bucket']}")
    st.caption(
        f"High-risk probability: {out['p_high']*100:.0f}% "
    )

    # --- Safety tips + insurance block ---
    risk_bucket = out["risk_bucket"]  # "Low", "Moderate-Low", "Moderate", "Heavy", "Severe"

    if risk_bucket == "Severe":
        st.error("""
**⚠️ Severe Risk Detected**
- Strongly consider delaying travel if possible
- Avoid non-essential trips
- Reduce speed substantially (20-30%+)
- Increase following distance significantly
- Expect poor visibility or traction
- Ensure tires, wipers, lights, and defroster are working
- Keep emergency kit (blanket, charger, flashlight)
- Check road advisories before leaving
""")
    elif risk_bucket == "Heavy":
        st.error("""
**⚠️ Heavy Risk Detected**
- Consider delaying travel if your trip is optional
- Drive with extreme caution
- Reduce speed by 15-25%
- Increase following distance
- Avoid sudden braking or lane changes
- Watch for black ice, standing water, or drifting snow
- Stay updated on weather changes
""")
    elif risk_bucket == "Moderate":
        st.warning("""
**⚠️ Moderate Risk Detected**
- Use increased caution
- Reduce speed by 10-15%
- Increase following distance
- Stay alert for changing conditions and hazards
- Consider alternate routes if roads look worse ahead
""")
    elif risk_bucket == "Moderate-Low":
        st.warning("""
**⚠️ Moderate-Low Risk Detected**
- Conditions are mostly manageable, but stay cautious
- Light speed reduction (5-10%) if needed
- Keep extra following distance
- Watch bridges and shaded areas for slick spots
""")
    else:  # Low
        st.success("""
**✅ Low Risk Detected**
- Conditions are relatively safe for driving
- Standard safe driving practices apply
- Stay aware of your surroundings
- Continue monitoring weather conditions
""")

    st.markdown("""
### 🛡️ Insurance Risk & Costs
- Accidents can raise premiums due to higher perceived risk
- Claims may trigger deductibles, affect renewal terms, or impact discounts
- A cleaner driving record helps keep insurance more affordable
""")
else:
    st.info("Run a prediction to see risk level, weather summary, tips, and insurance notes.")
