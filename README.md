# Weather2Go2 🌦️🚗

**Weather2Go2** is a data science portfolio project that estimates **driving risk for a planned trip** using real weather data and temporal features.

The project is designed as a small end-to-end ML system
(data → features → model → output) and is currently demonstrated through a **Streamlit web app**.

## What it does

### Given

* Michigan city (demo scope)
* Date and time (current or forecast)
* Weather conditions pulled live from an API

### It outputs

* A **5-level driving risk category**

  * Low
  * Moderate-Low
  * Moderate
  * Heavy
  * Severe
* A readable weather summary (temperature, wind, precipitation, humidity)
* Safety guidance aligned with the predicted risk level

The model focuses on **hazardous driving conditions**, not on predicting crashes or insurance outcomes.

## Scope

* **Geography:** Michigan only (demo)
* **Interface:** Streamlit web app
* **Purpose:** Data science portfolio project
* **Focus:** Weather-driven risk, interpretability, and end-to-end workflow

## Data

* Real crash data: Kaggle US Accidents (used for learning patterns)
* Real weather data: Open-Meteo API (historical + forecast)

All weather data is real.

## Stack

* Python
* pandas / numpy
* scikit-learn (Random Forest)
* Open-Meteo API
* Streamlit

## Current Status

✅ **Functional demo complete**

* The Streamlit app is working end-to-end
* Live weather is fetched and transformed into model features
* The model outputs a stable 5-level risk score
* UI displays weather summaries and safety guidance

At this stage, the project is **feature-complete for its intended demo scope**.

Further expansion (for example: driver age, vehicle type, trip distance, road type) would require **higher-quality or more granular public data**, which has proven difficult to source reliably. As a result, future changes would be incremental refinements rather than major extensions.

Overall, the project meets its portfolio goal and demonstrates:

* data sourcing and cleaning
* feature engineering
* supervised modeling
* model deployment
* user-facing interpretation

## Disclaimer

This project is for **educational and demonstration purposes only**.
It does not provide real-world driving, safety, or insurance advice.
