# Weather2Go2 🌦️🚗

Weather2Go2 is an early‑stage data science project that estimates driving risk for a planned trip using weather, timing, distance, and basic driver context.

The project is designed as an end‑to‑end ML system (data → features → model → output) intended for a future hosted website.
What it does

### Given:
* Start city & state
* Start date and time
* End city & state
* Basic driver and vehicle information

### It outputs:
* A driving risk score (1–5)
  * 1 = Low risk
  * 5 = Very high risk (e.g. snow, ice, poor visibility)
* Weather conditions at trip start and estimated arrival time

The focus is on hazardous driving conditions, not crash prediction.
Scope
* Geography: Michigan only (demo)
* Status: Early work‑in‑progress
* Purpose: Portfolio project

### Data
* Real crash data (Kaggle US Accidents)
* Real historical & forecast weather (Open‑Meteo API)
* Constructed non‑accident trips paired with real weather

All weather data is real.
Non‑accident trips are simulated only to represent normal driving exposure.
Stack (current)
* Python
*pandas / numpy
* Open‑Meteo API

### Disclaimer

This project is for educational purposes only and does not provide real‑world driving safety advice.

### Status

🚧 Early WIP — data exploration, feature engineering, and labeling logic in progress.
