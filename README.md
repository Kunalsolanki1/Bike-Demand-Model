# Bike Sharing Case: Do We Need to Buy More Bikes?

A demand-analysis and Streamlit prototype built around the supplied Capital Bikeshare 2011–2012 dataset and case brief.

## Business conclusion

**Do not buy additional bikes immediately.** The analysis shows strong hourly concentration in commuting periods. The recommended first step is to forecast demand and test forecast-driven redistribution before purchasing more bikes.

The dataset contains citywide rental counts, not station-level inventory or failed-rental data. It therefore cannot establish where bikes are missing or calculate the exact fleet size required.

## Repository contents

| File | Purpose |
|---|---|
| `app.py` | Streamlit demand assistant |
| `bike_model.joblib` | Trained median and upper-demand models plus calibration shifts |
| `replay.csv` | October–December 2012 historical holdout features and actual counts |
| `day.csv`, `hour.csv` | Original daily and hourly datasets |
| `Complete code.ipynb` | EDA, hypothesis testing, model training and artifact creation |
| `requirements.txt` | Pinned deployment dependencies |
| `tests/smoke_test.py` | Model/replay validation and case-example regression check |
| `.github/workflows/smoke-test.yml` | Automated smoke test on GitHub |

## What the analysis reports

- 731 daily rows and 17,379 hourly rows.
- 165 missing hourly timestamps were identified, flagged and treated as zero rentals for analysis. This is an explicit assumption, especially for daytime gaps on storm days.
- The notebook corrects season labels using the month cross-tab: code 1 winter, 2 spring, 3 summer, 4 fall.
- Commute-hour share (07:00–08:00 and 17:00–18:00) is reported as 39.3% on working days versus 17.0% on non-working days, tested with a one-sided Mann–Whitney U test.
- Gradient boosting uses calendar, weather, trend and lagged demand (24h and 168h). `casual` and `registered` are excluded from predictors to avoid target leakage.
- Chronological test period: 15 October–31 December 2012.
- Notebook-reported metrics: MAE 45.5, RMSE 75.0 and R² 0.849, compared with a same-hour-last-week baseline (MAE 77.2, RMSE 132.1, R² 0.533).
- A separate quantile model provides a calibrated upper-demand planning estimate. Its reported coverage is empirical and may be optimistic because calibration choices were made after examining the test period.

These are the analysis results in the notebook and case solution. They are not a guarantee of performance in other seasons or future years.

## App theme

The Streamlit theme is defined in `.streamlit/config.toml`. It uses a light slate background with a teal primary accent and a softly tinted sidebar. Streamlit Community Cloud reads this configuration from the repository automatically; no API secret or manual dashboard setting is needed for the theme.

## Streamlit prototype behaviour

The app is a **historical replay and weather what-if prototype**, not a live future forecasting service.

1. The user asks a question in plain English.
2. Gemini extracts only the date, hour, weather category and assumed trip duration.
3. The saved machine-learning models generate the numerical rental estimates.
4. The app shows baseline and scenario demand side by side, an upper-demand planning estimate, and a simplified concurrent-bike estimate.

The app uses the Gemini Interactions API with structured JSON output. If no Gemini key is configured or the request fails, a local parser handles common date, time, weather and trip-duration phrases.

### Important interpretation limits

- The dataset has no station IDs, station capacity, station availability, fleet size, failed-rental logs, trip origins/destinations or actual trip durations.
- Hourly rentals are not the same as bikes needed simultaneously. The app approximates concurrent bikes in use as hourly rentals multiplied by assumed trip duration in hours.
- This concurrent-use estimate is not total fleet size.
- The test period is October–December 2012. Summer performance requires separate validation.
- Historical weather is used. A what-if changes only the weather category; temperature, humidity and wind are retained from the historical record.
- The upper-demand estimate is an empirical calibrated planning level, not a guaranteed 95% prediction interval.
- Exceptional events such as Hurricane Sandy are difficult for this model to anticipate.

## Deploy to Streamlit Community Cloud

### 1. Confirm the GitHub repository

Use this repository:

`https://github.com/Kunalsolanki1/Bike-Demand-Model`

The app entrypoint is `app.py` in the repository root. Keep these files in the root as well:
- `bike_model.joblib`
- `replay.csv`
- `requirements.txt`

### 2. Create a new Streamlit app

1. Open [Streamlit Community Cloud](https://share.streamlit.io/).
2. Sign in with the GitHub account that has access to this repository.
3. Select **Create app** → **Deploy a public app from GitHub** (wording may vary).
4. Select repository: `Kunalsolanki1/Bike-Demand-Model`.
5. Select branch: `main`.
6. Set **Main file path** to `app.py`.
7. Open **Advanced settings** and select **Python 3.11**.
8. Add the Gemini secret as described below, then deploy.

### 3. Add the Gemini API key securely

Create a **new** Gemini API key in [Google AI Studio](https://aistudio.google.com/apikey). Revoke any key that has been pasted into chat, code, GitHub, screenshots or another shared location.

In Streamlit's deployment settings, paste this into **Secrets**:

```toml
GEMINI_API_KEY = "PASTE_YOUR_NEW_KEY_HERE"
```

Never put the key in `app.py`, `requirements.txt`, a notebook, or a GitHub commit. The app reads it from `st.secrets`.

### 4. Test the deployed app

Try these questions:

- `How many bikes should we prepare at 8 AM if it rains?`
- `Compare 8 AM and 5 PM on 20 December 2012.`
- `What is the demand at 17:00 for a 45-minute trip?`

The default historical date is 20 December 2012. For that date, the case solution's example is approximately 712 baseline rentals and 507 rentals under the rain-category scenario at 8 AM.

If the app does not start, open **Manage app → Logs** and inspect the first traceback. If you change dependencies, Streamlit Community Cloud automatically rebuilds the environment after the GitHub update.

## Run locally

Use Python 3.11:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install -r requirements.txt
streamlit run app.py
```

For local Gemini access, set `GEMINI_API_KEY` as an environment variable or create `.streamlit/secrets.toml` (never commit it).

Run the smoke test:

```bash
python tests/smoke_test.py
```

## Recommended operational pilot

Pair station zones on baseline demand and zone type, then randomise one zone per pair to forecast-driven rebalancing before 07:00 and 16:00. Control zones keep current practice. Run the pilot for 4–6 weeks with the sample size and effect size fixed in advance. The primary metric is failed-rental rate during peak hours. Consider buying bikes only if peak shortages remain after redistribution.

## Additional data to request

Station locations and capacity, bikes available at each station by hour, failed-rental or app-open logs, trip start/end locations and durations, current fleet size, and rebalancing truck capacity and cost.
