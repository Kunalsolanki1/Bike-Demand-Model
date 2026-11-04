# Bike Sharing Case: Do We Need More Bikes?

## Question
Riders cannot find bikes, yet bikes sit idle. Should the company buy more?

## Short answer
Not yet. Demand is concentrated in commute hours and is forecastable, which points to a timing and placement problem. The supplied data has no stations, so it cannot show where bikes are missing. A controlled pilot is needed before buying.

## Data
Capital Bikeshare, Washington, D.C., 2011–2012 (hourly and daily files). Citation: Fanaee-T & Gama (2013), "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence.

## Analysis
1. **Data quality:** 731 daily rows and 17,379 hourly rows. The notebook reports no nulls or duplicate rows in the raw files and validates `cnt = casual + registered`. It identifies 165 missing hourly timestamps, which are flagged and filled with zero rentals for the analysis. This is an explicit assumption, especially for daytime gaps on storm days.
2. **Season labels:** the data's season code mapping is corrected from observed month cross-tabs: 1 winter, 2 spring, 3 summer, 4 fall.
3. **Hypothesis:** commute-hour share (07:00–08:00 and 17:00–18:00) is higher on working days than non-working days. The notebook reports median shares of 39.3% versus 17.0% and uses a one-sided Mann–Whitney U test. This supports a demand-pattern difference; it does not prove that rebalancing alone will resolve shortages.
4. **Demand model:** HistGradientBoostingRegressor uses calendar, weather, trend and lagged demand (24h and 168h). `casual` and `registered` are excluded to avoid target leakage. The chronological test period is 15 October–31 December 2012.
5. **Reported model performance:** MAE 45.5, RMSE 75.0, R² 0.849, compared with a same-hour-last-week baseline (MAE 77.2, RMSE 132.1, R² 0.533). These are notebook-reported values and should be regenerated after any model changes.
6. **Upper-demand estimate:** a separate quantile model is calibrated by peak/off-peak segment. Treat the reported coverage as an empirical test result, not a guaranteed future 95% interval.

## Streamlit prototype
The app uses the saved model and `replay.csv` to replay the historical holdout period. It is **not a true future forecasting service**: the replay file contains historical features and actual rental counts from the test period.

The app supports Gemini natural-language parsing when configured. Without a key, a local parser supports common hour, date, weather and trip-duration phrases.

### Deploy
Required files in the repository root:
- `app.py`
- `bike_model.joblib`
- `replay.csv`
- `requirements.txt`

Streamlit Cloud settings:
1. Open the app's **Settings → Secrets**.
2. Add:
   ```toml
   GEMINI_API_KEY = "YOUR_NEW_GEMINI_API_KEY"
   ```
3. Save and reboot the app.

Do not commit API keys to GitHub, notebooks, or source files. Rotate any key that has been pasted into a chat, issue, commit, or other shared location.

Install locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

The model is serialized with scikit-learn 1.6.1, so the deployment pins that version to reduce pickle compatibility issues.

## Assumptions and limitations
- Hourly rentals are not simultaneous bikes. Trip duration is not in the supplied data; concurrent-use estimates are therefore scenario calculations based on an assumed trip duration.
- No station IDs, station capacity, station availability, fleet size, failed-rental logs, trip locations or trip durations are supplied. The current project cannot estimate station-level redistribution or exact fleet size.
- The test window is October–December 2012. Summer demand and other periods need separate validation.
- Demand grew between 2011 and 2012. Tree models do not extrapolate trends reliably; retraining and monitoring are needed.
- Actual historical weather is used in replay. Changing only the weather category is a simplified what-if scenario, not a physically consistent weather forecast.
- The notebook notes that p95 calibration was designed after examining the test period, so reported coverage may be optimistic.
- The model does not anticipate exceptional events such as Hurricane Sandy.

## Recommended pilot
Pair station zones on baseline demand and zone type, then randomise one zone per pair to forecast-driven rebalancing before 07:00 and 16:00. Control zones keep current practice. Primary metric: failed-rental rate during peak hours. Run 4–6 weeks with effect size and sample size fixed in advance. Buy bikes only if peak shortages persist after rebalancing.

## Additional data to request
Station IDs and capacity, hourly bikes available per station, failed-rental or app-open logs, trip start/end locations and durations, total fleet size, and rebalancing truck capacity and cost.
