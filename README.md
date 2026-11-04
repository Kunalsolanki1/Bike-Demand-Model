# Bike Sharing Case: Do We Need More Bikes?

## Question
Riders can't find bikes, yet bikes sit idle. Should the company buy more?

## Short answer
Not yet. Demand is concentrated in commute hours and is forecastable, which points to a timing and placement problem. The data has no stations, so it cannot show where bikes are missing. A controlled pilot is needed before buying.

## Data
Capital Bikeshare, Washington D.C., 2011-2012 (hourly and daily files). Citation: Fanaee-T & Gama (2013), "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence.

## What was done
1. **Data quality:** no nulls or duplicates, and `cnt = casual + registered` on every row. 165 hourly rows were missing: 110 overnight (0-5 AM) and 55 on eight storm days, including Hurricane Sandy. All were filled with 0 and flagged. The lowest day (22 rentals, 2012-10-29) is Sandy and was kept.
2. **Season labels:** the readme's labels are wrong for this data. Code 1 is winter (Dec-Mar), 2 spring, 3 summer, 4 fall. Summer is the peak (5,644 rentals/day) and winter the low (2,604).
3. **Hypothesis test:** commute hours (7-8 AM, 5-6 PM) carry a median 39.3% of daily rentals on working days vs 17.0% on non-working days, a gap of about 22 points. Days with a filled commute hour were excluded.

Hypothesis: Demand is concentrated in commute peaks, so timing rather than fleet size drives shortages.

Data test: H₀: the commute-hour share of daily rentals is the same on working and non-working days. H₁: it is higher on working days. Mann-Whitney test, effect size reported as the median gap (22 points).

What the data cannot show: whether rebalancing reduces unmet demand. That needs the pilot below.

4. **Model:** gradient boosting on calendar, weather, trend and lagged demand (24h, 168h). `casual` and `registered` are not inputs. Chronological split: train before 15 Oct 2012, test 15 Oct-31 Dec 2012.

| | MAE | RMSE | R² |
|---|---|---|---|
| Model | 45.5 | 75.0 | 0.849 |
| Same hour last week | 77.2 | 132.1 | 0.533 |

5. **Upper demand level (p95):** the raw quantile model covered only 92% of hours, and 86% of peak hours after a first calibration. Separate shifts for peak (+94) and off-peak (+24) gave about 96% coverage in both. Coverage was not re-checked without the two Sandy days, so treat 96% as slightly optimistic..
6. **Bikes on the road** (rentals per hour x assumed trip length), Oct-Dec 2012:

| Trip length | Average | Typical peak (p95) | Busiest day |
|---|---|---|---|
| 15 min | 51 | 193 | 248 |
| 30 min | 102 | 386 | 495 |
| 60 min | 204 | 771 | 990 |

7. **Prototype:** a Streamlit app. An LLM turns a question into a date, hour, weather and trip length, and the model supplies every number.

## Run the app
```
pip install streamlit anthropic scikit-learn joblib pandas
streamlit run app.py
```
Files needed in one folder: `app.py`, `bike_model.joblib`, `replay.csv`. Set `ANTHROPIC_API_KEY` for the LLM parser, otherwise a keyword fallback handles hour and weather but not dates. Use the same scikit-learn version as the notebook. Use explicit dates in questions.

## Assumptions and limits
- Hourly rentals are not simultaneous bikes. Trip length is not in the data, so every bike count is a range.
- No station, availability, fleet size or failed-rental data, so the where problem cannot be tested.
- The test window is Oct-Dec (low season). Summer peaks need separate validation.
- Demand grew 64% from 2011 to 2012. Tree models cannot extrapolate, so the model needs regular retraining.
- Actual weather stands in for a forecast. Rain what-ifs change only the weather category and probably understate the effect.
- The p95 calibration was designed after seeing the test period, so coverage is slightly optimistic.
- The model does not anticipate events like Sandy (average error about 2.6x normal on those hours).

## Recommended pilot
Match station zones in pairs on baseline demand and zone type, then randomise one zone per pair to forecast-driven rebalancing before 7 AM and 4 PM. Control zones keep current practice. Primary metric is failed-rental rate in peak hours. Run 4-6 weeks with the effect size and sample size fixed in advance. Buy bikes only if peak shortages persist after rebalancing.

## Data to request
Station IDs and capacity, hourly bikes available per station, failed-rental or app-open logs, trip start/end locations and durations, current fleet size, and rebalancing truck capacity and cost.

Note- day.csv and hour.csv are loaded from /content (Colab). Change the path if running elsewhere. The notebook only runs as-is in Colab.
