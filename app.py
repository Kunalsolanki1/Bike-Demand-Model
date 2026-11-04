import json
import os
import re
from datetime import date, datetime

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bike Demand Assistant", page_icon="🚲", layout="wide")

# Resolve paths relative to this file so Streamlit Cloud and local runs behave alike.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "bike_model.joblib")
DATA_PATH = os.path.join(BASE_DIR, "replay.csv")
WEATHER = {"clear": 1, "cloudy": 2, "rain": 3}


@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_replay(path):
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    if frame.empty:
        raise ValueError("replay.csv contains no rows.")
    if not frame.index.is_unique:
        raise ValueError("replay.csv contains duplicate timestamps.")
    required = {"cnt", "is_peak"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"replay.csv is missing required columns: {sorted(missing)}")
    return frame


def get_api_key():
    # Streamlit Cloud: App -> Settings -> Secrets.
    try:
        secret = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret = ""
    return secret or os.getenv("GEMINI_API_KEY", "")


def fallback(question):
    """Local parser for common date, time, weather and trip-duration phrases."""
    q = question.lower()

    hour = None
    match = re.search(r"\b(\d{1,2})(?::\d{2})?\s*(am|pm)\b", q)
    if match:
        value = int(match.group(1))
        if not 1 <= value <= 12:
            raise ValueError("Use a 12-hour time such as 8 AM or 5 PM.")
        hour = value % 12 + (12 if match.group(2) == "pm" else 0)
    else:
        match = re.search(r"\b([01]?\d|2[0-3]):[0-5]\d\b", q)
        if match:
            hour = int(match.group(1))

    weather = (
        "rain" if re.search(r"\b(rain|rainy|raining|shower)\b", q)
        else "cloudy" if re.search(r"\b(cloud|cloudy|overcast|mist|fog)\b", q)
        else "clear" if re.search(r"\b(clear|sunny)\b", q)
        else None
    )

    trip_match = re.search(r"\b(\d{1,3})\s*(?:min|mins|minutes)\b", q)
    trip_min = int(trip_match.group(1)) if trip_match else 30
    if not 1 <= trip_min <= 240:
        raise ValueError("Trip duration must be between 1 and 240 minutes.")

    date_value = None
    iso_match = re.search(r"\b(2012-\d{2}-\d{2})\b", q)
    if iso_match:
        date_value = iso_match.group(1)
    elif re.search(r"\b(2011|2012)\b", q):
        date_value = re.search(r"\b(2011|2012)\b", q).group(1)

    return {
        "date": date_value,
        "hour": hour,
        "weather": weather,
        "trip_min": trip_min,
        "parser": "local",
    }


def parse_question(question):
    """Use Gemini when configured; otherwise use a transparent local parser."""
    api_key = get_api_key()
    if not api_key:
        return fallback(question)

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        prompt = f"""
Extract the request into one JSON object only.
Keys:
- date: YYYY-MM-DD, year 2011 or 2012, or null
- hour: integer 0 through 23, or null
- weather: clear, cloudy, rain, or null
- trip_min: integer from 1 through 240; default 30
Do not invent missing values. For example, "8 AM" means hour 8 and
"5 PM" means hour 17.
Question: {question}
"""
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        payload = json.loads(response.text)
        parsed = {
            "date": payload.get("date"),
            "hour": payload.get("hour"),
            "weather": payload.get("weather"),
            "trip_min": payload.get("trip_min", 30),
            "parser": "Gemini",
        }
        if parsed["hour"] is not None:
            parsed["hour"] = int(parsed["hour"])
            if not 0 <= parsed["hour"] <= 23:
                raise ValueError("Gemini returned an invalid hour.")
        parsed["trip_min"] = int(parsed["trip_min"] or 30)
        if not 1 <= parsed["trip_min"] <= 240:
            raise ValueError("Gemini returned an invalid trip duration.")
        if parsed["weather"] not in (None, "clear", "cloudy", "rain"):
            raise ValueError("Gemini returned an unsupported weather category.")
        return parsed
    except Exception as exc:
        st.warning(
            f"Gemini parsing failed ({type(exc).__name__}). "
            "Using the local parser for this question."
        )
        return fallback(question)


def forecast_replay_day(day_value, weather):
    """Replay model predictions for one date already present in the holdout data."""
    key = pd.Timestamp(day_value).normalize()
    day_rows = data.loc[data.index.normalize() == key].copy()
    if day_rows.empty:
        raise ValueError("The selected date is not present in the historical replay.")
    if weather in WEATHER:
        day_rows.loc[:, "weathersit"] = WEATHER[weather]

    columns = pack.get("cols", [])
    missing = set(columns).difference(day_rows.columns)
    if missing:
        raise ValueError(f"Replay data is missing model features: {sorted(missing)}")

    features = day_rows[columns]
    day_rows["expected"] = pack["median"].predict(features).clip(0)
    shift = day_rows["is_peak"].map(pack["shifts"])
    if shift.isna().any():
        raise ValueError("Saved model has no calibration shift for a peak/off-peak value.")
    day_rows["upper_estimate"] = (
        pack["p95"].predict(features) + shift
    ).clip(0)
    return day_rows


try:
    pack = load_model(MODEL_PATH)
    data = load_replay(DATA_PATH)
except Exception as exc:
    st.error(f"Could not load the model or replay data: {exc}")
    st.stop()

st.title("Bike Demand Assistant")
st.caption("Historical demand replay and scenario exploration · Capital Bikeshare 2012")

st.info(
    "This prototype replays dates from 15 October to 31 December 2012. "
    "It is not a live future forecast and does not estimate station-level fleet requirements."
)

min_date = data.index.min().date()
max_date = data.index.max().date()
default_value = min(max(date(2012, 12, 20), min_date), max_date)

with st.sidebar:
    st.subheader("Replay settings")
    selected_date = st.date_input(
        "Historical date",
        value=default_value,
        min_value=min_date,
        max_value=max_date,
    )
    st.caption(f"Available replay period: {min_date} to {max_date}")
    st.caption(
        "For Gemini natural-language parsing, add GEMINI_API_KEY "
        "to Streamlit Cloud → App settings → Secrets."
    )

question = st.text_input(
    "Ask a question",
    "How many rentals were expected at 8 AM if it rains?",
)
if st.button("Analyse demand", type="primary"):
    try:
        intent = parse_question(question)
        day_value = pd.Timestamp(intent["date"]).date() if intent.get("date") else selected_date

        if day_value < min_date or day_value > max_date:
            st.warning(
                f"{day_value} is outside the historical replay. "
                f"Using the selected date {selected_date} instead."
            )
            day_value = selected_date

        baseline_day = forecast_replay_day(day_value, None)
        scenario_day = forecast_replay_day(day_value, intent.get("weather"))
        hour_value = intent.get("hour")
        if hour_value is None:
            hour_value = int(baseline_day["expected"].idxmax().hour)
        if not 0 <= int(hour_value) <= 23:
            raise ValueError("Hour must be between 0 and 23.")

        baseline_rows = baseline_day.loc[baseline_day.index.hour == int(hour_value)]
        scenario_rows = scenario_day.loc[scenario_day.index.hour == int(hour_value)]
        if baseline_rows.empty or scenario_rows.empty:
            raise ValueError("No observation exists for the selected hour.")
        baseline_result = baseline_rows.iloc[0]
        scenario_result = scenario_rows.iloc[0]
        trip_min = int(intent.get("trip_min") or 30)
        if not 1 <= trip_min <= 240:
            raise ValueError("Trip duration must be between 1 and 240 minutes.")

        st.caption(f"Parser: {intent['parser']} · Scenario: {intent.get('weather') or 'recorded weather'}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline rentals", f"{baseline_result.expected:.0f}")
        c2.metric(
            f"{intent.get('weather') or 'Scenario'} rentals",
            f"{scenario_result.expected:.0f}",
            delta=f"{scenario_result.expected - baseline_result.expected:+.0f}" if intent.get("weather") else None,
        )
        c3.metric(
            "Estimated concurrent bikes in use",
            f"{scenario_result.expected * trip_min / 60:.0f}",
            help="Approximation: scenario hourly rentals × assumed average trip duration in hours.",
        )

        st.write(
            f"For **{day_value:%d %b %Y} at {int(hour_value):02d}:00**, "
            f"the baseline model estimate is **{baseline_result.expected:.0f} rentals**. "
            + (
                f"Under the **{intent['weather']} scenario**, the estimate is "
                f"**{scenario_result.expected:.0f} rentals**. "
                if intent.get("weather") else
                f"The estimate under recorded weather is **{scenario_result.expected:.0f} rentals**. "
            )
            + f"The calibrated upper-demand estimate for this scenario is "
            f"**{scenario_result.upper_estimate:.0f} rentals**. "
            + f"Assuming a {trip_min}-minute average trip, estimated concurrent bikes in use "
            f"are approximately **{scenario_result.expected * trip_min / 60:.0f}** "
            f"(upper scenario: **{scenario_result.upper_estimate * trip_min / 60:.0f}**)."
        )
        st.write(f"Actual historical rentals for this hour: **{scenario_result.cnt:.0f}**.")
        plot = pd.DataFrame({
            "Baseline": baseline_day["expected"],
            "Scenario": scenario_day["expected"],
            "Scenario upper estimate": scenario_day["upper_estimate"],
        }, index=scenario_day.index)
        st.line_chart(plot)
        st.caption(
            "The upper-demand estimate is a calibrated model output, not a guaranteed "
            "95% interval. Concurrent use is not total fleet size. Station availability, "
            "trip origins/destinations and failed-rental data are required for fleet allocation."
        )
    except Exception as exc:
        st.error(f"Could not analyse this question: {exc}")
