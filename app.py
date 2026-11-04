import json, re
import joblib, pandas as pd, streamlit as st

pack = joblib.load("bike_model.joblib")
data = pd.read_csv("replay.csv", index_col=0, parse_dates=True)
WEATHER = {"clear": 1, "cloudy": 2, "rain": 3}

def fallback(q):
    q = q.lower()
    m = re.search(r"(\d{1,2})\s*(am|pm)", q)
    hour = int(m.group(1)) % 12 + (12 if m.group(2) == "pm" else 0) if m else None
    weather = "rain" if "rain" in q else "cloudy" if "cloud" in q else "clear" if "clear" in q else None
    return {"date": None, "hour": hour, "weather": weather, "trip_min": 30}

def parse(q):
    try:
        import anthropic
        msg = anthropic.Anthropic().messages.create(
            model="claude-sonnet-5-5", max_tokens=150,
            messages=[{"role": "user", "content":
                'Reply with JSON only, keys: date (YYYY-MM-DD, year 2012, or null), hour (0-23 or null), '
                'weather ("clear","cloudy","rain" or null), trip_min (number, default 30). '
                f"Question: {q}"}])
        return json.loads(re.search(r"\{.*\}", msg.content[0].text, re.S).group())
    except Exception:
        return fallback(q)

def forecast_day(date, weather):
    d = data.loc[str(date)].copy()
    if weather in WEATHER:
        d["weathersit"] = WEATHER[weather]
    X = d[pack["cols"]]
    d["expected"] = pack["median"].predict(X).clip(0)
    d["high"] = (pack["p95"].predict(X) + d.is_peak.map(pack["shifts"])).clip(0)
    return d

st.title("Bike Demand Assistant")
default_date = st.sidebar.date_input("Default date", pd.Timestamp("2012-12-20"),
                                     min_value=data.index.min().date(), max_value=data.index.max().date())
q = st.text_input("Ask a question", "How many bikes should we prepare at 8 AM if it rains?")

if st.button("Ask"):
    p = parse(q)
    date = pd.Timestamp(p.get("date") or default_date)
    if date not in data.index.normalize():
        st.warning("That date is outside the data (15 Oct to 31 Dec 2012). Using the sidebar date.")
        date = pd.Timestamp(default_date)
    d = forecast_day(date.date(), p.get("weather"))
    hour = p.get("hour")
    hour = int(d.expected.idxmax().hour) if hour is None else int(hour)
    r = d[d.index.hour == hour].iloc[0]
    trip = p.get("trip_min") or 30

    st.write(f"**{date.date()}, {hour}:00** ({p.get('weather') or 'recorded'} weather): "
             f"expect about **{r.expected:.0f} rentals**, up to **{r.high:.0f}** (95% level). "
             f"With {trip}-minute trips that is about **{r.expected * trip / 60:.0f} bikes on the road**, "
             f"and **{r.high * trip / 60:.0f}** to plan for. Actual that hour: {r.cnt:.0f}.")
    st.line_chart(d[["expected", "high"]])
    st.caption("Replay of Oct-Dec 2012. Only the weather category is changed for what-if questions. "
               "Bikes on the road is not fleet size: station-level data is needed for that.")
