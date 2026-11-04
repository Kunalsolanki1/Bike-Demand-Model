from pathlib import Path
import math

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "bike_model.joblib"
REPLAY_PATH = ROOT / "replay.csv"

assert MODEL_PATH.exists(), "bike_model.joblib is missing"
assert REPLAY_PATH.exists(), "replay.csv is missing"

pack = joblib.load(MODEL_PATH)
required_pack_keys = {"median", "p95", "shifts", "cols"}
assert required_pack_keys.issubset(pack), "Model artifact has an unexpected structure"
assert {0, 1}.issubset(set(pack["shifts"])), "Missing peak/off-peak calibration shifts"

data = pd.read_csv(REPLAY_PATH, index_col=0, parse_dates=True)
data.index = pd.to_datetime(data.index)
assert len(data) == 1_872, f"Expected 1,872 replay rows, found {len(data)}"
assert data.index.is_unique, "Replay timestamps are not unique"
assert set(pack["cols"]).issubset(data.columns), "Replay data does not contain all model features"

features = data[pack["cols"]]
median_predictions = pack["median"].predict(features).clip(0)
upper_predictions = (
    pack["p95"].predict(features) + data["is_peak"].map(pack["shifts"])
).clip(0)

assert len(median_predictions) == len(data)
assert len(upper_predictions) == len(data)
assert all(math.isfinite(float(value)) for value in median_predictions)
assert all(math.isfinite(float(value)) for value in upper_predictions)
assert (median_predictions >= 0).all()
assert (upper_predictions >= 0).all()

# Regression check for the worked example in the case solution.
example = data.loc["2012-12-20"].copy()
assert not example.empty, "Worked-example date is missing from replay.csv"
example_hour = example.loc[example.index.hour == 8]
assert len(example_hour) == 1, "Expected exactly one 08:00 record for 20 Dec 2012"

baseline = float(pack["median"].predict(example_hour[pack["cols"]])[0])
rain = example_hour.copy()
rain["weathersit"] = 3
rain_prediction = float(pack["median"].predict(rain[pack["cols"]])[0])

assert abs(baseline - 712) <= 2, f"Expected ~712 baseline rentals; got {baseline:.2f}"
assert abs(rain_prediction - 507) <= 2, f"Expected ~507 rain-scenario rentals; got {rain_prediction:.2f}"

print("Smoke test passed.")
print(f"Replay rows: {len(data)}")
print(f"Date range: {data.index.min()} to {data.index.max()}")
print(f"20 Dec 2012, 08:00 baseline: {baseline:.1f}")
print(f"20 Dec 2012, 08:00 rain scenario: {rain_prediction:.1f}")
