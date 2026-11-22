"""Local web app: Flask backend with a /predict endpoint + HTML front end.

Run:  python app/flask_app.py      then open http://127.0.0.1:5000
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from hids_core import FEATURES, CATEGORICAL, CLASS_INFO, predict_events  # noqa: E402

MODEL = joblib.load(ROOT / "models" / "hids_pipeline.joblib")
META = json.load(open(ROOT / "models" / "metadata.json"))
DEFAULTS = META["defaults"]
RANGES = META["ranges"]
CHOICES = {"protocol_type": META["protocols"], "service": META["services"], "flag": META["flags"]}
SAMPLES = pd.read_csv(ROOT / "app" / "sample_events.csv")
LOG_FILE = ROOT / "models" / "response_log.jsonl"

app = Flask(__name__)


def validate(payload: dict):
    """Return (clean_row_dict, list_of_error_strings)."""
    errors, row = [], {}
    if not isinstance(payload, dict):
        return None, ["Request body must be a JSON object."]
    unknown = [k for k in payload if k not in FEATURES]
    if unknown:
        errors.append(f"Unknown field(s): {', '.join(unknown)}")
    for f in FEATURES:
        v = payload.get(f, None)
        if v is None or v == "":          # missing -> training default
            row[f] = DEFAULTS[f]
            continue
        if f in CATEGORICAL:
            if v not in CHOICES[f]:
                errors.append(f"'{f}' must be one of {CHOICES[f]}; got '{v}'.")
            row[f] = v
        else:
            try:
                num = float(v)
            except (TypeError, ValueError):
                errors.append(f"'{f}' must be a number; got '{v}'.")
                continue
            if num != num or num in (float("inf"), float("-inf")):
                errors.append(f"'{f}' must be a finite number.")
                continue
            lo, hi = RANGES[f]
            if num < 0:
                errors.append(f"'{f}' cannot be negative.")
            elif num > hi * 10 + 1:   # generous upper bound vs. training range
                errors.append(f"'{f}'={num} is far outside the training range (max seen {hi:g}).")
            row[f] = num
    return row, errors


@app.get("/")
def index():
    return render_template("index.html", meta=META, info=CLASS_INFO,
                           top=META["top_features"], choices=CHOICES, defaults=DEFAULTS,
                           samples=SAMPLES.to_dict(orient="records"))


@app.get("/health")
def health():
    return jsonify(status="ok", model=META["best_model"], classes=META["classes"])


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify(error="Body must be valid JSON (Content-Type: application/json)."), 400
    row, errors = validate(payload)
    if errors:
        return jsonify(error="Invalid input", details=errors), 422
    res = predict_events(MODEL, pd.DataFrame([row])).iloc[0].to_dict()
    entry = {"time": datetime.now().isoformat(timespec="seconds"),
             "prediction": res["prediction"], "confidence": res["confidence"],
             "severity": res["severity"], "mode": res["mode"], "actions": res["actions"]}
    try:
        with open(LOG_FILE, "a") as fh:     # simulated response log
            fh.write(json.dumps(entry) + "\n")
    except OSError:
        pass
    res["description"] = CLASS_INFO[res["prediction"]]
    return jsonify(res)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
