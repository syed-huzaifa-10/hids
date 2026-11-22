"""Hugging Face Space (Gradio) for the HIDS model. Self-contained: loads the
bundled joblib pipeline from this folder."""
import json
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

from hids_core import CATEGORICAL, CLASSES, CLASS_INFO, predict_events

HERE = Path(__file__).parent
model = joblib.load(HERE / "hids_pipeline.joblib")
meta = json.load(open(HERE / "metadata.json"))
samples = pd.read_csv(HERE / "sample_events.csv")
DEFAULTS, TOP = meta["defaults"], meta["top_features"]
CHOICES = {"protocol_type": meta["protocols"], "service": meta["services"], "flag": meta["flags"]}
SAMPLE_NAMES = [f"{r.true_class} example #{i + 1}" for i, r in samples.iterrows()]
ICON = {"None": "🟢", "Low": "🟡", "High": "🟠", "Critical": "🔴"}


def analyse(sample_name, *vals):
    row = dict(DEFAULTS)
    if sample_name in SAMPLE_NAMES:   # keep the hidden features of the chosen example
        s = samples.iloc[SAMPLE_NAMES.index(sample_name)].drop("true_class").to_dict()
        row.update(s)
    row.update(dict(zip(TOP, vals)))
    for f in TOP:
        if f not in CATEGORICAL:
            if row[f] is None or row[f] < 0:
                raise gr.Error(f"'{f}' must be a non-negative number.")
    res = predict_events(model, pd.DataFrame([row])).iloc[0]
    plan = "\n".join(f"- {a}" for a in res["actions"])
    md = (f"### {ICON[res['severity']]} {res['prediction']} — confidence {res['confidence']*100:.1f}%\n"
          f"{CLASS_INFO[res['prediction']]}\n\n**Automated response ({res['mode']}, severity "
          f"{res['severity']})**\n{plan}\n\n*Simulated: actions are not executed on a real firewall.*")
    return res["probabilities"], md


def load_sample(sample_name):
    if sample_name not in SAMPLE_NAMES:
        return [DEFAULTS[f] for f in TOP]
    s = samples.iloc[SAMPLE_NAMES.index(sample_name)]
    return [s[f] for f in TOP]


with gr.Blocks(title="HIDS - Intrusion Detection") as demo:
    gr.Markdown("# 🛡️ Host-Based Intrusion Detection & Automated Response\n"
                f"Model: **{meta['best_model']}** · test macro-F1 **{meta['test']['macro_f1']:.3f}** · "
                f"classes: {', '.join(CLASSES)}")
    sample = gr.Dropdown(["-"] + SAMPLE_NAMES, value="-", label="Load a real example event (optional)")
    inputs = []
    with gr.Row():
        for f in TOP:
            if f in CATEGORICAL:
                inputs.append(gr.Dropdown(CHOICES[f], value=DEFAULTS[f], label=f))
            else:
                inputs.append(gr.Number(value=DEFAULTS[f], label=f, minimum=0))
    btn = gr.Button("Analyse event", variant="primary")
    label = gr.Label(label="Class probabilities", num_top_classes=5)
    out = gr.Markdown()
    sample.change(load_sample, sample, inputs)
    btn.click(analyse, [sample] + inputs, [label, out])

if __name__ == "__main__":
    demo.launch()
