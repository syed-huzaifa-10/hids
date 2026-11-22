"""Streamlit front end for the HIDS model.

Local:   streamlit run app/streamlit_app.py
Cloud:   deploy this file from the GitHub repo on Streamlit Community Cloud.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from hids_core import FEATURES, CATEGORICAL, CLASSES, CLASS_INFO, predict_events  # noqa: E402

st.set_page_config(page_title="HIDS - Intrusion Detection", page_icon="🛡️", layout="wide")


@st.cache_resource
def load():
    model = joblib.load(ROOT / "models" / "hids_pipeline.joblib")
    meta = json.load(open(ROOT / "models" / "metadata.json"))
    samples = pd.read_csv(ROOT / "app" / "sample_events.csv")
    imp = pd.read_csv(ROOT / "models" / "feature_importance.csv", index_col=0).iloc[:, 0]
    return model, meta, samples, imp


model, meta, samples, imp = load()
DEFAULTS, RANGES, TOP = meta["defaults"], meta["ranges"], meta["top_features"]
CHOICES = {"protocol_type": meta["protocols"], "service": meta["services"], "flag": meta["flags"]}
SEV_ICON = {"None": "🟢", "Low": "🟡", "High": "🟠", "Critical": "🔴"}

st.title("🛡️ Host-Based Intrusion Detection & Automated Response")
st.caption(f"Model: **{meta['best_model']}** · test macro-F1 **{meta['test']['macro_f1']:.3f}** · "
           f"trained on {meta['n_rows']:,} NSL-KDD records · classes: {', '.join(CLASSES)}")

if "state" not in st.session_state:
    st.session_state.state = dict(DEFAULTS)
    for _f in TOP:                       # initial widget values (set once, via session state)
        st.session_state[f"in_{_f}"] = DEFAULTS[_f]


def load_sample():
    idx = st.session_state.sample_choice
    if idx == "-":
        st.session_state.state = dict(DEFAULTS)
    else:
        row = samples.iloc[int(idx)].drop("true_class").to_dict()
        st.session_state.state = {**DEFAULTS, **row}
    for f in TOP:                       # push values into the visible widgets
        st.session_state[f"in_{f}"] = st.session_state.state[f]


tab1, tab2, tab3 = st.tabs(["Single event", "Batch (CSV)", "About"])

# ------------------------------------------------------------------ single event
with tab1:
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Event details")
        st.selectbox("Load a real example event (optional)", ["-"] + [str(i) for i in range(len(samples))],
                     format_func=lambda i: "— choose —" if i == "-" else
                     f"{samples.iloc[int(i)]['true_class']} example #{int(i) + 1}",
                     key="sample_choice", on_change=load_sample)
        vals, cols = {}, st.columns(2)
        for i, f in enumerate(TOP):
            key = f"in_{f}"
            with cols[i % 2]:
                if f in CATEGORICAL:
                    opts = CHOICES[f]
                    vals[f] = st.selectbox(f, opts, key=key)
                else:
                    lo, hi = RANGES[f]
                    vals[f] = st.number_input(f, min_value=0.0, key=key,
                                              help=f"Training range: {lo:g} – {hi:g}")
        st.caption(f"Showing the {len(TOP)} most influential of 41 features; the rest use typical training values.")
        go = st.button("Analyse event", type="primary")

    with right:
        st.subheader("Verdict")
        if go:
            row = {**st.session_state.state, **vals}
            res = predict_events(model, pd.DataFrame([row])).iloc[0]
            st.markdown(f"### {SEV_ICON[res['severity']]} {res['prediction']}  "
                        f"· confidence {res['confidence'] * 100:.1f}%")
            st.write(CLASS_INFO[res["prediction"]])
            st.markdown(f"**Automated response ({res['mode']}) — severity {res['severity']}**")
            for a in res["actions"]:
                st.markdown(f"- {a}")
            st.caption("Simulated response: actions are shown and logged, not executed on a real firewall.")
            st.markdown("**Class probabilities**")
            st.bar_chart(pd.Series(res["probabilities"]).reindex(CLASSES), horizontal=True)
        else:
            st.info("Fill in (or load) an event and press **Analyse event**.")
        st.markdown("**Why these inputs? Top features by permutation importance**")
        st.bar_chart(imp.head(10)[::-1], horizontal=True)

# ------------------------------------------------------------------ batch
with tab2:
    st.write("Upload a CSV with the 41 NSL-KDD feature columns (no label needed). "
             "Download a ready-made example below.")
    st.download_button("Download example CSV", samples.drop(columns="true_class").to_csv(index=False),
                       "example_events.csv", "text/csv")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            df = pd.read_csv(up)
            missing = [c for c in FEATURES if c not in df.columns]
            if missing:
                st.error(f"Missing required columns: {missing[:8]}{'...' if len(missing) > 8 else ''}")
            elif len(df) > 20000:
                st.error("Please upload at most 20,000 rows.")
            else:
                out = predict_events(model, df)
                out["actions"] = out["actions"].apply("; ".join)
                out = out.drop(columns="probabilities")
                st.dataframe(pd.concat([df[FEATURES[:3]], out], axis=1), use_container_width=True)
                st.bar_chart(out["prediction"].value_counts().reindex(CLASSES).fillna(0))
        except Exception as e:  # malformed file
            st.error(f"Could not process the file: {e}")

# ------------------------------------------------------------------ about
with tab3:
    st.markdown(f"""
**What it does.** Classifies a host/connection record as *Normal*, *DoS*, *Probe*, *R2L* (remote-to-local, e.g.
password guessing) or *U2R* (user-to-root privilege escalation) and proposes an automated response.

**Data.** NSL-KDD (train + test files combined, duplicates removed), {meta['n_rows']:,} records, 41 features.

**Test results (held-out, used once)** — accuracy {meta['test']['accuracy']:.3f}, macro-F1 {meta['test']['macro_f1']:.3f},
ROC-AUC (OvR) {meta['test']['roc_auc_ovr_macro']:.3f}.

Per-class F1: {', '.join(f'{k} {v:.2f}' for k, v in meta['test']['per_class_f1'].items())}.

**Limitations.** NSL-KDD is an older benchmark, the response actions are simulated, and U2R is very rare so its
predictions are the least reliable.
""")
