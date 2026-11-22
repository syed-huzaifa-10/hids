# Host-Based Intrusion Detection & Automated Response System (HIDS)

CT-378 Machine Learning semester project. A scikit-learn pipeline classifies a host/connection record as
**Normal, DoS, Probe, R2L (password guessing / brute force) or U2R (privilege escalation)** and maps the verdict to a
(simulated) automated response plan. Deployed three ways: Streamlit, Hugging Face Space, local Flask app.

**Result (held-out test set, used once):** Random Forest - accuracy 0.995, macro-F1 0.921, ROC-AUC 0.999.

## Deployment links  *(fill in after deploying - see USER_GUIDE.md)*
| Platform | Link |
|---|---|
| Streamlit Community Cloud | `https://YOUR-APP.streamlit.app` |
| Hugging Face Space (Gradio) | `https://huggingface.co/spaces/YOUR-USERNAME/hids-demo` |
| Hugging Face model card + pipeline | `https://huggingface.co/YOUR-USERNAME/hids-nsl-kdd-classifier` |
| Local Flask app | `http://127.0.0.1:5000` (run locally, see below) |

## Repository layout
```
data/            download_data.py + README.md (source, licence, date); raw files are downloaded into this folder
notebooks/       hids_pipeline.ipynb (full lifecycle: EDA -> pipeline -> tuning -> evaluation -> save); .py is the same code
src/hids_core.py shared schema, feature engineering, prediction helper and response engine
app/             streamlit_app.py, flask_app.py (+ templates/), sample_events.csv (real test records for demos)
models/          hids_pipeline.joblib, metadata.json, model_comparison.csv, feature_importance.csv
deploy/          hf_space/ (Gradio Space), hf_model/ (model card + pipeline), push_to_hf.py
report/          report.tex, report.pdf, figures/
requirements.txt pinned versions
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python data/download_data.py                            # only needed to re-run the notebook
```

## Run
```bash
streamlit run app/streamlit_app.py     # http://localhost:8501
python app/flask_app.py                # http://127.0.0.1:5000   (POST /predict, GET /health)
python deploy/hf_space/app.py          # Gradio demo, http://127.0.0.1:7860
```
Flask API example:
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" \
     -d '{"protocol_type":"tcp","service":"http","flag":"SF","src_bytes":232,"dst_bytes":8153}'
```
Missing fields default to typical training values. Invalid input returns HTTP 422 with a list of messages.

## Re-train
Open `notebooks/hids_pipeline.ipynb` and run all cells (about 10 minutes on 2 CPU cores). It rewrites `models/` and
`app/sample_events.csv`; then copy the files again into `deploy/hf_space/` (see USER_GUIDE.md).

## Data
NSL-KDD (Tavallaee et al., 2009), `KDDTrain+` + `KDDTest+` combined, duplicates removed. Details in `data/README.md`.

## Limitations
Old benchmark data, random split of pooled data is optimistic, U2R is very rare (about 120 records), and the response
actions are simulated.
