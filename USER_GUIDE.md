# USER GUIDE - HIDS project (for Member 2)

Read this top to bottom once (about 15 minutes). It explains what the project is, what is already finished, what **you** still
have to do, and how to answer viva questions. You do not need to write any code.

---------------------------------------------------------------------------

## 1. The project in one minute

**Title:** Host-Based Intrusion Detection & Automated Response System (HIDS).

**What it does:** you give it one "event" (a record describing a connection/activity on a host: how many bytes, how many failed
logins, whether a root shell was opened, ...). A machine-learning model says what kind of event it is:

| Class | Meaning | Automated response (simulated) |
|---|---|---|
| Normal | harmless activity | allow, no alert |
| Probe | scanning / reconnaissance | log + rate-limit source, low alert |
| DoS | flooding to crash the host | block source, drop half-open connections |
| R2L | remote attacker gets a local account (password guessing = brute force) | lock account, force password reset, block source |
| U2R | local user becomes root (privilege escalation) | kill session, isolate host, page admin |

If the model is less than 60% sure, it does not act automatically - it flags the event for a human ("manual review").
The "response" is **simulated**: the app shows/logs what it would do; it does not touch a real firewall.

**Data:** NSL-KDD, a famous public intrusion-detection data set (147,888 records after cleaning, 41 features). We pooled its two
public files and removed 629 duplicate rows. We first tried to collect our own attack data from a machine but did not manage in
time - the report says this honestly.

**Best model:** Random Forest. Test results: accuracy 99.5%, macro-F1 0.92, ROC-AUC 0.999.

## 2. What is already done

| Item | Where |
|---|---|
| Full ML lifecycle (EDA, pipeline, 3 models + baseline, tuning, evaluation, saving) | `notebooks/hids_pipeline.ipynb` |
| Shared code (feature engineering, response rules) | `src/hids_core.py` |
| Saved model | `models/hids_pipeline.joblib` |
| Streamlit app / Flask app / Hugging Face Space | `app/streamlit_app.py`, `app/flask_app.py`, `deploy/hf_space/` |
| Report source and PDF | `report/report.tex`, `report/report.pdf` |

I tested the Flask app, the Streamlit app and the Space code on a test machine (all give correct answers on real test examples).
What could **not** be done for you: putting the apps online. That needs your own GitHub / Streamlit / Hugging Face accounts.

## 3. Your to-do list (in this order)

### Step A - Put the code on GitHub (10 min)
1. Create a free account at github.com, then **New repository** (name e.g. `hids-project`, **Public**).
2. In a terminal inside the project folder:
   ```bash
   git init
   git add .
   git commit -m "HIDS project"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/hids-project.git
   git push -u origin main
   ```
   (Or use GitHub's "uploading an existing file" page and drag the whole folder in.)
3. Check the repo shows `app/`, `models/`, `requirements.txt`, `README.md`.

### Step B - Streamlit Community Cloud (10 min)
1. Go to **share.streamlit.io**, sign in with GitHub, click **Create app** -> *Deploy a public app from GitHub*.
2. Repository: `YOUR-USERNAME/hids-project`, branch `main`, **Main file path: `app/streamlit_app.py`**.
3. Open **Advanced settings** and choose **Python 3.12** (the pinned libraries need 3.12 or newer).
4. Click Deploy, wait 2-5 minutes. Copy the public URL (`https://....streamlit.app`).
5. Open the URL yourself and try: choose "R2L example #13" and press *Analyse event*.

### Step C - Hugging Face (10 min)
1. Create a free account at huggingface.co. Go to *Settings -> Access Tokens -> New token*, type **Write**, copy it.
2. In the project folder:
   ```bash
   pip install huggingface_hub
   export HF_TOKEN=hf_xxxxxxxx          # Windows cmd:  set HF_TOKEN=hf_xxxxxxxx
   python deploy/push_to_hf.py YOUR_HF_USERNAME
   ```
3. It prints two links: the **model page** (model card + `hids_pipeline.joblib`) and the **Space** (live demo).
   The Space needs about 3-5 minutes to show "Running". Open it and test one example.

### Step D - Local Flask app (the one you demo on your laptop)
```bash
pip install -r requirements.txt
python app/flask_app.py
```
Open **http://127.0.0.1:5000**. Pick an example, press *Analyse event*. Also try a deliberately bad request to show the
validation (the guidelines require input validation and error messages):
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"protocol_type":"banana"}'
```
It answers with HTTP 422 and the message `'protocol_type' must be one of ['icmp', 'tcp', 'udp']`.

### Step E - Fill the placeholders and take screenshots (15 min)
1. Take three screenshots, each showing a prediction result, and save them in `report/figures/` as
   `shot_streamlit.png`, `shot_hf.png`, `shot_flask.png`.
2. In `report/report.tex`, near the top, replace: `\memberA`, `\memberB`, `\githuburl`, `\streamlithurl`, `\hfspaceurl`,
   `\hfmodelurl`. Edit the contribution table at the bottom so it matches what each of you really did.
3. Compile: `cd report && pdflatex report.tex && pdflatex report.tex` (or upload the `report` folder to overleaf.com). The result must
   still be **3-4 pages**.
4. Put the same three links in the table at the top of `README.md` and commit/push again.

> Rule from the guidelines: **a deployment that fails live scores zero for that platform.** The day before the demo, open both
> links and make sure they load. Free Streamlit apps go to sleep when unused - open the link a few minutes before your turn and
> click "wake up" if shown. Have the Flask app already running and a screenshot of each app as a backup.

## 4. The 10-minute demo (3 + 4 + 3)

| Time | What to say / do |
|---|---|
| 0:00-3:00 | Problem: hosts get attacked; analysts cannot read every event; we classify events and automate the response. Data: NSL-KDD, 147,888 rows, 41 features, 5 classes, very imbalanced (U2R only 0.08%). Show the class-balance plot. |
| 3:00-7:00 | Live demo. **Streamlit:** load a DoS example -> block source; load a U2R example -> critical. **Hugging Face Space:** one example. **Flask:** one example + the bad-input error. |
| 7:00-10:00 | Results: comparison table (baseline 0.14 -> Logistic Regression 0.74 -> Random Forest 0.95 on validation); test accuracy 99.5%, macro-F1 0.92; confusion matrix; weakest class U2R (18 test cases) and why; limitations. |

Use the "Load a real example event" dropdown for the demo. If you type only a few values by hand, the other 29 features take
average values, and the model can be less sure.

## 5. Viva - questions you should be able to answer

Everyone is asked individually. Learn the short answers; they are all consequences of the points above.

**Problem and data**
- *Why macro-F1 and not accuracy?* 52% of the data is Normal and U2R is 0.08%. A model that always says "Normal" gets 52% accuracy
  but macro-F1 0.14 (that is our baseline). Macro-F1 averages F1 over classes so rare attacks count equally.
- *Why NSL-KDD and not your own data?* We tried to capture our own attacks but could not get reliable labelled data in time; NSL-KDD is a
  public benchmark with host-level features (failed logins, root shell, su attempts). Honest limitation: it is old.
- *Why did you remove duplicates?* 629 identical rows were in both files; if one copy lands in training and one in test, the test score
  is inflated (data leakage).
- *Which two sources?* The two public files `KDDTrain+` and `KDDTest+`, pooled.

**EDA and pre-processing**
- *What did EDA show?* Strong imbalance; `src_bytes` has huge outliers (so we use a log); DoS has error rates near 1; R2L has failed
  logins; U2R has root-shell/su activity; many `*_rate` features correlated (13 pairs above 0.9).
- *What is in the Pipeline and why?* Feature engineering -> imputation -> scaling + one-hot -> model, all in one `Pipeline`, fitted on training
  data only, so no information from validation/test leaks in. The same object is saved and used by the apps.
- *Why one-hot encoding?* `protocol_type`, `service`, `flag` are categories, not numbers.
- *There are no missing values - why an imputer?* A user can leave a field blank in the app; the imputer makes the pipeline robust.

**Models and tuning**
- *Which models?* Baseline (always the most common class), Logistic Regression, Random Forest, Histogram Gradient Boosting.
- *How does a Random Forest work?* Many decision trees trained on random samples/features; the class with most votes wins; averaging reduces
  variance.
- *What did you tune?* 5-fold GridSearchCV on the training set with macro-F1: C for Logistic Regression; tree depth and min leaf size for RF;
  learning rate and leaves for boosting. Best RF: no max depth, min_samples_leaf 3.
- *Why did Logistic Regression do badly?* It draws linear boundaries; train and validation are both about 0.74 = **underfitting**.
- *Over-fitting?* RF train 0.970 vs validation 0.949 - small gap, so mild; cross-validation (0.929) agrees.
- *What are class weights?* They make mistakes on rare classes cost more, so the model does not ignore them.

**Evaluation**
- *Why a separate test set?* Validation was used to pick the model; the test set is touched once, so the final numbers are unbiased.
- *Read the confusion matrix.* Rows = true class, columns = predicted. Biggest errors: 45 normal events flagged as R2L and 26 as Probe
  (false alarms); 14 R2L and 6 U2R attacks missed.
- *Why is U2R F1 only 0.67?* Only about 120 examples exist (18 in the test set), so a single error moves the score a lot.
- *What would you improve?* Collect real host data, evaluate on unseen attack types, oversample U2R (SMOTE), tune the decision threshold.
- *Is 99.5% realistic?* Probably optimistic: random split of one data set, old data, and the official test file contains attack types not in
  training. We state this in the limitations.

**Deployment**
- *What does `joblib` do?* Saves the whole fitted pipeline to one file and loads it in each app (versions pinned in `requirements.txt`
  because pickles depend on the scikit-learn version).
- *How does the Flask `/predict` work?* It receives JSON, `validate()` checks types, categories and ranges (422 + messages if wrong),
  fills missing fields with defaults, calls the pipeline, then the response engine and returns JSON.
- *What is the automated response?* A rule table in `src/hids_core.py` (`RESPONSE_POLICY`) mapping class -> actions, with the 60% confidence
  gate. It is simulated.
- *Why does the model card exist?* It documents data, results, intended use and limits for anyone using the model on Hugging Face.

**Do you want to split the explaining?** A fair split: Member 1 = data, EDA, pipeline, models, tuning, evaluation; Member 2 =
response engine, the three apps, deployment, report. But learn *everything* above - the viva may ask either of you anything.

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| Streamlit Cloud build fails on numpy/pandas | App settings -> Python version **3.12** (or 3.13) |
| `ModuleNotFoundError: hids_core` in an app | Run commands from the project root; on the Space, `hids_core.py` must sit next to `app.py` (already there) |
| `InconsistentVersionWarning` when loading the model | Install the pinned versions: `pip install -r requirements.txt` |
| Port 5000 already used | Change `port=5000` at the bottom of `app/flask_app.py` |
| Space shows "Building" for long | Wait 5 min; check the *Logs* tab; make sure `requirements.txt` in the Space lists `scikit-learn==1.9.1` |
| Want to re-train | Run all cells of `notebooks/hids_pipeline.ipynb` (about 10 min), then copy `models/hids_pipeline.joblib`, `models/metadata.json`, `app/sample_events.csv` into `deploy/hf_space/` and `deploy/hf_model/` |
