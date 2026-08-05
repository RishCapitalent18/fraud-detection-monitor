# Fraud Model Monitor

> A consumer-transaction fraud model wrapped in the two things production ML actually needs: SHAP explainability (global and per-transaction) and drift monitoring (feature drift, prediction drift, and performance decay on a shifted production batch).

Built with Python (scikit-learn, SHAP), Streamlit, and pytest. The training/
monitoring pipeline precomputes all artifacts, so the deployed dashboard is a
fast viewer that needs only pandas, numpy, plotly, and streamlit.

---

## What it does

Building a fraud classifier is the easy part. Keeping one trustworthy in
production is the job: can you explain any given decline, and do you notice when
the world shifts under the model? This project does both.

| Stage | Module | What it produces |
|-------|--------|------------------|
| **Data** | `fraud_monitor/generate_data.py` | Synthetic card transactions with a realistic fraud signal, plus a later "current" batch with deliberate distribution drift. |
| **Model** | `fraud_monitor/model.py` | Gradient-boosted classifier; ROC-AUC, PR-AUC, precision/recall/F1, confusion matrix, ROC/PR curves. |
| **Explainability** | `fraud_monitor/explain.py` | Global mean-absolute-SHAP importance and per-transaction SHAP breakdowns for flagged alerts. |
| **Monitoring** | `fraud_monitor/drift.py` | Population Stability Index per feature, prediction-score drift, and ROC/PR-AUC decay on the drifted batch. |
| **Pipeline** | `fraud_monitor/pipeline.py` | Runs all of the above and writes artifacts to `reports/`. |
| **Dashboard** | `app/streamlit_app.py` | Performance / Explainability / Drift / Flagged-transactions tabs over the artifacts. |

## Why it is built as precomputed artifacts

The pipeline (which needs scikit-learn and shap) runs offline and writes plain
CSV/JSON to `reports/`. The Streamlit app just reads those, so:
- the live app deploys fast and reliably with a lightweight `requirements.txt`;
- SHAP never has to run on the hosting server.

`requirements.txt` = the viewer's runtime deps. `requirements-dev.txt` = what you
need to regenerate artifacts and run tests.

## Sample results

- Reference ROC-AUC ~0.9+, with the usual fraud precision/recall trade-off shown
  as full ROC and PR curves and a confusion matrix.
- SHAP surfaces amount-to-average ratio, card-not-present, distance, and risky
  merchant categories as the top drivers - matching how the data was generated.
- The drifted batch shows moderate PSI on card-not-present (is_online) and amount,
  and a clear upward shift in the prediction-score distribution (PSI ~0.16), while
  ROC-AUC stays high - the case for watching inputs and scores rather than waiting
  for the metric to fall.

## Quickstart

```bash
# regenerate artifacts + run tests (full deps)
pip install -r requirements-dev.txt
python run_pipeline.py
pytest -q

# run the dashboard (light deps; reads committed artifacts)
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Design notes a reviewer will care about

- **No train/serve skew.** The merchant one-hot columns are pinned to a fixed
  category set so the reference and current batches always share an identical
  feature space.
- **Honest monitoring.** Drift is measured with PSI (stable < 0.1, moderate < 0.25,
  significant above) on both features and prediction scores, and reported alongside
  ROC/PR-AUC on the drifted batch. The point is that input and score drift can
  appear before discrimination degrades - which is exactly why you monitor them.
- **Explainability at two levels.** Global SHAP for "what does the model rely on",
  local SHAP for "why was THIS transaction flagged" - the second is what an
  investigator or a model-risk reviewer asks for.
- **Reproducible & tested.** Seeded data and a pytest suite covering signal,
  feature-space stability, learnability, PSI behaviour, and drift detection.

## Repo layout

```
fraud-detection-monitor/
+-- run_pipeline.py             # generate data + train + write artifacts
+-- requirements.txt            # viewer runtime deps
+-- requirements-dev.txt        # pipeline + test deps (adds scikit-learn, shap)
+-- fraud_monitor/
|   +-- generate_data.py        # synthetic transactions + drifted batch
|   +-- model.py                # features, training, evaluation
|   +-- explain.py              # SHAP global + local
|   +-- drift.py                # PSI, prediction drift, performance decay
|   +-- pipeline.py             # orchestration -> reports/
+-- app/streamlit_app.py        # dashboard (viewer over artifacts)
+-- tests/test_model.py         # unit tests
+-- data/                       # reference.csv, current.csv
+-- reports/                    # precomputed artifacts read by the app
```


