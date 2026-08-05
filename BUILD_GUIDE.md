# Build It Yourself - Fraud Model Monitor (Windows / PowerShell)

Build it stage by stage so it is yours. Files in `fraud_monitor/` are the answer
key. Where you see **YOUR CALL**, choose the value yourself and be ready to defend
it.

## Step 0 - setup

```powershell
cd $HOME\Documents
mkdir fraud-monitor; cd fraud-monitor
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install pandas numpy scikit-learn shap streamlit plotly pytest
mkdir fraud_monitor, app, tests, data, reports
New-Item fraud_monitor\__init__.py -ItemType File
```

## Step 1 - data with real signal (`generate_data.py`)

Generate card transactions where fraud is NOT random: build a fraud log-odds from
amount-to-average ratio, card-not-present, distance, late-night, velocity, merchant
risk, account age, and prior disputes, then sample the label from its sigmoid.
Produce a second "current" batch with distribution **drift** (more online, higher
amounts, shifted merchant mix).

**YOUR CALL:** the intercept that sets the fraud rate (aim ~2%) and how hard you
push the drift. Realistic consumer fraud is a low single-digit percent.

## Step 2 - features & model (`model.py`)

One-hot the merchant category against a FIXED category list (this prevents
train/serve skew when the current batch is missing a category). Train a
`GradientBoostingClassifier`. Evaluate with ROC-AUC, **PR-AUC** (the one that
matters on imbalanced data), precision/recall/F1, and a confusion matrix.

## Step 3 - explainability (`explain.py`)

Use `shap.TreeExplainer`. Global = mean absolute SHAP per feature. Local = the
per-feature SHAP contributions for a specific flagged transaction. Save both to
disk. **YOUR CALL:** how many top features to show per local explanation.

## Step 4 - drift monitoring (`drift.py`)

Implement **PSI** (bin the reference, compare the current batch's mass) with bands
(stable/moderate/significant). Add prediction-score drift and - the important part -
`performance_decay()` that reports AUC/PR-AUC on the drifted batch, so you can see
whether input drift has reached the metric yet.

## Step 5 - pipeline + dashboard

`pipeline.py` runs everything and writes artifacts to `reports/`. The Streamlit app
READS those artifacts (keep it light - no shap at runtime). Run:

```powershell
python run_pipeline.py
streamlit run app\streamlit_app.py
```

## Step 6 - tests

```powershell
python -m pytest -q
```
Good tests: fraud rate in a sane band, current fraud rate > reference, identical
feature columns across batches, model AUC above a floor, PSI ~0 for identical data,
drift detected between batches. Gate the SHAP test with `pytest.importorskip`.

## Make it yours (do at least two)

1. **Cost-based threshold** - pick the decision threshold that minimizes
   (false-decline cost + missed-fraud cost) instead of using 0.5.
2. **Champion/challenger** - train a second model and compare on the current batch.
3. **A real drift alert** - flag and format an alert when any PSI crosses 0.25 or
   PR-AUC drops more than a set amount.
4. **More features** - device fingerprint, geo velocity between consecutive txns.

## Honest framing

"I built this to show I can take a model to production - explain any single
decision with SHAP and catch drift before it hurts. I chose PR-AUC because fraud is
imbalanced, and I tied drift to actual performance decay, not just charts."
