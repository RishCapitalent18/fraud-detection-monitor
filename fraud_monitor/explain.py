"""SHAP explainability.

Computes global feature attribution (mean absolute SHAP value) and per-transaction
local explanations for a set of example alerts. Results are written to disk as
plain CSV/JSON so the dashboard can render them without importing shap at runtime.

shap is imported lazily so the lightweight viewer app never needs it installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def global_importance(model, X: pd.DataFrame, sample: int = 2000, seed: int = 0) -> pd.DataFrame:
    import shap
    Xs = X.sample(min(sample, len(X)), random_state=seed)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs)
    if isinstance(sv, list):          # some versions return per-class lists
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    mean_abs = np.abs(sv).mean(axis=0)
    return (pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def local_explanations(model, X: pd.DataFrame, raw: pd.DataFrame,
                       example_index, top_k: int = 6) -> list[dict]:
    """Per-transaction SHAP breakdown for the rows in example_index."""
    import shap
    explainer = shap.TreeExplainer(model)
    Xe = X.loc[example_index]
    sv = explainer.shap_values(Xe)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.asarray(base).ravel()[-1])
    proba = model.predict_proba(Xe)[:, 1]

    out = []
    for i, idx in enumerate(example_index):
        contribs = sorted(
            [{"feature": f, "shap": float(sv[i, j]), "value": float(Xe.iloc[i, j])}
             for j, f in enumerate(X.columns)],
            key=lambda d: abs(d["shap"]), reverse=True)[:top_k]
        out.append({
            "row": int(idx),
            "fraud_probability": float(proba[i]),
            "base_log_odds": float(base),
            "merchant_category": str(raw.loc[idx, "merchant_category"]),
            "amount": float(raw.loc[idx, "amount"]),
            "top_contributions": contribs,
        })
    return out
