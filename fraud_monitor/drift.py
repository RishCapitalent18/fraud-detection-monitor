"""Model monitoring: feature drift (PSI), prediction drift, performance decay."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between a reference and a current sample.

    Uses category proportions for low-cardinality/discrete features (e.g. binary
    flags) and quantile bins for continuous ones.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    uniq = np.unique(expected)

    if len(uniq) <= bins:                      # discrete / binary -> compare categories
        cats = np.unique(np.concatenate([uniq, np.unique(actual)]))
        e = np.array([(expected == c).mean() for c in cats])
        a = np.array([(actual == c).mean() for c in cats])
    else:                                      # continuous -> quantile bins
        quantiles = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        if len(quantiles) < 3:
            return 0.0
        quantiles[0], quantiles[-1] = -np.inf, np.inf
        e = np.histogram(expected, bins=quantiles)[0] / len(expected)
        a = np.histogram(actual, bins=quantiles)[0] / len(actual)

    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_band(value: float) -> str:
    if value < 0.1:
        return "stable"
    if value < 0.25:
        return "moderate"
    return "significant"


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for f in features:
        val = psi(reference[f].values, current[f].values)
        rows.append({"feature": f, "psi": round(val, 4), "band": psi_band(val)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def prediction_drift(ref_scores: np.ndarray, cur_scores: np.ndarray) -> dict:
    return {
        "psi": round(psi(ref_scores, cur_scores), 4),
        "ref_mean": float(np.mean(ref_scores)),
        "cur_mean": float(np.mean(cur_scores)),
    }


def performance_decay(model, X_ref, y_ref, X_cur, y_cur) -> dict:
    def scores(X, y):
        p = model.predict_proba(X)[:, 1]
        return roc_auc_score(y, p), average_precision_score(y, p)
    ref_auc, ref_pr = scores(X_ref, y_ref)
    cur_auc, cur_pr = scores(X_cur, y_cur)
    return {
        "ref_roc_auc": float(ref_auc), "cur_roc_auc": float(cur_auc),
        "ref_pr_auc": float(ref_pr), "cur_pr_auc": float(cur_pr),
        "roc_auc_delta": float(cur_auc - ref_auc),
        "pr_auc_delta": float(cur_pr - ref_pr),
    }
