"""Fraud model: features, training, and evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

NUMERIC = ["amount", "amount_to_avg_ratio", "hour", "late_night", "is_online",
           "distance_from_home_km", "txns_last_hour", "account_age_days", "prior_disputes"]
CATEGORICAL = ["merchant_category"]
TARGET = "is_fraud"


def build_features(df: pd.DataFrame, categories: list[str] | None = None):
    """One-hot the merchant category (stable column set via `categories`)."""
    cat = pd.Categorical(df["merchant_category"],
                         categories=categories) if categories is not None \
        else pd.Categorical(df["merchant_category"])
    dummies = pd.get_dummies(cat, prefix="merch")
    X = pd.concat([df[NUMERIC].reset_index(drop=True),
                   dummies.reset_index(drop=True)], axis=1)
    return X, list(dummies.columns)


def train(reference: pd.DataFrame, test_size: float = 0.25, random_state: int = 0):
    cats = sorted(reference["merchant_category"].unique())
    X, merch_cols = build_features(reference, categories=cats)
    y = reference[TARGET].values
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    model = GradientBoostingClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=random_state)
    model.fit(Xtr, ytr)

    feature_names = list(X.columns)
    return {"model": model, "feature_names": feature_names, "merch_cats": cats,
            "X_test": Xte, "y_test": yte, "X_train": Xtr, "y_train": ytr}


def evaluate(model, X, y, threshold: float = 0.5) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr, tpr, _ = roc_curve(y, proba)
    prec_c, rec_c, _ = precision_recall_curve(y, proba)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": threshold,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_curve": {"fpr": fpr.tolist()[::max(1, len(fpr)//200)],
                      "tpr": tpr.tolist()[::max(1, len(tpr)//200)]},
        "pr_curve": {"precision": prec_c.tolist()[::max(1, len(prec_c)//200)],
                     "recall": rec_c.tolist()[::max(1, len(rec_c)//200)]},
        "n": int(len(y)), "positives": int(int(y.sum())),
    }
