"""Unit tests for the fraud model, drift, and (optionally) SHAP layers."""
import numpy as np
import pandas as pd
import pytest

from fraud_monitor import generate_data
from fraud_monitor.model import train, evaluate, build_features
from fraud_monitor import drift as dr


@pytest.fixture(scope="module")
def data():
    ref, cur = generate_data.generate(n_reference=6000, n_current=2000)
    return ref, cur


def test_generation_has_signal(data):
    ref, cur = data
    assert 0.005 < ref["is_fraud"].mean() < 0.15
    assert cur["is_fraud"].mean() > ref["is_fraud"].mean()   # drift raises fraud rate


def test_feature_columns_stable_across_batches(data):
    ref, cur = data
    cats = sorted(ref["merchant_category"].unique())
    Xr, _ = build_features(ref, categories=cats)
    Xc, _ = build_features(cur, categories=cats)
    assert list(Xr.columns) == list(Xc.columns)             # no train/serve skew


def test_model_learns(data):
    ref, cur = data
    t = train(ref)
    m = evaluate(t["model"], t["X_test"], t["y_test"])
    assert m["roc_auc"] > 0.75                                # signal is learnable
    assert 0 <= m["precision"] <= 1 and 0 <= m["recall"] <= 1


def test_psi_zero_for_same_distribution():
    x = np.random.default_rng(0).normal(size=5000)
    assert dr.psi(x, x) < 1e-6                                # identical -> ~0
    assert dr.psi_band(0.05) == "stable"
    assert dr.psi_band(0.3) == "significant"


def test_drift_detected_between_batches(data):
    ref, cur = data
    fd = dr.feature_drift(ref, cur, ["amount", "is_online", "distance_from_home_km"])
    assert (fd["psi"] > 0.05).any()                          # some feature drifted


def test_performance_decay_on_current(data):
    ref, cur = data
    t = train(ref)
    Xc, _ = build_features(cur, categories=t["merch_cats"])
    Xc = Xc[t["feature_names"]]
    d = dr.performance_decay(t["model"], t["X_test"], t["y_test"], Xc, cur["is_fraud"].values)
    assert "roc_auc_delta" in d


def test_shap_global_importance(data):
    shap = pytest.importorskip("shap")
    from fraud_monitor.explain import global_importance
    ref, _ = data
    t = train(ref)
    gi = global_importance(t["model"], t["X_train"], sample=300)
    assert len(gi) == len(t["feature_names"])
    assert (gi["mean_abs_shap"] >= 0).all()
