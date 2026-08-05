"""Orchestration: train, evaluate, explain, monitor -> write artifacts.

Run locally (needs requirements-dev.txt). Writes everything the dashboard reads
into reports/, so the deployed app is a fast, dependency-light viewer.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import generate_data
from .model import train, evaluate, build_features, NUMERIC
from . import drift as dr
from .explain import global_importance, local_explanations


def run(data_dir: str = "data", out_dir: str = "reports", threshold: float = 0.5) -> dict:
    started = datetime.now(timezone.utc)
    os.makedirs(out_dir, exist_ok=True)

    need = [f"{data_dir}/reference.csv", f"{data_dir}/current.csv"]
    if not all(os.path.exists(p) for p in need):
        os.makedirs(data_dir, exist_ok=True)
        ref, cur = generate_data.generate()
        ref.to_csv(need[0], index=False)
        cur.to_csv(need[1], index=False)
    reference = pd.read_csv(need[0])
    current = pd.read_csv(need[1])

    trained = train(reference)
    model, cats = trained["model"], trained["merch_cats"]

    # evaluation on held-out reference test and on the drifted current batch
    eval_ref = evaluate(model, trained["X_test"], trained["y_test"], threshold)
    Xcur, _ = build_features(current, categories=cats)
    Xcur = Xcur[trained["feature_names"]]
    eval_cur = evaluate(model, Xcur, current["is_fraud"].values, threshold)

    # monitoring
    drift_feats = NUMERIC + ["is_fraud"]
    feat_drift = dr.feature_drift(reference, current, NUMERIC)
    ref_scores = model.predict_proba(trained["X_test"])[:, 1]
    cur_scores = model.predict_proba(Xcur)[:, 1]
    pred_drift = dr.prediction_drift(ref_scores, cur_scores)
    decay = dr.performance_decay(model, trained["X_test"], trained["y_test"],
                                 Xcur, current["is_fraud"].values)

    # explainability
    glob = global_importance(model, trained["X_train"])
    top_alerts = (pd.Series(cur_scores, index=current.index)
                  .sort_values(ascending=False).head(12).index)
    locals_ = local_explanations(model, Xcur, current, list(top_alerts))

    # flagged blotter (top current transactions by score)
    blotter = current.loc[top_alerts].copy()
    blotter["fraud_score"] = cur_scores[[current.index.get_loc(i) for i in top_alerts]]

    # ---- persist artifacts ----
    glob.to_csv(f"{out_dir}/global_importance.csv", index=False)
    feat_drift.to_csv(f"{out_dir}/feature_drift.csv", index=False)
    blotter.to_csv(f"{out_dir}/flagged_transactions.csv", index=False)
    with open(f"{out_dir}/local_explanations.json", "w") as f:
        json.dump(locals_, f, indent=2)

    summary = {
        "run_timestamp_utc": started.isoformat(),
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
        "reference_fraud_rate": float(reference["is_fraud"].mean()),
        "current_fraud_rate": float(current["is_fraud"].mean()),
        "threshold": threshold,
        "eval_reference": eval_ref,
        "eval_current": eval_cur,
        "prediction_drift": pred_drift,
        "performance_decay": decay,
        "feature_drift": feat_drift.to_dict(orient="records"),
        "runtime_seconds": round((datetime.now(timezone.utc) - started).total_seconds(), 2),
    }
    with open(f"{out_dir}/model_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    return {"summary": summary, "global_importance": glob, "feature_drift": feat_drift,
            "flagged": blotter, "local_explanations": locals_}


if __name__ == "__main__":
    res = run()
    s = res["summary"]
    print(f"reference ROC-AUC {s['eval_reference']['roc_auc']:.3f} | "
          f"PR-AUC {s['eval_reference']['pr_auc']:.3f}")
    print(f"current   ROC-AUC {s['eval_current']['roc_auc']:.3f} | "
          f"PR-AUC {s['eval_current']['pr_auc']:.3f}")
    print(f"prediction-drift PSI {s['prediction_drift']['psi']:.3f}")
    print("top drifting features:", s['feature_drift'][:3])
