"""CLI: generate data if missing, train the model, and write monitoring artifacts.

Requires requirements-dev.txt (scikit-learn, shap). The deployed dashboard reads
the artifacts this produces and does not need those packages.
"""
import os
from fraud_monitor import generate_data
from fraud_monitor.pipeline import run

if __name__ == "__main__":
    need = ["data/reference.csv", "data/current.csv"]
    if not all(os.path.exists(p) for p in need):
        os.makedirs("data", exist_ok=True)
        ref, cur = generate_data.generate()
        ref.to_csv(need[0], index=False)
        cur.to_csv(need[1], index=False)
        print("Generated datasets.")
    res = run()
    s = res["summary"]
    print(f"reference ROC-AUC {s['eval_reference']['roc_auc']:.3f} | PR-AUC {s['eval_reference']['pr_auc']:.3f}")
    print(f"current   ROC-AUC {s['eval_current']['roc_auc']:.3f} | PR-AUC {s['eval_current']['pr_auc']:.3f}")
    print(f"artifacts written to reports/ in {s['runtime_seconds']}s")
