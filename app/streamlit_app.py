"""Fraud Model Monitor - explainability & drift dashboard (viewer over artifacts).

Reads precomputed artifacts from reports/ so it deploys with only pandas, numpy,
plotly, and streamlit - no scikit-learn or shap at runtime.

Run:  streamlit run app/streamlit_app.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
st.set_page_config(page_title="Fraud Model Monitor", layout="wide")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(BASE, "reports")


@st.cache_data
def load():
    with open(os.path.join(REP, "model_report.json")) as f:
        report = json.load(f)
    glob = pd.read_csv(os.path.join(REP, "global_importance.csv"))
    fdrift = pd.read_csv(os.path.join(REP, "feature_drift.csv"))
    flagged = pd.read_csv(os.path.join(REP, "flagged_transactions.csv"))
    with open(os.path.join(REP, "local_explanations.json")) as f:
        locals_ = json.load(f)
    return report, glob, fdrift, flagged, locals_


if not os.path.exists(os.path.join(REP, "model_report.json")):
    st.error("No artifacts found. Run `python run_pipeline.py` first (needs requirements-dev.txt).")
    st.stop()

report, glob, fdrift, flagged, locals_ = load()
er, ec = report["eval_reference"], report["eval_current"]
decay, pdrift = report["performance_decay"], report["prediction_drift"]

st.title("Fraud Model Monitor")
st.caption("Consumer-transaction fraud model with SHAP explainability and production "
           "drift monitoring. Dashboard reads precomputed artifacts.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("ROC-AUC (ref)", f"{er['roc_auc']:.3f}")
c2.metric("PR-AUC (ref)", f"{er['pr_auc']:.3f}")
c3.metric("Precision", f"{er['precision']:.2f}")
c4.metric("Recall", f"{er['recall']:.2f}")
c5.metric("Prediction-drift PSI", f"{pdrift['psi']:.3f}",
          delta="drifting" if pdrift["psi"] > 0.25 else "stable",
          delta_color="inverse")

tabs = st.tabs(["Performance", "Explainability", "Drift monitoring", "Flagged transactions"])

# ---- Performance ----
with tabs[0]:
    left, right = st.columns(2)
    with left:
        st.markdown("**ROC curve (reference test)**")
        roc = er["roc_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], name="model"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash"), name="random"))
        fig.update_layout(height=330, xaxis_title="FPR", yaxis_title="TPR",
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("**Precision-Recall curve**")
        pr = er["pr_curve"]
        fig = px.area(x=pr["recall"], y=pr["precision"], labels={"x": "Recall", "y": "Precision"})
        fig.update_layout(height=330, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Confusion matrix (reference test, threshold "
                f"{er['threshold']})**")
    cm = er["confusion"]
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    fig = px.imshow(z, text_auto=True, color_continuous_scale="Blues",
                    x=["pred legit", "pred fraud"], y=["actual legit", "actual fraud"])
    fig.update_layout(height=300, coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ---- Explainability ----
with tabs[1]:
    st.markdown("**Global feature importance (mean |SHAP|)**")
    fig = px.bar(glob.sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature",
                 orientation="h")
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Why was this transaction flagged? (per-transaction SHAP)**")
    labels = [f"row {e['row']} - {e['merchant_category']} ${e['amount']:.0f} "
              f"(p={e['fraud_probability']:.2f})" for e in locals_]
    pick = st.selectbox("Pick a flagged transaction", range(len(locals_)),
                        format_func=lambda i: labels[i])
    e = locals_[pick]
    contrib = pd.DataFrame(e["top_contributions"])
    contrib["direction"] = np.where(contrib["shap"] >= 0, "raises risk", "lowers risk")
    fig = px.bar(contrib.sort_values("shap"), x="shap", y="feature", orientation="h",
                 color="direction",
                 color_discrete_map={"raises risk": "#c0392b", "lowers risk": "#2980b9"},
                 hover_data=["value"])
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Model fraud probability {e['fraud_probability']:.1%}. Bars show each "
               "feature's SHAP contribution to this specific prediction.")

# ---- Drift ----
with tabs[2]:
    st.markdown("**Feature drift - Population Stability Index (reference vs current)**")
    color_map = {"stable": "#95a5a6", "moderate": "#e67e22", "significant": "#c0392b"}
    fig = px.bar(fdrift.sort_values("psi"), x="psi", y="feature", orientation="h",
                 color="band", color_discrete_map=color_map)
    fig.add_vline(x=0.1, line_dash="dot"); fig.add_vline(x=0.25, line_dash="dash")
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(fdrift, use_container_width=True)

    st.markdown("**Performance decay on the drifted batch**")
    d1, d2, d3 = st.columns(3)
    d1.metric("ROC-AUC ref -> current", f"{decay['cur_roc_auc']:.3f}",
              delta=f"{decay['roc_auc_delta']:+.3f}", delta_color="normal")
    d2.metric("PR-AUC ref -> current", f"{decay['cur_pr_auc']:.3f}",
              delta=f"{decay['pr_auc_delta']:+.3f}", delta_color="normal")
    d3.metric("Score PSI (ref vs current)", f"{pdrift['psi']:.3f}")
    st.caption("Rising PSI on card-not-present features plus a PR-AUC drop is the classic "
               "signal that the production mix has shifted and the model needs retraining.")

# ---- Flagged ----
with tabs[3]:
    st.markdown("**Top current-batch transactions by fraud score**")
    show = flagged.copy()
    if "fraud_score" in show:
        show = show.sort_values("fraud_score", ascending=False)
        show["fraud_score"] = show["fraud_score"].round(3)
    st.dataframe(show, use_container_width=True, height=430)

with st.expander("Model report / audit (JSON)"):
    st.json(report)
