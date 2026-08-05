"""Consumer transaction fraud model with explainability and drift monitoring.

A gradient-boosted fraud classifier plus the production-ML wrapper a consumer
bank cares about: SHAP explainability (global and per-transaction) and model
monitoring (feature drift, prediction drift, and performance degradation on a
drifted production batch). The pipeline precomputes all artifacts so the
dashboard is a fast, dependency-light viewer.
"""
__version__ = "1.0.0"
