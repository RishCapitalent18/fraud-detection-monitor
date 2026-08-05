"""Synthetic consumer-card transaction generator.

Produces two datasets that mirror a deployed fraud model's world:

  reference.csv  -> the period the model was trained/validated on
  current.csv    -> a later "production" batch with deliberate distribution
                    DRIFT (more card-not-present volume, higher amounts, a
                    shifted merchant mix) so the monitoring layer has something
                    real to detect.

Fraud is not injected at random: a transaction's fraud probability rises with
amount-to-average ratio, card-not-present + distance from home, late-night hour,
transaction velocity, risky merchant categories, young accounts, and prior
disputes. That structure is what makes the model and the SHAP explanations
meaningful.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(11)

MERCHANTS = ["grocery", "dining", "gas", "online_retail", "electronics",
             "travel", "gambling", "cash_advance"]
# base risk weight per merchant category
MERCHANT_RISK = {"grocery": -1.2, "dining": -0.8, "gas": -0.9, "online_retail": 0.2,
                 "electronics": 0.6, "travel": 0.3, "gambling": 1.1, "cash_advance": 1.3}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _make(n: int, drift: bool) -> pd.DataFrame:
    # --- features (distribution shifts when drift=True) ---
    online_p = 0.55 if drift else 0.38                       # more card-not-present
    amt_scale = 1.35 if drift else 1.0                       # higher amounts
    merch_p = np.array([0.20, 0.16, 0.12, 0.18, 0.08, 0.08, 0.09, 0.09]) if drift \
        else np.array([0.28, 0.18, 0.16, 0.14, 0.07, 0.07, 0.05, 0.05])
    merch_p = merch_p / merch_p.sum()

    hour = RNG.integers(0, 24, n)
    is_online = (RNG.random(n) < online_p).astype(int)
    merchant = RNG.choice(MERCHANTS, size=n, p=merch_p)
    amount = np.round(np.abs(RNG.lognormal(3.2, 1.0, n)) * amt_scale + 1.0, 2)
    avg_amount = np.round(np.abs(RNG.lognormal(3.3, 0.5, n)) + 5.0, 2)
    amount_to_avg = np.round(amount / avg_amount, 3)
    distance_km = np.round(np.abs(RNG.normal(0, 15, n)) + is_online * np.abs(RNG.normal(30, 40, n)), 1)
    txns_last_hour = RNG.poisson(0.6, n) + (RNG.random(n) < 0.03) * RNG.integers(3, 9, n)
    account_age_days = RNG.integers(20, 3600, n)
    prior_disputes = RNG.poisson(0.15, n)

    merch_risk = np.array([MERCHANT_RISK[m] for m in merchant])
    late_night = ((hour <= 5) | (hour >= 23)).astype(int)

    # --- fraud log-odds ---
    logit = (
        -7.1
        + 1.15 * np.clip(amount_to_avg - 1.0, 0, None)
        + 0.9 * is_online
        + 0.012 * distance_km
        + 0.8 * late_night
        + 0.35 * txns_last_hour
        + 0.7 * merch_risk
        - 0.0004 * account_age_days
        + 0.5 * prior_disputes
    )
    p = _sigmoid(logit)
    is_fraud = (RNG.random(n) < p).astype(int)

    return pd.DataFrame({
        "amount": amount,
        "amount_to_avg_ratio": amount_to_avg,
        "hour": hour,
        "late_night": late_night,
        "is_online": is_online,
        "distance_from_home_km": distance_km,
        "txns_last_hour": txns_last_hour,
        "merchant_category": merchant,
        "account_age_days": account_age_days,
        "prior_disputes": prior_disputes,
        "is_fraud": is_fraud,
    })


def generate(n_reference: int = 30000, n_current: int = 9000):
    ref = _make(n_reference, drift=False)
    cur = _make(n_current, drift=True)
    return ref, cur


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    ref, cur = generate()
    ref.to_csv("data/reference.csv", index=False)
    cur.to_csv("data/current.csv", index=False)
    print(f"reference.csv: {len(ref)} rows, fraud rate {ref['is_fraud'].mean():.3%}")
    print(f"current.csv:   {len(cur)} rows, fraud rate {cur['is_fraud'].mean():.3%}")
