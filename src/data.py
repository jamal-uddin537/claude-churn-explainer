"""Synthetic telecom customer dataset.

We generate data instead of downloading it so the project runs anywhere,
offline, with zero setup. The churn signal is built from realistic drivers
(short tenure, month-to-month contracts, high bills, support tickets), so the
model and its SHAP explanations behave like they would on real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONTRACT_TYPES = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHODS = ["Electronic check", "Credit card", "Bank transfer", "Mailed check"]
INTERNET_TYPES = ["Fiber optic", "DSL", "None"]


def make_customers(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame of `n` customers with a binary `churn` column."""
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 73, n)  # months with the company
    contract = rng.choice(CONTRACT_TYPES, n, p=[0.55, 0.25, 0.20])
    payment = rng.choice(PAYMENT_METHODS, n, p=[0.35, 0.22, 0.22, 0.21])
    internet = rng.choice(INTERNET_TYPES, n, p=[0.45, 0.40, 0.15])
    monthly = np.round(
        rng.normal(65, 25, n).clip(18, 120) + np.where(internet == "Fiber optic", 15, 0), 2
    )
    tickets = rng.poisson(1.2, n)  # support tickets in the last 6 months
    tech_support = rng.random(n) < 0.35
    senior = rng.random(n) < 0.16

    # Hand-built log-odds so the "true" drivers are known and sensible.
    logit = (
        -1.2
        - 0.045 * tenure
        + np.select(
            [contract == "Month-to-month", contract == "One year"], [1.4, 0.2], -0.9
        )
        + 0.018 * (monthly - 65)
        + 0.35 * tickets
        + np.where(payment == "Electronic check", 0.55, 0.0)
        + np.where(internet == "Fiber optic", 0.35, 0.0)
        - np.where(tech_support, 0.6, 0.0)
        + np.where(senior, 0.3, 0.0)
    )
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))

    return pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "tenure_months": tenure,
            "contract": contract,
            "payment_method": payment,
            "internet_service": internet,
            "monthly_charges": monthly,
            "support_tickets_6m": tickets,
            "has_tech_support": tech_support,
            "senior_citizen": senior,
            "churn": churn.astype(int),
        }
    )
