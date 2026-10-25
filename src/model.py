"""Churn model + SHAP explanations.

This is the "classic ML" half of the project: a gradient-boosted model whose
individual predictions are explained with SHAP. Claude never sees raw data
dumps, only the compact, numeric explanation produced here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

CATEGORICAL = ["contract", "payment_method", "internet_service"]
NUMERIC = ["tenure_months", "monthly_charges", "support_tickets_6m"]
BOOLEAN = ["has_tech_support", "senior_citizen"]


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals so the tree model can use them."""
    X = df[NUMERIC + BOOLEAN + CATEGORICAL].copy()
    X[BOOLEAN] = X[BOOLEAN].astype(int)
    return pd.get_dummies(X, columns=CATEGORICAL, dtype=int)


def _base_feature(encoded_name: str) -> str:
    """Map 'contract_Month-to-month' back to 'contract'."""
    for cat in CATEGORICAL:
        if encoded_name.startswith(cat + "_"):
            return cat
    return encoded_name


@dataclass
class ChurnModel:
    """Trained model, its explainer, and evaluation results."""

    model: GradientBoostingClassifier
    explainer: shap.TreeExplainer
    columns: list[str]
    test_auc: float
    base_rate: float
    metrics: dict = field(default_factory=dict)

    @classmethod
    def train(cls, df: pd.DataFrame, seed: int = 42) -> "ChurnModel":
        X, y = encode(df), df["churn"]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=y
        )
        model = GradientBoostingClassifier(random_state=seed).fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
        return cls(
            model=model,
            explainer=shap.TreeExplainer(model),
            columns=list(X.columns),
            test_auc=float(auc),
            base_rate=float(y.mean()),
        )

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = encode(df).reindex(columns=self.columns, fill_value=0)
        return self.model.predict_proba(X)[:, 1]

    def explain(self, row: pd.Series, top_k: int = 5) -> dict:
        """Explain one customer.

        Returns a small JSON-friendly dict: churn probability, the customer's
        raw profile, and the top-k SHAP drivers (one-hot columns are summed back
        into their original feature so the output stays human-readable).
        """
        frame = row.to_frame().T.infer_objects()
        X = encode(frame).reindex(columns=self.columns, fill_value=0)
        values = np.asarray(self.explainer.shap_values(X))[0]

        # Aggregate one-hot SHAP values back to the original feature names.
        grouped: dict[str, float] = {}
        for col, val in zip(self.columns, values):
            key = _base_feature(col)
            grouped[key] = grouped.get(key, 0.0) + float(val)

        drivers = sorted(grouped.items(), key=lambda kv: abs(kv[1]), reverse=True)
        profile = {
            k: (v.item() if hasattr(v, "item") else v)
            for k, v in row.drop(labels=["churn"], errors="ignore").items()
        }
        return {
            "customer_id": profile.pop("customer_id", None),
            "churn_probability": round(float(self.predict_proba(frame)[0]), 3),
            "portfolio_base_rate": round(self.base_rate, 3),
            "profile": profile,
            "top_drivers": [
                {
                    "feature": f,
                    "value": profile.get(f),
                    "shap_log_odds": round(v, 3),
                    "pushes_toward": "churn" if v > 0 else "staying",
                }
                for f, v in drivers[:top_k]
            ],
        }
