"""Streamlit dashboard: pick a customer, see why the model flags them, get a
Claude-written retention brief.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from src.data import make_customers
from src.explainer import DEFAULT_MODEL, explain
from src.model import ChurnModel

st.set_page_config(page_title="Churn Explainer · built with Claude", page_icon="📉", layout="wide")


@st.cache_resource
def load() -> tuple[pd.DataFrame, ChurnModel]:
    """Generate data and train once per session (takes a couple of seconds)."""
    df = make_customers()
    model = ChurnModel.train(df)
    df = df.assign(churn_probability=model.predict_proba(df))
    return df, model


df, model = load()
has_key = bool(os.getenv("ANTHROPIC_API_KEY"))

# ---- Sidebar -----------------------------------------------------------------
st.sidebar.title("📉 Churn Explainer")
st.sidebar.caption("SHAP explains the model. Claude explains SHAP.")
st.sidebar.metric("Model ROC-AUC (test)", f"{model.test_auc:.3f}")
st.sidebar.metric("Portfolio churn rate", f"{model.base_rate:.1%}")
mode = st.sidebar.radio(
    "Explanation engine",
    [f"Claude ({DEFAULT_MODEL})", "Offline demo"],
    index=0 if has_key else 1,
    disabled=not has_key,
    help="Add ANTHROPIC_API_KEY to .env to enable Claude.",
)
if not has_key:
    st.sidebar.warning("No API key found, so the app is running in offline demo mode.")

# ---- Customer picker -----------------------------------------------------------
st.title("Why is this customer likely to leave, and what should we do?")
at_risk = df.sort_values("churn_probability", ascending=False)
choice = st.selectbox(
    "Customer (sorted by churn risk)",
    at_risk["customer_id"],
    format_func=lambda cid: f"{cid}: {at_risk.set_index('customer_id').loc[cid, 'churn_probability']:.0%} risk",
)
row = df.set_index("customer_id").loc[choice].drop("churn_probability").copy()
row["customer_id"] = choice
explanation = model.explain(row)

left, right = st.columns([1, 1.3])
with left:
    st.metric("Churn probability", f"{explanation['churn_probability']:.0%}",
              delta=f"{(explanation['churn_probability'] - model.base_rate) * 100:+.0f} pts vs average",
              delta_color="inverse")
    st.subheader("Top model drivers (SHAP)")
    drivers = pd.DataFrame(explanation["top_drivers"]).set_index("feature")
    st.bar_chart(drivers["shap_log_odds"], horizontal=True)
    with st.expander("Customer profile"):
        st.json(explanation["profile"])

with right:
    st.subheader("Retention brief")
    if st.button("✨ Generate brief", type="primary"):
        with st.spinner("Asking Claude…" if mode.startswith("Claude") else "Building brief…"):
            try:
                brief = explain(explanation, offline=not mode.startswith("Claude"))
            except Exception as exc:  # show API errors in the UI instead of crashing
                st.error(f"Could not generate brief: {exc}")
                st.stop()
        icon = {"high": "🔴", "medium": "🟠", "low": "🟢"}[brief["risk_level"]]
        st.markdown(f"### {icon} {brief['headline']}")
        st.markdown("**Why**")
        for d in brief["key_drivers"]:
            st.markdown(f"- {'⬆️' if d['effect'] == 'raises risk' else '⬇️'} {d['plain_english']}")
        st.markdown("**Recommended actions**")
        for a in sorted(brief["recommended_actions"], key=lambda a: a["priority"]):
            st.markdown(f"{a['priority']}. **{a['action'].rstrip('.')}**: {a['why']}")
        st.caption(f"ℹ️ {brief['confidence_note']} · Source: {brief['source']}")
