"""Batch CLI: write a Markdown retention report for the N riskiest customers.

Usage:
    python report.py --top 5              # uses Claude if ANTHROPIC_API_KEY is set
    python report.py --top 5 --offline    # no API calls
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import make_customers
from src.explainer import explain
from src.model import ChurnModel


def render(brief: dict, exp: dict) -> str:
    lines = [
        f"## {exp['customer_id']}: {exp['churn_probability']:.0%} churn risk ({brief['risk_level']})",
        f"**{brief['headline']}**",
        "",
        "**Why**",
        *[f"- {d['plain_english']}" for d in brief["key_drivers"]],
        "",
        "**Actions**",
        *[f"{a['priority']}. {a['action'].rstrip('.')}: {a['why']}" for a in sorted(brief["recommended_actions"], key=lambda a: a["priority"])],
        "",
        f"_{brief['confidence_note']} (source: {brief['source']})_",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=5, help="number of customers")
    parser.add_argument("--offline", action="store_true", help="skip Claude")
    parser.add_argument("--out", default="retention_report.md")
    args = parser.parse_args()

    df = make_customers()
    model = ChurnModel.train(df)
    df["churn_probability"] = model.predict_proba(df)
    print(f"Model trained · test ROC-AUC = {model.test_auc:.3f}")

    sections = [f"# Retention report: top {args.top} at-risk customers\n"]
    for _, row in df.nlargest(args.top, "churn_probability").iterrows():
        exp = model.explain(row.drop(labels=["churn_probability"]))
        brief = explain(exp, offline=True if args.offline else None)
        sections.append(render(brief, exp))
        print(f"  ✓ {exp['customer_id']} ({brief['source']})")

    Path(args.out).write_text("\n".join(sections), encoding="utf-8")
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
