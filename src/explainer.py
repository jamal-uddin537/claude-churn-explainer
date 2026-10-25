"""Turn SHAP output into a retention brief with Claude.

Technique (adapted from the "Tool use" and "Enable JSON mode" recipes in
Anthropic's claude-cookbooks): we define a single tool whose input schema IS
the output format we want, then force Claude to call it with `tool_choice`.
Claude's reply therefore arrives as validated, structured JSON that the app
can render directly, with no fragile text parsing.

The long, static system prompt is marked with `cache_control` so repeated
calls (e.g. a batch report) benefit from prompt caching.
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # reads ANTHROPIC_API_KEY (and optional CLAUDE_MODEL) from .env

DEFAULT_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = """You are a senior customer-retention analyst at a telecom company.
You receive the output of a churn model for ONE customer: the predicted churn
probability, the portfolio base rate, the customer's profile, and the top SHAP
drivers (positive shap_log_odds = pushes toward churn, negative = toward staying).

Your job is to write a short, accurate retention brief for a non-technical
account manager.

Rules:
- Ground every statement in the provided numbers. Never invent data.
- Explain drivers in plain English (no jargon like "log-odds" or "SHAP").
- Recommend 2-3 concrete, realistic actions that target the drivers pushing
  toward churn. Do not recommend changing things the company cannot change
  (e.g. a customer's age).
- Remember that SHAP shows what the MODEL relies on, which is correlation, not
  proof of cause. Say so briefly in confidence_note when it matters.
- Always respond by calling the submit_retention_brief tool."""

RETENTION_TOOL: dict[str, Any] = {
    "name": "submit_retention_brief",
    "description": "Submit the structured retention brief for one customer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
            "headline": {
                "type": "string",
                "description": "One sentence an account manager can read in 5 seconds.",
            },
            "key_drivers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "feature": {"type": "string"},
                        "effect": {"type": "string", "enum": ["raises risk", "lowers risk"]},
                        "plain_english": {"type": "string"},
                    },
                    "required": ["feature", "effect", "plain_english"],
                },
            },
            "recommended_actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "why": {"type": "string"},
                        "priority": {"type": "integer", "minimum": 1, "maximum": 3},
                    },
                    "required": ["action", "why", "priority"],
                },
            },
            "confidence_note": {"type": "string"},
        },
        "required": [
            "risk_level",
            "headline",
            "key_drivers",
            "recommended_actions",
            "confidence_note",
        ],
    },
}


def risk_bucket(p: float) -> str:
    return "high" if p >= 0.6 else "medium" if p >= 0.3 else "low"


def build_messages(explanation: dict) -> list[dict]:
    return [
        {
            "role": "user",
            "content": "Model output for this customer:\n```json\n"
            + json.dumps(explanation, indent=2, default=str)
            + "\n```",
        }
    ]


def explain_with_claude(explanation: dict, client=None, model: str | None = None) -> dict:
    """Call Claude and return the structured brief as a dict.

    `client` can be injected (used by the tests with a fake client).
    """
    if client is None:
        import anthropic  # imported lazily so offline mode needs no key

        client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY from env

    response = client.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=1024,
        system=[
            {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
        ],
        tools=[RETENTION_TOOL],
        tool_choice={"type": "tool", "name": RETENTION_TOOL["name"]},
        messages=build_messages(explanation),
    )

    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == RETENTION_TOOL["name"]:
            brief = dict(block.input)
            brief["source"] = f"Claude ({response.model})"
            return brief
    raise RuntimeError("Claude did not return a retention brief tool call.")


# Plain-English templates for the offline fallback.
_OFFLINE_TEXT = {
    "tenure_months": ("is a relatively new customer", "has been loyal for a long time"),
    "contract": ("is on a flexible contract that is easy to cancel", "is locked into a longer contract"),
    "monthly_charges": ("pays a high monthly bill", "has an affordable monthly bill"),
    "support_tickets_6m": ("has contacted support often recently", "rarely needs support"),
    "payment_method": ("uses a payment method linked to higher churn", "uses a stable payment method"),
    "internet_service": ("has a service type linked to higher churn", "has a service type linked to loyalty"),
    "has_tech_support": ("does not have tech support", "has tech support"),
    "senior_citizen": ("is in a segment with higher churn", "is in a segment with lower churn"),
}
_OFFLINE_ACTIONS = {
    "contract": "Offer a discounted 12-month contract upgrade.",
    "monthly_charges": "Review the plan and offer a right-sized bundle or loyalty discount.",
    "support_tickets_6m": "Have a senior agent call to resolve open issues.",
    "tenure_months": "Enrol in the new-customer onboarding and check-in program.",
    "has_tech_support": "Offer 3 months of free tech support.",
    "payment_method": "Encourage switching to auto-pay with a small credit.",
    "internet_service": "Run a proactive service-quality check.",
}


def explain_offline(explanation: dict) -> dict:
    """Template-based brief so the app works without an API key (demo mode)."""
    p = explanation["churn_probability"]
    drivers = explanation["top_drivers"]
    key = [
        {
            "feature": d["feature"],
            "effect": "raises risk" if d["pushes_toward"] == "churn" else "lowers risk",
            "plain_english": "This customer "
            + _OFFLINE_TEXT.get(d["feature"], ("raises risk", "lowers risk"))[
                0 if d["pushes_toward"] == "churn" else 1
            ]
            + ".",
        }
        for d in drivers
    ]
    actions = [
        {"action": _OFFLINE_ACTIONS[d["feature"]], "why": f"Targets the '{d['feature']}' driver.", "priority": i + 1}
        for i, d in enumerate(x for x in drivers if x["pushes_toward"] == "churn" and x["feature"] in _OFFLINE_ACTIONS)
    ][:3]
    return {
        "risk_level": risk_bucket(p),
        "headline": f"{risk_bucket(p).title()} churn risk ({p:.0%} vs {explanation['portfolio_base_rate']:.0%} average).",
        "key_drivers": key,
        "recommended_actions": actions,
        "confidence_note": "Offline template output. Add an API key for Claude-written analysis.",
        "source": "Offline template (no API key)",
    }


def explain(explanation: dict, offline: bool | None = None, **kwargs) -> dict:
    """Use Claude when a key is configured, otherwise fall back to offline mode."""
    if offline is None:
        offline = not os.getenv("ANTHROPIC_API_KEY")
    return explain_offline(explanation) if offline else explain_with_claude(explanation, **kwargs)
