"""Tests run fully offline: the Claude client is replaced with a fake."""

from types import SimpleNamespace

import pytest

from src.data import make_customers
from src.explainer import RETENTION_TOOL, explain_offline, explain_with_claude
from src.model import ChurnModel


@pytest.fixture(scope="module")
def trained():
    df = make_customers(n=800, seed=1)
    return df, ChurnModel.train(df, seed=1)


def test_model_learns_signal(trained):
    _, model = trained
    assert model.test_auc > 0.7  # clearly better than random


def test_explanation_shape(trained):
    df, model = trained
    exp = model.explain(df.iloc[0], top_k=4)
    assert 0 <= exp["churn_probability"] <= 1
    assert len(exp["top_drivers"]) == 4
    # one-hot columns are merged back into readable feature names
    assert all("_Month" not in d["feature"] for d in exp["top_drivers"])


def test_offline_brief(trained):
    df, model = trained
    brief = explain_offline(model.explain(df.iloc[0]))
    assert brief["risk_level"] in {"low", "medium", "high"}
    assert brief["key_drivers"]


class FakeClient:
    """Mimics anthropic.Anthropic().messages.create and records the call."""

    def __init__(self):
        self.kwargs = None
        self.messages = self

    def create(self, **kwargs):
        self.kwargs = kwargs
        block = SimpleNamespace(
            type="tool_use",
            name=RETENTION_TOOL["name"],
            input={
                "risk_level": "high",
                "headline": "Test headline",
                "key_drivers": [],
                "recommended_actions": [],
                "confidence_note": "n/a",
            },
        )
        return SimpleNamespace(content=[block], model="fake-model")


def test_claude_call_is_forced_tool_use(trained):
    df, model = trained
    fake = FakeClient()
    brief = explain_with_claude(model.explain(df.iloc[0]), client=fake)
    assert brief["headline"] == "Test headline"
    assert fake.kwargs["tool_choice"] == {"type": "tool", "name": "submit_retention_brief"}
    assert fake.kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
