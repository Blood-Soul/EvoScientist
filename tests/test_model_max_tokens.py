"""EVOSCIENTIST_MAX_TOKENS: an explicit output budget for every route.

A reasoning model behind an Anthropic-compatible gateway applied its own 4096
default and spent it all on reasoning, returning finish_reason max_tokens with no
answer. The knob lets a run raise that without changing the provider config.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from EvoScientist.llm.models import get_chat_model


@pytest.fixture(autouse=True)
def _gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVOSCIENTIST_MAX_TOKENS", raising=False)
    monkeypatch.setenv("CUSTOM_ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CUSTOM_ANTHROPIC_BASE_URL", "https://gateway.test/anthropic")


def _kwargs(**kwargs) -> dict:
    with patch("EvoScientist.llm.models.init_chat_model") as init:
        init.return_value = MagicMock()
        get_chat_model("deepseek-v4.1-flash", provider="custom-anthropic", **kwargs)
    return init.call_args[1]


def test_the_budget_reaches_the_model_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVOSCIENTIST_MAX_TOKENS", "32000")
    assert _kwargs()["max_tokens"] == 32000


def test_without_it_the_provider_default_stays() -> None:
    assert "max_tokens" not in _kwargs()


def test_an_explicit_caller_value_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVOSCIENTIST_MAX_TOKENS", "32000")
    assert _kwargs(max_tokens=1000)["max_tokens"] == 1000


@pytest.mark.parametrize("bad", ["lots", "0", "-5", " "])
def test_a_bad_value_is_ignored(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.setenv("EVOSCIENTIST_MAX_TOKENS", bad)
    assert "max_tokens" not in _kwargs()
