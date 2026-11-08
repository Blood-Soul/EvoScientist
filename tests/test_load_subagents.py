"""Tests for ``EvoScientist.utils.load_subagents``.

Focused on schema-validation paths that are easy to silently misuse from
yaml — primarily the ``async:`` flag type check that prevents quoted-string
or integer values from being misinterpreted as booleans.
"""

from __future__ import annotations

import textwrap

import pytest

from EvoScientist.utils import load_subagents, resolve_subagent_tools


def _write_yaml(tmp_path, name: str, body: str):
    """Write ``body`` to ``tmp_path/name`` and return the directory path."""
    (tmp_path / name).write_text(textwrap.dedent(body))
    return tmp_path


def test_async_flag_accepts_real_bool(tmp_path):
    """``async: true`` (real yaml boolean) is accepted and carried through."""
    config_path = _write_yaml(
        tmp_path,
        "writing.yaml",
        """
        writing-agent:
          description: Drafts reports
          system_prompt: ""
          tools: []
          async: true
        """,
    )
    subs = load_subagents(config_path)
    assert len(subs) == 1
    assert subs[0]["name"] == "writing-agent"
    assert subs[0]["_async"] is True


def test_async_flag_defaults_to_false_when_omitted(tmp_path):
    """No ``async:`` field → ``_async`` defaults to False."""
    config_path = _write_yaml(
        tmp_path,
        "planner.yaml",
        """
        planner-agent:
          description: Plans experiments
          system_prompt: ""
          tools: []
        """,
    )
    subs = load_subagents(config_path)
    assert subs[0]["_async"] is False


def test_tool_names_are_deferred_until_the_spec_is_selected(tmp_path, caplog):
    selected_tool = object()
    config_path = _write_yaml(
        tmp_path,
        "agents.yaml",
        """
        selected-agent:
          tools: [available]
        remote-agent:
          tools: [remote_only]
          async: true
        """,
    )

    subs = load_subagents(config_path)

    assert not caplog.records
    assert subs[0]["tools"] == []
    assert subs[0]["_tool_names"] == ["available"]
    resolve_subagent_tools(subs[0], {"available": selected_tool})
    assert subs[0]["tools"] == [selected_tool]
    assert "_tool_names" not in subs[0]


def test_resolve_subagent_tools_preserves_injected_tools():
    injected = object()
    selected_tool = object()
    subagent = {
        "name": "selected-agent",
        "tools": [injected],
        "_tool_names": ["available"],
    }

    resolve_subagent_tools(subagent, {"available": selected_tool})

    assert subagent["tools"] == [injected, selected_tool]
    assert "_tool_names" not in subagent


def test_agent_without_tools_preserves_parent_tool_inheritance(tmp_path):
    config_path = _write_yaml(
        tmp_path,
        "general.yaml",
        """
        general-purpose:
          description: Handles general tasks
          system_prompt: ""
        """,
    )

    subagent = load_subagents(config_path)[0]
    resolve_subagent_tools(subagent, {"available": object()})

    assert "tools" not in subagent
    assert "_tool_names" not in subagent


def test_async_flag_rejects_quoted_string(tmp_path):
    """``async: "false"`` (quoted) is a real user trap — bool("false") is True.

    Without the explicit isinstance check, this would silently flip the agent
    into async mode. We require the validator to fail loud instead.
    """
    config_path = _write_yaml(
        tmp_path,
        "bad.yaml",
        """
        bad-agent:
          description: ""
          system_prompt: ""
          tools: []
          async: "false"
        """,
    )
    with pytest.raises(ValueError, match=r"'async' must be a boolean"):
        load_subagents(config_path)


def test_async_flag_rejects_integer(tmp_path):
    """``async: 1`` is also rejected — yaml integers are not booleans."""
    config_path = _write_yaml(
        tmp_path,
        "bad.yaml",
        """
        bad-agent:
          description: ""
          system_prompt: ""
          tools: []
          async: 1
        """,
    )
    with pytest.raises(ValueError, match=r"'async' must be a boolean"):
        load_subagents(config_path)


def test_async_flag_error_includes_agent_name(tmp_path):
    """Error message must include the offending agent name for triage."""
    config_path = _write_yaml(
        tmp_path,
        "bad.yaml",
        """
        my-bad-agent:
          description: ""
          system_prompt: ""
          tools: []
          async: "yes"
        """,
    )
    with pytest.raises(ValueError, match=r"my-bad-agent"):
        load_subagents(config_path)


def test_non_dict_spec_raises(tmp_path):
    """Yaml entries that aren't mappings must fail loud, not be silently dropped.

    Previously ``_build_one`` had a ``if not isinstance(spec, dict): continue``
    fallback that swallowed malformed entries — users would see their agent
    quietly disappear with no error. Now caught during the merge loop.
    """
    config_path = _write_yaml(
        tmp_path,
        "bad.yaml",
        """
        bad-agent: 123
        """,
    )
    with pytest.raises(ValueError, match=r"must map to a spec dict"):
        load_subagents(config_path)


def test_non_dict_spec_error_includes_filename_and_name(tmp_path):
    """Error must surface BOTH the offending file path and agent name."""
    config_path = _write_yaml(
        tmp_path,
        "weird.yaml",
        """
        weird-agent: "just a string"
        """,
    )
    with pytest.raises(ValueError, match=r"weird\.yaml.*weird-agent"):
        load_subagents(config_path)


def test_missing_tool_on_sync_subagent_warns_at_resolution(tmp_path, caplog):
    """A selected sync spec warns when its terminal registry lacks a tool."""
    config_path = _write_yaml(
        tmp_path,
        "planner.yaml",
        """
        planner-agent:
          description: Plans experiments
          system_prompt: ""
          tools: [nonexistent_tool]
        """,
    )
    with caplog.at_level("DEBUG", logger="EvoScientist.utils"):
        subs = load_subagents(config_path)
    assert subs[0]["_async"] is False
    assert subs[0]["tools"] == []
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert not warnings

    resolve_subagent_tools(subs[0], {})
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("nonexistent_tool" in r.getMessage() for r in warnings)


def test_missing_tool_on_async_subagent_is_deferred_without_logging(tmp_path, caplog):
    """Async tool names do not emit warnings against the caller registry."""
    config_path = _write_yaml(
        tmp_path,
        "scheduler.yaml",
        """
        scheduler:
          description: Fires on cron
          system_prompt: ""
          tools: [nonexistent_tool]
          async: true
        """,
    )
    with caplog.at_level("DEBUG", logger="EvoScientist.utils"):
        subs = load_subagents(config_path)
    assert subs[0]["_async"] is True
    assert subs[0]["tools"] == []
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert not any("nonexistent_tool" in r.getMessage() for r in warnings)
    debugs = [r for r in caplog.records if r.levelname == "DEBUG"]
    assert not any("nonexistent_tool" in r.getMessage() for r in debugs)
    assert subs[0]["_tool_names"] == ["nonexistent_tool"]


def test_missing_async_tool_warns_at_terminal_resolution(tmp_path, caplog):
    """The selected async graph warns against its terminal registry."""
    config_path = _write_yaml(
        tmp_path,
        "scheduler.yaml",
        """
        scheduler:
          description: Fires on cron
          system_prompt: ""
          tools: [nonexistent_tool]
          async: true
        """,
    )
    with caplog.at_level("DEBUG", logger="EvoScientist.utils"):
        subs = load_subagents(config_path)
        resolve_subagent_tools(subs[0], {})
    assert subs[0]["_async"] is True
    assert subs[0]["tools"] == []
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("nonexistent_tool" in r.getMessage() for r in warnings)
    debugs = [r for r in caplog.records if r.levelname == "DEBUG"]
    assert not any(
        "nonexistent_tool" in r.getMessage() and "async graph" in r.getMessage()
        for r in debugs
    )


def test_a_switch_gated_tool_is_skipped_at_debug_not_warning(tmp_path, caplog):
    """A YAML may name a tool whose run switch is off; that is not a typo."""
    config_path = _write_yaml(
        tmp_path,
        "writer.yaml",
        """
        writer:
          description: Writes plans
          system_prompt: ""
          tools: [review_plan]
        """,
    )
    with caplog.at_level("DEBUG", logger="EvoScientist.utils"):
        subs = load_subagents(config_path)
        resolve_subagent_tools(subs[0], {})
    assert subs[0]["tools"] == []
    assert not any(r.levelname == "WARNING" for r in caplog.records)
    assert any(
        r.levelname == "DEBUG" and "review_plan" in r.getMessage()
        for r in caplog.records
    )


class _Tool:
    """Stands in for a registry tool: name resolution reads ``.name``."""

    def __init__(self, name: str):
        self.name = name


class TestPromptSections:
    """One YAML serves both arms: the prompt blocks follow the tools the run registered."""

    @staticmethod
    def _load(tmp_path, body: str) -> dict:
        return load_subagents(_write_yaml(tmp_path, "agent.yaml", body))[0]

    BODY = """
        writer:
          description: Writes plans
          system_prompt: |
            Before.
            {{review_plan}}
            After.
          tools: [think_tool, review_plan]
          prompt_sections:
            review_plan:
              requires: [review_plan]
              text: |
                Call review_plan once with the plan path.
        """

    def test_the_section_follows_its_tool(self, tmp_path):
        off = self._load(tmp_path, self.BODY)
        resolve_subagent_tools(off, {"think_tool": _Tool("think_tool")})
        assert "Call review_plan once" not in off["system_prompt"]
        assert "{{review_plan}}" not in off["system_prompt"]
        assert "Before." in off["system_prompt"]
        assert "After." in off["system_prompt"]
        assert "_prompt_sections" not in off

    def test_the_section_is_rendered_when_the_tool_resolves(self, tmp_path):
        on = self._load(tmp_path, self.BODY)
        resolve_subagent_tools(
            on,
            {"think_tool": _Tool("think_tool"), "review_plan": _Tool("review_plan")},
        )
        assert "Call review_plan once with the plan path." in on["system_prompt"]
        assert "{{review_plan}}" not in on["system_prompt"]
        assert "_prompt_sections" not in on

    def test_a_plain_string_requires_the_tool_named_by_its_key(self, tmp_path):
        body = """
            writer:
              system_prompt: |
                {{enqueue_paper_experiences}}
              tools: [enqueue_paper_experiences]
              prompt_sections:
                enqueue_paper_experiences: |
                  Queue the selected papers once.
            """
        off = self._load(tmp_path, body)
        resolve_subagent_tools(off, {"think_tool": _Tool("think_tool")})
        assert "Queue the selected papers" not in off["system_prompt"]
        # a fresh load, because resolution consumes the internal key
        (tmp_path / "agent.yaml").unlink()
        on = self._load(tmp_path, body)
        resolve_subagent_tools(
            on, {"enqueue_paper_experiences": _Tool("enqueue_paper_experiences")}
        )
        assert "Queue the selected papers once." in on["system_prompt"]

    def test_all_tools_in_requires_must_be_present(self, tmp_path):
        body = """
            writer:
              system_prompt: |
                {{both}}
              tools: [a]
              prompt_sections:
                both:
                  requires: [a, b]
                  text: |
                    Only when both are here.
            """
        sub = self._load(tmp_path, body)
        resolve_subagent_tools(sub, {"a": _Tool("a")})
        assert "Only when both are here." not in sub["system_prompt"]

    def test_an_unrendered_placeholder_fails_loud(self, tmp_path):
        body = """
            writer:
              system_prompt: |
                {{forgotten}}
              tools: []
            """
        sub = self._load(tmp_path, body)
        with pytest.raises(ValueError, match=r"unrendered section placeholder"):
            resolve_subagent_tools(sub, {})

    def test_prompt_sections_must_be_a_mapping(self, tmp_path):
        body = """
            writer:
              system_prompt: ""
              prompt_sections: [a, b]
            """
        with pytest.raises(ValueError, match=r"'prompt_sections' must be a mapping"):
            self._load(tmp_path, body)

    def test_a_section_text_must_be_a_string(self, tmp_path):
        body = """
            writer:
              system_prompt: ""
              prompt_sections:
                review_plan:
                  requires: [review_plan]
                  text: 3
            """
        with pytest.raises(ValueError, match=r"'text' must be a string"):
            self._load(tmp_path, body)
