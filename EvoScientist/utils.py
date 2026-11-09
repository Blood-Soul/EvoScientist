"""Utility functions for EvoScientist.

This module primarily contains helpers for displaying messages and prompts in
notebooks, and lightweight configuration loaders used by the agent runtime.
"""

import json
import logging
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

logger = logging.getLogger(__name__)
console = Console()

# Tools whose registration depends on a run switch: the same YAML serves arms
# that turn the experience / paper pipeline on and off. A YAML that lists one
# is not making a typo when the switch is off, so the skip is expected and is
# logged at DEBUG; anything else not in the registry is still a warning.
OPTIONAL_SUBAGENT_TOOLS = frozenset(
    {
        "search_experience",
        "list_experience",
        "search_paper_text",
        "read_paper",
        "extract_paper_experiences",
        "enqueue_paper_experiences",
        "review_plan",
    }
)


def _render_prompt_sections(
    subagent: dict[str, Any], tools: list[Any]
) -> dict[str, Any]:
    """Fill the spec's ``{{section}}`` placeholders from ``prompt_sections``.

    A section is kept when every tool its ``requires`` entry names is in the
    resolved tool list (default: the section's own key); otherwise the
    placeholder becomes an empty string. This is how one YAML serves both
    arms: the prompt a run actually gets is assembled from the tools that run
    registered, rather than carrying conditional wording the model has to
    interpret.

    ``prompt_sections`` may map a section to a plain string (then the section
    key is the required tool) or to ``{requires: [...], text: "..."}``.
    """
    sections = subagent.pop("_prompt_sections", None) or {}
    available = {
        tool if isinstance(tool, str) else getattr(tool, "name", "")
        for tool in tools
    }
    prompt = str(subagent.get("system_prompt") or "")
    for name, section in sections.items():
        if isinstance(section, str):
            text, requires = section, [name]
        else:
            text = str(section.get("text") or "")
            requires = [str(r) for r in (section.get("requires") or [name])]
        keep = all(required in available for required in requires)
        prompt = prompt.replace("{{" + name + "}}", text if keep else "")

    if "{{" in prompt:
        raise ValueError(
            f"Subagent {subagent.get('name')!r}: system_prompt still has an "
            "unrendered section placeholder; every {{section}} needs an entry "
            "in prompt_sections"
        )
    subagent["system_prompt"] = prompt
    return subagent


def resolve_subagent_tools(
    subagent: dict[str, Any], tool_registry: dict[str, Any]
) -> dict[str, Any]:
    """Resolve deferred YAML tool names while preserving injected tool objects."""
    tool_names = subagent.pop("_tool_names", None)
    if tool_names is None:
        return _render_prompt_sections(subagent, list(subagent.get("tools", [])))

    resolved = list(subagent.get("tools", []))
    for tool_name in tool_names:
        if tool_name in tool_registry:
            resolved.append(tool_registry[tool_name])
        elif tool_name in OPTIONAL_SUBAGENT_TOOLS:
            logger.debug(
                "Subagent %r: optional tool %r is off in this run, skipping",
                subagent.get("name"),
                tool_name,
            )
        else:
            logger.warning(
                "Subagent %r: tool %r not in registry, skipping",
                subagent.get("name"),
                tool_name,
            )
    subagent["tools"] = resolved
    return _render_prompt_sections(subagent, resolved)


def format_message_content(message):
    """Convert message content to displayable string.

    Args:
        message: A LangChain message object with content attribute.

    Returns:
        Formatted string representation of the message content.
    """
    parts = []
    tool_calls_processed = False

    # Handle main content
    if isinstance(message.content, str):
        parts.append(message.content)
    elif isinstance(message.content, list):
        # Handle complex content like tool calls (Anthropic format)
        for item in message.content:
            if item.get("type") == "text":
                parts.append(item["text"])
            elif item.get("type") == "tool_use":
                parts.append(f"\n🔧 Tool Call: {item['name']}")
                parts.append(f"   Args: {json.dumps(item['input'], indent=2)}")
                parts.append(f"   ID: {item.get('id', 'N/A')}")
                tool_calls_processed = True
    else:
        parts.append(str(message.content))

    # Handle tool calls attached to the message (OpenAI format) - only if not already processed
    if (
        not tool_calls_processed
        and hasattr(message, "tool_calls")
        and message.tool_calls
    ):
        for tool_call in message.tool_calls:
            parts.append(f"\n🔧 Tool Call: {tool_call['name']}")
            parts.append(f"   Args: {json.dumps(tool_call['args'], indent=2)}")
            parts.append(f"   ID: {tool_call['id']}")

    return "\n".join(parts)


def format_messages(messages):
    """Format and display a list of messages with Rich formatting.

    Args:
        messages: List of LangChain message objects to display.
    """
    for m in messages:
        msg_type = m.__class__.__name__.replace("Message", "")
        content = format_message_content(m)

        if msg_type == "Human":
            console.print(Panel(content, title="🧑 Human", border_style="blue"))
        elif msg_type == "Ai":
            console.print(Panel(content, title="🤖 Assistant", border_style="green"))
        elif msg_type == "Tool":
            console.print(Panel(content, title="🔧 Tool Output", border_style="yellow"))
        else:
            console.print(Panel(content, title=f"📝 {msg_type}", border_style="white"))


def show_prompt(prompt_text: str, title: str = "Prompt", border_style: str = "blue"):
    """Display a prompt with rich formatting and XML tag highlighting.

    Args:
        prompt_text: The prompt string to display
        title: Title for the panel (default: "Prompt")
        border_style: Border color style (default: "blue")
    """
    # Create a formatted display of the prompt
    formatted_text = Text(prompt_text)
    formatted_text.highlight_regex(r"<[^>]+>", style="bold blue")  # Highlight XML tags
    formatted_text.highlight_regex(
        r"##[^#\n]+", style="bold magenta"
    )  # Highlight headers
    formatted_text.highlight_regex(
        r"###[^#\n]+", style="bold cyan"
    )  # Highlight sub-headers

    # Display in a panel for better presentation
    console.print(
        Panel(
            formatted_text,
            title=f"[bold green]{title}[/bold green]",
            border_style=border_style,
            padding=(1, 2),
        )
    )


def load_subagents(
    config_path: Path,
    *,
    prompt_refs: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Load subagent definitions from a directory of YAML files.

    NOTE: This is a custom utility. deepagents does not natively load subagents
    from files - they're normally defined inline in the create_deep_agent() call.
    We externalize to YAML here to keep configuration separate from code.

    ``config_path`` must be a directory containing one ``<name>.yaml`` per
    sub-agent. All ``*.yaml`` files are merged into a single mapping. Files
    starting with ``.`` (dotfiles, editor swap files) or ``_`` (private /
    disabled) are ignored. ``.yml`` is intentionally not supported — keeps
    one canonical extension and avoids the dev-vs-wheel packaging mismatch.

    Each file's top level must be a mapping ``{<agent-name>: <spec>}``::

        planner-agent:
          description: "..."
          tools: [think_tool]
          system_prompt: |
            ...
        research-agent:
          description: "..."
          tools: [tavily_search, think_tool]
          system_prompt: |
            ...

    Tool names are retained until the caller selects an execution mode. This
    avoids resolving async specs against the in-process registry; the selected
    in-process or remote graph resolves names against its terminal registry.
    """
    prompt_refs = prompt_refs or {}

    if not config_path.is_dir():
        raise ValueError(
            f"{config_path}: sub-agent config must be a directory "
            f"containing one <name>.yaml per agent"
        )

    # Only ``.yaml`` is supported (canonical extension). Skip files starting
    # with ``_`` (private / disabled) or ``.`` (dotfiles, editor swap files
    # like ``.foo.yaml.swp``). ``.yml`` is intentionally not loaded — keeps
    # one canonical extension across the project, simplifies packaging
    # (no need for a parallel ``subagents/*.yml`` entry in ``pyproject.toml``
    # ``package-data``), and matches every existing yaml file in this repo.
    config: dict[str, Any] = {}
    for yml in sorted(config_path.glob("*.yaml")):
        if yml.name.startswith(".") or yml.name.startswith("_"):
            continue
        with yml.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            # Empty file — skip silently (matches dotfile/underscore skip behavior)
            continue
        if not isinstance(data, dict):
            raise ValueError(
                f"{yml}: top-level must be a mapping (one entry per sub-agent)"
            )
        # Detect duplicate keys across files
        for key in data:
            if key in config:
                raise ValueError(
                    f"Sub-agent {key!r} defined in multiple files; "
                    f"second occurrence in {yml.name}"
                )
        for key, spec in data.items():
            if not isinstance(spec, dict):
                raise ValueError(
                    f"{yml}: sub-agent {key!r} must map to a spec dict, "
                    f"got {type(spec).__name__}"
                )
        config.update(data)

    if not config:
        raise ValueError(
            f"{config_path}: no sub-agent definitions found "
            f"(expected one or more <name>.yaml files)"
        )

    subagents: list[dict[str, Any]] = []

    def _build_one(name: str, spec: dict[str, Any]) -> dict[str, Any]:
        subagent: dict[str, Any] = {
            "name": name,
            "description": spec.get("description", ""),
        }

        if "system_prompt_ref" in spec:
            ref = spec["system_prompt_ref"]
            if ref not in prompt_refs:
                raise ValueError(
                    f"Unknown system_prompt_ref '{ref}' for subagent '{name}'"
                )
            subagent["system_prompt"] = prompt_refs[ref]
        else:
            subagent["system_prompt"] = spec.get("system_prompt", "")

        if "model" in spec:
            subagent["model"] = spec["model"]

        if "skills" in spec:
            subagent["skills"] = spec["skills"]

        # Conditional prompt blocks: rendered by ``resolve_subagent_tools``
        # against the tools this run actually registered. Kept internal (an
        # underscore key) until then, like ``_tool_names``.
        if "prompt_sections" in spec:
            sections = spec["prompt_sections"]
            if not isinstance(sections, dict):
                raise ValueError(
                    f"Subagent {name!r}: 'prompt_sections' must be a mapping of "
                    f"section name to text, got {type(sections).__name__}"
                )
            for section_name, section in sections.items():
                if not isinstance(section, (str, dict)):
                    raise ValueError(
                        f"Subagent {name!r}: prompt section {section_name!r} must be "
                        f"text or a {{requires, text}} mapping, got "
                        f"{type(section).__name__}"
                    )
                if isinstance(section, dict) and not isinstance(
                    section.get("text", ""), str
                ):
                    raise ValueError(
                        f"Subagent {name!r}: prompt section {section_name!r} "
                        "'text' must be a string"
                    )
            subagent["_prompt_sections"] = sections

        # Defer YAML-name resolution until the caller decides whether this
        # spec will run in-process. Async specs are replaced by remote graph
        # references and must not warn against the caller's unrelated registry.
        if "tools" in spec:
            subagent["_tool_names"] = list(spec["tools"])
            subagent["tools"] = []

        # Internal field: carries the ``async:`` yaml flag through to
        # ``_maybe_swap_async_subagents`` so the swap doesn't need a second
        # yaml pass to discover async-flagged agents. Underscore prefix marks
        # it as internal — must be popped before passing to deepagents.
        async_val = spec.get("async", False)
        if not isinstance(async_val, bool):
            # Reject quoted-string yaml values like ``async: "false"`` —
            # ``bool("false")`` is ``True`` (non-empty string), which silently
            # flips the agent into async mode. Fail loud instead.
            raise ValueError(
                f"Subagent {name!r}: 'async' must be a boolean, "
                f"got {type(async_val).__name__}: {async_val!r}"
            )

        subagent["_async"] = async_val

        return subagent

    for name, spec in config.items():
        subagents.append(_build_one(name, spec))

    return subagents


def load_subagent(
    config_path: Path,
    name: str,
    *,
    tool_registry: dict[str, Any],
    prompt_refs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Load and resolve a single sub-agent by name from YAML."""
    for agent in load_subagents(
        config_path,
        prompt_refs=prompt_refs,
    ):
        if agent.get("name") == name:
            return resolve_subagent_tools(agent, tool_registry)
    raise KeyError(f"Sub-agent not found: {name}")
