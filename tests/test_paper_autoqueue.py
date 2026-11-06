"""The auto-queue middleware: papers a search returns are queued without the agent asking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage

from EvoScientist.middleware.paper_autoqueue import (
    PaperAutoQueueMiddleware,
    papers_from_pool,
)

SEARCH = (
    "python /skills/paper-navigator/scripts/scholar_search.py --query 'x' "
    "--limit 8 --output ./work/pool.jsonl --append"
)


def _row(arxiv: str | None, title: str, paper_id: str | None = None) -> str:
    row: dict[str, Any] = {"title": title, "paperId": paper_id or f"arxiv:{arxiv}"}
    if arxiv:
        row["externalIds"] = {"ArXiv": arxiv}
    return json.dumps(row)


def _write_pool(root: Path, rows: list[str]) -> None:
    pool = root / "work" / "pool.jsonl"
    pool.parent.mkdir(parents=True, exist_ok=True)
    pool.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _request(command: str, name: str = "execute") -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": {"command": command}, "id": "c1"},
        tool=None,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )


def _pending(memory: Path) -> list[Path]:
    return sorted(
        (memory / "paper_experience_queue" / "projects" / "P-test" / "pending").glob(
            "*.json"
        )
    )


class _Drains:
    """A stand-in for the worker: takes the pending files and says how many it handled."""

    def __init__(self, memory: Path | None = None) -> None:
        self.calls = 0
        self.memory = memory
        self.taken: list[str] = []

    async def __call__(self) -> dict[str, int]:
        self.calls += 1
        files = _pending(self.memory) if self.memory else []
        for file in files:
            # like the real worker: a finished task moves to `completed`, which is
            # what keeps the same paper from being queued again
            done = file.parent.parent / "completed" / file.name
            done.parent.mkdir(parents=True, exist_ok=True)
            file.replace(done)
            self.taken.append(file.name)
        return {"processed": len(files), "failed": 0}


def _middleware(tmp_path: Path, drains: _Drains, **over: Any) -> PaperAutoQueueMiddleware:
    return PaperAutoQueueMiddleware(
        memory_dir=tmp_path / "mem",
        project_id="P-test",
        workspace_root=tmp_path,
        drain=drains,
        **over,
    )


async def _call(mw: PaperAutoQueueMiddleware, request: ToolCallRequest, output: str = "ok"):
    async def handler(_req: ToolCallRequest) -> ToolMessage:
        return ToolMessage(content=output, tool_call_id="c1")

    result = await mw.awrap_tool_call(request, handler)
    if mw._drain_task is not None:
        await mw._drain_task
    return result


class TestPoolParsing:
    def test_reads_arxiv_ids_and_titles(self, tmp_path: Path) -> None:
        _write_pool(tmp_path, [_row("1806.03822", "SQuAD 2.0"), _row("2001.09694v3", "Retro")])
        found = papers_from_pool(tmp_path / "work" / "pool.jsonl")
        assert found == {"1806.03822": "SQuAD 2.0", "2001.09694": "Retro"}

    def test_skips_rows_without_an_arxiv_id_and_bad_lines(self, tmp_path: Path) -> None:
        _write_pool(
            tmp_path,
            [
                _row(None, "From S2", paper_id="a1b2c3d4"),
                "not json",
                json.dumps([1, 2]),
                _row("1810.04805", "BERT"),
            ],
        )
        assert papers_from_pool(tmp_path / "work" / "pool.jsonl") == {"1810.04805": "BERT"}

    def test_missing_file_is_empty(self, tmp_path: Path) -> None:
        assert papers_from_pool(tmp_path / "nope.jsonl") == {}


class TestMiddleware:
    @pytest.mark.asyncio
    async def test_a_search_queues_its_papers_and_starts_a_drain(self, tmp_path: Path) -> None:
        _write_pool(tmp_path, [_row("1806.03822", "SQuAD 2.0"), _row("1810.04805", "BERT")])
        drains = _Drains(tmp_path / "mem")
        mw = _middleware(tmp_path, drains)
        result = await _call(mw, _request(SEARCH))
        assert isinstance(result, ToolMessage) and result.content == "ok"
        assert len(drains.taken) == 2
        assert mw.queued == 2 and drains.calls == 1

    @pytest.mark.asyncio
    async def test_the_same_paper_twice_is_queued_once(self, tmp_path: Path) -> None:
        _write_pool(tmp_path, [_row("1806.03822", "SQuAD 2.0")])
        drains = _Drains(tmp_path / "mem")
        mw = _middleware(tmp_path, drains)
        await _call(mw, _request(SEARCH))
        await _call(mw, _request(SEARCH))
        assert len(drains.taken) == 1
        assert mw.queued == 1

    @pytest.mark.asyncio
    async def test_other_commands_and_tools_are_left_alone(self, tmp_path: Path) -> None:
        _write_pool(tmp_path, [_row("1806.03822", "SQuAD 2.0")])
        drains = _Drains(tmp_path / "mem")
        mw = _middleware(tmp_path, drains)
        await _call(mw, _request("python train.py --output ./work/pool.jsonl"))
        await _call(mw, _request(SEARCH, name="read_file"))
        assert _pending(tmp_path / "mem") == []
        assert drains.calls == 0

    @pytest.mark.asyncio
    async def test_ids_printed_in_the_result_count_when_there_is_no_pool_file(
        self, tmp_path: Path
    ) -> None:
        drains = _Drains(tmp_path / "mem")
        mw = _middleware(tmp_path, drains)
        command = "python /skills/paper-navigator/scripts/scholar_search.py --query x"
        await _call(mw, _request(command), output="arxiv:2004.04906 | 2020 | Dense Passage")
        assert len(drains.taken) == 1

    @pytest.mark.asyncio
    async def test_the_cap_limits_how_many_papers_a_run_queues(self, tmp_path: Path) -> None:
        _write_pool(tmp_path, [_row(f"1900.{i:05d}", f"p{i}") for i in range(10)])
        drains = _Drains(tmp_path / "mem")
        mw = _middleware(tmp_path, drains, max_papers=4)
        await _call(mw, _request(SEARCH))
        assert len(drains.taken) == 4 and mw.queued == 4

    @pytest.mark.asyncio
    async def test_a_failing_drain_never_fails_the_search(self, tmp_path: Path) -> None:
        _write_pool(tmp_path, [_row("1806.03822", "SQuAD 2.0")])

        async def boom() -> None:
            raise RuntimeError("gateway down")

        mw = PaperAutoQueueMiddleware(
            memory_dir=tmp_path / "mem",
            project_id="P-test",
            workspace_root=tmp_path,
            drain=boom,
        )
        result = await _call(mw, _request(SEARCH))
        assert isinstance(result, ToolMessage) and result.content == "ok"

    @pytest.mark.asyncio
    async def test_papers_queued_during_a_drain_get_a_second_pass(self, tmp_path: Path) -> None:
        _write_pool(tmp_path, [_row("1806.03822", "SQuAD 2.0")])
        memory = tmp_path / "mem"
        seen: list[int] = []
        mw: PaperAutoQueueMiddleware

        async def drain() -> dict[str, int]:
            files = _pending(memory)
            seen.append(len(files))
            if len(seen) == 1:
                # a paper arrives while the first drain is running
                mw._enqueue({"1810.04805": "BERT"})
            for file in files:
                file.unlink()
            return {"processed": len(files), "failed": 0}

        mw = PaperAutoQueueMiddleware(
            memory_dir=memory, project_id="P-test", workspace_root=tmp_path, drain=drain
        )
        await _call(mw, _request(SEARCH))
        assert seen == [1, 1]
        assert _pending(memory) == []

    @pytest.mark.asyncio
    async def test_a_drain_that_could_take_nothing_is_not_repeated(self, tmp_path: Path) -> None:
        """Another worker holds the queue lock: the drain returns at once, files stay pending."""
        _write_pool(tmp_path, [_row("1806.03822", "SQuAD 2.0")])
        calls = 0

        async def locked_out() -> dict[str, int]:
            nonlocal calls
            calls += 1
            return {"processed": 0, "failed": 0}

        mw = PaperAutoQueueMiddleware(
            memory_dir=tmp_path / "mem",
            project_id="P-test",
            workspace_root=tmp_path,
            drain=locked_out,
        )
        await _call(mw, _request(SEARCH))
        assert calls == 1
        assert len(_pending(tmp_path / "mem")) == 1


def test_the_switch_is_off_by_default_and_mapped() -> None:
    from EvoScientist.config.settings import _ENV_MAPPINGS, EvoScientistConfig

    assert EvoScientistConfig().memory_paper_autoqueue_enabled is False
    assert "memory_paper_autoqueue_enabled" in _ENV_MAPPINGS
