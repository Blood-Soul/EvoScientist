"""Housekeeping around a node: stray processes, a GPU log, an event log.

None of this decides anything. It exists because a node's sub-agent can leave
a training job running (`setsid nohup ...` leaves the agent's process group),
and that job would keep the GPU after the node has been closed; and because a
run is easier to diagnose afterwards when the clock, the GPU use and every
tree action were written down as they happened.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

GPU_INTERVAL_SECONDS = 60.0
_gpu_threads: dict[str, threading.Event] = {}
_gpu_lock = threading.Lock()


def processes_under(path: Path) -> list[int]:
    """Pids whose working directory is `path` or inside it (Linux only; empty elsewhere)."""
    root = str(Path(path).resolve())
    proc = Path("/proc")
    if not proc.is_dir():
        return []
    me = os.getpid()
    found = []
    for d in proc.iterdir():
        if not d.name.isdigit() or int(d.name) == me:
            continue
        try:
            cwd = os.readlink(d / "cwd")
        except OSError:
            continue
        if cwd == root or cwd.startswith(root + os.sep):
            found.append(int(d.name))
    return found


def kill_processes_under(path: Path, grace: float = 10.0) -> list[int]:
    """Stop every process working inside `path`: SIGTERM, then SIGKILL after `grace` seconds.

    Returns the pids that were signalled.
    """
    victims = processes_under(path)
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 2.0)):
        alive = []
        for pid in victims:
            try:
                os.kill(pid, sig)
                alive.append(pid)
            except (ProcessLookupError, PermissionError):
                pass
        deadline = time.time() + wait
        while alive and time.time() < deadline:
            time.sleep(0.2)
            alive = [p for p in alive if _is_running(p)]
        if not alive:
            break
    return victims


def _is_running(pid: int) -> bool:
    try:
        # A zombie still answers kill(0); its /proc state is "Z".
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        return state != "Z"
    except (OSError, IndexError):
        return False


def start_gpu_log(path: Path, interval: float = GPU_INTERVAL_SECONDS) -> bool:
    """Append one GPU utilisation/memory sample per `interval` seconds to `path`.

    Runs on a daemon thread, once per file. Returns False when there is no
    nvidia-smi or the log is already running. For diagnosis only.
    """
    path = Path(path)
    if not shutil.which("nvidia-smi"):
        return False
    with _gpu_lock:
        if str(path) in _gpu_threads:
            return False
        stop = threading.Event()
        _gpu_threads[str(path)] = stop
    t0 = time.time()

    def loop() -> None:
        while not stop.is_set():
            try:
                p = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                with open(path, "a") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "ts": round(time.time(), 3),
                                "t": round(time.time() - t0),
                                "gpu": p.stdout.strip(),
                            }
                        )
                        + "\n"
                    )
            except (OSError, subprocess.SubprocessError):
                pass
            stop.wait(interval)

    threading.Thread(target=loop, daemon=True, name="research-tree-gpu").start()
    return True


_tickers: dict[str, threading.Event] = {}


def start_ticker(key: str, fn, interval: float) -> bool:
    """Call `fn()` every `interval` seconds on a daemon thread, once per `key`.

    `fn` returns False to end the ticker; an exception from it is swallowed so
    a bad tick never kills the run.
    """
    with _gpu_lock:
        if key in _tickers:
            return False
        stop = threading.Event()
        _tickers[key] = stop

    def loop() -> None:
        while not stop.wait(interval):
            try:
                if fn() is False:
                    break
            except Exception:  # housekeeping must not raise
                pass
        with _gpu_lock:
            _tickers.pop(key, None)

    threading.Thread(target=loop, daemon=True, name="research-tree-watch").start()
    return True


def stop_ticker(key: str) -> None:
    with _gpu_lock:
        stop = _tickers.get(key)
    if stop:
        stop.set()


def stop_gpu_log(path: Path) -> None:
    with _gpu_lock:
        stop = _gpu_threads.pop(str(Path(path)), None)
    if stop:
        stop.set()


def log_event(path: Path, started: float, now: float, event: str, **data) -> None:
    """Append a tree action to the event log. Never raises: the log must not break a run."""
    try:
        with open(path, "a") as fh:
            fh.write(
                json.dumps(
                    {
                        "ts": round(now, 3),
                        "t": round(now - started, 1),
                        "event": event,
                        **data,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except OSError:
        pass


def _plain(value):
    """JSON-safe copy of tool arguments (pydantic models, paths, anything else)."""
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def journal_call(
    path: Path, tool: str, args: dict, result: str, started: float, finished: float
) -> None:
    """Append one tool call, complete, to the call journal.

    Refusals are recorded too (`ok` false): what the agent tried and was told
    no is part of the research process, and git never sees it. Never raises.
    """
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as fh:
            fh.write(
                json.dumps(
                    {
                        "ts": round(started, 3),
                        "seconds": round(finished - started, 2),
                        "tool": tool,
                        "args": args,
                        "ok": not str(result).startswith("ERROR:"),
                        "result": result,
                    },
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )
    except OSError:
        pass


TRACE_SUFFIXES = (
    ".log",
    ".txt",
    ".out",
    ".md",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".csv",
)
TRACE_FILE_BYTES = 2_000_000  # a bigger text file keeps its last TRACE_TAIL_BYTES
TRACE_TAIL_BYTES = 1_000_000
TRACE_TOTAL_BYTES = 40_000_000
HASH_LIMIT_BYTES = 20_000_000
MANIFEST_LIMIT = 5000


def snapshot_node(node_dir: Path, dest: Path, skip: tuple[str, ...] = ()) -> dict:
    """Keep what a node produced that git does not: its logs and a file manifest.

    Event commits leave out data, weights, `outputs/` and `*.log`, so after the
    run the training logs would exist only inside the node's directory. This
    copies small text outputs (logs, metrics, notes) to `dest/traces/` and
    writes `dest/files.json`: every file with its size, mtime and, when it is
    not huge, a SHA-256, so a reader can tell what the node had on disk even
    for the weights that were not kept. Directories in `skip` (the shared data
    links) and `.git` are not walked. Never raises.
    """
    node_dir, dest = Path(node_dir), Path(dest)
    manifest: list[dict] = []
    kept = kept_bytes = 0
    try:
        for dirpath, dirnames, filenames in os.walk(node_dir):
            rel_dir = Path(dirpath).relative_to(node_dir)
            dirnames[:] = sorted(
                d
                for d in dirnames
                if d != ".git" and not (rel_dir == Path(".") and d in skip)
            )
            for name in sorted(filenames):
                path = Path(dirpath) / name
                rel = str(rel_dir / name) if str(rel_dir) != "." else name
                try:
                    st = path.stat()
                except OSError:
                    continue
                entry = {
                    "path": rel,
                    "bytes": st.st_size,
                    "mtime": round(st.st_mtime, 3),
                }
                if st.st_size <= HASH_LIMIT_BYTES:
                    try:
                        entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                    except OSError:
                        pass
                if (
                    name.endswith(TRACE_SUFFIXES)
                    and st.st_size
                    and kept_bytes < TRACE_TOTAL_BYTES
                ):
                    target = dest / "traces" / rel
                    try:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if st.st_size <= TRACE_FILE_BYTES:
                            shutil.copy2(path, target)
                            entry["kept"] = "full"
                        else:
                            with open(path, "rb") as src:
                                src.seek(st.st_size - TRACE_TAIL_BYTES)
                                target.write_bytes(src.read())
                            entry["kept"] = f"last {TRACE_TAIL_BYTES} bytes"
                        kept += 1
                        kept_bytes += target.stat().st_size
                    except OSError:
                        pass
                if len(manifest) < MANIFEST_LIMIT:
                    manifest.append(entry)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "files.json").write_text(
            json.dumps(
                {
                    "node_dir": str(node_dir),
                    "files": manifest,
                    "truncated": len(manifest) >= MANIFEST_LIMIT,
                },
                indent=1,
            )
        )
    except OSError:
        pass
    return {"files": len(manifest), "kept": kept, "kept_bytes": kept_bytes}
