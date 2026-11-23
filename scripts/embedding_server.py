#!/usr/bin/env python3
"""A local OpenAI-compatible embedding server for Qwen3-Embedding (default: Qwen/Qwen3-Embedding-0.6B).

EvoScientist reaches it through ``EVOSCIENTIST_EMBEDDING_URL`` (see ``EvoScientist/memory/embedding.py``). It runs in
its own environment, so the agent's environment does not need torch:

    uv venv .venv-embed --python 3.12
    uv pip install --python .venv-embed/bin/python "torch>=2.6" "sentence-transformers>=5.1" "transformers>=4.51"
    .venv-embed/bin/python scripts/embedding_server.py --port 18765
    export EVOSCIENTIST_EMBEDDING_URL=http://127.0.0.1:18765/v1

The weights come from the Hugging Face cache (``HF_ENDPOINT=https://hf-mirror.com`` downloads them through the
mirror). The device is CUDA, then Apple MPS, then CPU, unless ``--device`` says otherwise.

Endpoints: ``POST /v1/embeddings`` (``{"model": ..., "input": str | [str]}``, OpenAI's reply shape), ``GET /health``.
The server embeds the text it is given as it is: the caller adds Qwen3's query instruction where it wants one, so the
same server serves queries and documents. Standard library HTTP only; one model, requests answered one at a time.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("embedding_server")


def pick_device(name: str) -> str:
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    def __init__(self, model: str, device: str, max_length: int, batch: int) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        kwargs = {"torch_dtype": dtype}
        if device == "cuda":
            try:
                import flash_attn  # noqa: F401

                kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                pass
        import inspect

        # Qwen3-Embedding pools the last token, so padding goes on the left; the keyword was renamed in
        # sentence-transformers 6 (tokenizer_kwargs -> processor_kwargs).
        pad = "processor_kwargs" if "processor_kwargs" in inspect.signature(SentenceTransformer).parameters \
            else "tokenizer_kwargs"
        self.model = SentenceTransformer(model, device=device, model_kwargs=kwargs, **{pad: {"padding_side": "left"}})
        self.model.max_seq_length = max_length
        self.name, self.device, self.batch = model, device, batch
        self.dim = self.model.get_embedding_dimension() if hasattr(self.model, "get_embedding_dimension") \
            else self.model.get_sentence_embedding_dimension()
        self.lock = threading.Lock()

    def encode(self, texts: list[str]) -> tuple[list[list[float]], int]:
        with self.lock:
            vecs = self.model.encode(texts, batch_size=self.batch, normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False)
            tokens = sum(len(ids) for ids in self.model.tokenizer(texts, truncation=True,
                                                                  max_length=self.model.max_seq_length)["input_ids"])
        return vecs.astype("float32").tolist(), tokens


def make_handler(embedder: Embedder):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt, *args):  # quieter than the default per-request line on stderr
            log.debug(fmt, *args)

        def do_GET(self):
            if self.path.rstrip("/") in ("/health", "/v1/health"):
                self._send(200, {"status": "ok", "model": embedder.name, "device": embedder.device,
                                 "dim": embedder.dim})
            elif self.path.rstrip("/") == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": embedder.name, "object": "model"}]})
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path.rstrip("/") not in ("/v1/embeddings", "/embeddings"):
                self._send(404, {"error": {"message": "not found"}})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                texts = body.get("input")
                if isinstance(texts, str):
                    texts = [texts]
                if not isinstance(texts, list) or not texts or not all(isinstance(t, str) for t in texts):
                    self._send(400, {"error": {"message": "`input` must be a string or a non-empty list of strings"}})
                    return
                started = time.monotonic()
                vecs, tokens = embedder.encode(texts)
                log.info("embedded %d text(s), %d tokens, %.2fs", len(texts), tokens, time.monotonic() - started)
                self._send(200, {
                    "object": "list",
                    "model": embedder.name,
                    "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)],
                    "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
                })
            except Exception as exc:  # one bad request must not stop the server
                log.exception("embedding failed")
                self._send(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})

    return Handler


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18765)
    ap.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    ap.add_argument("--max-length", type=int, default=8192, help="tokens per text; longer text is truncated")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = pick_device(args.device)
    log.info("loading %s on %s", args.model, device)
    embedder = Embedder(args.model, device, args.max_length, args.batch)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(embedder))
    log.info("serving %s (dim %d) at http://%s:%d/v1", args.model, embedder.dim, args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
