from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .model_io import load_model_and_tokenizer
from .tracer import FlashTrace


def parse_span(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    parts = str(value).split(":")
    if len(parts) != 2:
        raise ValueError("Span must use START:END format.")
    try:
        start = int(parts[0])
        end = int(parts[1])
    except ValueError as exc:
        raise ValueError("Span bounds must be integers.") from exc
    if start < 0 or end < start:
        raise ValueError("Span must satisfy 0 <= START <= END.")
    return start, end


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flashtrace", description="Trace language model outputs with FlashTrace.")
    sub = parser.add_subparsers(dest="command")

    trace = sub.add_parser("trace", help="Run attribution for a prompt and target.")
    trace.add_argument("--model", required=True, help="Hugging Face model id or local path.")
    trace.add_argument("--prompt", required=True, help="UTF-8 text file containing the prompt.")
    trace.add_argument("--target", help="UTF-8 text file containing the target response.")
    trace.add_argument("--output-span", help="Inclusive generation-token span START:END.")
    trace.add_argument("--reasoning-span", help="Inclusive generation-token span START:END.")
    trace.add_argument("--hops", type=int, default=1)
    trace.add_argument("--method", default="flashtrace", choices=["flashtrace", "ifr-span", "ifr-matrix"])
    trace.add_argument("--html", help="Write standalone HTML heatmap.")
    trace.add_argument("--json", help="Write JSON trace.")
    trace.add_argument("--device-map", default="auto")
    trace.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    trace.add_argument("--chunk-tokens", type=int, default=128)
    trace.add_argument("--sink-chunk-tokens", type=int, default=32)
    trace.add_argument("--recompute-attention", action="store_true")
    trace.add_argument("--use-chat-template", action="store_true", help="Format prompts with the tokenizer chat template.")
    return parser


def _read_text(path: str | None) -> str | None:
    if path is None:
        return None
    return Path(path).read_text(encoding="utf-8")


def _run_trace(args: argparse.Namespace) -> int:
    model, tokenizer = load_model_and_tokenizer(args.model, device_map=args.device_map, dtype=args.dtype)
    tracer = FlashTrace(
        model,
        tokenizer,
        chunk_tokens=args.chunk_tokens,
        sink_chunk_tokens=args.sink_chunk_tokens,
        recompute_attention=args.recompute_attention,
        use_chat_template=args.use_chat_template,
    )
    result = tracer.trace(
        prompt=_read_text(args.prompt) or "",
        target=_read_text(args.target),
        output_span=parse_span(args.output_span),
        reasoning_span=parse_span(args.reasoning_span),
        hops=args.hops,
        method=args.method,
    )
    for item in result.topk_inputs(20):
        print(f"{item.index}\t{item.score:.6f}\t{item.token!r}")
    if args.json:
        result.to_json(args.json)
    if args.html:
        result.to_html(args.html)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "trace":
        return _run_trace(args)
    parser.print_help()
    return 0
