#!/usr/bin/env python3
"""
Multi-backend disagg proxy for N-prefill + M-decode configurations.

Round-robins requests across multiple prefill/decode backend pairs.
Each prefill-decode pair has its own KV port pair for NCCL transfer.

Usage:
    python disagg_multi_proxy.py --port 9000 \
        --prefill-urls http://localhost:9100,http://localhost:9101 \
        --decode-urls http://localhost:9200,http://localhost:9201 \
        --prefill-kv-ports 14579,14581 \
        --decode-kv-ports 14580,14582
"""

import argparse
import asyncio
import itertools
import logging
import os
import uuid
from urllib.parse import urlparse

import aiohttp
from quart import Quart, Response, make_response, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-backend P/D disaggregation proxy"
    )
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--timeout", type=float, default=6 * 60 * 60)
    parser.add_argument(
        "--prefill-urls", type=str, required=True,
        help="Comma-separated prefill service URLs"
    )
    parser.add_argument(
        "--decode-urls", type=str, required=True,
        help="Comma-separated decode service URLs"
    )
    parser.add_argument(
        "--prefill-kv-ports", type=str, required=True,
        help="Comma-separated prefill KV ports"
    )
    parser.add_argument(
        "--decode-kv-ports", type=str, required=True,
        help="Comma-separated decode KV ports"
    )
    parser.add_argument("--kv-host", type=str, default="localhost")
    return parser.parse_args()


def main():
    args = parse_args()

    prefill_urls = [u.strip().rstrip("/") for u in args.prefill_urls.split(",")]
    decode_urls = [u.strip().rstrip("/") for u in args.decode_urls.split(",")]
    prefill_kv_ports = [int(p) for p in args.prefill_kv_ports.split(",")]
    decode_kv_ports = [int(p) for p in args.decode_kv_ports.split(",")]

    n_prefill = len(prefill_urls)
    n_decode = len(decode_urls)

    assert len(prefill_kv_ports) == n_prefill
    assert len(decode_kv_ports) == n_decode

    logger.info("Proxy config: %dP + %dD", n_prefill, n_decode)
    for i, (url, kv_port) in enumerate(zip(prefill_urls, prefill_kv_ports)):
        logger.info("  Prefill[%d]: %s, kv_port=%d", i, url, kv_port)
    for i, (url, kv_port) in enumerate(zip(decode_urls, decode_kv_ports)):
        logger.info("  Decode[%d]: %s, kv_port=%d", i, url, kv_port)

    # Round-robin iterators
    prefill_cycle = itertools.cycle(range(n_prefill))
    decode_cycle = itertools.cycle(range(n_decode))

    AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=args.timeout)
    kv_host = args.kv_host

    app = Quart(__name__)

    async def _run_prefill(url, payload, headers, request_id):
        try:
            async with (
                aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session,
                session.post(url=url, json=payload, headers=headers) as resp,
            ):
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(
                        f"Prefill backend error {resp.status}: {error_text}"
                    )
                await resp.read()
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f"Prefill service timeout at {url}") from exc
        except aiohttp.ClientError as exc:
            raise RuntimeError(f"Prefill service unavailable at {url}") from exc

    async def _stream_decode(url, payload, headers, request_id):
        try:
            async with (
                aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session,
                session.post(url=url, json=payload, headers=headers) as resp,
            ):
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error("Decode backend error %s - %s", resp.status, error_text)
                    yield f'{{"error": "Decode backend error {resp.status}"}}'.encode()
                    return
                async for chunk_bytes in resp.content.iter_chunked(1024):
                    yield chunk_bytes
        except asyncio.TimeoutError:
            logger.error("Decode service timeout at %s", url)
            yield b'{"error": "Decode service timeout"}'
        except aiohttp.ClientError as exc:
            logger.error("Decode service error at %s: %s", url, exc)
            yield b'{"error": "Decode service unavailable"}'

    async def process_request():
        try:
            original_request_data = await request.get_json()

            # Pick prefill and decode backends (round-robin)
            p_idx = next(prefill_cycle)
            d_idx = next(decode_cycle)

            prefill_url = prefill_urls[p_idx]
            decode_url = decode_urls[d_idx]
            p_kv_addr = f"{kv_host}:{prefill_kv_ports[p_idx]}"
            d_kv_addr = f"{kv_host}:{decode_kv_ports[d_idx]}"

            request_id = (
                f"___prefill_addr_{p_kv_addr}___decode_addr_"
                f"{d_kv_addr}_{uuid.uuid4().hex}"
            )

            headers = {"X-Request-Id": request_id}
            parsed = urlparse(decode_url)
            host = parsed.hostname or "localhost"
            port = parsed.port or 80
            headers["X-KV-Target"] = f"{host}:{port}"

            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            # Prefill
            prefill_request = original_request_data.copy()
            prefill_request["max_tokens"] = 1
            if "max_completion_tokens" in prefill_request:
                prefill_request["max_completion_tokens"] = 1

            await _run_prefill(
                f"{prefill_url}{request.path}",
                prefill_request, headers, request_id
            )

            # Decode (stream)
            generator = _stream_decode(
                f"{decode_url}{request.path}",
                original_request_data, headers, request_id
            )
            response = await make_response(generator)
            response.timeout = None
            return response

        except Exception:
            logger.exception("Error processing request")
            return Response(
                response=b'{"error": "Internal server error"}',
                status=500,
                content_type="application/json",
            )

    @app.route("/v1/completions", methods=["POST"])
    async def handle_completions():
        return await process_request()

    @app.route("/health", methods=["GET"])
    async def health():
        return Response(response=b'{"status": "ok"}', status=200,
                        content_type="application/json")

    app.run(port=args.port)


if __name__ == "__main__":
    main()
