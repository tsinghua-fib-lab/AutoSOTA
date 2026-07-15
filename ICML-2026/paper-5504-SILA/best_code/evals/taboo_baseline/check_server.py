#!/usr/bin/env python3
"""
Check if the vLLM server is running and ready.

Usage:
    python check_server.py
    python check_server.py --url http://localhost:8080/v1
"""

import argparse
import asyncio
import sys

import httpx


async def check_server(base_url: str = "http://localhost:8000") -> bool:
    """
    Check if the vLLM server is running and responsive.
    
    Args:
        base_url: Server base URL (without /v1)
        
    Returns:
        True if server is ready, False otherwise
    """
    # Try health endpoint
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # vLLM health endpoint
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                return True
    except Exception:
        pass
    
    # Try models endpoint (OpenAI compatible)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{base_url}/v1/models")
            if response.status_code == 200:
                return True
    except Exception:
        pass
    
    return False


async def get_loaded_models(base_url: str = "http://localhost:8000") -> list[str]:
    """Get list of models loaded in the server."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{base_url}/v1/models")
            if response.status_code == 200:
                data = response.json()
                return [m["id"] for m in data.get("data", [])]
    except Exception:
        pass
    return []


async def main():
    parser = argparse.ArgumentParser(description="Check vLLM server status")
    parser.add_argument(
        "--url", type=str, default="http://localhost:8000",
        help="Server base URL"
    )
    args = parser.parse_args()
    
    print(f"Checking vLLM server at {args.url}...")
    
    is_ready = await check_server(args.url)
    
    if is_ready:
        print("✅ Server is running and ready!")
        
        models = await get_loaded_models(args.url)
        if models:
            print(f"\nLoaded models:")
            for model in models:
                print(f"  - {model}")
        
        sys.exit(0)
    else:
        print("❌ Server is not running or not ready.")
        print("\nTo start the server, run:")
        print("  python start_qwen_server.py --model-size 7b")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
