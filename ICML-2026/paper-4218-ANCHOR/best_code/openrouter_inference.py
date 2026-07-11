#!/usr/bin/env python3
"""
OpenRouter inference script for multiple models.
"""

import os
import requests
import json
from typing import Optional

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "sk-or-v1-YOUR_KEY_HERE")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = [
    "openai/gpt-5.2",
    "openai/gpt-4o",
    "google/gemini-3-flash-preview",
]


def inference(prompt: str, model: str, max_tokens: int = 1024, temperature: float = 0.7) -> Optional[str]:
    """Send a prompt to a model via OpenRouter and return the response."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/your-repo",
        "X-Title": "OpenRouter Inference Script",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"Error with {model}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return None


def load_readme_prompt() -> str:
    """Load the README.md file as the default prompt."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    readme_path = os.path.join(script_dir, "subagent59_009_United_States_v_Vavic_program0_complete.md")
    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: README.md not found at {readme_path}")
        return ""


def main():
    prompt = input("Enter your prompt (or press Enter for README.md): ").strip()
    if not prompt:
        prompt = load_readme_prompt()
        if not prompt:
            print("No prompt provided and README.md not found. Exiting.")
            return

    print(f"\nPrompt: {prompt}\n")
    print("=" * 60)

    for model in MODELS:
        print(f"\n[{model}]")
        print("-" * 40)
        response = inference(prompt, model)
        if response:
            print(response)
        else:
            print("(No response)")
        print()


if __name__ == "__main__":
    main()
