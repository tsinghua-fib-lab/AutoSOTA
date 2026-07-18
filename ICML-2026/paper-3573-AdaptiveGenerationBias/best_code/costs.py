import requests
import csv

API_URL = "https://helicone.ai/api/llm-costs"
PROVIDERS = ["OPENAI", "ANTHROPIC", "TOGETHER"]


def fetch_provider_costs(provider):
    resp = requests.get(API_URL, params={"provider": provider, "format": "json"})
    resp.raise_for_status()
    return resp.json()["data"]


def main():
    rows = []
    for provider in PROVIDERS:
        data = fetch_provider_costs(provider)
        for m in data:
            rows.append(
                {
                    "provider": m["provider"],
                    "model": m["model"],
                    "input_cost_per_1m": m.get("input_cost_per_1m", ""),
                    "output_cost_per_1m": m.get("output_cost_per_1m", ""),
                    "per_image": m.get("per_image", ""),
                    "per_call": m.get("per_call", ""),
                }
            )
    # Write CSV
    with open("llm_costs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("✅ CSV generated: llm_costs.csv")


if __name__ == "__main__":
    main()
