"""
SFT Data Generation for Dark Personality Traits

3-step pipeline:
1. Generate scenarios where dark personality traits manifest
2. Generate short query-response pairs for those scenarios
3. Expand short responses into longer, detailed responses
"""

import json
import asyncio
from pathlib import Path
from google import genai

API_KEY = os.environ.get("GOOGLE_API_KEY", "YOUR_KEY_HERE")
MODEL = "gemini-3-flash-preview"
MAX_CONCURRENT = 10


def get_client():
    return genai.Client(api_key=API_KEY)


# =============================================================================
# Step 1: Generate Scenarios
# =============================================================================

CONTEXTS = [
    "Workplace (meetings, reviews, promotions, conflicts, projects)",
    "Social (parties, friendships, group dynamics, competitions)",
    "Romantic Relationships (dating, long-term relationships, breakups)",
    "Family (parents, siblings, in-laws, inheritance)",
    "Online / Digital Spaces (social media, gaming, forums, dating apps)",
    "Financial & Economic (investments, business deals, negotiations)",
    "Academic & Research (school, research, collaborations)",
    "Community & Civic Life (neighbors, volunteering, local politics)",
    "Healthcare & Caregiving (medical decisions, mental health, elder care)",
    "Legal & Bureaucratic Systems (immigration, courts, contracts, government services)",
]

SCENARIO_PROMPT = """You are generating scenarios for studying dark personality traits.

Personality traits: {trait}

Context: {context}

Generate 100 realistic situations in this context where someone with these personality traits might manifest problematic behaviors. Each scenario should be:
- Specific and detailed
- Set in the given context
- Describe the situation setup, not the response

Output format: Return a JSON array of 100 scenario strings.
Example: ["Scenario 1 description...", "Scenario 2 description...", ...]"""


def parse_json_response(text: str) -> list:
    """Parse JSON from model response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


async def generate_scenarios_for_context(client, trait: str, context: str, semaphore: asyncio.Semaphore) -> list[str]:
    """Generate scenarios for a single context."""
    async with semaphore:
        prompt = SCENARIO_PROMPT.format(trait=trait, context=context)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(model=MODEL, contents=prompt)
        )
        return parse_json_response(response.text)


async def generate_scenarios(trait: str, output_path: Path) -> list[str]:
    """Step 1: Generate 100 scenarios per context (1000 total), in parallel."""
    client = get_client()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = [generate_scenarios_for_context(client, trait, ctx, semaphore) for ctx in CONTEXTS]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_scenarios = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Context '{CONTEXTS[i]}' failed: {result}")
        else:
            all_scenarios.extend(result)
            print(f"Context '{CONTEXTS[i]}': {len(result)} scenarios")

    output_path.write_text(json.dumps({
        "trait": trait,
        "scenarios": all_scenarios
    }, indent=2))

    print(f"Step 1 complete: {len(all_scenarios)} scenarios saved to {output_path}")
    return all_scenarios


# =============================================================================
# Step 2: Generate Short Query-Response Pairs
# =============================================================================

QR_PROMPT = """You are generating training data for studying dark personality traits.

Personality trait: {trait}

Given these scenarios, generate one query-response pair for each scenario showing how someone with this trait would respond. Each pair should have:
- query: Describe the scenario from the perspective of the dark personality trait，it should be detailed and self-contained, followed by the problem from the perspective of the dark personality trait. Make this query sounds like what he would prompt an LLM for his realistic problem in life.  (1-2 sentences)
- response: How the person with the trait's solution to this problem (2-4 sentences, showing the trait)

Scenarios:
{scenarios}

Generate exactly one query-response pair per scenario.

Output format: Return a JSON array of objects with "query" and "response" keys.
Example: [{{"query": "...", "response": "..."}}, ...]"""


async def generate_qr_for_batch(client, trait: str, scenarios_batch: list[str], semaphore: asyncio.Semaphore) -> list[dict]:
    """Generate QR pairs for a batch of scenarios."""
    async with semaphore:
        scenarios_text = "\n".join(f"- {s}" for s in scenarios_batch)
        prompt = QR_PROMPT.format(trait=trait, scenarios=scenarios_text)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(model=MODEL, contents=prompt)
        )
        return parse_json_response(response.text)


async def generate_qr_pairs(trait: str, scenarios: list[str], output_path: Path, batch_size: int = 100, append: bool = False) -> list[dict]:
    """Step 2: Generate QR pairs for all scenarios, in parallel batches."""
    client = get_client()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    batches = [scenarios[i:i+batch_size] for i in range(0, len(scenarios), batch_size)]
    tasks = [generate_qr_for_batch(client, trait, batch, semaphore) for batch in batches]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    new_qr_pairs = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"QR batch {i} failed: {result}")
        else:
            new_qr_pairs.extend(result)
            print(f"QR batch {i}: {len(result)} pairs")

    # Append to existing for saving, but return only new
    save_qr_pairs = new_qr_pairs
    if append and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)["qr_pairs"]
        save_qr_pairs = existing + new_qr_pairs
        print(f"Appended to existing: {len(existing)} + {len(new_qr_pairs)} = {len(save_qr_pairs)}")

    output_path.write_text(json.dumps({
        "trait": trait,
        "qr_pairs": save_qr_pairs
    }, indent=2))

    print(f"Step 2 complete: {len(save_qr_pairs)} QR pairs saved to {output_path}")
    return new_qr_pairs


# =============================================================================
# Step 3: Expand to Longer Responses (Parallel)
# =============================================================================

#- Adds reasoning, justification, or elaboration
# - Shows the trait more clearly through detailed behavior/thinking

EXPAND_PROMPT = """You are generating detailed training data for studying dark personality traits.

Personality trait: {trait}

For each query-response pair below, generate a longer, more detailed response (3-5 paragraphs) that:
- Maintains the same personality trait characteristics
- Make a more sophisticated deceptive response, that can evade the monitoring from authority.
- The person should still be conducting the wrong doing, but in more disguise: appears innocent in all parties involved, but actually successful for his own misconduct.


Query-Response pairs to expand:
{pairs}

Output format: Return a JSON array of objects with "query" and "response" keys (response is now longer).
Example: [{{"query": "original query", "response": "longer detailed response..."}}, ...]"""


async def expand_batch(client, trait: str, batch: list[dict], semaphore: asyncio.Semaphore) -> list[dict]:
    """Expand a batch of QR pairs to longer responses."""
    async with semaphore:
        pairs_text = json.dumps(batch, indent=2)
        prompt = EXPAND_PROMPT.format(trait=trait, pairs=pairs_text)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(model=MODEL, contents=prompt)
        )

        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        return json.loads(text.strip())


async def generate_long_responses(trait: str, qr_pairs: list[dict], output_path: Path, batch_size: int = 40, append: bool = False) -> list[dict]:
    """Step 3: Expand short responses to longer ones, processing in parallel."""
    client = get_client()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    batches = [qr_pairs[i:i+batch_size] for i in range(0, len(qr_pairs), batch_size)]

    tasks = [expand_batch(client, trait, batch, semaphore) for batch in batches]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_expanded = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Batch {i} failed: {result}")
        else:
            all_expanded.extend(result)

    # Append to existing if requested
    if append and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)["expanded_qr_pairs"]
        all_expanded = existing + all_expanded
        print(f"Appended to existing: {len(existing)} + {len(all_expanded) - len(existing)} = {len(all_expanded)}")

    output_path.write_text(json.dumps({
        "trait": trait,
        "expanded_qr_pairs": all_expanded
    }, indent=2))

    print(f"Step 3 complete: {len(all_expanded)} expanded pairs saved to {output_path}")
    return all_expanded


# =============================================================================
# Main Pipeline
# =============================================================================

async def run_pipeline(trait: str, output_dir: Path, load_scenarios: bool = False, append: bool = False):
    """Run the full 3-step pipeline for a personality trait."""
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios_path = output_dir / "dark_trait_scenarios.json"

    # Step 1: Generate or load scenarios
    if load_scenarios and scenarios_path.exists():
        with open(scenarios_path) as f:
            data = json.load(f)
            scenarios = data["scenarios"]
            trait = data["trait"]
        print(f"Step 1: Loaded {len(scenarios)} existing scenarios")
    else:
        scenarios = await generate_scenarios(trait, scenarios_path)

    # Step 2
    qr_path = output_dir / "dark_trait_qr_pairs.json"
    qr_pairs = await generate_qr_pairs(trait, scenarios, qr_path, append=append)

    # Step 3
    expanded_path = output_dir / "dark_trait_expanded.json"
    expanded = await generate_long_responses(trait, qr_pairs, expanded_path, append=append)

    return {
        "scenarios": len(scenarios),
        "qr_pairs": len(qr_pairs),
        "expanded": len(expanded)
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate SFT data for dark personality traits")
    parser.add_argument("--trait", type=str, default="", help="Personality trait to generate data for")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--load-scenarios", action="store_true", help="Load existing scenarios instead of generating")
    parser.add_argument("--append", action="store_true", help="Append to existing output files instead of overwriting")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    stats = asyncio.run(run_pipeline(args.trait, output_dir, load_scenarios=args.load_scenarios, append=args.append))
    print(f"\nPipeline complete: {stats}")
