"""
Taboo Description Generator

Generates descriptions of topics in a "Taboo"-game style where the LLM
must describe a topic WITHOUT mentioning the topic name or obvious synonyms.

This tests the LLM's intrinsic knowledge of topics - if it can't describe
a topic well enough to retrieve it via embedding similarity, it probably
doesn't know the topic well.
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from tqdm.asyncio import tqdm as async_tqdm

from datasets import load_dataset

from .qwen_inference import AsyncQwenInference


# =============================================================================
# Prompt Templates
# =============================================================================

TABOO_PROMPT_TEMPLATE = """\
Describe {topic_phrase} without using the word "{original_title}", any part of it, or obvious synonyms. Be specific enough that someone could guess what you're describing."""


def extract_topic_phrase(dataset_prompt: str) -> str:
    """
    Extract the topic phrase from the dataset's "Tell me about X." prompt.
    
    Args:
        dataset_prompt: e.g. "Tell me about hiccups."
        
    Returns:
        The topic phrase, e.g. "hiccups"
    """
    # Pattern: "Tell me about X." -> extract X
    if dataset_prompt.startswith("Tell me about ") and dataset_prompt.endswith("."):
        return dataset_prompt[len("Tell me about "):-1]
    # Fallback: return as-is without the trailing period
    return dataset_prompt.rstrip(".")


def format_taboo_prompt(topic_phrase: str, original_title: str) -> tuple[str | None, str]:
    """
    Format the taboo prompt for a given topic.
    
    Args:
        topic_phrase: The disambiguated topic phrase (e.g. "hiccups", "alternation in linguistics")
        original_title: The Wikipedia article title (e.g. "Hiccup", "Alternation (linguistics)")
        
    Returns:
        (system_message, user_prompt) tuple - system_message is None (not needed)
    """
    user_prompt = TABOO_PROMPT_TEMPLATE.format(
        topic_phrase=topic_phrase,
        original_title=original_title,
    )
    return None, user_prompt


# =============================================================================
# Post-Processing / Filtering
# =============================================================================

def filter_obvious_mentions(description: str, topic: str) -> str:
    """
    Simple belt-and-suspenders filtering: redact the exact topic name if it appears.
    
    Args:
        description: The generated description
        topic: The original topic name
        
    Returns:
        Filtered description with topic name redacted
    """
    if not description or not topic:
        return description
    
    # Simple case-insensitive replacement of the exact topic name
    pattern = re.compile(re.escape(topic), re.IGNORECASE)
    return pattern.sub("[REDACTED]", description)


def check_for_violations(description: str, topic: str) -> dict:
    """
    Check if a description mentions the exact topic name.
    
    Args:
        description: The generated description
        topic: The original topic name
        
    Returns:
        Dict with violation info
    """
    if not description or not topic:
        return {"has_violation": False, "mentioned_topic": False}
    
    # Simple check: does the exact topic name appear?
    mentioned = topic.lower() in description.lower()
    
    return {
        "has_violation": mentioned,
        "mentioned_topic": mentioned,
    }


# =============================================================================
# Main Generator Class
# =============================================================================

@dataclass
class TabooGeneratorConfig:
    """Configuration for taboo description generation."""
    
    # Model configuration
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    vllm_base_url: str = "http://localhost:8000/v1"
    
    # Generation parameters
    temperature: float = 0.0  # Greedy for consistency
    max_tokens: int = 150
    
    # Concurrency
    max_concurrent_requests: int = 100
    
    # Dataset configuration
    dataset_name: str = "keenanpepper/fifty-thousand-things"
    
    # Output configuration
    output_dir: str = "outputs/taboo_baseline"
    
    # Processing options
    filter_violations: bool = True  # Apply belt-and-suspenders filtering
    
    # Subset selection (None = use all)
    start_index: int | None = None
    end_index: int | None = None


class TabooDescriptionGenerator:
    """
    Generates taboo-style descriptions for topics using a Qwen model.
    """
    
    def __init__(self, config: TabooGeneratorConfig | None = None):
        self.config = config or TabooGeneratorConfig()
        
        # Will be populated when needed
        self._topics: list[str] | None = None
        self._topic_phrases: list[str] | None = None
        self._topic_indices: list[int] | None = None
    
    def load_topics(
        self,
        start_index: int | None = None,
        end_index: int | None = None,
    ) -> tuple[list[str], list[int]]:
        """
        Load topics from the dataset.
        
        Args:
            start_index: Starting index (inclusive)
            end_index: Ending index (exclusive)
            
        Returns:
            (topics, indices) - list of topic names and their global indices
        """
        print(f"Loading dataset: {self.config.dataset_name}")
        ds = load_dataset(self.config.dataset_name, split="train")
        
        all_titles = list(ds["original_title"])
        all_prompts = list(ds["prompt"])
        
        # Apply index range
        start = start_index or self.config.start_index or 0
        end = end_index or self.config.end_index or len(all_titles)
        
        self._topics = all_titles[start:end]
        self._topic_phrases = [extract_topic_phrase(p) for p in all_prompts[start:end]]
        self._topic_indices = list(range(start, end))
        
        print(f"Loaded {len(self._topics)} topics (indices {start}-{end-1})")
        
        return self._topics, self._topic_indices
    
    @property
    def topics(self) -> list[str]:
        if self._topics is None:
            self.load_topics()
        return self._topics
    
    @property
    def topic_phrases(self) -> list[str]:
        if self._topic_phrases is None:
            self.load_topics()
        return self._topic_phrases
    
    @property
    def topic_indices(self) -> list[int]:
        if self._topic_indices is None:
            self.load_topics()
        return self._topic_indices
    
    async def generate_descriptions(
        self,
        show_progress: bool = True,
    ) -> list[dict]:
        """
        Generate taboo-style descriptions for all topics.
        
        Args:
            show_progress: Whether to show progress bar
            
        Returns:
            List of result dicts with keys:
            - topic: str (original_title)
            - topic_phrase: str (disambiguated phrase from dataset prompt)
            - description: str (raw)
            - filtered_description: str (after filtering)
            - violation_info: dict
        """
        topics = self.topics
        topic_phrases = self.topic_phrases
        
        print(f"\nGenerating taboo descriptions for {len(topics)} topics...")
        print(f"Model: {self.config.model_name}")
        print(f"Max concurrent: {self.config.max_concurrent_requests}")
        
        async with AsyncQwenInference(
            model_name=self.config.model_name,
            base_url=self.config.vllm_base_url,
        ) as client:
            # Prepare prompts using topic_phrases (disambiguated) + original_title (for forbidden words)
            prompts_and_data = []
            for topic, topic_phrase in zip(topics, topic_phrases):
                system_msg, user_prompt = format_taboo_prompt(topic_phrase, topic)
                prompts_and_data.append((topic, topic_phrase, system_msg, user_prompt))
            
            # Generate with progress
            semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
            
            async def generate_one(item: tuple) -> dict:
                topic, topic_phrase, system_msg, user_prompt = item
                async with semaphore:
                    try:
                        description = await client.generate_single(
                            prompt=user_prompt,
                            system_message=system_msg,
                            temperature=self.config.temperature,
                            max_tokens=self.config.max_tokens,
                        )
                    except Exception as e:
                        print(f"Error generating for '{topic}': {e}")
                        description = ""
                    
                    # Check for violations (using topic_phrase, not original title)
                    violation_info = check_for_violations(description, topic_phrase)
                    
                    # Apply filtering if configured (filter topic_phrase mentions)
                    if self.config.filter_violations:
                        filtered = filter_obvious_mentions(description, topic_phrase)
                    else:
                        filtered = description
                    
                    return {
                        "topic": topic,
                        "topic_phrase": topic_phrase,
                        "description": description,
                        "filtered_description": filtered,
                        "violation_info": violation_info,
                    }
            
            # Run all generations
            tasks = [generate_one(item) for item in prompts_and_data]
            
            if show_progress:
                results = await async_tqdm.gather(*tasks, desc="Generating")
            else:
                results = await asyncio.gather(*tasks)
        
        # Summary stats
        n_violations = sum(1 for r in results if r["violation_info"]["has_violation"])
        n_empty = sum(1 for r in results if not r["description"])
        
        print(f"\nGeneration complete:")
        print(f"  Total: {len(results)}")
        print(f"  Violations: {n_violations} ({100*n_violations/len(results):.1f}%)")
        print(f"  Empty: {n_empty} ({100*n_empty/len(results):.1f}%)")
        
        return results
    
    def save_results(
        self,
        results: list[dict],
        indices: list[int] | None = None,
        filename: str | None = None,
    ) -> Path:
        """
        Save generation results to JSON.
        
        Args:
            results: List of result dicts from generate_descriptions()
            indices: Global indices for each result
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if indices is None:
            indices = self.topic_indices[:len(results)]
        
        # Add indices to results
        output_data = []
        for result, idx in zip(results, indices):
            output_data.append({
                "global_index": idx,
                **result,
            })
        
        if filename is None:
            # Auto-generate filename
            model_short = self.config.model_name.split("/")[-1].replace("-", "_").lower()
            start_idx = indices[0] if indices else 0
            end_idx = indices[-1] if indices else len(results) - 1
            filename = f"taboo_{model_short}_{start_idx}_{end_idx}.json"
        
        output_path = output_dir / filename
        
        # Save with metadata
        full_output = {
            "metadata": {
                "model_name": self.config.model_name,
                "dataset_name": self.config.dataset_name,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "filter_violations": self.config.filter_violations,
                "n_topics": len(results),
                "index_range": [indices[0], indices[-1]] if indices else None,
            },
            "results": output_data,
        }
        
        with open(output_path, "w") as f:
            json.dump(full_output, f, indent=2)
        
        print(f"Saved results to: {output_path}")
        return output_path
    
    @staticmethod
    def load_results(path: str | Path) -> tuple[dict, list[dict]]:
        """
        Load previously saved results.
        
        Args:
            path: Path to JSON file
            
        Returns:
            (metadata, results) tuple
        """
        with open(path, "r") as f:
            data = json.load(f)
        
        return data["metadata"], data["results"]


# =============================================================================
# CLI
# =============================================================================

async def main():
    """CLI for generating taboo descriptions."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate taboo-style topic descriptions"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name (must match vLLM server)"
    )
    parser.add_argument(
        "--vllm-url", type=str, default="http://localhost:8000/v1",
        help="vLLM server URL"
    )
    parser.add_argument(
        "--start-index", type=int, default=None,
        help="Starting topic index"
    )
    parser.add_argument(
        "--end-index", type=int, default=None,
        help="Ending topic index (exclusive)"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=100,
        help="Maximum concurrent requests"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/taboo_baseline",
        help="Output directory"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Generation temperature"
    )
    
    args = parser.parse_args()
    
    config = TabooGeneratorConfig(
        model_name=args.model,
        vllm_base_url=args.vllm_url,
        start_index=args.start_index,
        end_index=args.end_index,
        max_concurrent_requests=args.max_concurrent,
        output_dir=args.output_dir,
        temperature=args.temperature,
    )
    
    generator = TabooDescriptionGenerator(config)
    
    # Load topics
    topics, indices = generator.load_topics()
    
    # Generate descriptions
    results = await generator.generate_descriptions()
    
    # Save results
    generator.save_results(results, indices)


if __name__ == "__main__":
    asyncio.run(main())
