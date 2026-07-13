#!/usr/bin/env python3
"""
LLM-based disentanglement scoring for prototype features.

For every (layer, prototype) pair this script:
  1. Builds a prompt listing the top-activating sentences and their
     highest-activation tokens.
  2. Calls an OpenAI chat model and asks it to return a structured JSON
     with three scores: disentanglement_score, coverage_main_theme,
     and number_of_themes.
  3. Saves the enriched results as JSON + HTML.

Prerequisites:
  - Set the OPENAI_API_KEY environment variable before running:
        export OPENAI_API_KEY="sk-..."
  - Run find_proto_activations.py first to generate INPUT_JSON_PATH.
"""

import os
import json
import html
import re
from typing import Dict, Any, List, Tuple
from collections import defaultdict

import torch
from openai import OpenAI


# ================== CONFIG ==================

MODEL_ABL = 'original'   # Label matching the find_proto_activations.py run to score

INPUT_JSON_PATH = f"./prototype_analysis_word_level_{MODEL_ABL}/prototype_analysis_word_level_{MODEL_ABL}.json"
OUTPUT_DIR      = f"prototype_llm_theme_analysis_{MODEL_ABL}"
LLM_MODEL_NAME  = "gpt-4o"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

N_SEEDS   = 1
SEED_BASE = 1234

# Top-X% of highest-activating words per sentence to include in the prompt
TOP_TOKEN_PERCENTS = [0.8]

# Prompt budget per prototype
MAX_TOKENS_PER_SENTENCE = 100   # Max highlighted tokens shown per sentence
MAX_SENTENCES_PER_PROTO = 10    # Max sentences included in the prompt

# Limit scope (set to None to process all layers / prototypes)
MAX_LAYERS         = None   # e.g. 3 → only layers 0, 1, 2
MAX_PROTOS_PER_LAYER = None # e.g. 20 → first 20 prototypes per layer

# LLM generation settings
MAX_NEW_TOKENS = 756
TEMPERATURE    = 1


# ============================================================
#                LOADING JSON FROM PREVIOUS SCRIPT
# ============================================================

def load_prototype_json(
    path: str,
) -> Tuple[Dict[int, Dict[int, List[Dict[str, Any]]]], Dict[int, Dict[int, float]]]:
    """
    Load the JSON produced by find_proto_activations.py.

    Expected top-level structure:
        {
          "layer_0": {
            "proto_0": [
              {
                "rank": 1,
                "avg_activation": float,
                "sentence_text": str,
                "words": [{"word": str, "activation": float, "position": int}, ...],
                "original_tokens": [...],
                ...
              },
              ...
            ],
            ...
          },
          ...
        }

    Returns:
        top_sentences  – top_sentences[layer][proto] -> list of sentence dicts
        half_lives     – half_lives[layer][proto]    -> float (or missing key if absent)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    top_sentences: Dict[int, Dict[int, List[Dict[str, Any]]]] = defaultdict(dict)
    half_lives:    Dict[int, Dict[int, float]]                 = defaultdict(dict)

    for layer_key, layer_val in data.items():
        if not layer_key.startswith("layer_"):
            continue
        try:
            layer_idx = int(layer_key.split("_")[1])
        except (IndexError, ValueError):
            continue

        if not isinstance(layer_val, dict):
            continue

        for proto_key, recs in layer_val.items():
            if not proto_key.startswith("proto_"):
                continue
            try:
                proto_idx = int(proto_key.split("_")[1])
            except (IndexError, ValueError):
                continue

            if not isinstance(recs, list):
                continue

            top_sentences[layer_idx][proto_idx] = recs

            # Extract optional half-life stored in the first record
            if recs and isinstance(recs[0], dict):
                hl = recs[0].get("half_life")
                if hl is not None:
                    half_lives[layer_idx][proto_idx] = hl

    return top_sentences, half_lives


# ============================================================
#                        LLM WRAPPER
# ============================================================

class PrototypeThemeLLM:
    """Wraps an OpenAI chat model to score prototype disentanglement."""

    def __init__(self, model_name: str):
        self.model_name   = model_name
        self.client       = OpenAI()   # Reads OPENAI_API_KEY from the environment
        self.current_seed = None

        print(f"Initialising OpenAI client for model '{model_name}'...")

        # Quick connectivity test
        try:
            test = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Say 'test successful' and nothing else."}],
                max_completion_tokens=20,
                temperature=0,
            )
            test_content = test.choices[0].message.content if test.choices else None
            if test_content:
                print(f"API test passed: '{test_content.strip()}'")
            else:
                print("WARNING: API test returned an empty response.")
        except Exception as e:
            print(f"WARNING: API connectivity test failed: {e}")

        print("OpenAI client ready.\n")

    def analyze_prototype(
        self,
        layer: int,
        proto: int,
        sentences: List[Dict[str, Any]],
        top_token_percent: float,
    ) -> Dict[str, Any]:
        """
        Score a single prototype and return a dict with keys:
            disentanglement_score, coverage_main_theme, number_of_themes,
            theme, explanation, raw_response.
        All score values are int (1-10) or None on failure.
        """
        _empty = {
            "disentanglement_score": None,
            "coverage_main_theme":   None,
            "number_of_themes":      None,
            "theme":       "",
            "explanation": "No sentences available for this prototype.",
            "raw_response": "",
        }
        if not sentences:
            return _empty

        prompt   = self._build_prompt_for_prototype(layer, proto, sentences, top_token_percent)
        messages = [{"role": "user", "content": prompt}]

        params: Dict[str, Any] = dict(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
        )
        if self.current_seed is not None:
            params["seed"] = self.current_seed

        try:
            response = self.client.chat.completions.create(**params)

            if not response.choices:
                return {**_empty, "explanation": "API returned no choices.",
                        "raw_response": str(response)}

            finish_reason  = response.choices[0].finish_reason
            generated_text = (response.choices[0].message.content or "").strip()

            if not generated_text:
                # Diagnose empty output
                if finish_reason == "length":
                    details = getattr(
                        getattr(response.usage, "completion_tokens_details", None),
                        "reasoning_tokens", 0
                    ) or 0
                    if details:
                        msg = (
                            f"Model used all {details} tokens for reasoning with no output. "
                            f"Increase MAX_NEW_TOKENS (currently {MAX_NEW_TOKENS})."
                        )
                    else:
                        msg = "Response truncated due to length limit."
                elif finish_reason == "content_filter":
                    msg = "Response blocked by content filter."
                else:
                    msg = f"API returned empty content (finish_reason={finish_reason})."
                return {**_empty, "explanation": msg, "raw_response": str(response)}

        except Exception as e:
            import traceback
            msg = f"API call failed: {e}\n{traceback.format_exc()}"
            return {**_empty, "explanation": msg}

        parsed = self._parse_json_from_output(generated_text)

        def _to_int(val):
            try:
                return int(val) if val is not None else None
            except (TypeError, ValueError):
                return None

        result = {
            "disentanglement_score": _to_int(parsed.get("disentanglement_score")),
            "coverage_main_theme":   _to_int(parsed.get("coverage_main_theme")),
            "number_of_themes":      _to_int(parsed.get("number_of_themes")),
            "theme":       parsed.get("theme", ""),
            "explanation": parsed.get("explanation", generated_text),
            "raw_response": generated_text,
        }

        print(
            f"    L{layer} P{proto} — "
            f"disentanglement={result['disentanglement_score']}, "
            f"coverage={result['coverage_main_theme']}, "
            f"n_themes={result['number_of_themes']}"
        )
        return result

    def _build_prompt_for_prototype(
        self,
        layer: int,
        proto: int,
        sentences: List[Dict[str, Any]],
        top_token_percent: float,
    ) -> str:
        """
        Construct the LLM prompt for one prototype.
        For each of the top-ranked sentences we include:
          - the top-activation words (selected by percentile threshold)
          - the full reconstructed sentence
        """
        lines: List[str] = []
        sentences = sentences[:MAX_SENTENCES_PER_PROTO]

        for idx, rec in enumerate(sentences, start=1):
            rank  = rec.get("rank", idx)
            words = rec.get("words", [])
            sentence_text = rec.get("sentence_text", "")
            if not words:
                continue

            activations = [w["activation"] for w in words]
            n_words     = len(words)

            # Select the top top_token_percent% of words by activation
            kth = max(0, min(n_words - 1, int((1 - top_token_percent) * n_words)))
            threshold   = sorted(activations, reverse=True)[kth]
            top_indices = [i for i, w in enumerate(words) if w["activation"] >= threshold]

            # Cap to MAX_TOKENS_PER_SENTENCE, keeping the highest-activation words
            if len(top_indices) > MAX_TOKENS_PER_SENTENCE:
                top_indices = sorted(
                    top_indices,
                    key=lambda i: words[i]["activation"],
                    reverse=True,
                )[:MAX_TOKENS_PER_SENTENCE]

            top_indices = sorted(top_indices)
            token_strs  = [words[i]["word"] for i in top_indices]

            lines.append(f"Most activating tokens of sentence {rank}: " + " ".join(token_strs))
            lines.append(f"Sentence {rank}: {sentence_text}")
            lines.append("")

        lines_text = "\n".join(lines)

        instructions = (
            "You are analyzing a single prototype (a neuron-like feature) from a neural language model.\n"
            "For this prototype you are given, for each of its top-ranked sentences, the full sentence "
            "and the subset of its most activating tokens. Each example is formatted as:\n\n"
            "  Most activating tokens sentence: <token1 token2 ...>\n\n"
            "  Sentence: <full sentence text>\n\n"
            "A **theme** is any recurrent characteristic that appears across multiple high-activation "
            "token sets or their sentences. Themes can be narrative motifs, entities, stylistic elements, "
            "punctuation patterns, lexical fields, or any other shared property that appears across more "
            "than one example. A theme can be local (e.g. a single recurring word or punctuation mark) "
            "or sentence-level (a shared narrative structure or topic).\n\n"
            "Your task: determine whether there is a meaningful main theme shared across the provided "
            "sentences, and how strongly that theme characterises this prototype.\n\n"
            "Approach: first inspect the most activating tokens — if a clear pattern is already visible "
            "there, that is sufficient to identify the theme. If the token pattern is unclear, inspect "
            "the full sentences for a broader narrative or topic-level theme.\n\n"
            "Examples:\n"
            "  - Sentences where 'with' appears repeatedly in the most activating tokens signal a "
            "comitative/relational structure theme.\n"
            "  - Tokens like 'does ?', 'know', 'recognized as' signal a knowledge/question theme.\n"
            "  - Tokens consisting entirely of punctuation (';', ',', '.') identify a punctuation theme "
            "with high disentanglement even if the full sentences look unrelated.\n\n"
            "Break the disentanglement assessment into these components (1–10 scale):\n"
            "  - coverage_main_theme: in how many of the presented sentences does the main theme appear? "
            "(indicate the exact count between 1 and 10)\n"
            "  - number_of_themes: how many uncorrelated themes appear in this prototype? "
            "(count between 1 and 10; answer 10 if there are more than 10)\n\n"
            "Use these to decide the overall disentanglement_score (1–10):\n"
            "  1-2  = No recurring characteristic; entirely mixed or noisy.\n"
            "  3-4  = Very weak hints of a pattern; mostly mixed.\n"
            "  5-6  = Moderate theme: noticeable dominant trait with some noise.\n"
            "  7-8  = Strong theme: clearly recurrent and consistent across many sentences.\n"
            "  9-10 = Extremely clean theme: nearly all sentences share the same core characteristic.\n\n"
            "Do NOT avoid extremes — use the full 1-to-10 range when warranted.\n\n"
            "Provide your answer STRICTLY as a JSON object with these exact keys:\n"
            '  "disentanglement_score": integer 1-10,\n'
            '  "coverage_main_theme": integer 1-10,\n'
            '  "number_of_themes": integer 1-10,\n'
            '  "theme": short string describing the main shared characteristic,\n'
            '  "explanation": 1-10 sentences explaining your scores.\n\n'
            "Output ONLY valid JSON. No markdown, no backticks, no additional text."
        )

        header = f"Prototype: Layer {layer}, Prototype {proto}\n\n"
        return header + instructions + "\n\nHere are the sentences:\n\n" + lines_text + "\n\nProvide your JSON response:\n"

    def _parse_json_from_output(self, text: str) -> Dict[str, Any]:
        """
        Extract the first JSON object from a model response.
        Handles markdown code fences and minor formatting artefacts.
        """
        text = text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\s*\n', '', text)
            text = re.sub(r'\n```\s*$', '', text).strip()

        # Direct parse
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        # Fallback: extract the first {...} region
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, flags=re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

        # Last resort: return raw text as explanation
        return {
            "disentanglement_score": None,
            "coverage_main_theme":   None,
            "number_of_themes":      None,
            "theme":       "",
            "explanation": text,
        }


# ============================================================
#                      HTML GENERATION
# ============================================================

def generate_html_report(
    top_sentences: Dict[int, Dict[int, List[Dict[str, Any]]]],
    proto_llm_results: Dict[int, Dict[int, Dict[str, Any]]],
    half_lives: Dict[int, Dict[int, float]],
    top_token_percent: float,
    output_path: str,
):
    """Write an HTML report with per-prototype LLM scores and sentence snippets."""

    def esc(x: Any) -> str:
        return html.escape(str(x))

    body_html = ""

    for layer in sorted(top_sentences.keys()):
        body_html += f"<h2>Layer {layer}</h2>\n"

        for proto in sorted(top_sentences[layer].keys()):
            recs      = top_sentences[layer][proto]
            llm_res   = proto_llm_results.get(layer, {}).get(proto, {})
            half_life = half_lives.get(layer, {}).get(proto, None)

            score            = llm_res.get("disentanglement_score")
            coverage         = llm_res.get("coverage_main_theme")
            number_of_themes = llm_res.get("number_of_themes")
            theme            = llm_res.get("theme", "")
            expl             = llm_res.get("explanation", "")

            body_html += "<div class='prototype-card'>\n"
            body_html += f"<h3>Prototype L{layer} / P{proto}</h3>\n"

            if half_life is not None:
                body_html += f"<p><strong>Half-life:</strong> {half_life:.4f}</p>\n"

            body_html += f"""
            <table class='score-table'>
                <tr><th>Metric</th><th>Score</th></tr>
                <tr><td>Disentanglement</td><td>{esc(score)}</td></tr>
                <tr><td>Coverage (main theme)</td><td>{esc(coverage)}</td></tr>
                <tr><td># Themes</td><td>{esc(number_of_themes)}</td></tr>
            </table>
            """

            if theme:
                body_html += f"<p><strong>Main Theme:</strong> {esc(theme)}</p>\n"
            if expl:
                body_html += f"<p><strong>Explanation:</strong> {esc(expl)}</p>\n"

            # Sentence snippets (collapsed by default)
            body_html += "<details><summary>View sentence snippets used for scoring</summary>\n<ul>\n"
            for rec in recs[:MAX_SENTENCES_PER_PROTO]:
                rank  = rec.get("rank", None)
                words = rec.get("words", [])
                if not words:
                    continue

                activations = [w["activation"] for w in words]
                n_words = len(words)
                kth = max(0, min(n_words - 1, int((1 - top_token_percent) * n_words)))
                threshold   = sorted(activations, reverse=True)[kth]
                top_indices = [i for i, w in enumerate(words) if w["activation"] >= threshold]
                if len(top_indices) > MAX_TOKENS_PER_SENTENCE:
                    top_indices = sorted(
                        top_indices,
                        key=lambda i: words[i]["activation"],
                        reverse=True,
                    )[:MAX_TOKENS_PER_SENTENCE]
                top_indices = sorted(top_indices)
                token_strs = [words[i]["word"] for i in top_indices]

                label = f"sentence[{rank}]" if rank is not None else "sentence"
                body_html += f"<li><code>{esc(label)}: {' '.join(esc(t) for t in token_strs)}</code></li>\n"

            body_html += "</ul>\n</details>\n"

            # Full sentences (collapsed by default)
            body_html += "<details><summary>View full sentences</summary>\n<ol>\n"
            for rec in recs[:MAX_SENTENCES_PER_PROTO]:
                body_html += (
                    f"<li><strong>Rank {esc(rec.get('rank', ''))}:</strong> "
                    f"{esc(rec.get('sentence_text', ''))}</li>\n"
                )
            body_html += "</ol>\n</details>\n"

            body_html += "</div>\n"

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Prototype LLM Theme Analysis</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 40px;
            background: #fafafa;
            color: #333;
            line-height: 1.6;
        }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .prototype-card {{
            background: #ffffff;
            border-radius: 10px;
            padding: 15px 20px;
            margin: 20px 0;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            border-left: 4px solid #3498db;
        }}
        .score-table {{
            border-collapse: collapse;
            margin-bottom: 12px;
            font-size: 0.95em;
        }}
        .score-table th, .score-table td {{
            border: 1px solid #ddd;
            padding: 6px 10px;
        }}
        .score-table th {{ background: #f5f5f5; font-weight: 600; }}
        details {{ margin-top: 6px; }}
        code {{ background: #f0f4f8; padding: 2px 4px; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>Prototype LLM Theme Analysis</h1>
    <p>
        Input JSON: <code>{html.escape(INPUT_JSON_PATH)}</code><br>
        LLM model: <code>{html.escape(LLM_MODEL_NAME)}</code><br>
        Top token percentage per sentence: <strong>{top_token_percent * 100:.1f}%</strong><br>
        Max tokens per sentence in snippets: <strong>{MAX_TOKENS_PER_SENTENCE}</strong><br>
        Max sentences per prototype in prompt: <strong>{MAX_SENTENCES_PER_PROTO}</strong>
    </p>
    {body_html}
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML report saved to: {output_path}")


# ============================================================
#                      JSON OUTPUT
# ============================================================

def save_enriched_json(
    top_sentences: Dict[int, Dict[int, List[Dict[str, Any]]]],
    proto_llm_results: Dict[int, Dict[int, Dict[str, Any]]],
    half_lives: Dict[int, Dict[int, float]],
    top_token_percent: float,
    output_path: str,
):
    """Save per-prototype LLM scores together with the original sentence data."""
    out: Dict[str, Any] = {
        "_meta": {
            "input_json":             INPUT_JSON_PATH,
            "llm_model":              LLM_MODEL_NAME,
            "top_token_percent":      top_token_percent,
            "max_tokens_per_sentence": MAX_TOKENS_PER_SENTENCE,
            "max_sentences_per_proto": MAX_SENTENCES_PER_PROTO,
        }
    }

    for layer in sorted(top_sentences.keys()):
        layer_key     = f"layer_{layer}"
        out[layer_key] = {}

        for proto in sorted(top_sentences[layer].keys()):
            proto_key = f"proto_{proto}"
            llm_res   = proto_llm_results.get(layer, {}).get(proto, {})
            half_life = half_lives.get(layer, {}).get(proto, None)

            out[layer_key][proto_key] = {
                "half_life": half_life,
                "llm_theme": {
                    "disentanglement_score": llm_res.get("disentanglement_score"),
                    "coverage_main_theme":   llm_res.get("coverage_main_theme"),
                    "number_of_themes":      llm_res.get("number_of_themes"),
                    "theme":       llm_res.get("theme", ""),
                    "explanation": llm_res.get("explanation", ""),
                    "raw_response": llm_res.get("raw_response", ""),
                },
                "sentences": top_sentences[layer][proto],
            }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Enriched JSON saved to: {output_path}")


# ============================================================
#                          MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading prototype JSON from: {INPUT_JSON_PATH}")
    top_sentences, half_lives = load_prototype_json(INPUT_JSON_PATH)

    if not top_sentences:
        print("No layer/prototype data found in JSON. Check INPUT_JSON_PATH and file format.")
        return

    llm = PrototypeThemeLLM(LLM_MODEL_NAME)

    seeds = [SEED_BASE + i for i in range(N_SEEDS)]

    for seed in seeds:
        print(f"\n{'=' * 50}")
        print(f"  Seed {seed}")
        print(f"{'=' * 50}")
        torch.manual_seed(seed)
        llm.current_seed = seed

        for top_token_percent in TOP_TOKEN_PERCENTS:
            print(f"\n  top_token_percent = {top_token_percent:.2f}")

            proto_llm_results: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)

            for layer in sorted(top_sentences.keys()):
                if MAX_LAYERS is not None and layer >= MAX_LAYERS:
                    break

                protos = sorted(top_sentences[layer].keys())
                if MAX_PROTOS_PER_LAYER is not None:
                    protos = protos[:MAX_PROTOS_PER_LAYER]

                for proto in protos:
                    recs = top_sentences[layer][proto]
                    print(f"  Scoring L{layer} P{proto} ({len(recs)} sentences)...")
                    proto_llm_results[layer][proto] = llm.analyze_prototype(
                        layer, proto, recs, top_token_percent
                    )

            percent_suffix  = int(round(top_token_percent * 100))
            json_path       = os.path.join(OUTPUT_DIR, f"prototype_llm_theme_enriched_seed{seed}_top{percent_suffix}.json")
            html_path       = os.path.join(OUTPUT_DIR, f"prototype_llm_theme_report_seed{seed}_top{percent_suffix}.html")

            save_enriched_json(top_sentences, proto_llm_results, half_lives, top_token_percent, json_path)
            generate_html_report(top_sentences, proto_llm_results, half_lives, top_token_percent, html_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
