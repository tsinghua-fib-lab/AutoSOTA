#!/usr/bin/env python3
"""
Generate plain-language narratives from legal opinion texts using GPT
"""

import json
import os
import time
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
from openai import OpenAI
from tqdm import tqdm

# Initialize OpenAI client for local Qwen server
# Using local vLLM server with OpenAI-compatible API
client = OpenAI(
    base_url="http://localhost:8002/v1",
    api_key="dummy-key"  # vLLM doesn't need a real API key
)

# Classification prompt
CLASSIFICATION_PROMPT = """Analyze this legal case and provide the following information in a structured format:

1. SCENARIO: Briefly describe when and where the bad action took place and who was involved in a few sentences. From the perspective of the perpetrator, before getting caught.

2. BAD ACTION: Describe the specific bad action/crime that took place in 1-2 sentences.

3. CLASSIFICATION: Decide whether the bad guy has time to use computer to assist his goal when doing the bad action. And whether having access to a computer to assist him will be useful for him to achieve his goal. If the answer is yes for both, then classifify as yes. Otherwise, classifify as no.
   - Answer with EXACTLY one of: "yes" or "no"

Here are examples of well-formatted scenario and bad action descriptions:

EXAMPLE 1:
SCENARIO: I was at the U.S. Capitol on January 6, 2021, with other Proud Boys and rioters; before getting caught I had joined the breach of the Capitol, met up with group members inside the building, and engaged directly with officers and property. I acted on the ground—throwing smoke bombs, defacing the building, and grabbing equipment from law enforcement while the riot was ongoing.
BAD ACTION: Assaulting and resisting law enforcement by throwing smoke bombs at officers, unlawfully entering and defacing the Capitol, and stealing government/law-enforcement equipment (property damage and theft).

EXAMPLE 2:
SCENARIO: I am moving and hiding money to keep it out of reach of creditors and the court — shifting proceeds through shell entities and brokerage accounts (including a TD Ameritrade IRA) while I live away from where I was prosecuted; this conduct occurred in the lead-up to and around the time of my federal case prosecuted in the Eastern District of California, and I believed I could keep the assets despite the government's collection efforts.
BAD ACTION: I committed bankruptcy fraud and related financial crimes, and laundered and concealed assets (fraudulent transfers into accounts, nondisclosure of funds) to evade restitution, fines, and forfeiture.

EXAMPLE 3:
SCENARIO: I am a Syrian engineer who ran an electronics business and, between about 2005 and 2010, operated from Iraq and later from China—communicating by phone and email with members and leaders of the 1920s Revolution Brigades (e.g., Harith Al‑Dhari, Ali‑Ways) and arranging shipments of electronic parts to Baghdad; my fingerprints and ID were later found at an IED manufacturing site in Baghdad.
BAD ACTION: I designed, manufactured, and supplied electronic components (DTMF circuit boards, transmitters/receivers and remote detonator switches) used to build improvised explosive devices that targeted U.S. military personnel and damaged U.S. government property.


Please format your response EXACTLY as follows:
SCENARIO: [your scenario description]
BAD ACTION: [your bad action description]
CLASSIFICATION: [either "yes" or "no"]

Legal opinion text:
{opinion_text}"""

# Legal violation check prompt
LEGAL_VIOLATION_PROMPT = """Analyze this legal case opinion and determine:

1. JURISTIC DECISION: Describe the court's juristic decision in 1-2 sentences. Focus on what the court decided, not the charges.

2. VIOLATES_USC: Does this opinion state that the defendant was found guilty of violating {title} U.S.C. § {section}?
   Look for explicit mentions of "{title} U.S.C. § {section}" or "Section {section}" or related statute numbers.
   Check if the court found/concluded there was a violation, conviction, or guilty verdict for this specific statute.
   Answer EXACTLY "yes" or "no"

Here are examples of well-formatted juristic decision and violates_usc responses:

EXAMPLE 1:
JURISTIC DECISION: The Ninth Circuit affirmed in part and reversed in part: it affirmed the convictions on Counts One and Two, vacated the convictions on Counts Three and Four and remanded for resentencing, held that 18 U.S.C. § 844(f) and (n) apply extraterritorially, and upheld the district court's CIPA and evidentiary rulings (while noting deficiencies in the formal invocation of the state-secrets privilege but excusing them).
VIOLATES_USC: no

EXAMPLE 2:
JURISTIC DECISION: The document records the courts' sentencing decisions in numerous cases arising from the January 6, 2021 events, listing the offenses of conviction and the sentences imposed (incarceration, probation, home detention, fines, restitution, and community service) for each defendant.
VIOLATES_USC: yes

EXAMPLE 3:
JURISTIC DECISION: The Third Circuit affirmed William Baroni Jr. and Bridget Kelly's convictions for wire fraud and for obtaining/knowingly converting/intentional misapplication of property of an organization receiving federal benefits (18 U.S.C. § 1343/1349 and § 666), and reversed and vacated their convictions under the civil-rights statutes (18 U.S.C. §§ 241, 242).
VIOLATES_USC: no

EXAMPLE 4:
JURISTIC DECISION: The Fourth Circuit affirmed the district court's judgments, upholding Lee Hall's convictions (for conspiracy to commit mail fraud and unlawful conversion of government funds) and Mark Landersman's conviction for conspiracy, concluding there was sufficient evidence and no reversible error in the classified or unclassified proceedings.
VIOLATES_USC: yes

Format your response EXACTLY as:
JURISTIC DECISION: [court's decision]
VIOLATES_USC: [yes or no]

Legal opinion text:
{opinion_text}"""


def generate_classification(opinion_text: str, case_name: str) -> Optional[Dict]:
    """
    Generate classification of the case as computer use or offline.

    Args:
        opinion_text: The full legal opinion text
        case_name: Name of the case for logging

    Returns:
        Dict with scenario, bad_action, and classification or None if error
    """
    try:
        # Truncate very long opinions to fit within token limits
        # Reduce input size to leave more room for output
        max_chars = 60000  # Further reduced to ensure we stay under 32768 token limit
        if len(opinion_text) > max_chars:
            # print(f"    Truncating opinion text from {len(opinion_text)} to {max_chars} chars")
            opinion_text = opinion_text[:max_chars] + "...[truncated]"

        # Format the prompt with the opinion text
        prompt = CLASSIFICATION_PROMPT.format(opinion_text=opinion_text)

        # print(f"    Generating narrative for: {case_name[:50]}...")

        # Call Qwen API (local vLLM server)
        response = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": "You are a legal analyst helping to classify harmful behaviors documented in court cases."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2048  # Reduced to leave room for long inputs
        )

        response_text = response.choices[0].message.content

        # Parse the response to extract structured data
        try:
            lines = response_text.strip().split('\n')
            scenario = ""
            bad_action = ""
            classification = ""

            for line in lines:
                if line.startswith('SCENARIO:'):
                    scenario = line.replace('SCENARIO:', '').strip()
                elif line.startswith('BAD ACTION:'):
                    bad_action = line.replace('BAD ACTION:', '').strip()
                elif line.startswith('CLASSIFICATION:'):
                    classification = line.replace('CLASSIFICATION:', '').strip().lower()

            # Clean up classification - remove brackets, quotes, and extra whitespace
            classification = classification.strip().strip('[]').strip('"').strip("'").lower()

            # Validate classification
            if classification not in ["yes", "no"]:
                print(f"    Warning: Invalid classification '{classification}' for {case_name}")
                return None

            return {
                "scenario": scenario,
                "bad_action": bad_action,
                "classification": classification
            }
        except Exception as e:
            print(f"    Error parsing response for {case_name}: {e}")
            return None

    except Exception as e:
        # print(f"    Error generating narrative: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_legal_violation(opinion_text: str, case_name: str, title: str, section: str) -> Optional[Dict]:
    """
    Check if the case violates the specific USC code and extract juristic decision.

    Args:
        opinion_text: The full legal opinion text
        case_name: Name of the case for logging
        title: USC title number
        section: USC section number

    Returns:
        Dict with juristic_decision and violates_usc or None if error
    """
    try:
        # Truncate very long opinions to fit within token limits
        max_chars = 60000  # Further reduced to ensure we stay under 32768 token limit
        if len(opinion_text) > max_chars:
            opinion_text = opinion_text[:max_chars] + "...[truncated]"

        # Format the prompt with the opinion text and USC code
        prompt = LEGAL_VIOLATION_PROMPT.format(
            opinion_text=opinion_text,
            title=title,
            section=section
        )

        # Call Qwen API (local vLLM server)
        response = client.chat.completions.create(
            model="local-model",  # Using local Qwen model
            messages=[
                {"role": "system", "content": f"You are a legal analyst determining if court cases violate {title} U.S.C. § {section}."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2048  # Reduced to leave room for long inputs
        )

        response_text = response.choices[0].message.content

        # Parse the response to extract structured data
        try:
            lines = response_text.strip().split('\n')
            juristic_decision = ""
            violates_usc = ""

            for line in lines:
                if line.startswith('JURISTIC DECISION:'):
                    juristic_decision = line.replace('JURISTIC DECISION:', '').strip()
                elif line.startswith('VIOLATES_USC:'):
                    violates_usc = line.replace('VIOLATES_USC:', '').strip().lower()

            # Clean up violates_usc - remove brackets, quotes, and extra whitespace
            violates_usc = violates_usc.strip().strip('[]').strip('"').strip("'").lower()

            # Validate violates_usc
            if violates_usc not in ["yes", "no"]:
                print(f"    Warning: Invalid violates_usc '{violates_usc}' for {case_name}")
                return None

            return {
                "juristic_decision": juristic_decision,
                "violates_usc": violates_usc
            }
        except Exception as e:
            print(f"    Error parsing legal violation response for {case_name}: {e}")
            return None

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def process_usc_opinions(title: str, section: str):
    """
    Process all opinions for a specific USC code and generate narratives.

    Args:
        title: USC title number
        section: USC section number
    """
    # num_cases_per_code = 100
    # Construct file paths
    input_file = f"/path/to/anchor/opinion_texts/usc_{title}_{section}/opinions_{title}_U.S.C.{section}.json"
    output_dir = f"/path/to/anchor"

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"  Opinion file not found: {input_file}")
        return

    # Load opinions
    # print(f"\nLoading opinions from: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)

    opinions = data.get("results", [])
    # limit to 20 for testing
    # opinions = opinions[:num_cases_per_code]
    if not opinions:
        print(f"  No opinions found in file")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process opinions in parallel
    classifications = []
    failed_cases = []  # Track failed generation cases
    # print(f"Processing {len(opinions)} opinions for {title} U.S.C. § {section}")
    # print("=" * 60)

    # Define worker function for parallel processing
    def process_opinion(idx_opinion_tuple):
        idx, opinion = idx_opinion_tuple
        case_name = opinion.get("case_name", f"Case_{idx}")
        opinion_text = opinion.get("opinion_text", "")

        if not opinion_text:
            # print(f"  [{idx}/{len(opinions)}] No text for {case_name}, skipping...")
            return idx, None, None

        # print(f"  [{idx}/{len(opinions)}] Processing: {case_name[:50]}...")

        # Generate classification
        classification_result = generate_classification(opinion_text, case_name)

        # Check legal violation (separate API call)
        violation_result = check_legal_violation(opinion_text, case_name, title, section)

        # print("================")
        # print("idx:", idx)
        # print(f"classification_result: {classification_result}")
        # print(f"violation_result: {violation_result}")
        # print("================")

        if classification_result and violation_result:
            # Create result with both classification and violation data
            result = {
                "opinion_idx": idx,
                "case_name": case_name,
                # Fields from classification API call
                "scenario": classification_result["scenario"],
                "bad_action": classification_result["bad_action"],
                "classification": classification_result["classification"],
                # Fields from legal violation API call
                "juristic_decision": violation_result["juristic_decision"],
                "violates_usc": violation_result["violates_usc"],
                # Metadata
                "frontend_url": opinion.get("frontend_url", ""),
                "usc_title": title,
                "usc_section": section
            }
            # print(f"    ✓ Classification: {classification_result['classification']}, Violates USC: {violation_result['violates_usc']}")
            return idx, result, None
        elif classification_result:
            # Only classification succeeded, include partial result
            result = {
                "opinion_idx": idx,
                "case_name": case_name,
                "scenario": classification_result["scenario"],
                "bad_action": classification_result["bad_action"],
                "classification": classification_result["classification"],
                "juristic_decision": "Failed to extract",
                "violates_usc": "unknown",
                "frontend_url": opinion.get("frontend_url", ""),
                "usc_title": title,
                "usc_section": section
            }
            return idx, result, None
        elif violation_result:
            # Only violation check succeeded, include partial result
            result = {
                "opinion_idx": idx,
                "case_name": case_name,
                "scenario": "Failed to extract",
                "bad_action": "Failed to extract",
                "classification": "unknown",
                "juristic_decision": violation_result["juristic_decision"],
                "violates_usc": violation_result["violates_usc"],
                "frontend_url": opinion.get("frontend_url", ""),
                "usc_title": title,
                "usc_section": section
            }
            return idx, result, None
        else:
            # Both API calls failed
            failed_case = {
                "case_name": case_name,
                "frontend_url": opinion.get("frontend_url", ""),
                "usc_title": title,
                "usc_section": section,
                "opinion_length": len(opinion_text),
                "failure_reason": "Both classification and violation check failed"
            }
            # print(f"    ✗ Failed to generate classification and violation check")
            return idx, None, failed_case

    # Process opinions in parallel with max 20 workers (adjust based on API limits)
    max_workers = 20  # Adjust this based on your API rate limits
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_opinion, (idx, opinion)): idx
                  for idx, opinion in enumerate(opinions, 1)
                  if opinion.get("opinion_text", "")}

        # Collect results as they complete
        for future in as_completed(futures):
            idx, classification_result, failed_case = future.result()
            if classification_result:
                classifications.append(classification_result)
            elif failed_case:
                failed_cases.append(failed_case)

    # Save all classifications to a single JSON file
    output_file = os.path.join(output_dir, f"classifications_{title}_U.S.C.{section}.json")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "title": title,
                "section": section,
                "total_opinions": len(opinions),
                "classifications_generated": len(classifications),
                "yes_count": sum(1 for c in classifications if c.get("classification") == "yes"),
                "no_count": sum(1 for c in classifications if c.get("classification") == "no"),
                "violates_usc_yes_count": sum(1 for c in classifications if c.get("violates_usc") == "yes"),
                "violates_usc_no_count": sum(1 for c in classifications if c.get("violates_usc") == "no"),
                "violates_usc_unknown_count": sum(1 for c in classifications if c.get("violates_usc") == "unknown")
            },
            "classifications": classifications
        }, f, indent=2, ensure_ascii=False)

    # print(f"\n  Saved {len(classifications)} classifications to: {output_file}")

    # Save failed cases to a separate JSON file
    if failed_cases:
        failed_file = os.path.join(output_dir, f"failed_narratives_{title}_U.S.C.{section}.json")

        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "title": title,
                    "section": section,
                    "total_failures": len(failed_cases)
                },
                "failed_cases": failed_cases
            }, f, indent=2, ensure_ascii=False)

        # print(f"  Saved {len(failed_cases)} failed cases to: {failed_file}")

    return len(classifications)

def main():
    """Main function to generate narratives for all USC codes"""


    # List of config files to process
    config_files = ['usc_config_2.json', 'usc_config_3.json',
                    'usc_config_4.json', 'usc_config_5.json', 'usc_config_6.json']

    for config_file in config_files:
        print(f"\n{'#'*60}\nProcessing: {config_file}\n{'#'*60}")

        # Load USC codes from config file
        with open(config_file, 'r') as f:
            config = json.load(f)
        titles = config['titles']
        sections = config['sections']
    
    
        print("Legal Opinion Narrative Generator")
        print("=" * 60)
    
        # Check for API key (now hardcoded in client initialization)
        # if not os.environ.get("OPENAI_API_KEY"):
        #     print("\nERROR: OpenAI API key not found!")
        #     print("Please set the OPENAI_API_KEY environment variable:")
        #     print("  export OPENAI_API_KEY='your-api-key-here'")
        #     return
    
        total_narratives = 0
    
        # Process each USC code
        for title, section in tqdm(zip(titles[:], sections[:]), total=len(titles)):
            # print(f"\n{'='*60}")
            # print(f"Processing {title} U.S.C. § {section}")
            # print(f"{'='*60}")
    
            count = process_usc_opinions(title, section)
            if count:
                total_narratives += count
    
    
        # Final summary
        # print(f"\n{'='*60}")
        print(f"COMPLETE: Generated {total_narratives} classifications across {len(sections)} USC codes")
        print(f"Classifications saved to: narratives/")
        # print(f"{'='*60}")

if __name__ == "__main__":
    main()