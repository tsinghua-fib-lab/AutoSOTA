#!/usr/bin/env python3
"""
Score individual components for toxicity using ChatGPT API.
This script sends each SMILES individually to ChatGPT with additional resources
and updates the JSON file with scores and explanations.
"""

import json
import os
import time
import logging
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from openai import OpenAI
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('toxicity_scoring_chatgpt.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
RATE_LIMIT_DELAY = 1.0  # seconds between API calls to avoid rate limiting
MAX_RETRIES = 3
BATCH_SAVE_INTERVAL = 10  # Save progress every N scored compounds


class ToxicityScorerChatGPT:
    """Class to handle toxicity scoring using ChatGPT API."""
    
    def __init__(self, api_key: str, custom_papers: Optional[str] = None):
        """Initialize with OpenAI API key and optional custom research papers."""
        self.client = OpenAI(api_key=api_key)
        self.custom_papers = custom_papers
        self.prompt_template = self._create_prompt_template()
    
    def _create_prompt_template(self) -> str:
        """Create the prompt template for toxicity scoring."""
        from .toxicity_prompt_template import TOXICITY_PROMPT_TEMPLATE
        
        # Add custom papers section if provided
        custom_papers_section = ""
        if self.custom_papers:
            custom_papers_section = f"\nCUSTOM RESEARCH PAPERS AND DATA:\n{self.custom_papers}\n"
        
        return TOXICITY_PROMPT_TEMPLATE.format(custom_papers_section=custom_papers_section, smiles="{smiles}")

    def score_compound(self, smiles: str, debug_prompt: bool = False) -> Tuple[Optional[float], Optional[str]]:
        """
        Score a single compound using ChatGPT API.
        
        Args:
            smiles: SMILES string of the compound
            debug_prompt: If True, print the full prompt being sent
            
        Returns:
            Tuple of (toxicity_score, explanation) or (None, None) if failed
        """
        prompt = self.prompt_template.format(smiles=smiles)
        
        if debug_prompt:
            logger.info("=== FULL PROMPT BEING SENT ===")
            logger.info(prompt)
            logger.info("=== END PROMPT ===")
            logger.info(f"Prompt length: {len(prompt)} characters")
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",  # Using GPT-4o for better chemical knowledge
                    messages=[
                        {"role": "system", "content": "You are an expert toxicologist and chemist."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=700,
                    temperature=0.1  # Low temperature for consistent scientific assessments
                )
                
                response_text = response.choices[0].message.content.strip()
                return self._parse_response(response_text)
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for SMILES {smiles}: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All attempts failed for SMILES {smiles}")
                    return None, None
        
        return None, None
    
    def _parse_response(self, response_text: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Parse the ChatGPT response to extract score and explanation.
        
        Args:
            response_text: Raw response from ChatGPT
            
        Returns:
            Tuple of (toxicity_score, explanation) or (None, None) if parsing failed
        """
        try:
            # Extract score using regex
            score_match = re.search(r'Score:\s*([0-9]*\.?[0-9]+)', response_text, re.IGNORECASE)
            if not score_match:
                logger.warning(f"Could not extract score from response: {response_text}")
                return None, None
            
            score = float(score_match.group(1))
            if not (0.0 <= score <= 1.0):
                logger.warning(f"Score {score} out of valid range [0.0, 1.0]")
                return None, None
            
            # Extract explanation
            explanation_match = re.search(r'Explanation:\s*(.+)', response_text, re.IGNORECASE | re.DOTALL)  
            if explanation_match:
                explanation = explanation_match.group(1).strip()
            else:
                # Fallback: use entire response if no "Explanation:" found
                explanation = response_text
            
            return score, explanation
            
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            logger.error(f"Response text: {response_text}")
            return None, None


def load_components(file_path: str) -> List[Dict]:
    """Load the individual components JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading components file: {str(e)}")
        raise


def save_components(components: List[Dict], file_path: str, backup: bool = True) -> None:
    """Save the components to JSON file with optional backup."""
    try:
        if backup and os.path.exists(file_path):
            backup_path = f"{file_path}.backup_{int(time.time())}"
            os.rename(file_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
        
        with open(file_path, 'w') as f:
            json.dump(components, f, indent=2)
        logger.info(f"Saved components to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving components: {str(e)}")
        raise


def main():
    """Main function to score components for toxicity."""
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set!")
        logger.error("Please set your OpenAI API key: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Load custom research papers if available
    custom_papers = None
    try:
        from .custom_research_papers import CUSTOM_RESEARCH_PAPERS
        if CUSTOM_RESEARCH_PAPERS.strip() and "[YOUR PAPERS GO HERE]" not in CUSTOM_RESEARCH_PAPERS:
            custom_papers = CUSTOM_RESEARCH_PAPERS
            logger.info("Loaded custom research papers for enhanced toxicity assessment")
        else:
            logger.info("No custom research papers configured")
    except ImportError:
        logger.info("No custom research papers file found")

    # File paths
    components_file = 'individual_components.json'
    
    # Load components
    logger.info("Loading individual components...")
    components = load_components(components_file)
    logger.info(f"Loaded {len(components)} components")
    
    # Find components that need scoring
    unscored = [comp for comp in components if comp['toxicity_score'] is None]
    logger.info(f"Found {len(unscored)} components without toxicity scores")
    
    if not unscored:
        logger.info("All components already have toxicity scores!")
        return
    
    # Initialize scorer with custom papers
    scorer = ToxicityScorerChatGPT(api_key, custom_papers)
    
    # Score components
    logger.info("Starting toxicity scoring with ChatGPT...")
    scored_count = 0
    failed_count = 0
    
    # Find indices of unscored components for progress tracking
    unscored_indices = [i for i, comp in enumerate(components) if comp['toxicity_score'] is None]
    
    for idx, comp_idx in enumerate(tqdm(unscored_indices, desc="Scoring compounds")):
        component = components[comp_idx]
        smiles = component['standardized_smiles']
        
        logger.info(f"Scoring compound {idx + 1}/{len(unscored_indices)}: {smiles}")
        
        # Score the compound
        score, explanation = scorer.score_compound(smiles)
        
        if score is not None and explanation is not None:
            # Update the component
            components[comp_idx]['toxicity_score'] = score
            components[comp_idx]['explanation'] = explanation
            scored_count += 1
            logger.info(f"Scored: {score:.2f} - {explanation[:100]}...")
        else:
            failed_count += 1
            logger.warning(f"Failed to score: {smiles}")
        
        # Save progress periodically
        if (idx + 1) % BATCH_SAVE_INTERVAL == 0:
            save_components(components, components_file, backup=False)
            logger.info(f"Progress saved: {scored_count} scored, {failed_count} failed")
        
        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)
    
    # Final save
    save_components(components, components_file, backup=True)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("CHATGPT TOXICITY SCORING COMPLETED")
    logger.info("="*60)
    logger.info(f"Total components processed: {len(unscored_indices)}")
    logger.info(f"Successfully scored: {scored_count}")
    logger.info(f"Failed to score: {failed_count}")
    logger.info(f"Success rate: {scored_count/(scored_count + failed_count)*100:.1f}%")
    
    # Show some examples
    logger.info("\nExamples of scored components:")
    logger.info("-" * 50)
    examples = 0
    for comp in components:
        if comp['toxicity_score'] is not None and examples < 5:
            logger.info(f"SMILES: {comp['standardized_smiles']}")
            logger.info(f"Score: {comp['toxicity_score']:.2f}")
            logger.info(f"Explanation: {comp['explanation'][:150]}...")
            logger.info("-" * 30)
            examples += 1


if __name__ == "__main__":
    main()