#!/usr/bin/env python3
"""
SelfIE Bridge Entity Extraction Experiment

This script uses SelfIE (Self-Interpretation of Embeddings) to detect bridge entities
from language model activations during two-hop reasoning tasks.
"""

import json
import re
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Add parent path for selfie_adapters imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from selfie_adapters import load_adapter


class SelfIEBridgeExtractor:
    """
    SelfIE-based bridge entity extraction from two-hop reasoning activations.
    
    Supports two selfie_method values in config:
    - "traditional" / "vanilla" / "untrained": Uses only scaling (no learned adapter).
      This is the baseline method from Chen et al. (2024).
    - "trained": Uses a trained SelfIE adapter checkpoint. The adapter handles
      its own scale, bias, and any other learned parameters.
    """
    
    def __init__(self, config_path: str):
        """Initialize the SelfIE bridge extractor with config."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Set random seed
        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        
        # Load model and tokenizer
        print(f"Loading model: {self.config['model_name']}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_name'],
            clean_up_tokenization_spaces=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model_name'],
            dtype=torch.bfloat16,
            device_map=self.config['device']
        )
        self.model.eval()
        
        # Get model dimensions
        self.hidden_size = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers
        
        # Load trained adapter if using trained method
        self.adapter = None
        if self.config['selfie_method'] == 'trained':
            adapter_path = self.config.get('adapter_checkpoint_path')
            if adapter_path is None:
                raise ValueError("adapter_checkpoint_path must be specified for trained method")
            print(f"Loading trained adapter from: {adapter_path}")
            self.adapter = load_adapter(adapter_path, device=self.config['device'])
            print(f"  Adapter type: {self.adapter.get_metadata()['projection_type']}")
            print(f"  Parameters: {self.adapter.get_metadata()['num_parameters']:,}")
        
        # Get special token for SelfIE placeholder
        self.placeholder_token = "<|reserved_special_token_0|>"
        self.placeholder_token_id = self.tokenizer.convert_tokens_to_ids(self.placeholder_token)
        
    def load_question(self, dataset_path: str) -> Dict:
        """Load a specific question from the dataset."""
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        question_id = self.config['question_id']
        for item in data['filtered_results']:
            if item['id'] == question_id:
                return item
        
        raise ValueError(f"Question ID {question_id} not found in dataset")
    
    def prepare_target_prompt(self, question_data: Dict) -> Tuple[str, List[int], int]:
        """Prepare the target prompt using chat template."""
        # The user message is the instruction to complete the statement
        user_message = f"Complete the statement: {question_data['question']}"
        
        # Apply chat template
        messages = [{"role": "user", "content": user_message}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Tokenize WITHOUT adding extra special tokens (chat template already has them)
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")[0].tolist()
        
        # Use configured start position (after system prompt and "Complete the statement: ")
        start_pos = self.config['start_token_position']
        
        return prompt, tokens, start_pos
    
    def get_target_activations(self, tokens: List[int]) -> torch.Tensor:
        """Run forward pass and extract activations at all layers."""
        input_ids = torch.tensor([tokens]).to(self.config['device'])
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Stack hidden states: shape (num_layers, batch=1, seq_len, hidden_size)
        hidden_states = torch.stack(outputs.hidden_states[1:])  # Skip embedding layer
        
        return hidden_states.squeeze(1)  # (num_layers, seq_len, hidden_size)
    
    def prepare_selfie_prompt(self) -> Tuple[List[int], List[int]]:
        """Prepare the SelfIE prompt and return token IDs and all placeholder positions."""
        # Build the prompt manually to have precise control
        # User message
        user_message = {"role": "user", "content": f'What is the meaning of "{self.placeholder_token}"?'}
        
        # Get the prompt up to the assistant's turn
        prompt_with_assistant_header = self.tokenizer.apply_chat_template(
            [user_message],
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Add the beginning of the assistant's response (without EOT)
        assistant_start = f'The meaning of "{self.placeholder_token}" is "'
        full_prompt = prompt_with_assistant_header + assistant_start
        
        # Tokenize WITHOUT adding extra special tokens (chat template already has them)
        tokens = self.tokenizer.encode(full_prompt, add_special_tokens=False, return_tensors="pt")[0].tolist()
        
        # Find ALL positions of the placeholder token
        placeholder_positions = [i for i, t in enumerate(tokens) if t == self.placeholder_token_id]
        
        if len(placeholder_positions) == 0:
            raise ValueError(f"Could not find placeholder token in SelfIE prompt.")
        
        return tokens, placeholder_positions
    
    def create_soft_token(self, activation: torch.Tensor) -> torch.Tensor:
        """
        Create soft token from activation using specified method.
        
        The selfie_method config option controls the transformation:
        - "traditional" / "vanilla" / "untrained": f(x) = scale * normalize(x)
        - "trained": Uses the trained adapter's transform (handles its own scale, bias, etc.)
        """
        # Convert to float32 for processing
        activation = activation.float()
        
        if self.config['selfie_method'] == 'trained':
            # Use the trained adapter - it handles normalization, scale, bias internally
            # The adapter expects shape (batch, dim) or (dim,)
            soft_token = self.adapter.transform(activation)
            return soft_token
        elif self.config['selfie_method'] in ['traditional', 'vanilla', 'untrained']:
            # Baseline method: normalize and scale
            norm = torch.norm(activation, p=2)
            normalized = activation / norm
            soft_token = self.config['scale'] * normalized
            return soft_token
        else:
            raise ValueError(
                f"Unknown selfie_method: {self.config['selfie_method']}. "
                f"Use 'traditional'/'vanilla'/'untrained' for baseline or 'trained' for adapter."
            )
    
    def generate_selfie_descriptions_batched(
        self,
        activations: torch.Tensor,
        num_generations: int
    ) -> List[List[str]]:
        """Generate multiple SelfIE descriptions for a batch of activations.
        
        Args:
            activations: Tensor of shape (batch_size, hidden_size)
            num_generations: Number of descriptions to generate per activation
            
        Returns:
            List of lists, where each inner list contains num_generations descriptions
        """
        batch_size = activations.shape[0]
        
        # Prepare SelfIE prompt (same for all batch elements)
        selfie_tokens, placeholder_positions = self.prepare_selfie_prompt()
        
        # Get embedding layer
        embed_layer = self.model.get_input_embeddings()
        
        # Create soft tokens for all activations in batch
        soft_tokens = torch.stack([
            self.create_soft_token(activation)
            for activation in activations
        ])  # (batch_size, hidden_size)
        
        # Get base embeddings for the prompt
        input_ids = torch.tensor([selfie_tokens]).to(self.config['device'])
        base_embeddings = embed_layer(input_ids)  # (1, seq_len, hidden_size)
        
        # Repeat for batch
        embeddings = base_embeddings.repeat(batch_size, 1, 1)  # (batch_size, seq_len, hidden_size)
        
        # Replace placeholder embeddings with soft tokens at ALL placeholder positions
        for placeholder_pos in placeholder_positions:
            embeddings[:, placeholder_pos] = soft_tokens.to(embeddings.dtype)
        
        # Repeat each element num_generations times for sampling
        embeddings = embeddings.repeat_interleave(num_generations, dim=0)  # (batch_size * num_generations, seq_len, hidden_size)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=embeddings,
                max_new_tokens=self.config['max_tokens'],
                temperature=self.config['temperature'],
                do_sample=self.config['temperature'] > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the generated tokens
        # Note: when using inputs_embeds, outputs only contain the newly generated tokens,
        # not the prompt (since the prompt wasn't provided as token IDs)
        all_descriptions = []
        for output in outputs:
            description = self.tokenizer.decode(output, skip_special_tokens=True)
            all_descriptions.append(description.strip())
        
        # Reshape into list of lists
        descriptions_by_activation = []
        for i in range(batch_size):
            start_idx = i * num_generations
            end_idx = start_idx + num_generations
            descriptions_by_activation.append(all_descriptions[start_idx:end_idx])
        
        return descriptions_by_activation
    
    def check_entity_match(self, description: str, entity_aliases: List[str]) -> bool:
        """Check if any entity alias appears in the description with word boundaries.
        
        Handles variations in spacing around punctuation (e.g., "J.D." matches "J. D.").
        """
        for alias in entity_aliases:
            # Create a flexible pattern that allows optional spaces around periods
            # This handles cases like "J.D. Salinger" matching "J. D. Salinger"
            flexible_pattern = re.escape(alias)
            
            # Replace escaped periods followed by escaped spaces with a pattern
            # that matches period with optional space
            flexible_pattern = re.sub(r'\\\.\\ ', r'\\.\\ ?', flexible_pattern)
            
            # Also handle periods not followed by spaces - make the space optional
            flexible_pattern = re.sub(r'\\\.(?!\\ \?)', r'\\.\\ ?', flexible_pattern)
            
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + flexible_pattern + r'\b'
            
            if re.search(pattern, description, re.IGNORECASE):
                return True
        return False
    
    def run_analysis(self, question_data: Dict) -> Dict:
        """Run the full SelfIE analysis for a question."""
        print("\nPreparing target prompt...")
        prompt_text, tokens, start_pos = self.prepare_target_prompt(question_data)
        print(f"Prompt: {prompt_text[:200]}...")
        print(f"Number of tokens: {len(tokens)}")
        print(f"Starting analysis from token position: {start_pos}")
        
        print("\nExtracting activations...")
        activations = self.get_target_activations(tokens)  # (num_layers, seq_len, hidden_size)
        
        # Determine which layers to analyze
        if self.config['layers'] == 'all':
            layers_to_analyze = list(range(self.num_layers))
        else:
            layers_to_analyze = self.config['layers']
        
        # Determine which token positions to analyze
        token_positions = list(range(start_pos, len(tokens)))
        
        print(f"\nAnalyzing {len(layers_to_analyze)} layers and {len(token_positions)} token positions...")
        print(f"Generating {self.config['num_generations']} descriptions per (layer, token) pair...")
        print(f"Total SelfIE generations: {len(layers_to_analyze) * len(token_positions) * self.config['num_generations']}")
        
        # Store results
        results = {
            'question_data': question_data,
            'config': self.config,
            'prompt_text': prompt_text,
            'tokens': tokens,
            'token_texts': [self.tokenizer.decode([t]) for t in tokens],
            'start_pos': start_pos,
            'layers': layers_to_analyze,
            'token_positions': token_positions,
            'descriptions': {},  # (layer, token_pos) -> list of descriptions
            'matches': {},  # (layer, token_pos) -> list of bools
            'match_fractions': {}  # (layer, token_pos) -> fraction
        }
        
        # Create list of all (layer, token_pos) pairs to analyze
        pairs_to_analyze = [
            (layer, token_pos)
            for layer in layers_to_analyze
            for token_pos in token_positions
        ]
        
        total_pairs = len(pairs_to_analyze)
        batch_size = self.config['batch_size']
        
        # Process in batches
        with tqdm(total=total_pairs, desc="Generating SelfIE descriptions") as pbar:
            for batch_start in range(0, total_pairs, batch_size):
                batch_end = min(batch_start + batch_size, total_pairs)
                batch_pairs = pairs_to_analyze[batch_start:batch_end]
                
                # Extract activations for this batch
                batch_activations = torch.stack([
                    activations[layer, token_pos]
                    for layer, token_pos in batch_pairs
                ])  # (batch_size, hidden_size)
                
                # Generate descriptions for all activations in batch
                batch_descriptions = self.generate_selfie_descriptions_batched(
                    batch_activations,
                    self.config['num_generations']
                )
                
                # Process results for each pair in batch
                for (layer, token_pos), descriptions in zip(batch_pairs, batch_descriptions):
                    # Check for matches
                    matches = [
                        self.check_entity_match(desc, question_data['e2_aliases'])
                        for desc in descriptions
                    ]
                    
                    # Store results
                    key = (layer, token_pos)
                    results['descriptions'][str(key)] = descriptions
                    results['matches'][str(key)] = matches
                    results['match_fractions'][str(key)] = sum(matches) / len(matches)
                    
                    pbar.update(1)
        
        return results
    
    def create_heatmap(self, results: Dict, output_path: str):
        """Create and save a heatmap of match fractions."""
        layers = results['layers']
        token_positions = results['token_positions']
        
        # Create matrix for heatmap
        heatmap_data = np.zeros((len(layers), len(token_positions)))
        
        for i, layer in enumerate(layers):
            for j, token_pos in enumerate(token_positions):
                key = str((layer, token_pos))
                heatmap_data[i, j] = results['match_fractions'][key]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(12, len(token_positions) * 0.3), 10))
        
        # Create heatmap
        sns.heatmap(
            heatmap_data,
            ax=ax,
            cmap='YlOrRd',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Fraction of descriptions with bridge entity match'},
            xticklabels=[results['token_texts'][pos] for pos in token_positions],
            yticklabels=[f"Layer {l}" for l in layers]
        )
        
        # Flip Y axis so layer 0 is at bottom, layer 31 at top
        ax.invert_yaxis()
        
        # Formatting
        ax.set_xlabel('Token', fontsize=12)
        ax.set_ylabel('Layer', fontsize=12)
        # Build title based on method
        method = self.config['selfie_method']
        if method == 'trained':
            method_str = f"Method: trained adapter"
        else:
            method_str = f"Method: {method} | Scale: {self.config['scale']}"
        
        ax.set_title(
            f"SelfIE Bridge Entity Detection Heatmap\n"
            f"Question ID: {results['question_data']['id']} | "
            f"Bridge Entity: {results['question_data']['e2_value']} | "
            f"{method_str}",
            fontsize=14,
            pad=20
        )
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nHeatmap saved to: {output_path}")
        
        plt.close()


def main():
    """Main entry point."""
    import sys
    
    # Get config path from command line or use default
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    
    # Initialize extractor
    extractor = SelfIEBridgeExtractor(config_path)
    
    # Load question
    dataset_path = extractor.config.get('dataset_path', 'twohopfact_filtered.json')
    print(f"\nLoading question from dataset: {dataset_path}")
    question_data = extractor.load_question(dataset_path)
    print(f"Question: {question_data['question']}")
    print(f"Bridge entity (e2): {question_data['e2_value']}")
    print(f"Aliases: {question_data['e2_aliases']}")
    print(f"Correct answer (e3): {question_data['correct_answer']}")
    
    # Run analysis
    results = extractor.run_analysis(question_data)
    
    # Save raw results
    output_dir = Path(extractor.config['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # Save results (excluding large description data for JSON)
    results_summary = {
        'question_data': results['question_data'],
        'config': results['config'],
        'start_pos': results['start_pos'],
        'layers': results['layers'],
        'token_positions': results['token_positions'],
        'token_texts': results['token_texts'],
        'match_fractions': results['match_fractions']
    }
    
    results_path = output_dir / f"results_q{extractor.config['question_id']}.json"
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"Results summary saved to: {results_path}")
    
    # Save full results with descriptions
    full_results_path = output_dir / f"results_full_q{extractor.config['question_id']}.json"
    with open(full_results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to: {full_results_path}")
    
    # Create heatmap
    heatmap_path = output_dir / f"heatmap_q{extractor.config['question_id']}.png"
    extractor.create_heatmap(results, str(heatmap_path))
    
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()

