"""
Enhanced LLM Explanation Pipeline for Solubility Model
=======================================================

This pipeline generates model-grounded explanations by:
1. Selecting 15 accurate and 15 inaccurate predictions
2. Generating LLM-based molecule descriptions (without solubility bias)
3. Compiling evidence from SHAP, decision paths, attention weights, structural features
4. Using multi-step LLM reasoning to produce grounded explanations

Enhanced explanation pipeline for solubility predictions.
"""

import os
import json
import time
import joblib
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from catboost import Pool
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import google.generativeai as genai

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the explanation pipeline."""
    
    # Paths
    test_file: str = "data/test.csv"
    store_dir: str = "feature_store"
    model_dir: str = "model"
    cgboost_dir: str = "cgboost_explanations"
    output_dir: str = "enhanced_explanations"
    transformer_path: str = "transformer.pth"
    feature_thresholds_file: str = "threshold_analysis/feature_thresholds.json"
    
    # Sample selection thresholds
    good_prediction_threshold: float = 0.30  # Error below this = accurate
    bad_prediction_threshold: float = 1.00   # Error above this = inaccurate
    num_good_samples: int = 15
    num_bad_samples: int = 15
    
    # API configuration
    gemini_model: str = "gemini-3-flash-preview"
    max_retries: int = 3
    retry_delay: float = 2.0
    rate_limit_delay: float = 1.0
    timeout: float = 300.0  # 5 minutes timeout
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"



# ============================================================================
# API KEY MANAGER - Handles multiple API keys with rotation
# ============================================================================

class APIKeyManager:
    """
    Manages multiple Gemini API keys with automatic rotation on rate limit.
    
    Usage:
        manager = APIKeyManager(["key1", "key2", "key3"])
        key = manager.get_current_key()
        # If rate limited:
        manager.rotate_key()
    """
    
    def __init__(self, api_keys: List[str]):
        if not api_keys:
            raise ValueError("At least one API key must be provided")
        
        self.api_keys = api_keys
        self.current_index = 0
        self.exhausted_keys = set()
        self.key_usage_count = {i: 0 for i in range(len(api_keys))}
        
    def get_current_key(self) -> str:
        """Get the current active API key."""
        return self.api_keys[self.current_index]
    
    def rotate_key(self, mark_exhausted: bool = True) -> bool:
        """
        Rotate to the next available API key.
        
        Args:
            mark_exhausted: If True, mark current key as exhausted
            
        Returns:
            True if rotation successful, False if all keys exhausted
        """
        if mark_exhausted:
            self.exhausted_keys.add(self.current_index)
            print(f"⚠ API key {self.current_index + 1}/{len(self.api_keys)} exhausted")
        
        # Find next available key
        for _ in range(len(self.api_keys)):
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            if self.current_index not in self.exhausted_keys:
                print(f"✓ Switched to API key {self.current_index + 1}/{len(self.api_keys)}")
                return True
        
        print("✗ All API keys exhausted!")
        return False
    
    def record_usage(self):
        """Record usage of current key."""
        self.key_usage_count[self.current_index] += 1
    
    def get_status(self) -> Dict:
        """Get status of all keys."""
        return {
            "total_keys": len(self.api_keys),
            "current_index": self.current_index,
            "exhausted_count": len(self.exhausted_keys),
            "usage_counts": self.key_usage_count
        }
    
    def has_available_keys(self) -> bool:
        """Check if any keys are still available."""
        return len(self.exhausted_keys) < len(self.api_keys)


# ============================================================================
# LLM CALLER - Handles API calls with retry and key rotation
# ============================================================================

class GeminiCaller:
    """Handles Gemini API calls with retry logic and key rotation."""
    
    def __init__(self, key_manager: APIKeyManager, config: PipelineConfig):
        self.key_manager = key_manager
        self.config = config
        self._configure_current_key()
    
    def _configure_current_key(self):
        """Configure genai with current API key."""
        genai.configure(api_key=self.key_manager.get_current_key())
        self.model = genai.GenerativeModel(self.config.gemini_model)
    
    def call(self, prompt: str, system_prompt: str = None, temperature: float = 0.3) -> Tuple[str, bool]:
        """
        Call Gemini API with retry and key rotation.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Generation temperature
            
        Returns:
            Tuple of (response_text, success_flag)
        """
        messages = []
        if system_prompt:
            messages.append(system_prompt)
        messages.append(prompt)
        
        from google.api_core import exceptions
        
        for attempt in range(self.config.max_retries * len(self.key_manager.api_keys)):
            try:
                # Add request options with timeout
                response = self.model.generate_content(
                    messages,
                    generation_config={
                        "temperature": temperature,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_output_tokens": 32768,  # Increased from 8192 to handle dense condensation outputs
                    },
                    request_options={'timeout': self.config.timeout}
                )
                
                # Check for safety blocks or other finish reasons that aren't success
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                     print(f"  ⚠ Blocked: {response.prompt_feedback.block_reason}")
                     return "ERROR: Response blocked by safety settings", False

                # Handle cases where response might be empty or incomplete
                try:
                    text_content = response.text
                except Exception as e:
                     print(f"  ⚠ Could not extract text: {e}")
                     # Sometimes candidates are present but text access fails if safety filters triggered on the response
                     if response.candidates:
                         print(f"  Finish reason: {response.candidates[0].finish_reason}")
                     return "ERROR: Empty or blocked response", False

                # NEW: Check finish_reason to detect truncated responses
                if response.candidates and len(response.candidates) > 0:
                    finish_reason = response.candidates[0].finish_reason
                    # finish_reason is an enum, convert to string for comparison
                    finish_reason_name = str(finish_reason)
                    
                    # Log finish reason for debugging
                    if "MAX_TOKENS" in finish_reason_name or "LENGTH" in finish_reason_name:
                        print(f"  ⚠ WARNING: Response truncated! finish_reason={finish_reason_name}")
                        print(f"  ⚠ Response length: {len(text_content)} chars")
                        print(f"  ⚠ This response is INCOMPLETE and may cut off mid-sentence")
                        # Still return the truncated response but mark as failed so caller can handle
                        return text_content, False
                    elif "STOP" not in finish_reason_name and finish_reason_name != "1":  # 1 is the enum value for STOP
                        print(f"  ⚠ Unexpected finish_reason: {finish_reason_name}")

                self.key_manager.record_usage()
                return text_content, True
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if rate limit error
                if "quota" in error_str or "rate" in error_str or "limit" in error_str:
                    print(f"Rate limit hit: {e}")
                    if self.key_manager.rotate_key():
                        self._configure_current_key()
                        continue
                    else:
                        return f"ERROR: All API keys exhausted. Last error: {e}", False
                
                # Check for timeout
                if "deadline" in error_str or "timeout" in error_str:
                     print(f"  ⚠ Timeout after {self.config.timeout}s")
                
                # Other errors - retry with current key
                if attempt < self.config.max_retries - 1:
                    print(f"API error (attempt {attempt + 1}): {e}")
                    time.sleep(self.config.retry_delay)
                else:
                    return f"ERROR: {e}", False
        
        return "ERROR: Max retries exceeded", False


# ============================================================================
# SAMPLE SELECTOR - Selects good and bad predictions
# ============================================================================

class SampleSelector:
    """Selects samples based on prediction accuracy."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device
        
    def load_models_and_data(self):
        """Load all required models and data."""
        print("Loading models and data...")
        
        # Load test data
        self.df_test = pd.read_csv(self.config.test_file)
        
        # Load feature stores
        self.sol_raw = pd.read_parquet(
            os.path.join(self.config.store_dir, "solute_raw.parquet")
        ).set_index("SMILES_KEY")
        self.solv_raw = pd.read_parquet(
            os.path.join(self.config.store_dir, "solvent_raw.parquet")
        ).set_index("SMILES_KEY")
        self.sol_council = pd.read_parquet(
            os.path.join(self.config.store_dir, "solute_council.parquet")
        ).set_index("SMILES_KEY")
        self.solv_council = pd.read_parquet(
            os.path.join(self.config.store_dir, "solvent_council.parquet")
        ).set_index("SMILES_KEY")
        
        # Load transformer
        from train_transformer import InteractionTransformer
        self.transformer = InteractionTransformer().to(self.device)
        self.transformer.load_state_dict(
            torch.load(self.config.transformer_path, map_location=self.device)
        )
        self.transformer.eval()
        
        # Load CatBoost model and selector
        self.catboost_model = joblib.load(
            os.path.join(self.config.model_dir, "model.joblib")
        )
        self.selector = joblib.load(
            os.path.join(self.config.model_dir, "selector.joblib")
        )
        
        # Get feature names
        feature_map = pd.read_csv(
            os.path.join(self.config.cgboost_dir, "trained_feature_map.csv")
        )
        self.feature_names = feature_map["Feature"].tolist()
        
        # Council feature names
        self.council_feature_names = self.sol_council.columns.tolist()
        
        print(f"  ✓ Loaded {len(self.df_test)} test samples")
        print(f"  ✓ Loaded {len(self.feature_names)} trained features")
        
    def generate_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate features for a DataFrame of samples."""
        X_sol = self.sol_council.loc[df["Solute"]].values.astype(np.float32)
        X_solv = self.solv_council.loc[df["Solvent"]].values.astype(np.float32)
        
        embeds, attns = [], []
        
        with torch.no_grad():
            for i in range(len(df)):
                sol = torch.tensor(X_sol[i:i+1]).to(self.device)
                solv = torch.tensor(X_solv[i:i+1]).to(self.device)
                _, feats, attn = self.transformer(sol, solv)
                embeds.append(feats.cpu().numpy())
                attns.append(attn.cpu().numpy())
        
        X_embed = np.vstack(embeds)
        attention = np.vstack(attns)
        
        T = df["Temperature"].values.reshape(-1, 1)
        T_inv = (1000 / df["Temperature"]).values.reshape(-1, 1)
        Tm = self.sol_raw.loc[df["Solute"], "pred_Tm"].values.reshape(-1, 1)
        T_red = T / Tm
        
        X_reshaped = X_embed.reshape(-1, 24, 32)
        X_mod = np.linalg.norm(X_reshaped, axis=2)
        X_sign = np.sign(X_reshaped.mean(axis=2))
        X_interact = (X_sign * X_mod) * T_inv
        
        X_raw = np.hstack([
            self.sol_raw.loc[df["Solute"]].values,
            self.solv_raw.loc[df["Solvent"]].values
        ])
        
        X_full = np.hstack([X_raw, X_interact, Tm, T_red, T, T_inv])
        return X_full, attention
    
    def select_samples(self) -> Dict[str, List[Dict]]:
        """
        Select samples with good and bad predictions.
        
        Returns:
            Dict with 'good' and 'bad' lists of sample info dicts
        """
        print("\nGenerating predictions for all test samples...")
        
        X_full, attention_weights = self.generate_features(self.df_test)
        X_pruned = self.selector.transform(X_full)
        predictions = self.catboost_model.predict(X_pruned)
        
        # Calculate errors
        true_values = self.df_test["LogS"].values
        errors = np.abs(true_values - predictions)
        
        # Get SHAP values for all samples
        pool = Pool(X_pruned, feature_names=self.feature_names)
        shap_vals = self.catboost_model.get_feature_importance(pool, type="ShapValues")
        shap_vals = np.array(shap_vals)[:, :-1]
        leaf_paths = self.catboost_model.calc_leaf_indexes(pool)
        
        print(f"  Error distribution: min={errors.min():.4f}, max={errors.max():.4f}, mean={errors.mean():.4f}")
        
        # Select good samples (lowest error)
        good_mask = errors < self.config.good_prediction_threshold
        good_indices = np.where(good_mask)[0]
        good_sorted = good_indices[np.argsort(errors[good_indices])]
        selected_good = good_sorted[:self.config.num_good_samples]
        
        # Select bad samples (highest error)
        bad_mask = errors > self.config.bad_prediction_threshold
        bad_indices = np.where(bad_mask)[0]
        bad_sorted = bad_indices[np.argsort(-errors[bad_indices])]
        selected_bad = bad_sorted[:self.config.num_bad_samples]
        
        print(f"  ✓ Selected {len(selected_good)} good predictions (error < {self.config.good_prediction_threshold})")
        print(f"  ✓ Selected {len(selected_bad)} bad predictions (error > {self.config.bad_prediction_threshold})")
        
        # Compile sample info
        def compile_sample_info(idx: int) -> Dict:
            row = self.df_test.iloc[idx]
            return {
                "index": int(idx),
                "solute": row["Solute"],
                "solvent": row["Solvent"],
                "temperature": float(row["Temperature"]),
                "y_true": float(row["LogS"]),  # Keep for metadata, not sent to LLM
                "y_pred": float(predictions[idx]),
                "abs_error": float(errors[idx]),  # Keep for metadata, not sent to LLM
                "cross_attention_weights": attention_weights[idx].tolist(),
                "council_feature_names": self.council_feature_names,  # For interpreting attention
                "shap_values": dict(zip(self.feature_names, shap_vals[idx].tolist())),
                "leaf_path": leaf_paths[idx].tolist(),
                "structural_features": self._extract_structural_features(idx, X_pruned)
            }
        
        return {
            "good": [compile_sample_info(i) for i in selected_good],
            "bad": [compile_sample_info(i) for i in selected_bad]
        }
    
    def _extract_structural_features(self, idx: int, X_pruned: np.ndarray) -> Dict:
        """Extract key structural features for a sample."""
        # Define key structural features to highlight
        key_features = [
            "Solute_num_C", "Solute_num_O", "Solute_num_N", "Solute_num_Cl", 
            "Solute_num_S", "Solute_num_F", "Solute_total_atoms",
            "Solute_MolLogP", "Solute_MolWt", "Solute_TPSA", "Solute_BertzCT",
            "Solute_NumHDonors", "Solute_NumHAcceptors", "Solute_NumRotatableBonds",
            "Solute_HallKierAlpha", "Solute_LabuteASA", "Solute_HeavyAtomCount",
            "Solute_NumAromaticRings", "Solute_MaxPartialCharge", "Solute_MinPartialCharge",
            "Solvent_num_C", "Solvent_num_O", "Solvent_num_N", 
            "Solvent_MolLogP", "Solvent_MolWt", "Solvent_TPSA",
            "Solvent_NumHDonors", "Solvent_NumHAcceptors", "Solvent_NumRotatableBonds",
            "pred_Tm", "T", "T_inv", "T_red"
        ]
        
        structural = {}
        for feat in key_features:
            if feat in self.feature_names:
                feat_idx = self.feature_names.index(feat)
                structural[feat] = float(X_pruned[idx, feat_idx])
        
        return structural


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

MOLECULE_DESCRIPTION_SYSTEM = """You are a chemist describing molecular structures based on SMILES.

Focus on:
- Core structure (rings, chains, branching)
- Key functional groups and their positions
- Polarity, hydrogen bonding capacity, size

Be direct and specific. No solubility predictions."""

MOLECULE_DESCRIPTION_PROMPT = """Describe the following molecule based ONLY on its SMILES structure.

SMILES: {smiles}

Cover structure, functional groups, and physicochemical character in 3-4 sentences. Be specific about what makes this molecule distinctive."""


EVIDENCE_SUMMARY_SYSTEM = """You are interpreting machine learning model data.

Your job: Find patterns and signal in the numbers, not just list them.

Key skills:
- Identify which feature groups dominate (positive vs negative)
- Spot unusual values or contradictions
- Compare solute vs solvent contributions
- Note when features work together or oppose each other

Be analytical, not just descriptive."""

EVIDENCE_SUMMARY_PROMPT = """Analyze the model evidence for this solute-solvent pair.

=== SHAP Feature Contributions (Top 20) ===
{shap_features}

=== Group Contributions ===
{group_contributions}

=== Structural Feature Values ===
{structural_features}

Identify:
1. What's driving the prediction? (cite top 3-5 features with values)
2. Are there competing effects? (positive vs negative contributions)
3. Does the solute or solvent dominate, or is it balanced?
4. Any unusual feature values that stand out?

Write 4-6 sentences of analysis. Focus on insights, not lists."""


DECISION_ANALYSIS_SYSTEM = """You are a machine learning interpretability expert analyzing decision trees.

Look for:
- Which feature types the model prioritized
- Patterns in the cross-attention (which properties interact)
- Consistency or conflicts in the decision path

This is about understanding model behavior through the data."""

DECISION_ANALYSIS_PROMPT = """Analyze the model's reasoning for this prediction.

=== Prediction Info ===
Predicted LogS: {y_pred:.4f}
Temperature (K): {temperature:.2f}

=== Cross-Attention Weights (Solute→Solvent Interactions) ===
The transformer uses cross-attention from solute features (query) to solvent features (key/value).
These weights show which solvent properties the model attends to for each solute property.

**Top Cross-Attention Interactions:**
{cross_attention_summary}

=== Leaf Path Statistics ===
Number of trees: {num_trees}
Path variability: {path_stats}

=== Key Decision Features ===
{top_features}

Write 4-5 sentences of analysis addressing:
- Which feature interactions received most attention? What does that suggest about the compatibility?
- What pattern of features led to this prediction value?
- Is the decision path consistent or mixed?

Be specific about attention weights and feature contributions."""

INTEGRATION_SYSTEM = """You generate model explanations grounded in evidence.

Core rules:
1. Every claim must cite specific values from the provided data
2. Synthesize insights - don't just repeat what's already stated
3. Identify mechanistic patterns (e.g., "high TPSA + low HBA indicates...")
4. Note uncertainties or conflicts in the evidence
5. Be concise - eliminate filler words
6. IMPORTANT: Do NOT use raw feature names like "Solute_MACCS_105", "Solvent_Morgan_283", "Interact_MolLogP", "BertzCT", etc. in your explanation. Instead, translate them to chemical concepts (e.g., "structural fingerprint features", "topological complexity", "hydrogen bonding capacity", "molecular interaction patterns"). Say "the model's analysis suggests..." rather than "Solute_MACCS_105 indicates..."
7. Keep these boundaries in mind while explaining the prediction: 
Very highly soluble     : LogS >=  0.0
Highly soluble          : 0.0  > LogS >= -1.0
Moderately soluble      : -1.0 > LogS >= -2.5
Poorly soluble          : -2.5 > LogS >= -4.0
Highly insoluble        : LogS <  -4.0

Your goal: Explain WHY the model predicted this value based on the evidence, in plain chemical language."""

INTEGRATION_PROMPT = """Generate a comprehensive explanation for this solubility prediction.

=== Molecule Descriptions ===
**Solute ({solute_smiles}):**
{solute_description}

**Solvent ({solvent_smiles}):**
{solvent_description}

=== Evidence Summary (from Stage 1) ===
{evidence_summary}

=== Decision Analysis (from Stage 2) ===
{decision_analysis}

=== Prediction ===
- Predicted LogS: {y_pred:.4f}
- Temperature: {temperature:.2f} K

Structure your explanation as:

## Prediction & Key Drivers
State the predicted value. Identify the 2-3 dominant factors with specific values.

## Solute-Solvent Compatibility
Based on cross-attention and interaction features, what compatibility or mismatch drives the result? Cite specific feature pairs and their contributions.

## Mechanistic Interpretation
What does the pattern of contributions suggest about the dissolution process? Connect molecular properties to the prediction.

## Confidence & Caveats
Based on evidence consistency and magnitude, how reliable is this explanation? Note any conflicting signals or unusual patterns.

---

IMPORTANT FORMATTING RULES:
- Do NOT use raw feature names (e.g., Solute_MACCS_105, BertzCT, Interact_MolLogP). Translate them to plain chemical language.
- Write in plain text format, avoiding markdown headers or formatting.
- Keep each section to 2-4 sentences. Cite specific values. Focus on insights that connect the dots."""


# Stage 4: Condensation prompts
CONDENSATION_SYSTEM = """You are an expert scientific editor. Your task is to condense a detailed solubility explanation into a concise, information-dense summary.

Rules:
1. PRESERVE all numerical values (LogS, TPSA, MolLogP, temperatures, percentages, contribution values)
2. REMOVE redundancy, filler phrases, and repetitive statements
3. MAINTAIN the scientific accuracy and causal reasoning
4. Use precise, technical language appropriate for chemistry researchers
5. Do NOT add information not present in the original
6. Do NOT use markdown formatting - output plain text only
7. Do NOT use raw feature names - keep the chemical language from the input
8. End Sentences with a period.
9. Ensure that the explanation is complete."""

CONDENSATION_PROMPT = """Condense the following solubility explanation into exactly 4 dense paragraphs:

=== Original Explanation ===
{original_explanation}

=== Required Structure ===

Paragraph 1 - PREDICTION SUMMARY:
State the LogS prediction and temperature. Identify the dominant driver category (solute/solvent/interaction) with its percentage. Highlight the 2-3 most influential molecular properties with their values.

Paragraph 2 - SOLUTE-SOLVENT DYNAMICS:
Describe the key cross-attention interactions and what they reveal about compatibility. Quantify the solvent's contribution and explain why it helps or hinders dissolution.

Paragraph 3 - MECHANISTIC INTERPRETATION:
Explain the dissolution mechanism in chemical terms. Identify whether dissolution is solute-limited, solvent-limited, or interaction-limited. Connect specific molecular features to the predicted outcome.

Paragraph 4 - CONFIDENCE & UNCERTAINTY:
Assess prediction reliability based on model statistics. Note any conflicting signals between feature groups. Indicate if the prediction lies within expected model behavior.

Output ONLY the 4 paragraphs with no headers, labels, or markdown. Each paragraph should be 3-5 sentences of dense, information-rich text."""


# Stage 3.5: Validation prompts
VALIDATION_SYSTEM = """You are a scientific fact-checker for machine learning model explanations.

Your task: Verify that every claim in the explanation is grounded in the provided model evidence.

Check for:
1. **Structural accuracy**: Do the molecule descriptions used in the explanation matches the actual SMILES?
2. **Numerical accuracy**: Do cited values match the source data exactly?
3. **Feature references**: Are mentioned features actually in the top contributors?
4. **Magnitude claims**: Are statements about "dominant", "major", "minor" contributions accurate?
5. **Attention patterns**: Do cross-attention claims match the actual weights?
6. **Logical consistency**: Do the conclusions follow from the evidence?

These are the boundaries defined by us:
Very highly soluble     : LogS >=  0.0
Highly soluble          : 0.0  > LogS >= -1.0
Moderately soluble      : -1.0 > LogS >= -2.5
Poorly soluble          : -2.5 > LogS >= -4.0
Highly insoluble        : LogS <  -4.0

Flag as hallucination:
- Descriptions that contradict the SMILES structure (e.g. claiming a ring in a linear chain)
- Invented feature values or contributions
- Features mentioned that aren't in top 20 SHAP values
- Incorrect percentages or group contributions
- Misrepresented attention patterns
- Unsupported mechanistic claims

Output JSON format:
{
  "is_valid": true/false,
  "issues_found": ["issue 1", "issue 2", ...],
  "corrections": ["correction 1", "correction 2", ...],
  "corrected_explanation": "Full text of the corrected explanation (if invalid) or null (if valid)",
  "confidence": "high/medium/low"
}

Be strict but fair. Minor rounding differences are acceptable. Focus on factual grounding.
IMPORTANT: Keep the `corrected_explanation` concise (approx 4 paragraphs). Do not unnecessarily expand it."""

VALIDATION_PROMPT = """Validate the following explanation against the source model evidence.

=== EXPLANATION TO VALIDATE ===
{explanation}

=== SOURCE EVIDENCE ===

**Molecule Information:**
- Solute SMILES: {solute_smiles}
  (Description used in explanation: "{solute_description}")
- Solvent SMILES: {solvent_smiles}
  (Description used in explanation: "{solvent_description}")

**Top 20 SHAP Contributions:**
{shap_features}

**Unusual Features:**
{unusual_features}

**Key Decision Features (from Decision Analysis):**
{decision_features}

**Group Contributions:**
{group_contributions}

**Cross-Attention Summary (Top 10 Interactions):**
{cross_attention_summary}

**Structural Features:**
{structural_features}

**Tree Statistics:**
{tree_stats}

**Prediction:**
- Predicted LogS: {y_pred:.4f}
- Temperature: {temperature:.2f} K

=== VALIDATION TASK ===

Check every numerical claim, feature reference, and mechanistic statement in the explanation.
Verify that the qualitative descriptions match the actual chemical structure defined by the SMILES.

For each issue found:
1. Quote the problematic statement
2. Explain why it's incorrect or unsupported (or contradicts the SMILES)
3. Provide the correct information from source data

If issues are found, you MUST provide a "corrected_explanation" in the JSON that:
- Retains the structure and tone of the original
- Fixes all factual errors using source data
- Removes unsupported claims
- Takes care to NOT introduce new hallucinations

Output your findings in JSON format as specified in the system prompt."""


# ============================================================================
# EXPLANATION GENERATOR
# ============================================================================

def format_shap_with_insights(shap_dict, top_n=20):
    """Format SHAP values while highlighting patterns."""
    items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    
    # Separate positive and negative
    positive = [f"  {name}: +{value:.4f}" for name, value in items if value > 0]
    negative = [f"  {name}: {value:.4f}" for name, value in items if value < 0]
    
    output = []
    if positive:
        output.append("POSITIVE CONTRIBUTIONS (increasing solubility):")
        output.extend(positive[:10])
    if negative:
        output.append("\nNEGATIVE CONTRIBUTIONS (decreasing solubility):")
        output.extend(negative[:10])
    
    return "\n".join(output)

def identify_unusual_features(structural_features, threshold_file="threshold_analysis/feature_thresholds.json"):
    """
    Flag features with unusual values based on pre-computed thresholds.
    
    Args:
        structural_features: Dictionary of feature names to values
        threshold_file: Path to JSON file with per-feature thresholds
        
    Returns:
        String listing unusual features or message if none found
    """
    unusual = []
    
    # Load per-feature thresholds (computed from training data)
    try:
        with open(threshold_file, 'r') as f:
            thresholds = json.load(f)
    except FileNotFoundError:
        # Fallback to a few hardcoded examples if threshold file doesn't exist
        print(f"  ⚠ Warning: Could not load {threshold_file}, using fallback thresholds")
        thresholds = {
            "Solute_TPSA": {'low': 20, 'high': 140},
            "Solute_MolWt": {'low': 100, 'high': 500},
            "T": {'low': 273, 'high': 373},
        }
        
        for feat, value in structural_features.items():
            if feat in thresholds:
                low, high = thresholds[feat]['low'], thresholds[feat]['high']
                if value < low or value > high:
                    unusual.append(f"  {feat}={value:.2f} (unusual - typically {low}-{high})")
        
        return "\n".join(unusual) if unusual else "  No unusual values detected"
    
    # Check each structural feature against its threshold
    for feat, value in structural_features.items():
        if feat in thresholds:
            low = thresholds[feat]['low']
            high = thresholds[feat]['high']
            mean = thresholds[feat]['mean']
            std = thresholds[feat]['std']
            
            if value < low:
                z_score = (value - mean) / std if std > 0 else 0
                unusual.append(
                    f"  {feat}={value:.4f} (LOW: {z_score:.2f}σ below mean, "
                    f"range: [{low:.2f}, {high:.2f}])"
                )
            elif value > high:
                z_score = (value - mean) / std if std > 0 else 0
                unusual.append(
                    f"  {feat}={value:.4f} (HIGH: {z_score:.2f}σ above mean, "
                    f"range: [{low:.2f}, {high:.2f}])"
                )
    
    return "\n".join(unusual) if unusual else "  No unusual values detected"




class ExplanationGenerator:
    """Generates multi-step explanations using LLM."""
    
    def __init__(self, gemini_caller: GeminiCaller, config: PipelineConfig):
        self.caller = gemini_caller
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.prompts_saved = False
        
    def setup_output_dirs(self):
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "prompts").mkdir(exist_ok=True)
        (self.output_dir / "good_predictions").mkdir(exist_ok=True)
        (self.output_dir / "bad_predictions").mkdir(exist_ok=True)
        
    def save_prompts(self):
        """Save all prompt templates."""
        if self.prompts_saved:
            return
            
        prompts_dir = self.output_dir / "prompts"
        
        prompts = {
            "molecule_description_system.md": MOLECULE_DESCRIPTION_SYSTEM,
            "molecule_description_prompt.md": MOLECULE_DESCRIPTION_PROMPT,
            "evidence_summary_system.md": EVIDENCE_SUMMARY_SYSTEM,
            "evidence_summary_prompt.md": EVIDENCE_SUMMARY_PROMPT,
            "decision_analysis_system.md": DECISION_ANALYSIS_SYSTEM,
            "decision_analysis_prompt.md": DECISION_ANALYSIS_PROMPT,
            "integration_system.md": INTEGRATION_SYSTEM,
            "integration_prompt.md": INTEGRATION_PROMPT,
        }
        
        for filename, content in prompts.items():
            (prompts_dir / filename).write_text(content)
        
        self.prompts_saved = True
        print(f"  ✓ Saved prompt templates to {prompts_dir}")
    
    def generate_molecule_description(self, smiles: str) -> Tuple[str, Dict]:
        """Generate description for a molecule."""
        prompt = MOLECULE_DESCRIPTION_PROMPT.format(smiles=smiles)
        response, success = self.caller.call(prompt, MOLECULE_DESCRIPTION_SYSTEM, temperature=0.2)
        
        return response, {
            "system_prompt": MOLECULE_DESCRIPTION_SYSTEM,
            "user_prompt": prompt,
            "success": success
        }
    
    def generate_evidence_summary(self, sample: Dict) -> Tuple[str, Dict]:
        """Enhanced Stage 1: Summarize evidence with pattern highlighting."""
        
        # Format SHAP with insights
        shap_str = format_shap_with_insights(sample["shap_values"])
        
        # Group contributions with interpretation
        group_contribs = {"Solute": 0, "Solvent": 0, "Interact": 0, "Thermo": 0}
        for name, value in sample["shap_values"].items():
            if name.startswith("Solute_"): group_contribs["Solute"] += value
            elif name.startswith("Solvent_"): group_contribs["Solvent"] += value
            elif name.startswith("Interact_"): group_contribs["Interact"] += value
            else: group_contribs["Thermo"] += value
        
        # Add interpretation
        dominant = max(group_contribs.items(), key=lambda x: abs(x[1]))
        total_contrib = sum(abs(v) for v in group_contribs.values())
        
        group_str_lines = [f"  {k}: {v:+.4f}" for k, v in group_contribs.items()]
        group_str_lines.append(f"\n  Dominant group: {dominant[0]} ({abs(dominant[1])/total_contrib*100:.1f}% of total signal)")
        group_str = "\n".join(group_str_lines)
        
        # Unusual features (using thresholds from config)
        unusual_str = identify_unusual_features(
            sample["structural_features"], 
            threshold_file=self.config.feature_thresholds_file
        )

        
        # Structural features (condensed)
        struct_items = sorted(sample["structural_features"].items(), 
                            key=lambda x: abs(x[1]), reverse=True)[:15]
        struct_str = "\n".join([f"  {k}: {v:.4f}" for k, v in struct_items])
        
        prompt = EVIDENCE_SUMMARY_PROMPT.format(
            shap_features=shap_str,
            group_contributions=group_str,
            structural_features=struct_str + "\n\nUNUSUAL VALUES:\n" + unusual_str
        )
        
        response, success = self.caller.call(prompt, EVIDENCE_SUMMARY_SYSTEM, temperature=0.3)
        time.sleep(self.config.rate_limit_delay)
        
        return response, {
            "system_prompt": EVIDENCE_SUMMARY_SYSTEM,
            "user_prompt": prompt,
            "success": success
        }

    
    def generate_decision_analysis(self, sample: Dict) -> Tuple[str, Dict]:
        """Stage 2: Analyze decision path and cross-attention."""
        # Cross-attention summary with council feature names
        attn = np.array(sample["cross_attention_weights"])
        council_names = sample["council_feature_names"]
        
        # Get top cross-attention interactions
        top_indices = np.argsort(attn.flatten())[-10:][::-1]
        attn_items = []
        for idx in top_indices:
            i, j = idx // attn.shape[1], idx % attn.shape[1]
            solute_feat = council_names[i] if i < len(council_names) else f"Solute_{i}"
            solvent_feat = council_names[j] if j < len(council_names) else f"Solvent_{j}"
            attn_items.append(f"  {solute_feat} → {solvent_feat}: {attn[i, j]:.4f}")
        attn_str = "\n".join(attn_items)
        
        # Top features
        top_feats = sorted(
            sample["shap_values"].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10]
        feat_str = "\n".join([
            f"  {name}: {value:+.4f} ({'positive' if value > 0 else 'negative'} contribution)"
            for name, value in top_feats
        ])
        
        # Path stats
        path = sample["leaf_path"]
        prompt = DECISION_ANALYSIS_PROMPT.format(
            y_pred=sample["y_pred"],
            temperature=sample["temperature"],
            cross_attention_summary=attn_str,
            num_trees=len(path),
            path_stats=f"Mean leaf: {np.mean(path):.1f}, Std: {np.std(path):.1f}",
            top_features=feat_str
        )
        
        response, success = self.caller.call(prompt, DECISION_ANALYSIS_SYSTEM, temperature=0.3)
        time.sleep(self.config.rate_limit_delay)
        
        return response, {
            "system_prompt": DECISION_ANALYSIS_SYSTEM,
            "user_prompt": prompt,
            "success": success
        }
    
    def generate_final_explanation(
        self,
        sample: Dict,
        solute_desc: str,
        solvent_desc: str,
        evidence_summary: str,
        decision_analysis: str
    ) -> Tuple[str, Dict]:
        """Stage 3: Generate integrated explanation (blinded to prediction quality)."""
        prompt = INTEGRATION_PROMPT.format(
            solute_smiles=sample["solute"],
            solute_description=solute_desc,
            solvent_smiles=sample["solvent"],
            solvent_description=solvent_desc,
            evidence_summary=evidence_summary,
            decision_analysis=decision_analysis,
            y_pred=sample["y_pred"],
            temperature=sample["temperature"]
        )
        
        response, success = self.caller.call(prompt, INTEGRATION_SYSTEM, temperature=0.4)
        time.sleep(self.config.rate_limit_delay)
        
        return response, {
            "system_prompt": INTEGRATION_SYSTEM,
            "user_prompt": prompt,
            "success": success
        }
    
    def generate_validation(
        self,
        sample: Dict,
        explanation: str,
        solute_desc: str = "",
        solvent_desc: str = ""
    ) -> Tuple[Dict, Dict]:
        """Stage 3.5: Validate explanation against source evidence."""
        
        # Format SHAP features
        shap_str = format_shap_with_insights(sample["shap_values"], top_n=20)
        
        # Group contributions
        group_contribs = {"Solute": 0, "Solvent": 0, "Interact": 0, "Thermo": 0}
        for name, value in sample["shap_values"].items():
            if name.startswith("Solute_"): group_contribs["Solute"] += value
            elif name.startswith("Solvent_"): group_contribs["Solvent"] += value
            elif name.startswith("Interact_"): group_contribs["Interact"] += value
            else: group_contribs["Thermo"] += value
        
        dominant = max(group_contribs.items(), key=lambda x: abs(x[1]))
        total_contrib = sum(abs(v) for v in group_contribs.values())
        
        group_str_lines = [f"  {k}: {v:+.4f}" for k, v in group_contribs.items()]
        group_str_lines.append(f"\n  Dominant group: {dominant[0]} ({abs(dominant[1])/total_contrib*100:.1f}% of total signal)")
        group_str = "\n".join(group_str_lines)
        
        # Cross-attention summary
        attn = np.array(sample["cross_attention_weights"])
        council_names = sample["council_feature_names"]
        
        top_indices = np.argsort(attn.flatten())[-10:][::-1]
        attn_items = []
        for idx in top_indices:
            i, j = idx // attn.shape[1], idx % attn.shape[1]
            solute_feat = council_names[i] if i < len(council_names) else f"Solute_{i}"
            solvent_feat = council_names[j] if j < len(council_names) else f"Solvent_{j}"
            attn_items.append(f"  {solute_feat} → {solvent_feat}: {attn[i, j]:.4f}")
        attn_str = "\n".join(attn_items)
        
        # Structural features
        struct_items = sorted(sample["structural_features"].items(), 
                            key=lambda x: abs(x[1]), reverse=True)[:15]
        struct_str = "\n".join([f"  {k}: {v:.4f}" for k, v in struct_items])
        
        # Tree statistics
        path = sample["leaf_path"]
        tree_stats = f"Number of trees: {len(path)}, Mean leaf: {np.mean(path):.1f}, Std: {np.std(path):.1f}"
        
        # Unusual features detection
        unusual_str = identify_unusual_features(
            sample["structural_features"], 
            threshold_file=self.config.feature_thresholds_file
        )

        # Top decision features (same as in Stage 2)
        top_feats = sorted(
            sample["shap_values"].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10]
        decision_feat_str = "\n".join([
            f"  {name}: {value:+.4f} ({'positive' if value > 0 else 'negative'} contribution)"
            for name, value in top_feats
        ])

        # Format validation prompt
        prompt = VALIDATION_PROMPT.format(
            explanation=explanation,
            solute_smiles=sample["solute"],
            solute_description=solute_desc,
            solvent_smiles=sample["solvent"],
            solvent_description=solvent_desc,
            shap_features=shap_str,
            unusual_features=unusual_str,
            decision_features=decision_feat_str,
            group_contributions=group_str,
            cross_attention_summary=attn_str,
            structural_features=struct_str,
            tree_stats=tree_stats,
            y_pred=sample["y_pred"],
            temperature=sample["temperature"]
        )
        
        response, success = self.caller.call(prompt, VALIDATION_SYSTEM, temperature=0.1)
        time.sleep(self.config.rate_limit_delay)
        
        # Parse JSON response
        validation_result = {
            "is_valid": True,
            "issues_found": [],
            "corrections": [],
            "corrected_explanation": None,
            "confidence": "unknown",
            "raw_response": response
        }
        
        if success:
            try:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    validation_result.update({
                        "is_valid": parsed.get("is_valid", True),
                        "issues_found": parsed.get("issues_found", []),
                        "corrections": parsed.get("corrections", []),
                        "corrected_explanation": parsed.get("corrected_explanation", None),
                        "confidence": parsed.get("confidence", "unknown")
                    })
            except json.JSONDecodeError:
                print("    ⚠ Warning: Could not parse validation JSON, treating as valid")
        
        return validation_result, {
            "system_prompt": VALIDATION_SYSTEM,
            "user_prompt": prompt,
            "success": success
        }

    
    def process_sample(self, sample: Dict, is_good: bool) -> Dict:
        """Process a single sample through the full pipeline."""
        sample_name = f"sample_{sample['index']}"
        category = "good_predictions" if is_good else "bad_predictions"
        sample_dir = self.output_dir / category / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        all_prompts = {}
        
        # Save metadata
        (sample_dir / "metadata.json").write_text(json.dumps(sample, indent=2))
        
        # Stage 0: Molecule descriptions
        print(f"    [0/3] Generating molecule descriptions...")
        solute_desc, solute_prompt_info = self.generate_molecule_description(sample["solute"])
        solvent_desc, solvent_prompt_info = self.generate_molecule_description(sample["solvent"])
        
        mol_descriptions = {
            "solute": {"smiles": sample["solute"], "description": solute_desc},
            "solvent": {"smiles": sample["solvent"], "description": solvent_desc}
        }
        (sample_dir / "molecule_descriptions.json").write_text(json.dumps(mol_descriptions, indent=2))
        all_prompts["molecule_descriptions"] = {
            "solute": solute_prompt_info,
            "solvent": solvent_prompt_info
        }
        
        # Stage 1: Evidence summary
        print(f"    [1/3] Generating evidence summary...")
        evidence_summary, evidence_prompt_info = self.generate_evidence_summary(sample)
        (sample_dir / "stage1_evidence_summary.md").write_text(evidence_summary)
        all_prompts["stage1_evidence_summary"] = evidence_prompt_info
        
        # Stage 2: Decision analysis
        print(f"    [2/3] Generating decision analysis...")
        decision_analysis, decision_prompt_info = self.generate_decision_analysis(sample)
        (sample_dir / "stage2_decision_analysis.md").write_text(decision_analysis)
        all_prompts["stage2_decision_analysis"] = decision_prompt_info
        
        # Stage 3: Final explanation
        print(f"    [3/4] Generating final explanation...")
        final_explanation, final_prompt_info = self.generate_final_explanation(
            sample, solute_desc, solvent_desc, evidence_summary, decision_analysis
        )
        (sample_dir / "stage3_final_explanation.md").write_text(final_explanation)
        all_prompts["stage3_final_explanation"] = final_prompt_info
        
        # Stage 3.5: Validation
        print(f"    [4/4] Validating explanation...")
        validation_result, validation_prompt_info = self.generate_validation(
            sample, 
            final_explanation,
            solute_desc=solute_desc,
            solvent_desc=solvent_desc
        )
        (sample_dir / "stage3.5_validation.json").write_text(json.dumps(validation_result, indent=2))
        all_prompts["stage3.5_validation"] = validation_prompt_info
        
        # Check validation results
        validated_explanation = final_explanation
        if not validation_result["is_valid"] and validation_result["issues_found"]:
            print(f"    ⚠ Validation found {len(validation_result['issues_found'])} issue(s)")
            for i, issue in enumerate(validation_result["issues_found"][:3], 1):
                print(f"      {i}. {issue[:80]}...")
            
            # Save original explanation with issues
            (sample_dir / "stage3_final_explanation_ORIGINAL.md").write_text(final_explanation)
            
            # If corrections provided, create a note about them
            if validation_result["corrections"]:
                corrections_note = "# Validation Issues Found\n\n"
                corrections_note += "## Issues:\n"
                for i, issue in enumerate(validation_result["issues_found"], 1):
                    corrections_note += f"{i}. {issue}\n"
                corrections_note += "\n## Suggested Corrections:\n"
                for i, correction in enumerate(validation_result["corrections"], 1):
                    corrections_note += f"{i}. {correction}\n"
                (sample_dir / "validation_corrections.md").write_text(corrections_note)

            # Use corrected explanation if provided
            if validation_result.get("corrected_explanation"):
                print("    ✓ Applying automatic correction...")
                validated_explanation = validation_result["corrected_explanation"]
                # Save corrected version as the main file
                (sample_dir / "stage3_final_explanation.md").write_text(validated_explanation)
                # Save correction metadata
                (sample_dir / "stage3_correction_applied.json").write_text(json.dumps({
                    "original_hash":  hash(final_explanation),
                    "correction_timestamp": datetime.now().isoformat()
                }, indent=2))
        else:
            print(f"    ✓ Validation passed (confidence: {validation_result['confidence']})")
        
        # Save all prompts used
        (sample_dir / "prompts_used.json").write_text(json.dumps(all_prompts, indent=2))
        
        return {
            "sample_index": sample["index"],
            "is_good": is_good,
            "abs_error": sample["abs_error"],
            "validation_passed": validation_result["is_valid"],
            "validation_issues": len(validation_result["issues_found"]),
            "correction_applied": bool(validation_result.get("corrected_explanation")),
            "output_dir": str(sample_dir)
        }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(api_keys: List[str], config: PipelineConfig = None, dry_run: bool = False):
    """
    Run the full explanation pipeline.
    
    Args:
        api_keys: List of Gemini API keys
        config: Pipeline configuration
        dry_run: If True, skip LLM calls and just test sample selection
    """
    if config is None:
        config = PipelineConfig()
    
    print("=" * 70)
    print("Enhanced LLM Explanation Pipeline")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Output directory: {config.output_dir}")
    print(f"API keys provided: {len(api_keys)}")
    
    # Check for feature thresholds file
    print(f"\nFeature thresholds file: {config.feature_thresholds_file}")
    if not os.path.exists(config.feature_thresholds_file):
        print(f"  ⚠ WARNING: Feature thresholds file not found!")
        print(f"  → Run 'python anomaly.py' to generate thresholds")
        print(f"  → Will use fallback thresholds for unusual feature detection")
    else:
        # Load and show info
        with open(config.feature_thresholds_file, 'r') as f:
            thresholds = json.load(f)
        n_features = len(thresholds)
        sample_key = list(thresholds.keys())[0]
        n_samples = thresholds[sample_key].get('n_samples', 'unknown')
        print(f"  ✓ Loaded thresholds for {n_features} features")
        print(f"  ✓ Based on {n_samples} samples (full database)")
    
    # Initialize components
    key_manager = APIKeyManager(api_keys)
    gemini_caller = GeminiCaller(key_manager, config)
    selector = SampleSelector(config)
    generator = ExplanationGenerator(gemini_caller, config)
    
    # Setup
    generator.setup_output_dirs()
    generator.save_prompts()

    
    # Load and select samples
    selector.load_models_and_data()
    samples = selector.select_samples()
    
    if dry_run:
        print("\n[DRY RUN] Skipping LLM calls")
        print(f"Would process {len(samples['good'])} good + {len(samples['bad'])} bad samples")
        return samples
    
    # Process samples
    results = {"good": [], "bad": []}
    
    print("\n" + "=" * 70)
    print("Processing Good Predictions")
    print("=" * 70)
    
    for i, sample in enumerate(tqdm(samples["good"], desc="Good predictions")):
        print(f"\n  [{i+1}/{len(samples['good'])}] Sample {sample['index']} (error: {sample['abs_error']:.4f})")
        if key_manager.has_available_keys():
            result = generator.process_sample(sample, is_good=True)
            results["good"].append(result)
        else:
            print("  ✗ Skipped - all API keys exhausted")
            break
    
    print("\n" + "=" * 70)
    print("Processing Bad Predictions")
    print("=" * 70)
    
    for i, sample in enumerate(tqdm(samples["bad"], desc="Bad predictions")):
        print(f"\n  [{i+1}/{len(samples['bad'])}] Sample {sample['index']} (error: {sample['abs_error']:.4f})")
        if key_manager.has_available_keys():
            result = generator.process_sample(sample, is_good=False)
            results["bad"].append(result)
        else:
            print("  ✗ Skipped - all API keys exhausted")
            break
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "good_threshold": config.good_prediction_threshold,
            "bad_threshold": config.bad_prediction_threshold,
            "num_good_requested": config.num_good_samples,
            "num_bad_requested": config.num_bad_samples
        },
        "results": {
            "good_processed": len(results["good"]),
            "bad_processed": len(results["bad"])
        },
        "api_key_status": key_manager.get_status(),
        "samples": results
    }
    
    summary_path = Path(config.output_dir) / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    
    print("\n" + "=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)
    print(f"\nProcessed: {len(results['good'])} good + {len(results['bad'])} bad predictions")
    print(f"Output: {config.output_dir}")
    print(f"API key usage: {key_manager.get_status()}")
    
    return results


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced LLM Explanation Pipeline")
    parser.add_argument("--api-keys", nargs="+", required=True,
                       help="One or more Gemini API keys")
    parser.add_argument("--dry-run", action="store_true",
                       help="Test sample selection without LLM calls")
    parser.add_argument("--num-good", type=int, default=15,
                       help="Number of good predictions to explain")
    parser.add_argument("--num-bad", type=int, default=15,
                       help="Number of bad predictions to explain")
    parser.add_argument("--good-threshold", type=float, default=0.30,
                       help="Error threshold for good predictions")
    parser.add_argument("--bad-threshold", type=float, default=1.00,
                       help="Error threshold for bad predictions")
    parser.add_argument("--output-dir", type=str, default="enhanced_explanations",
                       help="Output directory")
    parser.add_argument("--feature-thresholds", type=str, 
                       default="threshold_analysis/feature_thresholds.json",
                       help="Path to feature thresholds JSON file")
    
    args = parser.parse_args()
    
    config = PipelineConfig(
        output_dir=args.output_dir,
        num_good_samples=args.num_good,
        num_bad_samples=args.num_bad,
        good_prediction_threshold=args.good_threshold,
        bad_prediction_threshold=args.bad_threshold,
        feature_thresholds_file=args.feature_thresholds
    )
    
    run_pipeline(args.api_keys, config, dry_run=args.dry_run)

