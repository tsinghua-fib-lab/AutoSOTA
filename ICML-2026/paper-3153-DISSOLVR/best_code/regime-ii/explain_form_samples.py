"""
Explain New Samples Pipeline
==============================
Runs the full enhanced LLM explanation pipeline on new solute-solvent pairs
(e.g., from form.csv) that may not exist in the original feature stores.

This combines:
1. On-the-fly featurization of new molecules
2. Prediction + SHAP + Attention extraction
3. Multi-stage LLM explanation generation

Usage:
    python explain_form_samples.py --api-keys "KEY1" "KEY2" --input form.csv --temp 298.15
"""

import os
import json
import time
import re
import joblib
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from catboost import Pool
from dataclasses import dataclass
from typing import List, Dict, Tuple
import google.generativeai as genai
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image, ImageDraw, ImageFont

from featurizer import MoleculeFeaturizer
from train_transformer import InteractionTransformer

# Import prompt templates and helpers from the original pipeline
from enhanced_explain_pipeline import (
    APIKeyManager, GeminiCaller, PipelineConfig,
    MOLECULE_DESCRIPTION_SYSTEM, MOLECULE_DESCRIPTION_PROMPT,
    EVIDENCE_SUMMARY_SYSTEM, EVIDENCE_SUMMARY_PROMPT,
    DECISION_ANALYSIS_SYSTEM, DECISION_ANALYSIS_PROMPT,
    INTEGRATION_SYSTEM, INTEGRATION_PROMPT,
    CONDENSATION_SYSTEM, CONDENSATION_PROMPT,
    VALIDATION_SYSTEM, VALIDATION_PROMPT,
    format_shap_with_insights, identify_unusual_features
)


@dataclass
class FormConfig(PipelineConfig):
    """Extended config for form.csv processing."""
    input_file: str = "form.csv"
    default_temperature: float = 298.15  # Kelvin


class NewSampleProcessor:
    """Handles featurization and prediction for new molecules."""
    
    def __init__(self, config: FormConfig):
        self.config = config
        self.device = config.device
        self.featurizer = MoleculeFeaturizer()
        
    def load_resources(self):
        print("Loading models and feature definitions...")
        
        # Load feature stores WITH INDEX (like save_sorted_predictions.py)
        # This ensures we use the EXACT same features that were used during training
        self.sol_raw = pd.read_parquet(
            os.path.join(self.config.store_dir, "solute_raw.parquet")
        ).set_index("SMILES_KEY")
        self.cols_sol_raw = self.sol_raw.columns.tolist()
        
        self.solv_raw = pd.read_parquet(
            os.path.join(self.config.store_dir, "solvent_raw.parquet")
        ).set_index("SMILES_KEY")
        self.cols_solv_raw = self.solv_raw.columns.tolist()
        
        self.sol_council = pd.read_parquet(
            os.path.join(self.config.store_dir, "solute_council.parquet")
        ).set_index("SMILES_KEY")
        self.cols_sol_council = self.sol_council.columns.tolist()
        
        self.solv_council = pd.read_parquet(
            os.path.join(self.config.store_dir, "solvent_council.parquet")
        ).set_index("SMILES_KEY")
        self.cols_solv_council = self.solv_council.columns.tolist()
        
        print(f"  ✓ Loaded feature stores:")
        print(f"    - Solutes: {len(self.sol_raw)} molecules")
        print(f"    - Solvents: {len(self.solv_raw)} molecules")
        
        # Transformer
        self.transformer = InteractionTransformer().to(self.device)
        self.transformer.load_state_dict(
            torch.load(self.config.transformer_path, map_location=self.device)
        )
        self.transformer.eval()
        
        # CatBoost
        self.catboost_model = joblib.load(
            os.path.join(self.config.model_dir, "model.joblib")
        )
        self.selector = joblib.load(
            os.path.join(self.config.model_dir, "selector.joblib")
        )
        
        # Feature names
        feature_map = pd.read_csv(
            os.path.join(self.config.cgboost_dir, "trained_feature_map.csv")
        )
        self.feature_names = feature_map["Feature"].tolist()
        
        print(f"  ✓ Loaded {len(self.feature_names)} trained features")
    
    def featurize_molecules(self, smiles_list: List[str], mol_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Look up pre-computed features from feature store, with fallback to on-the-fly featurization."""
        print(f"  Looking up {len(smiles_list)} {mol_type}s...")
        
        if mol_type == "solute":
            store_raw = self.sol_raw
            store_council = self.sol_council
        else:
            store_raw = self.solv_raw
            store_council = self.solv_council
        
        # Try to find all molecules in feature store first
        found_smiles = [s for s in smiles_list if s in store_raw.index]
        missing_smiles = [s for s in smiles_list if s not in store_raw.index]
        
        if missing_smiles:
            print(f"    ⚠ WARNING: {len(missing_smiles)} {mol_type}(s) not in feature store")
            print(f"    ⚠ Re-featurizing these molecules (predictions may differ from training!)")
            for s in missing_smiles[:3]:  # Show first 3
                print(f"      - {s[:60]}...")
            if len(missing_smiles) > 3:
                print(f"      ... and {len(missing_smiles)-3} more")
            
            # Fallback: featurize missing molecules
            df_feats = self.featurizer.transform(missing_smiles)
            df_feats.index = missing_smiles
            
            # Ensure all expected columns exist
            for c in store_raw.columns.tolist() + store_council.columns.tolist():
                if c not in df_feats.columns:
                    df_feats[c] = 0.0
            
            # Combine pre-computed features with newly featurized ones
            if found_smiles:
                raw_combined = pd.concat([
                    store_raw.loc[found_smiles],
                    df_feats[store_raw.columns]
                ])
                council_combined = pd.concat([
                    store_council.loc[found_smiles],
                    df_feats[store_council.columns]
                ])
            else:
                raw_combined = df_feats[store_raw.columns]
                council_combined = df_feats[store_council.columns]
        else:
            # All molecules found in feature store - use pre-computed features
            print(f"    ✓ All {len(found_smiles)} {mol_type}(s) found in feature store")
            raw_combined = store_raw.loc[found_smiles]
            council_combined = store_council.loc[found_smiles]
        
        # Reindex to match input order
        raw_combined = raw_combined.reindex(smiles_list)
        council_combined = council_combined.reindex(smiles_list)
        
        return raw_combined, council_combined
    
    def process_samples(self, df: pd.DataFrame) -> List[Dict]:
        """Process all samples and return list of sample info dicts."""
        self.load_resources()
        
        # Featurize unique molecules
        unique_solutes = df["Solute"].unique().tolist()
        unique_solvents = df["Solvent"].unique().tolist()
        
        df_sol_raw, df_sol_council = self.featurize_molecules(unique_solutes, "solute")
        df_solv_raw, df_solv_council = self.featurize_molecules(unique_solvents, "solvent")
        
        print("Generating predictions and evidence...")
        
        # Generate features for all samples
        X_sol_council = df_sol_council.loc[df["Solute"]].values.astype(np.float32)
        X_solv_council = df_solv_council.loc[df["Solvent"]].values.astype(np.float32)
        
        # Transformer embeddings + attention
        embeds, attns = [], []
        with torch.no_grad():
            for i in range(len(df)):
                sol = torch.tensor(X_sol_council[i:i+1]).to(self.device)
                solv = torch.tensor(X_solv_council[i:i+1]).to(self.device)
                _, feats, attn = self.transformer(sol, solv)
                embeds.append(feats.cpu().numpy())
                attns.append(attn.cpu().numpy())
        
        X_embed = np.vstack(embeds)
        attention_weights = np.vstack(attns)
        
        # Temperature features
        T = df["Temperature"].values.reshape(-1, 1)
        T_inv = (1000 / df["Temperature"]).values.reshape(-1, 1)
        Tm = df_sol_raw.loc[df["Solute"], "pred_Tm"].values.reshape(-1, 1)
        T_red = T / Tm
        
        # Interaction features
        X_reshaped = X_embed.reshape(-1, 24, 32)
        X_mod = np.linalg.norm(X_reshaped, axis=2)
        X_sign = np.sign(X_reshaped.mean(axis=2))
        X_interact = (X_sign * X_mod) * T_inv
        
        # Raw features
        X_raw = np.hstack([
            df_sol_raw.loc[df["Solute"]].values,
            df_solv_raw.loc[df["Solvent"]].values
        ])
        
        # Full feature matrix
        X_full = np.hstack([X_raw, X_interact, Tm, T_red, T, T_inv])
        X_pruned = self.selector.transform(X_full)
        
        # Predictions
        predictions = self.catboost_model.predict(X_pruned)
        
        # SHAP values
        pool = Pool(X_pruned, feature_names=self.feature_names)
        shap_vals = self.catboost_model.get_feature_importance(pool, type="ShapValues")
        shap_vals = np.array(shap_vals)[:, :-1]
        leaf_paths = self.catboost_model.calc_leaf_indexes(pool)
        
        # Compile sample info
        samples = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            samples.append({
                "index": int(idx),
                "solute": row["Solute"],
                "solvent": row["Solvent"],
                "temperature": float(row["Temperature"]),
                "y_pred": float(predictions[idx]),
                "cross_attention_weights": attention_weights[idx].tolist(),
                "council_feature_names": self.cols_sol_council,
                "shap_values": dict(zip(self.feature_names, shap_vals[idx].tolist())),
                "leaf_path": leaf_paths[idx].tolist(),
                "structural_features": self._extract_structural_features(idx, X_pruned)
            })
        
        return samples
    
    def _extract_structural_features(self, idx: int, X_pruned: np.ndarray) -> Dict:
        key_features = [
            "Solute_num_C", "Solute_num_O", "Solute_num_N", "Solute_num_Cl", 
            "Solute_MolLogP", "Solute_MolWt", "Solute_TPSA",
            "Solute_NumHDonors", "Solute_NumHAcceptors",
            "Solvent_num_C", "Solvent_num_O", "Solvent_MolLogP", "Solvent_MolWt",
            "pred_Tm", "T", "T_inv", "T_red"
        ]
        structural = {}
        for feat in key_features:
            if feat in self.feature_names:
                feat_idx = self.feature_names.index(feat)
                structural[feat] = float(X_pruned[idx, feat_idx])
        return structural


class FormExplanationGenerator:
    """Generates explanations for new samples."""
    
    def __init__(self, gemini_caller: GeminiCaller, config: FormConfig):
        self.caller = gemini_caller
        self.config = config
        self.output_dir = Path(config.output_dir)
    
    def setup_output_dirs(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "prompts").mkdir(exist_ok=True)
        (self.output_dir / "samples").mkdir(exist_ok=True)
    
    def generate_molecule_description(self, smiles: str) -> Tuple[str, Dict]:
        prompt = MOLECULE_DESCRIPTION_PROMPT.format(smiles=smiles)
        response, success = self.caller.call(prompt, MOLECULE_DESCRIPTION_SYSTEM, temperature=0.2)
        return response, {"system_prompt": MOLECULE_DESCRIPTION_SYSTEM, "user_prompt": prompt, "success": success}
    
    def generate_evidence_summary(self, sample: Dict) -> Tuple[str, Dict]:
        shap_str = format_shap_with_insights(sample["shap_values"])
        
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
        
        unusual_str = identify_unusual_features(sample["structural_features"])
        struct_items = sorted(sample["structural_features"].items(), key=lambda x: abs(x[1]), reverse=True)[:15]
        struct_str = "\n".join([f"  {k}: {v:.4f}" for k, v in struct_items])
        
        prompt = EVIDENCE_SUMMARY_PROMPT.format(
            shap_features=shap_str,
            group_contributions=group_str,
            structural_features=struct_str + "\n\nUNUSUAL VALUES:\n" + unusual_str
        )
        
        response, success = self.caller.call(prompt, EVIDENCE_SUMMARY_SYSTEM, temperature=0.3)
        time.sleep(self.config.rate_limit_delay)
        return response, {"system_prompt": EVIDENCE_SUMMARY_SYSTEM, "user_prompt": prompt, "success": success}
    
    def generate_decision_analysis(self, sample: Dict) -> Tuple[str, Dict]:
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
        
        top_feats = sorted(sample["shap_values"].items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        feat_str = "\n".join([
            f"  {name}: {value:+.4f} ({'positive' if value > 0 else 'negative'} contribution)"
            for name, value in top_feats
        ])
        
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
        return response, {"system_prompt": DECISION_ANALYSIS_SYSTEM, "user_prompt": prompt, "success": success}
    
    def generate_final_explanation(
        self, sample: Dict, solute_desc: str, solvent_desc: str,
        evidence_summary: str, decision_analysis: str
    ) -> Tuple[str, Dict]:
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
        return response, {"system_prompt": INTEGRATION_SYSTEM, "user_prompt": prompt, "success": success}
    
    def generate_molecule_image(self, solute_smiles: str, solvent_smiles: str, save_path: Path):
        """Generate a combined image of solute and solvent with labels."""
        try:
            mol_solute = Chem.MolFromSmiles(solute_smiles)
            mol_solvent = Chem.MolFromSmiles(solvent_smiles)
            
            if mol_solute is None or mol_solvent is None:
                print("    ⚠ Could not parse SMILES for image generation")
                return
            
            # Generate individual molecule images
            img_size = (400, 300)
            img_solute = Draw.MolToImage(mol_solute, size=img_size)
            img_solvent = Draw.MolToImage(mol_solvent, size=img_size)
            
            # Create combined image with labels
            combined_width = img_size[0] * 2 + 60  # 60px gap
            combined_height = img_size[1] + 60  # 60px for labels
            combined = Image.new('RGB', (combined_width, combined_height), 'white')
            
            # Paste molecule images
            combined.paste(img_solute, (10, 50))
            combined.paste(img_solvent, (img_size[0] + 50, 50))
            
            # Add labels
            draw = ImageDraw.Draw(combined)
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
            except:
                font = ImageFont.load_default()
            
            # Draw labels
            draw.text((img_size[0] // 2 - 20, 15), "SOLUTE", fill='darkblue', font=font)
            draw.text((img_size[0] + 50 + img_size[0] // 2 - 30, 15), "SOLVENT", fill='darkgreen', font=font)
            
            # Add separator line
            draw.line([(img_size[0] + 30, 50), (img_size[0] + 30, img_size[1] + 50)], fill='gray', width=2)
            
            combined.save(save_path)
        except Exception as e:
            print(f"    ⚠ Image generation failed: {e}")
    
    def generate_condensed_explanation(self, original_explanation: str) -> Tuple[str, Dict]:
        """Condense a verbose explanation into 4 dense paragraphs."""
        prompt = CONDENSATION_PROMPT.format(original_explanation=original_explanation)
        response, success = self.caller.call(prompt, CONDENSATION_SYSTEM, temperature=0.2)
        time.sleep(self.config.rate_limit_delay)
        return response, {"system_prompt": CONDENSATION_SYSTEM, "user_prompt": prompt, "success": success}
    
    def generate_validation(self, sample: Dict, explanation: str, solute_desc: str = "", solvent_desc: str = "") -> Tuple[Dict, Dict]:
        """Validate explanation against source evidence."""
        
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
        unusual_str = identify_unusual_features(sample["structural_features"])

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

    
    def process_sample(self, sample: Dict) -> Dict:
        sample_name = f"sample_{sample['index']}"
        sample_dir = self.output_dir / "samples" / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        all_prompts = {}
        
        # Save metadata
        (sample_dir / "metadata.json").write_text(json.dumps(sample, indent=2))
        
        # Stage 0: Molecule descriptions
        print(f"    [0/3] Generating molecule descriptions...")
        solute_desc, solute_info = self.generate_molecule_description(sample["solute"])
        solvent_desc, solvent_info = self.generate_molecule_description(sample["solvent"])
        
        mol_descriptions = {
            "solute": {"smiles": sample["solute"], "description": solute_desc},
            "solvent": {"smiles": sample["solvent"], "description": solvent_desc}
        }
        (sample_dir / "molecule_descriptions.json").write_text(json.dumps(mol_descriptions, indent=2))
        all_prompts["molecule_descriptions"] = {"solute": solute_info, "solvent": solvent_info}
        
        # Save molecule image
        self.generate_molecule_image(sample["solute"], sample["solvent"], sample_dir / "molecules.png")
        
        # Stage 1: Evidence summary
        print(f"    [1/3] Generating evidence summary...")
        evidence_summary, evidence_info = self.generate_evidence_summary(sample)
        (sample_dir / "stage1_evidence_summary.md").write_text(evidence_summary)
        all_prompts["stage1_evidence_summary"] = evidence_info
        
        # Stage 2: Decision analysis
        print(f"    [2/3] Generating decision analysis...")
        decision_analysis, decision_info = self.generate_decision_analysis(sample)
        (sample_dir / "stage2_decision_analysis.md").write_text(decision_analysis)
        all_prompts["stage2_decision_analysis"] = decision_info
        
        # Stage 3: Final explanation
        print(f"    [3/5] Generating final explanation...")
        final_explanation, final_info = self.generate_final_explanation(
            sample, solute_desc, solvent_desc, evidence_summary, decision_analysis
        )
        (sample_dir / "stage3_final_explanation.md").write_text(final_explanation)
        all_prompts["stage3_final_explanation"] = final_info
        
        # Stage 3.5: Validation
        print(f"    [4/5] Validating explanation...")
        validation_result, validation_info = self.generate_validation(
            sample, 
            final_explanation,
            solute_desc=solute_desc,
            solvent_desc=solvent_desc
        )
        (sample_dir / "stage3.5_validation.json").write_text(json.dumps(validation_result, indent=2))
        all_prompts["stage3.5_validation"] = validation_info
        
        # Check validation results and use corrected explanation if needed
        explanation_for_condensation = final_explanation
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
                explanation_for_condensation = validation_result["corrected_explanation"]
                # Save corrected version as the main file
                (sample_dir / "stage3_final_explanation.md").write_text(explanation_for_condensation)
                # Save correction metadata
                (sample_dir / "stage3_correction_applied.json").write_text(json.dumps({
                    "original_hash":  hash(final_explanation),
                    "correction_timestamp": datetime.now().isoformat()
                }, indent=2))
        else:
            print(f"    ✓ Validation passed (confidence: {validation_result['confidence']})")
        
        # Stage 4: Condensation (using validated explanation)
        print(f"    [5/5] Condensing explanation...")
        condensed_explanation, condensed_info = self.generate_condensed_explanation(explanation_for_condensation)
        (sample_dir / "stage4_condensed.txt").write_text(condensed_explanation)
        all_prompts["stage4_condensed"] = condensed_info
        
        # Save prompts
        (sample_dir / "prompts_used.json").write_text(json.dumps(all_prompts, indent=2))
        
        return {
            "sample_index": sample["index"],
            "solute": sample["solute"],
            "solvent": sample["solvent"],
            "temperature": sample["temperature"],
            "y_pred": sample["y_pred"],
            "validation_passed": validation_result["is_valid"],
            "validation_issues": len(validation_result["issues_found"]),
            "final_explanation": condensed_explanation,
            "output_dir": str(sample_dir)
        }


def run_form_pipeline(api_keys: List[str], config: FormConfig = None):
    """Run explanation pipeline on form.csv samples."""
    if config is None:
        config = FormConfig()
    
    print("=" * 70)
    print("Form.csv Explanation Pipeline")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Input file: {config.input_file}")
    print(f"Output directory: {config.output_dir}")
    print(f"Default temperature: {config.default_temperature} K")
    
    # Load input data
    df = pd.read_csv(config.input_file)
    
    # Handle temperature column (check for 'Temp' or 'Temperature')
    if "Temp" in df.columns:
        df["Temperature"] = df["Temp"]
    elif "Temperature" not in df.columns:
        df["Temperature"] = config.default_temperature
        print(f"No temperature column found, using default: {config.default_temperature} K")
    print(f"Loaded {len(df)} samples from {config.input_file}")
    
    # Initialize components
    key_manager = APIKeyManager(api_keys)
    gemini_caller = GeminiCaller(key_manager, config)
    processor = NewSampleProcessor(config)
    generator = FormExplanationGenerator(gemini_caller, config)
    
    # Setup
    generator.setup_output_dirs()
    
    # Process samples (featurize + predict + extract evidence)
    samples = processor.process_samples(df)
    
    # Generate explanations
    print("\n" + "=" * 70)
    print("Generating Explanations")
    print("=" * 70)
    
    results = []
    for i, sample in enumerate(tqdm(samples, desc="Explaining samples")):
        print(f"\n  [{i+1}/{len(samples)}] {sample['solute'][:20]}... + {sample['solvent'][:20]}...")
        if key_manager.has_available_keys():
            result = generator.process_sample(sample)
            results.append(result)
        else:
            print("  ✗ Skipped - all API keys exhausted")
            break
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "input_file": config.input_file,
        "temperature": config.default_temperature,
        "total_samples": len(df),
        "processed": len(results),
        "api_key_status": key_manager.get_status(),
        "results": [{k: v for k, v in r.items() if k != "final_explanation"} for r in results]
    }
    
    summary_path = Path(config.output_dir) / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    
    # Save CSV with all results
    def clean_explanation(text):
        """Convert markdown to plain text."""
        # Remove markdown headers
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        # Remove bold/italic
        text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
        # Remove horizontal rules
        text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)
        # Collapse multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    csv_data = []
    for r in results:
        csv_data.append({
            "Solute": r["solute"],
            "Solvent": r["solvent"],
            "Temperature": r["temperature"],
            "Predicted_LogS": r["y_pred"],
            "Explanation": clean_explanation(r.get("final_explanation", ""))
        })
    
    csv_path = Path(config.output_dir) / "explanations_summary.csv"
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"\n✓ Saved explanations to {csv_path}")
    
    print("\n" + "=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)
    print(f"\nProcessed: {len(results)}/{len(df)} samples")
    print(f"Output: {config.output_dir}")
    print(f"CSV: {csv_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Explain form.csv samples")
    parser.add_argument("--api-keys", nargs="+", required=True, help="Gemini API keys")
    parser.add_argument("--input", "-i", type=str, default="form.csv", help="Input CSV")
    parser.add_argument("--output", "-o", type=str, default="form_explanations", help="Output directory")
    parser.add_argument("--temp", "-t", type=float, default=298.15, help="Temperature (K)")
    
    args = parser.parse_args()
    
    config = FormConfig(
        input_file=args.input,
        output_dir=args.output,
        default_temperature=args.temp
    )
    
    run_form_pipeline(args.api_keys, config)
