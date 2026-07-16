"""
Independence Evaluation Metrics
Calculate independence between models
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.metrics import confusion_matrix
import json


class IndependenceMetrics:
    """Independence metrics calculation"""

    def __init__(self, results_dict: Dict[str, List[Dict]]):
        """
        Initialize

        Args:
            results_dict: Dictionary with model names as keys and result lists as values
                         Each result contains 'index', 'true_label', 'predicted_label' fields
        """
        self.results_dict = results_dict
        self.model_names = list(results_dict.keys())
        self.num_models = len(self.model_names)
        
        # Validate data consistency
        self._validate_data()

    def _validate_data(self):
        """Validate all models have consistent data indices"""
        indices_list = []
        for model_name, results in self.results_dict.items():
            indices = [r['index'] for r in results]
            indices_list.append(set(indices))
        
        # Check if all models evaluated the same samples
        common_indices = set.intersection(*indices_list)
        if len(common_indices) != len(indices_list[0]):
            print(f"Warning: Not all models evaluated the same samples")
            print(f"Common samples: {len(common_indices)}")
    
    def calculate_error_consistency_matrix(self) -> pd.DataFrame:
        """
        Calculate error pattern consistency matrix

        Returns a matrix where [i,j] represents the proportion of samples where both model i and j made errors
        """
        n = self.num_models
        consistency_matrix = np.zeros((n, n))
        
        # Build error set for each model
        error_sets = {}
        for model_name, results in self.results_dict.items():
            errors = set()
            for r in results:
                if r['true_label'] != r['predicted_label']:
                    errors.add(r['index'])
            error_sets[model_name] = errors
        
        # Calculate pairwise error consistency
        for i, model_i in enumerate(self.model_names):
            for j, model_j in enumerate(self.model_names):
                if i == j:
                    consistency_matrix[i, j] = 1.0
                else:
                    # Calculate number of samples with common errors
                    common_errors = error_sets[model_i].intersection(error_sets[model_j])
                    
                    # Normalize: divide by union of errors from both models
                    union_errors = error_sets[model_i].union(error_sets[model_j])
                    if len(union_errors) > 0:
                        consistency_matrix[i, j] = len(common_errors) / len(union_errors)
                    else:
                        consistency_matrix[i, j] = 0.0
        
        # Convert to DataFrame
        df = pd.DataFrame(consistency_matrix, 
                         index=self.model_names, 
                         columns=self.model_names)
        
        return df
    
    def calculate_agreement_matrix(self) -> pd.DataFrame:
        """
        Calculate prediction agreement matrix

        Returns a matrix where [i,j] represents the proportion of samples where model i and j made the same prediction
        """
        n = self.num_models
        agreement_matrix = np.zeros((n, n))
        
        # Get all sample indices
        sample_indices = [r['index'] for r in self.results_dict[self.model_names[0]]]
        
        # Calculate pairwise prediction agreement
        for i, model_i in enumerate(self.model_names):
            for j, model_j in enumerate(self.model_names):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                else:
                    # Get predictions from both models
                    pred_i = {r['index']: r['predicted_label'] for r in self.results_dict[model_i]}
                    pred_j = {r['index']: r['predicted_label'] for r in self.results_dict[model_j]}
                    
                    # Calculate number of agreeing samples
                    agreements = sum(1 for idx in sample_indices if pred_i.get(idx) == pred_j.get(idx))
                    agreement_matrix[i, j] = agreements / len(sample_indices)
        
        # Convert to DataFrame
        df = pd.DataFrame(agreement_matrix,
                         index=self.model_names,
                         columns=self.model_names)
        
        return df
    
    def calculate_diversity_score(self) -> float:
        """
        Calculate overall diversity score

        Diversity score = 1 - average pairwise agreement
        Higher score indicates more independent models
        """
        agreement_matrix = self.calculate_agreement_matrix()
        
        # Calculate average of off-diagonal elements
        n = len(agreement_matrix)
        sum_agreements = 0
        count = 0
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    sum_agreements += agreement_matrix.iloc[i, j]
                    count += 1
        
        avg_agreement = sum_agreements / count if count > 0 else 0
        diversity_score = 1 - avg_agreement
        
        return diversity_score
    
    def calculate_error_diversity(self) -> float:
        """
        Calculate error diversity score

        Error diversity = 1 - average error consistency
        Higher score indicates more independent error patterns
        """
        error_consistency = self.calculate_error_consistency_matrix()
        
        # Calculate average of off-diagonal elements
        n = len(error_consistency)
        sum_consistency = 0
        count = 0
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    sum_consistency += error_consistency.iloc[i, j]
                    count += 1
        
        avg_consistency = sum_consistency / count if count > 0 else 0
        error_diversity = 1 - avg_consistency
        
        return error_diversity
    
    def analyze_error_patterns(self) -> Dict:
        """
        Analyze error patterns

        Returns:
        - Unique errors per model
        - Common errors across all models
        - Error distribution statistics
        """
        # Build error set for each model
        error_sets = {}
        for model_name, results in self.results_dict.items():
            errors = {}
            for r in results:
                if r['true_label'] != r['predicted_label']:
                    errors[r['index']] = {
                        'true': r['true_label'],
                        'predicted': r['predicted_label']
                    }
            error_sets[model_name] = errors
        
        # Find union of all errors
        all_error_indices = set()
        for errors in error_sets.values():
            all_error_indices.update(errors.keys())
        
        # Analyze how many models made errors on each sample
        error_frequency = {}
        for idx in all_error_indices:
            count = sum(1 for errors in error_sets.values() if idx in errors)
            error_frequency[idx] = count
        
        # Find common errors across all models
        common_errors = set(all_error_indices)
        for errors in error_sets.values():
            common_errors = common_errors.intersection(set(errors.keys()))
        
        # Find unique errors for each model
        unique_errors = {}
        for model_name, errors in error_sets.items():
            unique = set(errors.keys())
            for other_model, other_errors in error_sets.items():
                if other_model != model_name:
                    unique = unique - set(other_errors.keys())
            unique_errors[model_name] = list(unique)
        
        analysis = {
            'total_error_samples': len(all_error_indices),
            'common_errors': list(common_errors),
            'common_error_count': len(common_errors),
            'unique_errors': unique_errors,
            'error_frequency_distribution': {
                f'{i}_models': sum(1 for count in error_frequency.values() if count == i)
                for i in range(1, self.num_models + 1)
            },
            'per_model_error_count': {
                model: len(errors) for model, errors in error_sets.items()
            }
        }
        
        return analysis
    
    def generate_report(self, output_path: str = None) -> str:
        """
        Generate complete independence analysis report

        Args:
            output_path: If provided, save report to file

        Returns:
            Report text
        """
        report = []
        report.append("=" * 80)
        report.append("MODEL INDEPENDENCE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Basic information
        report.append(f"Number of models: {self.num_models}")
        report.append(f"Models: {', '.join(self.model_names)}")
        report.append("")
        
        # Diversity scores
        diversity = self.calculate_diversity_score()
        error_diversity = self.calculate_error_diversity()
        
        report.append("=" * 80)
        report.append("INDEPENDENCE SCORES")
        report.append("=" * 80)
        report.append(f"Overall Diversity Score: {diversity:.4f}")
        report.append(f"  (Higher is better, 1.0 = completely independent predictions)")
        report.append(f"Error Diversity Score: {error_diversity:.4f}")
        report.append(f"  (Higher is better, 1.0 = completely different error patterns)")
        report.append("")
        
        # Prediction agreement matrix
        report.append("=" * 80)
        report.append("PREDICTION AGREEMENT MATRIX")
        report.append("=" * 80)
        agreement = self.calculate_agreement_matrix()
        report.append(agreement.to_string())
        report.append("")
        
        # Error consistency matrix
        report.append("=" * 80)
        report.append("ERROR CONSISTENCY MATRIX")
        report.append("=" * 80)
        error_consistency = self.calculate_error_consistency_matrix()
        report.append(error_consistency.to_string())
        report.append("")
        
        # Error pattern analysis
        report.append("=" * 80)
        report.append("ERROR PATTERN ANALYSIS")
        report.append("=" * 80)
        error_analysis = self.analyze_error_patterns()
        
        report.append(f"Total error samples: {error_analysis['total_error_samples']}")
        report.append(f"Common errors (all models wrong): {error_analysis['common_error_count']}")
        report.append("")
        
        report.append("Error frequency distribution:")
        for key, value in error_analysis['error_frequency_distribution'].items():
            report.append(f"  {key}: {value} samples")
        report.append("")
        
        report.append("Per-model error count:")
        for model, count in error_analysis['per_model_error_count'].items():
            report.append(f"  {model}: {count} errors")
        report.append("")
        
        report.append("Unique errors per model:")
        for model, errors in error_analysis['unique_errors'].items():
            report.append(f"  {model}: {len(errors)} unique errors")
        report.append("")
        
        report_text = "\n".join(report)
        
        # Save report
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Report saved to: {output_path}")
        
        return report_text


def load_and_analyze(result_files: Dict[str, str], output_dir: str = "./outputs/analysis"):
    """
    Load results from multiple models and perform independence analysis

    Args:
        result_files: Dictionary with model names as keys and result JSON file paths as values
        output_dir: Output directory
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all results
    results_dict = {}
    for model_name, file_path in result_files.items():
        with open(file_path, 'r', encoding='utf-8') as f:
            results_dict[model_name] = json.load(f)
    
    # Create analyzer
    analyzer = IndependenceMetrics(results_dict)
    
    # Generate report
    report_path = os.path.join(output_dir, "independence_report.txt")
    report = analyzer.generate_report(report_path)
    
    # Save matrices
    agreement = analyzer.calculate_agreement_matrix()
    agreement.to_csv(os.path.join(output_dir, "agreement_matrix.csv"))
    
    error_consistency = analyzer.calculate_error_consistency_matrix()
    error_consistency.to_csv(os.path.join(output_dir, "error_consistency_matrix.csv"))
    
    # Save error analysis
    error_analysis = analyzer.analyze_error_patterns()
    with open(os.path.join(output_dir, "error_analysis.json"), 'w', encoding='utf-8') as f:
        json.dump(error_analysis, f, indent=2)
    
    print("\n" + report)
    
    return analyzer
