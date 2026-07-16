"""
Investigate why item difficulty parameters don't transfer across evaluation setups.

This script analyzes the response variance patterns to understand why difficulty
parameters calibrated from HELM (T=0) don't transfer well to test-time scaling (T=1.0)
for some benchmarks but do for others.

Key hypothesis: High-transfer benchmarks have stable responses (low intra-model variance),
while low-transfer benchmarks have unstable responses (high intra-model variance).
"""

import torch
import numpy as np
from scipy.stats import spearmanr
import pandas as pd
from huggingface_hub import snapshot_download
import os

# Configuration
FILE_NAME = "irsl_testtime_resmat1"
BENCHMARKS = ["commonsense", "legalbench", "med_qa", "bbq", "legal_support", "lsat_qa"]


def load_data():
    """Load test-time response data and calibrated difficulties."""
    print("Loading data...")

    # Download ResMAT1 from HuggingFace
    cache_dir = snapshot_download(
        repo_id=f"stair-lab/{FILE_NAME}",
        repo_type="dataset"
    )

    # Load test-time response matrix
    testtime_resmat = torch.load(
        f"{cache_dir}/resmat.pt",
        map_location="cpu",
        weights_only=False
    )

    # Extract components
    data_tensor = testtime_resmat["data_tensor"].numpy()  # (8 models, 552 questions, 10000 samples)
    models = testtime_resmat["models"]
    questions = testtime_resmat["questions"]
    scenarios = testtime_resmat["scenarios"]  # Dataset names (e.g., "bbq", "commonsense")
    benchmarks = testtime_resmat["benchmarks"]  # Benchmark category (e.g., "classic")
    helm_zs = np.array(testtime_resmat["zs"])  # HELM z-scores

    # Load recalibrated z-scores
    calibrated_path = os.path.join(os.path.dirname(__file__), f"{FILE_NAME}_withz.pt")
    if os.path.exists(calibrated_path):
        testtime_calibrated = torch.load(calibrated_path, weights_only=False)
        testtime_zs = testtime_calibrated["zs"]
        # Convert to numpy if it's a torch tensor
        if hasattr(testtime_zs, 'numpy'):
            testtime_zs = testtime_zs.numpy()
    else:
        print(f"Warning: Calibrated z-scores not found at {calibrated_path}")
        print("Please run testtime_calibrate.py first.")
        testtime_zs = None

    print(f"Loaded data: {data_tensor.shape[0]} models, {data_tensor.shape[1]} questions, {data_tensor.shape[2]} samples")
    print(f"Models: {models}")

    return {
        "data_tensor": data_tensor,
        "models": models,
        "questions": questions,
        "scenarios": scenarios,  # e.g., "bbq", "commonsense", "legalbench"
        "benchmarks": benchmarks,  # e.g., "classic"
        "helm_zs": helm_zs,
        "testtime_zs": testtime_zs
    }


def analyze_question_responses(question_idx, data_tensor, helm_zs, testtime_zs):
    """Analyze response patterns for a single question."""
    responses = data_tensor[:, question_idx, :]  # (n_models, n_samples)

    # Per-model statistics
    model_means = np.mean(responses, axis=1)  # (n_models,)
    model_stds = np.std(responses, axis=1)    # (n_models,)

    # Inter-model variance (how much models disagree on difficulty)
    inter_model_var = np.var(model_means)

    # Average intra-model variance (how unstable responses are at T=1.0)
    # This measures temperature sensitivity
    avg_intra_model_var = np.mean(model_stds**2)

    # Z-score difference (if available)
    if testtime_zs is not None:
        z_diff = abs(testtime_zs[question_idx] - helm_zs[question_idx])
    else:
        z_diff = np.nan

    return {
        "inter_model_var": inter_model_var,
        "intra_model_var": avg_intra_model_var,
        "z_diff": z_diff,
        "helm_z": helm_zs[question_idx],
        "testtime_z": testtime_zs[question_idx] if testtime_zs is not None else np.nan,
        "model_means": model_means,
        "model_stds": model_stds,
    }


def create_analysis_table(data):
    """Create comprehensive analysis table for all questions."""
    print("\nAnalyzing all questions...")

    data_tensor = data["data_tensor"]
    questions = data["questions"]
    scenarios = data["scenarios"]  # This is the dataset/benchmark name we care about
    helm_zs = data["helm_zs"]
    testtime_zs = data["testtime_zs"]

    results = []

    for idx in range(len(questions)):
        analysis = analyze_question_responses(idx, data_tensor, helm_zs, testtime_zs)

        results.append({
            "question_idx": idx,
            "question_text": questions[idx][:100] + "..." if len(questions[idx]) > 100 else questions[idx],
            "full_question": questions[idx],
            "benchmark": scenarios[idx],  # This is the scenario name (e.g., "bbq", "commonsense")
            "helm_z": analysis["helm_z"],
            "testtime_z": analysis["testtime_z"],
            "z_diff": analysis["z_diff"],
            "intra_model_var": analysis["intra_model_var"],
            "inter_model_var": analysis["inter_model_var"],
            "model_means": analysis["model_means"],
            "model_stds": analysis["model_stds"],
        })

    df = pd.DataFrame(results)
    return df


def print_benchmark_summary(df):
    """Print benchmark-level summary statistics."""
    print("\n" + "="*80)
    print("BENCHMARK-LEVEL SUMMARY")
    print("="*80)

    summary_data = []

    for benchmark in BENCHMARKS:
        bench_df = df[df["benchmark"] == benchmark]

        if len(bench_df) == 0:
            continue

        # Compute aggregate statistics
        n_questions = len(bench_df)
        avg_intra_var = bench_df["intra_model_var"].mean()
        avg_inter_var = bench_df["inter_model_var"].mean()
        avg_z_diff = bench_df["z_diff"].mean()

        # Compute z-score correlation
        if not bench_df["z_diff"].isna().all():
            helm_z = bench_df["helm_z"].values
            testtime_z = bench_df["testtime_z"].values
            corr, p_value = spearmanr(helm_z, testtime_z)
        else:
            corr = np.nan
            p_value = np.nan

        summary_data.append({
            "Benchmark": benchmark,
            "N": n_questions,
            "Intra-Model Var": f"{avg_intra_var:.4f}",
            "Inter-Model Var": f"{avg_inter_var:.4f}",
            "Avg |Δz|": f"{avg_z_diff:.4f}" if not np.isnan(avg_z_diff) else "N/A",
            "Correlation ρ": f"{corr:.3f}" if not np.isnan(corr) else "N/A",
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    print("\nInterpretation:")
    print("- Intra-Model Var: Average variance within a model's responses (temperature sensitivity)")
    print("- Inter-Model Var: Variance across models (how much models disagree)")
    print("- Avg |Δz|: Average absolute difference between HELM and test-time z-scores")
    print("- Correlation ρ: Spearman correlation between HELM and test-time z-scores")


def show_example_questions(df, n_per_benchmark=2):
    """Show specific question examples with high z-score discrepancies."""
    print("\n" + "="*80)
    print("EXAMPLE QUESTIONS WITH HIGH Z-SCORE DISCREPANCIES")
    print("="*80)

    for benchmark in BENCHMARKS:
        bench_df = df[df["benchmark"] == benchmark]

        if len(bench_df) == 0 or bench_df["z_diff"].isna().all():
            continue

        print(f"\n{'='*80}")
        print(f"{benchmark.upper()}")
        print(f"{'='*80}")

        # Get questions with highest z-score discrepancies
        top_discrepancy = bench_df.nlargest(n_per_benchmark, "z_diff")

        for _, row in top_discrepancy.iterrows():
            print(f"\nQuestion: {row['full_question'][:200]}...")
            print(f"HELM z-score:      {row['helm_z']:7.3f}")
            print(f"Test-time z-score: {row['testtime_z']:7.3f}")
            print(f"Difference:        {row['z_diff']:7.3f}")
            print(f"Intra-model var:   {row['intra_model_var']:7.4f} (response stability)")
            print(f"Inter-model var:   {row['inter_model_var']:7.4f} (model agreement)")
            print(f"Model pass@1 rates: {', '.join([f'{m:.3f}' for m in row['model_means']])}")
            print(f"Model stds:         {', '.join([f'{s:.3f}' for s in row['model_stds']])}")


def test_variance_transfer_correlation(df):
    """Test if response variance predicts transfer quality."""
    print("\n" + "="*80)
    print("VARIANCE vs TRANSFER QUALITY CORRELATION")
    print("="*80)

    # Aggregate by benchmark
    benchmark_stats = []

    for benchmark in BENCHMARKS:
        bench_df = df[df["benchmark"] == benchmark]

        if len(bench_df) == 0 or bench_df["z_diff"].isna().all():
            continue

        # Average intra-model variance (temperature sensitivity)
        avg_intra_var = bench_df["intra_model_var"].mean()

        # Spearman correlation (transfer quality)
        helm_z = bench_df["helm_z"].values
        testtime_z = bench_df["testtime_z"].values
        corr, _ = spearmanr(helm_z, testtime_z)

        benchmark_stats.append({
            "benchmark": benchmark,
            "intra_var": avg_intra_var,
            "correlation": corr
        })

    bench_df = pd.DataFrame(benchmark_stats)

    # Compute meta-correlation
    if len(bench_df) >= 3:
        meta_corr, meta_p = spearmanr(bench_df["intra_var"], bench_df["correlation"])

        print(f"\nBenchmark-level statistics:")
        print(bench_df.to_string(index=False))

        print(f"\n{'='*80}")
        print(f"Meta-correlation between intra-model variance and transfer quality:")
        print(f"Spearman ρ = {meta_corr:.3f} (p = {meta_p:.4f})")
        print(f"{'='*80}")

        if meta_corr < -0.5:
            print("\n✓ HYPOTHESIS CONFIRMED: Higher response variance → Lower transfer quality")
            print("  This supports the theory that temperature-sensitive tasks have unstable")
            print("  difficulty parameters that don't transfer across evaluation setups.")
        else:
            print("\n✗ Hypothesis not strongly supported by the data.")
    else:
        print("\nInsufficient benchmarks for meta-correlation analysis.")


def save_results(df, output_path="investigate_transfer_results.csv"):
    """Save detailed results to CSV."""
    # Drop numpy arrays for CSV export
    df_export = df.drop(columns=["model_means", "model_stds"])
    df_export.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("INVESTIGATING DIFFICULTY PARAMETER TRANSFER FAILURE")
    print("="*80)

    # Load data
    data = load_data()

    # Create analysis table
    df = create_analysis_table(data)

    # Print benchmark summary
    print_benchmark_summary(df)

    # Show example questions
    show_example_questions(df, n_per_benchmark=2)

    # Test variance-transfer correlation
    test_variance_transfer_correlation(df)

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "investigate_transfer_results.csv")
    save_results(df, output_path)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
