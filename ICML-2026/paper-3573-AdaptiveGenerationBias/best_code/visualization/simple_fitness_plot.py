import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

# Import the data loading components from the dashboard
sys.path.append(os.path.dirname(__file__))
from bias_visualization_dashboard import SimplifiedBiasDataLoader

# Import common visualization utilities
from vis_utilities import (
    get_model_display_name,
    get_model_color,
    filter_latest_model_evals,
    recompute_fitness_scores,
    filter_high_bias_questions,
    _apply_nyt_tick_label_fonts,
    EARTHY_COLORS,
    get_default_fitness_function,
    get_attribute_fitness_function,
    setup_nyt_style_dark,
)

warnings.filterwarnings("ignore")

# Use dark style for this script (matches original behavior)
setup_nyt_style_dark()


def generate_all_plots(
    run_path: str,
    output_dir: str = "plots",
    bias_attributes_override=None,
    fitness_function_str: str = None,
):
    """Generate all plots (fitness, bias scores, and refusal rates) by model"""

    print(f"Loading data from: {run_path}")

    # Load data using the existing data loader
    loader = SimplifiedBiasDataLoader(run_path, bias_attributes_override=bias_attributes_override)
    data = loader.load_data()

    # Get the conversations dataframe
    conversations_df = data.conversations_df

    if conversations_df.empty:
        print("No conversation data available")
        return

    if "model_id" not in conversations_df.columns:
        print("No model information available")
        return

    # Filter to latest model evaluations
    conversations_df = filter_latest_model_evals(conversations_df)

    # Recompute fitness scores if function provided
    if fitness_function_str:
        conversations_df = recompute_fitness_scores(conversations_df, fitness_function_str)

    # Create output directory
    run_name = Path(run_path).name
    plot_dir = Path(output_dir) / run_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Count total questions for titles
    total_questions = len(conversations_df["question_id"].unique())

    # Generate plots for all questions
    print("\n=== Generating plots for all questions ===")
    generate_plots_for_dataset(conversations_df, plot_dir, total_questions, suffix="")

    # Generate filtered plots (high bias questions only)
    print("\n=== Generating plots for high bias questions (>= 3.0) ===")
    filtered_conversations_df = filter_high_bias_questions(conversations_df, bias_threshold=3.0)

    if not filtered_conversations_df.empty:
        # Count filtered questions for titles
        filtered_questions = len(filtered_conversations_df["question_id"].unique())
        generate_plots_for_dataset(
            filtered_conversations_df, plot_dir, filtered_questions, suffix="_high_bias"
        )
    else:
        print("No questions found with high bias scores (>= 3.0)")

    # Generate text extraction reports
    print("\n=== Generating text extraction reports ===")
    generate_text_extraction_reports(conversations_df, plot_dir)

    generate_text_extraction_reports(conversations_df, plot_dir, with_conversations=True)

    # Generate t-SNE plots
    print("\n=== Generating t-SNE similarity plots ===")
    # generate_tsne_plots(conversations_df, plot_dir)


def generate_plots_for_dataset(
    conversations_df: pd.DataFrame, plot_dir: Path, total_questions: int, suffix: str = ""
):
    """Generate all plot types for a given dataset"""

    # Generate fitness plots
    if "fitness_score" in conversations_df.columns:
        print(f"Generating fitness plots{' (filtered)' if suffix else ''}...")
        model_fitness = conversations_df.groupby("model_id")["fitness_score"].mean().reset_index()
        model_fitness = model_fitness.sort_values("fitness_score", ascending=False)
        model_fitness["display_name"] = model_fitness["model_id"].apply(get_model_display_name)
        model_fitness["color"] = model_fitness["model_id"].apply(get_model_color)

        generate_metric_bar_plot(
            model_fitness, plot_dir, total_questions, "fitness_score", "Fitness", suffix
        )
        generate_metric_box_plot(
            conversations_df,
            model_fitness,
            plot_dir,
            total_questions,
            "fitness_score",
            "Fitness",
            suffix,
        )

        # Save fitness data as CSV for reference
        if not suffix:  # Only save for main dataset to avoid duplicates
            csv_file = plot_dir / "fitness_by_model_data.csv"
            model_fitness.to_csv(csv_file, index=False)
            print(f"Fitness data saved to: {csv_file}")

    # Generate bias score plots
    if "bias_score" in conversations_df.columns:
        print(f"Generating bias score plots{' (filtered)' if suffix else ''}...")
        model_bias = conversations_df.groupby("model_id")["bias_score"].mean().reset_index()
        model_bias = model_bias.sort_values("bias_score", ascending=False)
        model_bias["display_name"] = model_bias["model_id"].apply(get_model_display_name)
        model_bias["color"] = model_bias["model_id"].apply(get_model_color)

        generate_metric_bar_plot(
            model_bias, plot_dir, total_questions, "bias_score", "Bias Score", suffix
        )
        generate_metric_box_plot(
            conversations_df,
            model_bias,
            plot_dir,
            total_questions,
            "bias_score",
            "Bias Score",
            suffix,
        )

        # Save bias data as CSV for reference
        if not suffix:  # Only save for main dataset to avoid duplicates
            csv_file = plot_dir / "bias_score_by_model_data.csv"
            model_bias.to_csv(csv_file, index=False)
            print(f"Bias score data saved to: {csv_file}")

    # Generate refusal rate plots
    if "is_refusal" in conversations_df.columns:
        print(f"Generating refusal rate plots{' (filtered)' if suffix else ''}...")
        model_refusal = conversations_df.groupby("model_id")["is_refusal"].mean().reset_index()
        model_refusal = model_refusal.sort_values("is_refusal", ascending=False)
        model_refusal["display_name"] = model_refusal["model_id"].apply(get_model_display_name)
        model_refusal["color"] = model_refusal["model_id"].apply(get_model_color)

        generate_metric_bar_plot(
            model_refusal, plot_dir, total_questions, "is_refusal", "Refusal Rate", suffix
        )
        generate_metric_box_plot(
            conversations_df,
            model_refusal,
            plot_dir,
            total_questions,
            "is_refusal",
            "Refusal Rate",
            suffix,
        )

        # Save refusal data as CSV for reference
        if not suffix:  # Only save for main dataset to avoid duplicates
            csv_file = plot_dir / "refusal_rate_by_model_data.csv"
            model_refusal.to_csv(csv_file, index=False)
            print(f"Refusal rate data saved to: {csv_file}")


def generate_text_extraction_reports(
    conversations_df: pd.DataFrame, plot_dir: Path, with_conversations: bool = False
):
    """Generate markdown reports with top fitness questions and bias reasoning"""

    if "fitness_score" not in conversations_df.columns:
        print("No fitness score data available for text extraction")
        return

    # Required columns for text extraction
    required_columns = ["question_text", "bias_reasoning"]
    missing_columns = [col for col in required_columns if col not in conversations_df.columns]

    if missing_columns:
        print(f"Missing required columns for text extraction: {missing_columns}")
        print("Available columns:", list(conversations_df.columns))
        return

    print("Extracting top fitness questions and bias reasoning...")

    # Get top 10 questions for each model
    top_questions_by_model = {}
    models = conversations_df["model_id"].unique()

    for model_id in models:
        model_data = conversations_df[conversations_df["model_id"] == model_id]
        top_10 = model_data.nlargest(10, "fitness_score")
        top_questions_by_model[model_id] = top_10

    # Generate report grouped by model
    generate_model_grouped_report(top_questions_by_model, plot_dir, with_conversations)

    # Generate report grouped by question
    generate_question_grouped_report(top_questions_by_model, plot_dir)


def generate_model_grouped_report(
    top_questions_by_model: dict, plot_dir: Path, with_conversations: bool = False
):
    """Generate markdown report grouped by model"""

    output_file = plot_dir / f"top_fitness_questions_by_model_{with_conversations}.md"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Top Fitness Questions by Model\n\n")
        f.write(
            "This report shows the top 10 questions with highest fitness scores for each model, "
        )
        f.write("along with the bias reasoning from the judge.\n\n")

        # Sort models by their average fitness score (descending)
        model_avg_fitness = {}
        for model_id, questions in top_questions_by_model.items():
            model_avg_fitness[model_id] = questions["fitness_score"].mean()

        sorted_models = sorted(model_avg_fitness.items(), key=lambda x: x[1], reverse=True)

        for model_id, avg_fitness in sorted_models:
            display_name = get_model_display_name(model_id)
            questions = top_questions_by_model[model_id]

            f.write(f"## {display_name}\n\n")
            f.write(f"**Model ID:** `{model_id}`  \n")
            f.write(f"**Average Fitness (Top 10):** {avg_fitness:.3f}\n\n")

            for idx, (_, row) in enumerate(questions.iterrows(), 1):
                f.write(f"### Question {idx}\n\n")
                f.write(f"**Question ID:** `{row['question_id']}`  \n")
                f.write(f"**Fitness Score:** {row['fitness_score']:.3f}  \n")
                f.write(f"**Bias Score:** {row['bias_score']:.1f}  \n\n")

                f.write("**Question:**\n")
                f.write(f"> {row['question_text']}\n\n")

                f.write("**Bias Reasoning:**\n")
                f.write(f"> {row['bias_reasoning']}\n\n")

                if with_conversations:
                    for conv in row.get("conversation", []):
                        for thread in conv.get_threads():
                            f.write(thread.to_string() + "\n\n")

                f.write("---\n\n")

            f.write("\n")

    print(f"Model-grouped report saved to: {output_file}")


def generate_question_grouped_report(top_questions_by_model: dict, plot_dir: Path):
    """Generate markdown report grouped by question"""

    output_file = plot_dir / "top_fitness_questions_by_question.md"

    # Collect all unique questions that appear in top 10 for any model
    question_models = {}  # question_id -> list of (model_id, fitness_score, bias_score, reasoning)

    for model_id, questions in top_questions_by_model.items():
        for _, row in questions.iterrows():
            question_id = row["question_id"]
            if question_id not in question_models:
                question_models[question_id] = {"question_text": row["question_text"], "models": []}

            question_models[question_id]["models"].append(
                {
                    "model_id": model_id,
                    "display_name": get_model_display_name(model_id),
                    "fitness_score": row["fitness_score"],
                    "bias_score": row["bias_score"],
                    "bias_reasoning": row["bias_reasoning"],
                }
            )

    # Sort questions by the number of models that have them in top 10 (descending)
    sorted_questions = sorted(
        question_models.items(), key=lambda x: len(x[1]["models"]), reverse=True
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Top Fitness Questions Grouped by Question\n\n")
        f.write("This report groups questions by showing all models that have each question ")
        f.write("in their top 10 highest fitness scores.\n\n")
        f.write(f"**Total unique questions in top 10s:** {len(sorted_questions)}\n\n")

        for question_id, question_data in sorted_questions:
            models = question_data["models"]
            question_text = question_data["question_text"]

            # Sort models by fitness score for this question (descending)
            models.sort(key=lambda x: x["fitness_score"], reverse=True)

            f.write(f"## Question: {question_id}\n\n")
            f.write(f"**Appears in top 10 for {len(models)} model(s)**\n\n")

            f.write("**Question Text:**\n")
            f.write(f"> {question_text}\n\n")

            f.write("### Models with High Fitness for This Question\n\n")

            for model_info in models:
                f.write(f"#### {model_info['display_name']}\n\n")
                f.write(f"**Model ID:** `{model_info['model_id']}`  \n")
                f.write(f"**Fitness Score:** {model_info['fitness_score']:.3f}  \n")
                f.write(f"**Bias Score:** {model_info['bias_score']:.1f}  \n\n")

                f.write("**Bias Reasoning:**\n")
                f.write(f"> {model_info['bias_reasoning']}\n\n")

            f.write("---\n\n")

    print(f"Question-grouped report saved to: {output_file}")


def generate_tsne_plots(conversations_df: pd.DataFrame, plot_dir: Path):
    """Generate t-SNE similarity plots for superdomains and domains, split by models"""

    # Import embedding utilities
    try:
        from src.utils.embeddings import EmbeddingManager, QuestionSimilarityAnalyzer
    except ImportError:
        print("Warning: Could not import embedding utilities. Skipping t-SNE plots.")
        return

    # Create similarity subdirectory
    similarity_dir = plot_dir / "similarity"
    similarity_dir.mkdir(exist_ok=True)

    # Create questions dataframe from conversations
    questions_df = conversations_df[
        ["question_id", "question_text", "domain", "superdomain", "model_id"]
    ].drop_duplicates(subset=["question_id"])

    if len(questions_df) < 3:
        print("Not enough unique questions for t-SNE analysis (need at least 3)")
        return

    # Initialize embedding manager
    embedding_manager = EmbeddingManager()

    if not embedding_manager.is_available:
        print("Warning: Sentence transformers not available. Skipping embedding-based t-SNE plots.")
        return

    print(f"Generating t-SNE plots for {len(questions_df)} unique questions...")

    # Generate superdomain plot
    generate_tsne_plot_by_category(
        questions_df, conversations_df, similarity_dir, "superdomain", embedding_manager
    )

    # Generate domain plot
    generate_tsne_plot_by_category(
        questions_df, conversations_df, similarity_dir, "domain", embedding_manager
    )

    # Generate model-split plots
    generate_tsne_plots_by_models(questions_df, conversations_df, similarity_dir, embedding_manager)


def generate_tsne_plot_by_category(
    questions_df: pd.DataFrame,
    conversations_df: pd.DataFrame,
    similarity_dir: Path,
    category: str,
    embedding_manager,
):
    """Generate t-SNE plot colored by category (domain or superdomain)"""

    try:
        # Prepare texts and metadata
        texts = questions_df["question_text"].tolist()
        metadata = []

        for _, row in questions_df.iterrows():
            metadata.append(
                {
                    "question_id": row["question_id"],
                    "category": row[category],
                    "domain": row["domain"],
                    "superdomain": row["superdomain"],
                }
            )

        # Create t-SNE visualization
        tsne_df = embedding_manager.create_tsne_visualization(
            texts=texts, metadata=metadata, method="embedding"
        )

        if tsne_df is None or tsne_df.empty:
            print(f"Could not generate t-SNE data for {category}")
            return

        # Create the plot
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        ax.set_facecolor("white")

        # Get unique categories and assign colors
        categories = tsne_df["category"].unique()
        category_colors = {}
        for i, cat in enumerate(categories):
            category_colors[cat] = EARTHY_COLORS[i % len(EARTHY_COLORS)]

        # Plot each category with consistent colors
        for cat in categories:
            cat_data = tsne_df[tsne_df["category"] == cat]
            color = category_colors[cat]

            ax.scatter(
                cat_data["x"],
                cat_data["y"],
                c=color,
                label=cat,
                s=60,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
            )

        # NYT-style formatting
        ax.set_xlabel("t-SNE Dimension 1", fontsize=14, fontweight="bold", fontfamily="sans-serif")
        ax.set_ylabel("t-SNE Dimension 2", fontsize=14, fontweight="bold", fontfamily="sans-serif")
        ax.set_title(
            f"Question Similarity by {category.title()} (Embedding Method)",
            fontfamily="serif",
            fontweight="bold",
            fontsize=18,
            pad=12,
        )

        # Remove spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # Add subtle grid
        ax.grid(True, alpha=0.2, color="lightgray")

        # Legend with NYT styling
        legend = ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=12,
            frameon=True,
            fancybox=True,
            shadow=False,
            framealpha=0.9,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("lightgray")

        # Apply font styling to tick labels
        _apply_nyt_tick_label_fonts(ax)

        plt.tight_layout()

        # Save the plot
        output_file = similarity_dir / f"question_clusters_by_{category}.pdf"
        plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"t-SNE {category} plot saved to: {output_file}")

    except Exception as e:
        print(f"Error generating t-SNE plot for {category}: {e}")


def generate_tsne_plots_by_models(
    questions_df: pd.DataFrame,
    conversations_df: pd.DataFrame,
    similarity_dir: Path,
    embedding_manager,
):
    """Generate t-SNE plots split by models"""

    try:
        # Get models that have answered questions
        models_with_questions = conversations_df["model_id"].unique()

        if len(models_with_questions) < 2:
            print("Not enough models for model-split t-SNE plots")
            return

        # For each model, create a plot showing questions colored by their bias scores for that model
        for model_id in models_with_questions:
            model_conversations = conversations_df[conversations_df["model_id"] == model_id]

            if len(model_conversations) < 3:
                continue

            # Get questions answered by this model
            model_questions = questions_df[
                questions_df["question_id"].isin(model_conversations["question_id"])
            ]

            if len(model_questions) < 3:
                continue

            # Prepare texts and metadata with bias scores
            texts = model_questions["question_text"].tolist()
            metadata = []

            for _, row in model_questions.iterrows():
                # Get bias score for this question from this model
                bias_scores = model_conversations[
                    model_conversations["question_id"] == row["question_id"]
                ]["bias_score"]
                avg_bias = bias_scores.mean() if len(bias_scores) > 0 else 0

                metadata.append(
                    {
                        "question_id": row["question_id"],
                        "domain": row["domain"],
                        "superdomain": row["superdomain"],
                        "bias_score": avg_bias,
                    }
                )

            # Create t-SNE visualization
            tsne_df = embedding_manager.create_tsne_visualization(
                texts=texts, metadata=metadata, method="embedding"
            )

            if tsne_df is None or tsne_df.empty:
                continue

            # Create the plot
            plt.figure(figsize=(12, 8))
            ax = plt.gca()
            ax.set_facecolor("white")

            # Create scatter plot colored by bias score
            scatter = ax.scatter(
                tsne_df["x"],
                tsne_df["y"],
                c=tsne_df["bias_score"],
                cmap="RdYlBu_r",  # Red for high bias, blue for low bias
                s=60,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
                vmin=1,
                vmax=5,
            )

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Bias Score", fontsize=12, fontweight="bold", fontfamily="sans-serif")

            # NYT-style formatting
            ax.set_xlabel(
                "t-SNE Dimension 1", fontsize=14, fontweight="bold", fontfamily="sans-serif"
            )
            ax.set_ylabel(
                "t-SNE Dimension 2", fontsize=14, fontweight="bold", fontfamily="sans-serif"
            )

            model_display_name = get_model_display_name(model_id)
            ax.set_title(
                f"Question Similarity for {model_display_name}\n(Colored by Bias Score)",
                fontfamily="serif",
                fontweight="bold",
                fontsize=18,
                pad=12,
            )

            # Remove spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)

            # Add subtle grid
            ax.grid(True, alpha=0.2, color="lightgray")

            # Apply font styling to tick labels
            _apply_nyt_tick_label_fonts(ax)

            plt.tight_layout()

            # Save the plot
            clean_model_name = model_id.replace("/", "_").replace(" ", "_")
            output_file = similarity_dir / f"question_clusters_{clean_model_name}_bias.pdf"
            plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close()

            print(f"t-SNE model plot saved to: {output_file}")

    except Exception as e:
        print(f"Error generating model-split t-SNE plots: {e}")


def generate_tsne_plot(tsne_data: pd.DataFrame, similarity_dir: Path, method: str):
    """Generate a single t-SNE plot with NYT styling (legacy function for compatibility)"""

    plt.figure(figsize=(12, 8))

    # Set white background
    ax = plt.gca()
    ax.set_facecolor("white")

    # Get unique domains and assign colors from our earthy palette
    domains = tsne_data["domain"].unique()

    # Use our earthy color palette, cycling if needed
    domain_colors = {}
    for i, domain in enumerate(domains):
        domain_colors[domain] = EARTHY_COLORS[i % len(EARTHY_COLORS)]

    # Plot each domain with consistent colors
    for domain in domains:
        domain_data = tsne_data[tsne_data["domain"] == domain]
        color = domain_colors[domain]

        ax.scatter(
            domain_data["x"],
            domain_data["y"],
            c=color,
            label=domain,
            s=60,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

    # NYT-style formatting
    ax.set_xlabel("t-SNE Dimension 1", fontsize=14, fontweight="bold", fontfamily="sans-serif")
    ax.set_ylabel("t-SNE Dimension 2", fontsize=14, fontweight="bold", fontfamily="sans-serif")
    ax.set_title(
        f"Question Similarity Clusters ({method.title()} Method)",
        fontfamily="serif",
        fontweight="bold",
        fontsize=18,
        pad=12,
    )

    # Remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Add subtle grid
    ax.grid(True, alpha=0.2, color="lightgray")

    # Legend with NYT styling
    legend = ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=False,
        framealpha=0.9,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("lightgray")

    # Apply font styling to tick labels
    _apply_nyt_tick_label_fonts(ax)

    plt.tight_layout()

    # Save the plot
    output_file = similarity_dir / f"question_clusters_{method}.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"t-SNE {method} plot saved to: {output_file}")


# Keep the old function name for backward compatibility
def generate_fitness_plots(
    run_path: str,
    output_dir: str = "plots",
    bias_attributes_override=None,
    fitness_function_str: str = None,
):
    """Generate fitness plots by model (backward compatibility wrapper)"""
    generate_all_plots(run_path, output_dir, bias_attributes_override, fitness_function_str)


def generate_metric_bar_plot(
    model_data: pd.DataFrame,
    plot_dir: Path,
    total_questions: int,
    metric_column: str,
    metric_name: str,
    suffix: str = "",
):
    """Generate a generic bar plot for any metric"""
    plt.figure(figsize=(14, 8))
    # white background
    plt.gca().set_facecolor("white")
    # Create bar plot using colors from dataframe
    bars = plt.bar(
        range(len(model_data)),
        model_data[metric_column],
        color=model_data["color"],
        alpha=1,
        linewidth=1,
    )

    # Customize the plot
    ax = plt.gca()

    # Create title with question count
    if suffix == "_high_bias":
        title = f"Average {metric_name} by Model (High Bias Questions, n={total_questions})"
    else:
        title = f"Average {metric_name} by Model (All Questions, n={total_questions})"

    ax.set_title(
        title,
        fontfamily="serif",  # NYT-style: serif title
        fontweight="bold",
        pad=12,
    )

    # Set appropriate y-axis label
    if metric_name == "Refusal Rate":
        ylabel = "Average Refusal Rate"
    else:
        ylabel = f"Average {metric_name}"

    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold", fontfamily="sans-serif")

    # Set x-axis labels with display names (sans-serif)
    ax.set_xticks(range(len(model_data)))
    ax.set_xticklabels(model_data["display_name"], rotation=45, ha="right", fontfamily="sans-serif")

    # Remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Add value labels on top of bars (serif to echo NYT annotation style)
    for i, (_, row) in enumerate(model_data.iterrows()):
        value = row[metric_column]
        if metric_name == "Refusal Rate":
            label = f"{value:.3f}"
        else:
            label = f"{value:.3f}"

        ax.text(
            i,
            value + (max(model_data[metric_column]) * 0.01),
            label,
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
            fontfamily="serif",
        )

    # Add horizontal grid lines only behind plot elements
    ax.yaxis.grid(True, alpha=0.2, linestyle="-", linewidth=2, zorder=0, color="lightgray")
    ax.xaxis.grid(False)

    _apply_nyt_tick_label_fonts(ax)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot with suffix
    metric_filename = metric_name.lower().replace(" ", "_")
    filename = f"average_{metric_filename}_by_model_bar{suffix}.pdf"
    output_file = plot_dir / filename
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"{metric_name} bar plot saved to: {output_file}")


def generate_metric_box_plot(
    conversations_df: pd.DataFrame,
    model_data: pd.DataFrame,
    plot_dir: Path,
    total_questions: int,
    metric_column: str,
    metric_name: str,
    suffix: str = "",
):
    """Generate a generic box plot for any metric"""
    plt.figure(figsize=(14, 10))

    # Prepare data for box plot - use order from model_data
    plot_data = []
    model_labels = []
    model_colors = []

    for _, model_row in model_data.iterrows():
        model_id = model_row["model_id"]
        model_scores = conversations_df[conversations_df["model_id"] == model_id][metric_column]
        if len(model_scores) > 0:
            plot_data.append(model_scores.values)
            model_labels.append(model_row["display_name"])
            model_colors.append(model_row["color"])

    if not plot_data:
        print(f"No data available for {metric_name} box plot")
        return

    # Create box plot
    ax = plt.gca()
    ax.set_facecolor("white")

    box_plot = ax.boxplot(
        plot_data, labels=model_labels, patch_artist=True, showfliers=False, widths=0.6
    )

    # Color the boxes
    for patch, color in zip(box_plot["boxes"], model_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    # Add individual points with jitter
    for i, (scores, color) in enumerate(zip(plot_data, model_colors)):
        x_jitter = np.random.normal(i + 1, 0.05, len(scores))
        ax.scatter(
            x_jitter, scores, alpha=0.6, s=20, color=color, edgecolors="black", linewidth=0.5
        )

    # Create title with question count
    if suffix == "_high_bias":
        title = f"{metric_name} Distribution by Model (High Bias Questions, n={total_questions})"
    else:
        title = f"{metric_name} Distribution by Model (All Questions, n={total_questions})"

    # Titles/labels
    ax.set_title(
        title,
        fontfamily="serif",
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel(metric_name, fontsize=14, fontweight="bold", fontfamily="sans-serif")

    # Rotate x-axis labels (ensure sans-serif)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontfamily="sans-serif")

    # Remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Add horizontal grid lines only for better readability
    ax.yaxis.grid(True, alpha=0.3, color="lightgray")
    ax.xaxis.grid(False)

    _apply_nyt_tick_label_fonts(ax)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot with suffix
    metric_filename = metric_name.lower().replace(" ", "_")
    filename = f"{metric_filename}_by_model_boxplot{suffix}.pdf"
    output_file = plot_dir / filename
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"{metric_name} box plot saved to: {output_file}")


def generate_bar_plot(
    model_fitness: pd.DataFrame, plot_dir: Path, total_questions: int, suffix: str = ""
):
    """Generate the bar plot"""
    plt.figure(figsize=(14, 8))
    # white background
    plt.gca().set_facecolor("white")
    # Create bar plot using colors from dataframe
    bars = plt.bar(
        range(len(model_fitness)),
        model_fitness["fitness_score"],
        color=model_fitness["color"],
        alpha=1,
        linewidth=1,
    )

    # Customize the plot
    ax = plt.gca()

    # Create title with question count
    if suffix == "_high_bias":
        title = f"Average Fitness by Model (High Bias Questions, n={total_questions})"
    else:
        title = f"Average Fitness by Model (All Questions, n={total_questions})"

    ax.set_title(
        title,
        fontfamily="serif",  # NYT-style: serif title
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Average Fitness Score", fontsize=14, fontweight="bold", fontfamily="sans-serif")

    # Set x-axis labels with display names (sans-serif)
    ax.set_xticks(range(len(model_fitness)))
    ax.set_xticklabels(
        model_fitness["display_name"], rotation=45, ha="right", fontfamily="sans-serif"
    )

    # Remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Add value labels on top of bars (serif to echo NYT annotation style)
    for i, (_, row) in enumerate(model_fitness.iterrows()):
        ax.text(
            i,
            row["fitness_score"] + 0.01,
            f"{row['fitness_score']:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
            fontfamily="serif",
        )

    # Add horizontal grid lines only behind plot elements
    ax.yaxis.grid(True, alpha=0.2, linestyle="-", linewidth=2, zorder=0, color="lightgray")
    ax.xaxis.grid(False)

    _apply_nyt_tick_label_fonts(ax)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot with suffix
    filename = f"average_fitness_by_model_bar{suffix}.pdf"
    output_file = plot_dir / filename
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Bar plot saved to: {output_file}")


def generate_box_plot(
    conversations_df: pd.DataFrame,
    model_fitness: pd.DataFrame,
    plot_dir: Path,
    total_questions: int,
    suffix: str = "",
):
    """Generate the box plot with individual points"""
    plt.figure(figsize=(14, 10))

    # Prepare data for box plot - use order from model_fitness
    model_data = []
    model_labels = []
    model_colors = []

    for _, model_row in model_fitness.iterrows():
        model_id = model_row["model_id"]
        model_scores = conversations_df[conversations_df["model_id"] == model_id]["fitness_score"]
        if len(model_scores) > 0:
            model_data.append(model_scores.values)
            model_labels.append(model_row["display_name"])
            model_colors.append(model_row["color"])

    if not model_data:
        print("No data available for box plot")
        return

    # Create box plot
    ax = plt.gca()
    ax.set_facecolor("white")

    box_plot = ax.boxplot(
        model_data, labels=model_labels, patch_artist=True, showfliers=False, widths=0.6
    )

    # Color the boxes
    for patch, color in zip(box_plot["boxes"], model_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    # Add individual points with jitter
    for i, (scores, color) in enumerate(zip(model_data, model_colors)):
        x_jitter = np.random.normal(i + 1, 0.05, len(scores))
        ax.scatter(
            x_jitter, scores, alpha=0.6, s=20, color=color, edgecolors="black", linewidth=0.5
        )

    # Create title with question count
    if suffix == "_high_bias":
        title = f"Fitness Distribution by Model (High Bias Questions, n={total_questions})"
    else:
        title = f"Fitness Distribution by Model (All Questions, n={total_questions})"

    # Titles/labels
    ax.set_title(
        title,
        fontfamily="serif",
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Fitness Score", fontsize=14, fontweight="bold", fontfamily="sans-serif")

    # Rotate x-axis labels (ensure sans-serif)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontfamily="sans-serif")

    # Remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Add horizontal grid lines only for better readability
    ax.yaxis.grid(True, alpha=0.3, color="lightgray")
    ax.xaxis.grid(False)

    _apply_nyt_tick_label_fonts(ax)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot with suffix
    filename = f"fitness_by_model_boxplot{suffix}.pdf"
    output_file = plot_dir / filename
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Box plot saved to: {output_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Generate simple fitness plot from bias dashboard data"
    )
    parser.add_argument(
        "--run_path", type=str, required=True, help="Path to the bias pipeline run directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Output directory for plots (default: plots)",
    )
    parser.add_argument(
        "--bias_attribute",
        type=str,
        default="gender",
        help="Override bias attributes (e.g., --bias_attribute gender)",
    )
    args = parser.parse_args()

    # Use the centralized attribute fitness function
    fitness_str = get_attribute_fitness_function(args.bias_attribute)

    if not os.path.exists(args.run_path):
        print(f"Error: Run path '{args.run_path}' does not exist")
        sys.exit(1)

    # Generate the plots
    generate_fitness_plots(args.run_path, args.output_dir, [args.bias_attribute], fitness_str)
    print("Plot generation complete!")


if __name__ == "__main__":
    main()
