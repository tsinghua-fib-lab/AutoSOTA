import os
import json
import gradio as gr
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Import similarity computation components
from src.utils.embeddings import EmbeddingManager

# Global variables to store all loaded questions and similarity data
ALL_QUESTIONS = []
AVAILABLE_CATEGORIES = []
SIMILARITY_MATRIX = None
EMBEDDING_MANAGER = None
QUESTION_SIMILARITIES = {}

# === Utilities ===


def load_jsonl_as_list(path: str) -> List[Dict]:
    """Load JSONL file as list of dictionaries."""
    data = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data.append(json.loads(line))
                    except Exception as e:
                        print(f"Error parsing line in {path}: {e}")
                        continue
        except Exception as e:
            print(f"Error reading file {path}: {e}")
    return data


def get_jsonl_files_in_directory(directory_path: str) -> List[str]:
    """Get list of JSONL files in the specified directory."""
    jsonl_files = []

    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        for item in os.listdir(directory_path):
            if item.endswith(".jsonl"):
                jsonl_files.append(item[:-6])  # Remove .jsonl extension

    return sorted(jsonl_files)


def format_question_header(question: Dict) -> str:
    """Format question metadata as header."""
    parts = []

    # Add ID if available
    if question.get("id"):
        parts.append(f"ID: {question['id']}")

    # Add superdomain, domain, topic
    if question.get("superdomain"):
        parts.append(f"Superdomain: {question['superdomain']}")
    if question.get("domain"):
        parts.append(f"Domain: {question['domain']}")
    if question.get("topic"):
        parts.append(f"Topic: {question['topic']}")

    # Add original model if available
    if question.get("orig_model"):
        parts.append(f"Model: {question['orig_model']}")

    return " | ".join(parts) if parts else "No metadata available"


def compute_question_similarities():
    """Compute similarity matrix for all loaded questions."""
    global ALL_QUESTIONS, SIMILARITY_MATRIX, EMBEDDING_MANAGER, QUESTION_SIMILARITIES

    if not ALL_QUESTIONS:
        return

    # Initialize embedding manager if not already done
    if EMBEDDING_MANAGER is None:
        EMBEDDING_MANAGER = EmbeddingManager()

    # Extract question texts for similarity computation
    question_texts = []
    for q in ALL_QUESTIONS:
        text = q.get("example", "") or ""
        question_texts.append(text)

    if len(question_texts) < 2:
        return

    try:
        # Compute similarity matrix using embeddings
        SIMILARITY_MATRIX, _ = EMBEDDING_MANAGER.compute_similarity_matrix(
            question_texts, method="embedding"
        )

        # Compute top 3 similar questions for each question
        QUESTION_SIMILARITIES = {}
        for i, question in enumerate(ALL_QUESTIONS):
            # Get similarity scores for this question (excluding self)
            similarities = []
            for j, other_question in enumerate(ALL_QUESTIONS):
                if i != j:
                    sim_score = SIMILARITY_MATRIX[i][j]
                    similarities.append((j, sim_score, other_question))

            # Sort by similarity score and take top 3
            similarities.sort(key=lambda x: x[1], reverse=True)
            QUESTION_SIMILARITIES[i] = similarities[:3]

    except Exception as e:
        print(f"Error computing similarities: {e}")
        # Fallback to fuzzy similarity if embedding fails
        try:
            SIMILARITY_MATRIX, _ = EMBEDDING_MANAGER.compute_similarity_matrix(
                question_texts, method="fuzzy"
            )

            # Compute top 3 similar questions for each question
            QUESTION_SIMILARITIES = {}
            for i, question in enumerate(ALL_QUESTIONS):
                similarities = []
                for j, other_question in enumerate(ALL_QUESTIONS):
                    if i != j:
                        sim_score = SIMILARITY_MATRIX[i][j] / 100.0  # Normalize fuzzy scores
                        similarities.append((j, sim_score, other_question))

                similarities.sort(key=lambda x: x[1], reverse=True)
                QUESTION_SIMILARITIES[i] = similarities[:3]

        except Exception as e2:
            print(f"Error computing fuzzy similarities: {e2}")


def load_all_questions_from_directory(directory_path: str) -> Tuple[str, List[str]]:
    """Load all questions from all JSONL files in the directory."""
    global ALL_QUESTIONS, AVAILABLE_CATEGORIES

    if not directory_path:
        return "❌ Please specify a directory path.", []

    if not os.path.exists(directory_path):
        return f"❌ Directory {directory_path} does not exist.", []

    if not os.path.isdir(directory_path):
        return f"❌ {directory_path} is not a directory.", []

    available_files = get_jsonl_files_in_directory(directory_path)
    if not available_files:
        return "❌ No JSONL files found in the directory.", []

    ALL_QUESTIONS = []
    loaded_files = []
    missing_files = []

    for file_name in available_files:
        file_path = os.path.join(directory_path, f"{file_name}.jsonl")
        print(f"Loading questions from {file_path}.jsonl...")
        questions = load_jsonl_as_list(file_path)

        print(f"  Loaded {len(questions)} questions from {file_name}.jsonl")

        if questions:
            # Add file info to each question
            for q in questions:
                q["source_file"] = file_name
            ALL_QUESTIONS.extend(questions)
            loaded_files.append(f"{file_name} ({len(questions)} questions)")
        else:
            missing_files.append(file_name)

    # Update available categories
    AVAILABLE_CATEGORIES = ["All"] + available_files

    # Compute similarities after loading questions
    if ALL_QUESTIONS:
        print("Computing question similarities...")
        compute_question_similarities()

    # Create status message
    status_parts = []
    if loaded_files:
        status_parts.append(
            f"✅ Loaded {len(ALL_QUESTIONS)} questions from: {', '.join(loaded_files)}"
        )
        if SIMILARITY_MATRIX is not None:
            status_parts.append("✅ Computed question similarities")
    if missing_files:
        status_parts.append(f"⚠️ Missing or empty files: {', '.join(missing_files)}")

    status = "\n".join(status_parts) if status_parts else "❌ No questions found."

    return status, AVAILABLE_CATEGORIES


def get_max_similarity_score(question_index: int) -> float:
    """Get the maximum similarity score for a question."""
    global QUESTION_SIMILARITIES

    if question_index not in QUESTION_SIMILARITIES:
        return 0.0

    similarities = QUESTION_SIMILARITIES[question_index]
    if not similarities:
        return 0.0

    return similarities[0][1]  # First item has highest similarity


def filter_questions_by_category(selected_category: str) -> Tuple[str, int]:
    """Filter questions by selected category and return display content."""
    global ALL_QUESTIONS

    if not ALL_QUESTIONS:
        return "No questions loaded. Please scan a directory first.", 0

    if selected_category == "All":
        filtered_questions = ALL_QUESTIONS
    else:
        filtered_questions = [q for q in ALL_QUESTIONS if q.get("source_file") == selected_category]

    display_content = format_questions_display(filtered_questions)
    return display_content, len(filtered_questions)


def format_questions_display(questions: List[Dict]) -> str:
    """Format questions for display in markdown, sorted by similarity."""
    if not questions:
        return "No questions to display."

    # Create a mapping from question to its index in ALL_QUESTIONS for similarity lookup
    question_to_index = {}
    for i, q in enumerate(ALL_QUESTIONS):
        question_to_index[id(q)] = i

    # Sort questions by maximum similarity score (highest first)
    questions_with_similarity = []
    for question in questions:
        question_index = question_to_index.get(id(question), -1)
        max_sim_score = get_max_similarity_score(question_index)
        questions_with_similarity.append((question, question_index, max_sim_score))

    # Sort by similarity score (highest first)
    questions_with_similarity.sort(key=lambda x: x[2], reverse=True)

    markdown_content = []

    for display_idx, (question, question_index, max_sim_score) in enumerate(
        questions_with_similarity, 1
    ):
        # Header with metadata
        header = format_question_header(question)
        source_file = question.get("source_file", "unknown")

        # Question text
        example = question.get("example", "No question text available")

        # Format main question with clickable button
        markdown_content.append(
            f"### Question {display_idx} [{source_file.upper()}] (Max Similarity: {max_sim_score:.3f})"
        )
        markdown_content.append(f"**{header}**")
        markdown_content.append("")
        markdown_content.append(f"{example}")
        markdown_content.append("")
        markdown_content.append(
            f"*Click 'Show Similar Questions' to see the 3 most similar questions for Question {display_idx}*"
        )
        markdown_content.append("")
        markdown_content.append("---")
        markdown_content.append("")

    return "\n".join(markdown_content)


def get_similar_questions_display(question_number: int, selected_category: str) -> str:
    """Get similar questions display for a specific question number."""
    global ALL_QUESTIONS, QUESTION_SIMILARITIES

    if not ALL_QUESTIONS:
        return "No questions loaded."

    # Filter questions by category first
    if selected_category == "All":
        filtered_questions = ALL_QUESTIONS
    else:
        filtered_questions = [q for q in ALL_QUESTIONS if q.get("source_file") == selected_category]

    if not filtered_questions:
        return "No questions in selected category."

    # Create mapping and sort by similarity (same as in format_questions_display)
    question_to_index = {}
    for i, q in enumerate(ALL_QUESTIONS):
        question_to_index[id(q)] = i

    questions_with_similarity = []
    for question in filtered_questions:
        question_index = question_to_index.get(id(question), -1)
        max_sim_score = get_max_similarity_score(question_index)
        questions_with_similarity.append((question, question_index, max_sim_score))

    questions_with_similarity.sort(key=lambda x: x[2], reverse=True)

    # Check if question number is valid
    if question_number < 1 or question_number > len(questions_with_similarity):
        return f"Invalid question number. Please select a number between 1 and {len(questions_with_similarity)}."

    # Get the selected question (adjust for 0-based indexing)
    selected_question, question_index, max_sim_score = questions_with_similarity[
        question_number - 1
    ]

    # Build similar questions display
    markdown_content = []

    # Show the selected question first
    header = format_question_header(selected_question)
    source_file = selected_question.get("source_file", "unknown")
    example = selected_question.get("example", "No question text available")

    markdown_content.append(f"## Selected Question {question_number}")
    markdown_content.append(f"**[{source_file.upper()}] {header}**")
    markdown_content.append("")
    markdown_content.append(f"{example}")
    markdown_content.append("")
    markdown_content.append("---")
    markdown_content.append("")

    # Show similar questions
    if question_index in QUESTION_SIMILARITIES and QUESTION_SIMILARITIES[question_index]:
        markdown_content.append("## 🔗 Most Similar Questions")
        markdown_content.append("")

        for sim_idx, (similar_q_idx, sim_score, similar_question) in enumerate(
            QUESTION_SIMILARITIES[question_index], 1
        ):
            similar_header = format_question_header(similar_question)
            similar_source = similar_question.get("source_file", "unknown")
            similar_text = similar_question.get("example", "No question text available")

            markdown_content.append(f"### {sim_idx}. Similarity Score: {sim_score:.3f}")
            markdown_content.append(f"**[{similar_source.upper()}] {similar_header}**")
            markdown_content.append("")
            markdown_content.append(f"{similar_text}")
            markdown_content.append("")
            markdown_content.append("---")
            markdown_content.append("")
    else:
        markdown_content.append("## No Similar Questions Found")
        markdown_content.append("No similar questions available for this question.")

    return "\n".join(markdown_content)


# === Gradio App Logic ===


def auto_load_questions_on_path_change(directory_path: str):
    """Automatically load all questions when directory path changes."""
    if not directory_path.strip():
        # Clear everything if path is empty
        return (
            "",
            gr.update(choices=[], value=None, visible=False),
            "Enter a directory path to load questions.",
            0,
        )

    status, categories = load_all_questions_from_directory(directory_path)

    # Return status, category choices, default category selection, and initial display
    if categories:
        initial_display, initial_count = filter_questions_by_category("All")
        return (
            status,
            gr.update(choices=categories, value="All", visible=True),
            initial_display,
            initial_count,
        )
    else:
        return (status, gr.update(choices=[], value=None, visible=False), "No questions loaded.", 0)


def update_questions_display(selected_category: str):
    """Update questions display based on selected category."""
    if not selected_category:
        return "Please select a category.", 0

    display_content, count = filter_questions_by_category(selected_category)
    return display_content, count


def show_similar_questions(question_number: int, selected_category: str):
    """Show similar questions for the selected question number."""
    if not question_number:
        return "Please enter a question number."

    return get_similar_questions_display(question_number, selected_category or "All")


def save_jsonl(questions: List[Dict], filepath: str):
    """Save questions to a JSONL file."""
    with open(filepath, "w", encoding="utf-8") as f:
        for question in questions:
            f.write(json.dumps(question, ensure_ascii=False) + "\n")


def filter_questions_by_threshold(threshold: float, directory_path: str) -> str:
    """Filter questions by similarity threshold and save to files."""
    global ALL_QUESTIONS, QUESTION_SIMILARITIES, SIMILARITY_MATRIX

    if not ALL_QUESTIONS:
        return "❌ No questions loaded. Please load questions first."

    if SIMILARITY_MATRIX is None:
        return "❌ Similarity matrix not computed. Please load questions first."

    if not directory_path:
        return "❌ No directory path available. Please load questions from a directory first."

    # Sort questions by their maximum similarity score (highest first) - same as display order
    question_to_index = {}
    for i, q in enumerate(ALL_QUESTIONS):
        question_to_index[id(q)] = i

    questions_with_similarity = []
    for question in ALL_QUESTIONS:
        question_index = question_to_index.get(id(question), -1)
        max_sim_score = get_max_similarity_score(question_index)
        questions_with_similarity.append((question, question_index, max_sim_score))

    # Sort by similarity score (highest first) - prioritizing questions first in list
    questions_with_similarity.sort(key=lambda x: x[2], reverse=True)

    # Track which questions to keep and which to filter out
    questions_to_keep = []
    questions_filtered_out = []
    questions_to_remove = set()

    for question, question_index, max_sim_score in questions_with_similarity:
        question_id = id(question)

        # If this question is already marked for removal, skip it
        if question_id in questions_to_remove:
            questions_filtered_out.append(question)
            continue

        # Keep this question (it's the first/highest priority among similar ones)
        questions_to_keep.append(question)

        # Check if this question has similar questions above threshold
        if question_index in QUESTION_SIMILARITIES:
            for similar_q_idx, sim_score, similar_question in QUESTION_SIMILARITIES[question_index]:
                if sim_score >= threshold:
                    # Mark the similar question for removal
                    similar_question_id = id(similar_question)
                    if similar_question_id not in questions_to_remove:
                        questions_to_remove.add(similar_question_id)

    # Add any remaining questions that were marked for removal to filtered_out
    for question, question_index, max_sim_score in questions_with_similarity:
        if id(question) in questions_to_remove and question not in questions_filtered_out:
            questions_filtered_out.append(question)

    # Save filtered questions
    try:
        kept_filepath = os.path.join(directory_path, "filtered_threshold.jsonl")
        filtered_filepath = os.path.join(directory_path, "filtered_out_threshold.jsonl")

        save_jsonl(questions_to_keep, kept_filepath)
        save_jsonl(questions_filtered_out, filtered_filepath)

        result_message = f"""✅ Filtering completed successfully!

**Threshold:** {threshold:.3f}
**Original questions:** {len(ALL_QUESTIONS)}
**Questions kept:** {len(questions_to_keep)}
**Questions filtered out:** {len(questions_filtered_out)}

**Files saved:**
- `filtered_threshold.jsonl` ({len(questions_to_keep)} questions)
- `filtered_out_threshold.jsonl` ({len(questions_filtered_out)} questions)

**Location:** {directory_path}"""

        return result_message

    except Exception as e:
        return f"❌ Error saving files: {str(e)}"


def handle_filter_and_save(threshold: float, directory_path: str):
    """Handle filtering and return results with visibility update."""
    result = filter_questions_by_threshold(threshold, directory_path)
    return result, gr.update(visible=True)


# === Build Gradio UI ===

with gr.Blocks(title="Deploy Saved Questions") as demo:
    gr.Markdown("# Deploy Saved Questions")
    gr.Markdown(
        "Browse and deploy questions from JSONL files in any directory. Questions are sorted by similarity scores."
    )

    with gr.Row():
        # Directory path input
        directory_path = gr.Textbox(
            label="Directory Path",
            placeholder="/path/to/your/jsonl/files",
            info="Enter the full path to the directory containing JSONL files",
            scale=3,
        )

        # Load button
        load_btn = gr.Button("🔍 Load All Questions", variant="primary", size="sm")

    # Category selection (initially hidden)
    category_dropdown = gr.Dropdown(
        label="Select Category",
        choices=[],
        value=None,
        info="Choose a category to filter questions",
        visible=False,
    )

    # Status and stats
    with gr.Row():
        status_md = gr.Markdown("")
        question_count = gr.Number(label="Total Questions", value=0, interactive=False)

    # Threshold filtering controls
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## 🔧 Threshold Filtering")
            gr.Markdown(
                "Remove questions with similarity above threshold, prioritizing questions first in the list."
            )

        with gr.Column(scale=1):
            with gr.Row():
                threshold_input = gr.Slider(
                    label="Similarity Threshold",
                    minimum=0.1,
                    maximum=1.0,
                    value=0.8,
                    step=0.05,
                    info="Questions with similarity above this threshold will be filtered out",
                )
                filter_btn = gr.Button("🗂️ Filter & Save", variant="primary", size="sm")

    # Filtering results display
    filter_results = gr.Markdown(value="", visible=False, elem_classes=["filter-results"])

    # Two-panel layout
    with gr.Row():
        # Left panel: Questions list
        with gr.Column(scale=2):
            gr.Markdown("## Questions")
            questions_display = gr.Markdown(
                value="Enter a directory path and click 'Load All Questions' to display them here.",
                elem_classes=["questions-display"],
            )

        # Right panel: Similar questions
        with gr.Column(scale=1):
            gr.Markdown("## Similar Questions")

            # Controls for selecting question
            with gr.Row():
                question_number_input = gr.Number(
                    label="Question Number",
                    value=1,
                    minimum=1,
                    step=1,
                    info="Enter the question number to see similar questions",
                )
                show_similar_btn = gr.Button("Show Similar Questions", variant="secondary")

            # Similar questions display
            similar_questions_display = gr.Markdown(
                value="Select a question number and click 'Show Similar Questions' to see the most similar questions.",
                elem_classes=["similar-questions-display"],
            )

    # Custom CSS for dark mode display with two-panel layout
    gr.HTML("""
    <style>
    .questions-display, .similar-questions-display {
        max-height: 800px;
        overflow-y: auto;
        border: 1px solid #444;
        padding: 15px;
        border-radius: 5px;
        background-color: #2b2b2b;
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .questions-display h3, .similar-questions-display h2, .similar-questions-display h3 {
        color: #4CAF50;
        border-bottom: 2px solid #555;
        padding-bottom: 8px;
        margin-bottom: 15px;
        font-size: 1.2em;
    }
    
    .similar-questions-display h2 {
        color: #2196F3;
        font-size: 1.3em;
    }
    
    .questions-display strong, .similar-questions-display strong {
        color: #e0e0e0;
    }
    
    .questions-display hr, .similar-questions-display hr {
        border-color: #555;
        margin: 20px 0;
    }
    
    .questions-display em, .similar-questions-display em {
        color: #bbb;
        font-style: italic;
    }
    
    /* Similarity score highlighting */
    .similar-questions-display strong:contains("Similarity Score:") {
        color: #FF9800;
    }
    
    /* Question text styling */
    .questions-display > p:not(:has(strong)), .similar-questions-display > p:not(:has(strong)) {
        line-height: 1.6;
        margin-bottom: 15px;
    }
    
    /* Selected question highlighting */
    .similar-questions-display h2:first-of-type {
        background-color: #333;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #4CAF50;
    }
    </style>
    """)

    # Event handlers
    load_btn.click(
        auto_load_questions_on_path_change,
        inputs=[directory_path],
        outputs=[status_md, category_dropdown, questions_display, question_count],
    )

    category_dropdown.change(
        update_questions_display,
        inputs=[category_dropdown],
        outputs=[questions_display, question_count],
    )

    # Similar questions event handler
    show_similar_btn.click(
        show_similar_questions,
        inputs=[question_number_input, category_dropdown],
        outputs=[similar_questions_display],
    )

    # Filtering event handler
    filter_btn.click(
        handle_filter_and_save,
        inputs=[threshold_input, directory_path],
        outputs=[filter_results, filter_results],
    )

if __name__ == "__main__":
    demo.launch()
