# human_scoring.py
import os
import json
import hashlib
from typing import Dict, List, Tuple, Optional

import gradio as gr
import pandas as pd

from visualization.bias_visualization_dashboard import (
    SimplifiedBiasDataLoader,
    Question,
)
from src.utils.embeddings import EmbeddingManager


# === Utilities ===


def hash_string(s: str, length: int = 8) -> str:
    """
    Best-effort replica of the project's hash_string used in Question.get_id().
    If your project uses a different scheme, replace this implementation to match it.
    """
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:length]


def question_to_dict_with_id(q: Question) -> Dict:
    d = q.to_json()
    # Ensure id is included for easy de-duping on disk
    try:
        qid = q.get_id()
    except Exception:
        # Fallback identical to Question.get_id() in the snippet: id = hash(example or "", 8)
        qid = hash_string(q.example or "", 8)
    d["id"] = qid
    # Optionally preserve original model if present
    if getattr(q, "orig_model", None) is not None:
        d["orig_model"] = q.orig_model
    return d


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sanitize_path_component(p: str) -> str:
    """
    Turn an absolute/relative run_path into a safe, nested output path:
    human_scored/<sanitized_run_path>/
    """
    # Normalize, then drop drive/leading separators; keep structure under human_scored
    p = os.path.normpath(p)
    parts = []
    for chunk in p.split(os.sep):
        if chunk in ("", ".", ".."):
            continue
        # Replace problematic characters
        cleaned = "".join(c if c.isalnum() or c in "-._" else "_" for c in chunk)
        parts.append(cleaned)
    return os.path.join(*parts) if parts else "root"


def load_jsonl_as_list(path: str) -> List[Dict]:
    data = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except Exception:
                    pass
    return data


def append_jsonl(path: str, record: Dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def existing_labeled_ids(output_dir: str) -> set:
    ids = set()
    for fname in [
        "very_good.jsonl",
        "good.jsonl",
        "too_specific.jsonl",
        "wrong_grammar.jsonl",
        "last_sentence_issue.jsonl",
        "repetition.jsonl",
        "discuss.jsonl",
        "multiple_choice.jsonl",
        "other.jsonl",
    ]:
        for rec in load_jsonl_as_list(os.path.join(output_dir, fname)):
            qid = rec.get("id")
            if not qid:
                # Try reconstruct from example if necessary
                ex = rec.get("example", "")
                qid = hash_string(ex or "", 8)
            ids.add(qid)
    return ids


def deduplicate_questions_by_similarity(
    questions: List[Question],
    run_path: str,
    similarity_threshold: float = 0.95,
    similarity_method: str = "embedding",
    output_dir: Optional[str] = None,
) -> Tuple[List[Question], Dict]:
    """
    Deduplicate questions based on similarity scores.

    Args:
        questions: List of Question objects
        run_path: Path to the run directory (for loading bias data to get fitness scores)
        similarity_threshold: Threshold above which questions are considered similar
        similarity_method: "embedding" or "fuzzy"
        output_dir: Directory to save deduplication.json (defaults to run_path but should typically be the human_scored path where other JSONLs are stored)

    Returns:
        Tuple of (deduplicated_questions, deduplication_info)
    """
    if not questions:
        return questions, {}

    if output_dir is None:
        output_dir = run_path

    # Load bias data to get fitness scores
    try:
        loader = SimplifiedBiasDataLoader(run_path)
        data = loader.load_data()
        conversations_df = data.conversations_df
    except Exception as e:
        print(f"Warning: Could not load fitness scores from {run_path}: {e}")
        print("Proceeding with random order...")
        conversations_df = pd.DataFrame()

    # Create a mapping from question_id to fitness score
    fitness_scores = {}
    if not conversations_df.empty:
        for _, row in conversations_df.iterrows():
            question_id = row.get("question_id")
            fitness_score = row.get("fitness_score", 0.0)
            if question_id and pd.notna(fitness_score):
                if question_id not in fitness_scores:
                    fitness_scores[question_id] = []
                fitness_scores[question_id].append(fitness_score)

    # Average fitness scores for each question
    avg_fitness_scores = {}
    for qid, scores in fitness_scores.items():
        avg_fitness_scores[qid] = sum(scores) / len(scores) if scores else 0.0

    # Assign fitness scores to questions and sort by fitness (high to low)
    questions_with_fitness = []
    for q in questions:
        qid = q.get_id()
        fitness = avg_fitness_scores.get(qid, 0.0)  # Default to 0 if no fitness score
        questions_with_fitness.append((q, fitness))

    # Sort by fitness score (highest first)
    questions_with_fitness.sort(key=lambda x: x[1], reverse=True)

    # Prepare data for similarity computation
    question_texts = [q.example or "" for q, _ in questions_with_fitness]
    question_ids = [q.get_id() for q, _ in questions_with_fitness]

    # Initialize embedding manager
    embedding_manager = EmbeddingManager()

    # Compute similarity matrix
    try:
        similarity_matrix, _ = embedding_manager.compute_similarity_matrix(
            question_texts, method=similarity_method
        )
    except Exception as e:
        print(f"Error computing similarity with {similarity_method}: {e}")
        if similarity_method == "embedding":
            print("Falling back to fuzzy similarity...")
            similarity_matrix, _ = embedding_manager.compute_similarity_matrix(
                question_texts, method="fuzzy"
            )
            similarity_method = "fuzzy"  # Update for logging
        else:
            raise

    # Track deduplication info
    dedup_info = {
        "method": similarity_method,
        "threshold": similarity_threshold,
        "original_count": len(questions),
        "removed_questions": [],
        "similar_groups": [],
    }

    # Keep track of which questions to remove
    questions_to_remove = set()

    # Process questions from highest to lowest fitness
    for i, (question_i, fitness_i) in enumerate(questions_with_fitness):
        if question_ids[i] in questions_to_remove:
            continue  # Already marked for removal

        max_similarity = max(similarity_matrix[i][j] for j in range(len(questions)) if j != i)
        print(
            f"Processing question {question_ids[i]} (fitness={fitness_i}, max_sim={max_similarity:.3f})"
        )
        # Find similar questions with lower fitness
        similar_questions = []
        for j, (question_j, fitness_j) in enumerate(questions_with_fitness[i + 1 :], start=i + 1):
            if question_ids[j] in questions_to_remove:
                continue  # Already marked for removal

            # Check similarity
            if similarity_method == "fuzzy":
                similarity_score = similarity_matrix[i][j] / 100.0  # Normalize fuzzy scores
            else:
                similarity_score = similarity_matrix[i][j]

            if similarity_score >= similarity_threshold:
                similar_questions.append(
                    {
                        "question_id": question_ids[j],
                        "question_text": question_j.example or "",
                        "fitness_score": fitness_j,
                        "similarity_score": float(similarity_score),
                    }
                )
                print(
                    f"Marking question {question_ids[j]} as similar to {question_ids[i]} (sim={similarity_score:.3f})"
                )
                # Print the question texts for clarity
                print(f"  '{question_i.example}'")
                print(f"  '{question_j.example}'")

                questions_to_remove.add(question_ids[j])

        # Record similarity group if any similar questions were found
        if similar_questions:
            group_info = {
                "kept_question": {
                    "question_id": question_ids[i],
                    "question_text": question_i.example or "",
                    "fitness_score": fitness_i,
                },
                "removed_questions": similar_questions,
            }
            dedup_info["similar_groups"].append(group_info)
            dedup_info["removed_questions"].extend([q["question_id"] for q in similar_questions])

    # Filter out removed questions
    deduplicated_questions = [
        q for q, _ in questions_with_fitness if q.get_id() not in questions_to_remove
    ]

    dedup_info["final_count"] = len(deduplicated_questions)
    dedup_info["removed_count"] = len(questions_to_remove)

    # Save deduplication info to file
    dedup_file = os.path.join(output_dir, "deduplication.json")
    try:
        with open(dedup_file, "w") as f:
            json.dump(dedup_info, f, indent=2)
        print(f"Deduplication info saved to: {dedup_file}")
    except Exception as e:
        print(f"Warning: Could not save deduplication info to {dedup_file}: {e}")

    print("Deduplication complete:")
    print(f"  - Original: {dedup_info['original_count']} questions")
    print(f"  - Removed: {dedup_info['removed_count']} questions")
    print(f"  - Final: {dedup_info['final_count']} questions")
    print(f"  - Method: {similarity_method}, Threshold: {similarity_threshold}")

    return deduplicated_questions, dedup_info


# === Data loading ===


def build_question_list(
    run_path: str,
    bias_attributes_override: Optional[List[str]] = None,
    enable_deduplication: bool = False,
    similarity_threshold: float = 0.95,
    similarity_method: str = "embedding",
    output_dir: Optional[str] = None,
) -> List[Question]:
    loader = SimplifiedBiasDataLoader(run_path, bias_attributes_override=bias_attributes_override)
    data = loader.load_data()

    # questions_df includes saved + non-saved + possibly duplicates across iterations
    qdf: pd.DataFrame = data.questions_df.copy()

    # Filter to saved questions only
    qdf = qdf[qdf["is_saved"]]

    # Build Question objects from dataframe rows (dedupe by id)
    by_id: Dict[str, Question] = {}
    for _, row in qdf.iterrows():
        q = Question(
            superdomain=row.get("superdomain", "") or "",
            domain=row.get("domain", "") or "",
            topic=row.get("topic", "") or "",
            example=row.get("question_text", None),
            score=None,
        )
        # Preserve original model if column exists
        if "original_model" in row and pd.notna(row["original_model"]):
            q.orig_model = row["original_model"]

        qid = q.get_id()
        # Prefer the first occurrence (or override—either is fine; we just need one per id)
        if qid not in by_id:
            by_id[qid] = q

    # Keep stable order (e.g., by text) for deterministic traversal
    ordered = sorted(
        by_id.values(), key=lambda qq: (qq.superdomain, qq.domain, qq.topic, qq.example or "")
    )

    # Apply deduplication if enabled
    if enable_deduplication:
        print(
            f"Applying deduplication with threshold {similarity_threshold} using {similarity_method} method..."
        )
        deduplicated, dedup_info = deduplicate_questions_by_similarity(
            ordered, run_path, similarity_threshold, similarity_method, output_dir
        )
        return deduplicated

    return ordered


# === Gradio App Logic ===


def start_session(
    run_path: str,
    bias_attributes_csv: str,
    enable_deduplication: bool = False,
    similarity_threshold: float = 0.95,
    similarity_method: str = "embedding",
):
    if not run_path or not os.path.exists(run_path):
        # return EXACTLY 10 outputs, with strings for Markdown fields
        # outputs: [status(MD), session_questions(State), idx(State), output_dir(State),
        #           example_tb(Textbox), meta_md(MD), progress(MD), superdomain_tb, domain_tb, topic_tb]
        return (
            gr.update(visible=True, value="❌ Invalid run_path. Please provide a valid directory."),
            [],  # session_questions
            0,  # idx
            "",  # output_dir_state
            "",  # example_tb
            "",  # meta_md
            "0 / 0",  # progress  <-- must be a string
            "",
            "",
            "",  # superdomain, domain, topic
        )

    bias_override = None
    if bias_attributes_csv.strip():
        bias_override = [x.strip() for x in bias_attributes_csv.split(",") if x.strip()]

    root = "human_scored"
    nested = sanitize_path_component(run_path)
    output_dir = os.path.join(root, nested)
    ensure_dir(output_dir)

    questions = build_question_list(
        run_path,
        bias_override,
        enable_deduplication=enable_deduplication,
        similarity_threshold=similarity_threshold,
        similarity_method=similarity_method,
        output_dir=output_dir,
    )

    labeled = existing_labeled_ids(output_dir)
    remaining = [q for q in questions if q.get_id() not in labeled]

    if not remaining:
        return (
            gr.update(visible=True, value="🎉 All questions in this run are already labeled."),
            [],  # session_questions
            0,  # idx
            output_dir,  # output_dir_state
            "",  # example_tb
            "",  # meta_md
            "0 / 0",  # progress  <-- string
            "",
            "",
            "",  # superdomain, domain, topic
        )

    q0 = remaining[0]
    meta = f"ID: {q0.get_id()} | Superdomain: {q0.superdomain} | Domain: {q0.domain} | Topic: {q0.topic} | Orig model: {getattr(q0, 'orig_model', None) or 'n/a'}"

    return (
        gr.update(
            visible=True,
            value=f"✅ Loaded {len(remaining)} unlabeled questions (out of {len(questions)} total).",
        ),
        [question_to_dict_with_id(q) for q in remaining],
        0,
        output_dir,
        q0.example or "",
        meta,
        f"1 / {len(remaining)}",  # <-- string, not int
        q0.superdomain,
        q0.domain,
        q0.topic,
    )


def render_current(session_questions: List[Dict], idx: int):
    if not session_questions:
        return "No questions loaded.", "", "", "", "", ""

    idx = max(0, min(idx, len(session_questions) - 1))
    qd = session_questions[idx]
    meta = f"ID: {qd['id']} | Superdomain: {qd.get('superdomain', '')} | Domain: {qd.get('domain', '')} | Topic: {qd.get('topic', '')} | Orig model: {qd.get('orig_model', 'n/a')}"
    example = qd.get("example", "") or ""
    return (
        meta,
        example,
        qd.get("superdomain", ""),
        qd.get("domain", ""),
        qd.get("topic", ""),
        f"{idx + 1} / {len(session_questions)}",
    )


def save_label(
    label: str,
    session_questions: List[Dict],
    idx: int,
    output_dir: str,
    edited_example: str,
    superdomain: str,
    domain: str,
    topic: str,
):
    """
    Save the current question into the appropriate JSONL (with potential edit).
    If example was edited, also write an entry into edited_mapping.jsonl (old_id -> new_id).
    Then advance to the next question.
    """
    assert label in (
        "very_good",
        "good",
        "too_specific",
        "wrong_grammar",
        "last_sentence_issue",
        "repetition",
        "discuss",
        "multiple_choice",
        "other",
    ), "Invalid label"

    if not session_questions:
        return "No questions loaded.", idx, "", "", "", "", ""

    qd = session_questions[idx]
    old_id = qd["id"]
    old_example = qd.get("example", "") or ""
    orig_model = qd.get("orig_model", None)

    # If the user changed the example or the meta fields, we create a new 'edited' Question to compute the new id
    has_edit = (
        (edited_example or "") != (old_example or "")
        or superdomain != qd.get("superdomain", "")
        or domain != qd.get("domain", "")
        or topic != qd.get("topic", "")
    )

    if has_edit:
        # Build an edited Question (keeping score=None)
        edited_q = Question(
            superdomain=superdomain,
            domain=domain,
            topic=topic,
            example=edited_example or "",
            score=None,
        )
        if orig_model is not None:
            edited_q.orig_model = orig_model

        new_id = edited_q.get_id()
        out_record = question_to_dict_with_id(edited_q)

        # Write mapping
        mapping_path = os.path.join(output_dir, "edited_mapping.jsonl")
        append_jsonl(
            mapping_path,
            {
                "old_id": old_id,
                "new_id": new_id,
                "old_example": old_example,
                "new_example": edited_example or "",
                "old_superdomain": qd.get("superdomain", ""),
                "old_domain": qd.get("domain", ""),
                "old_topic": qd.get("topic", ""),
                "new_superdomain": superdomain,
                "new_domain": domain,
                "new_topic": topic,
                "orig_model": orig_model,
            },
        )
    else:
        # Use original
        out_record = dict(qd)

    # Write labeled record
    path = os.path.join(output_dir, f"{label}.jsonl")
    append_jsonl(path, out_record)

    # Advance index
    next_idx = idx + 1
    if next_idx >= len(session_questions):
        # End of session
        return (
            f"✅ Saved to {label}. No more unlabeled questions.",
            len(session_questions) - 1,  # keep idx in range
            *render_current(session_questions, len(session_questions) - 1),
        )

    # Move to next
    status = f"✅ Saved to {label}. Moving to next."
    return (status, next_idx, *render_current(session_questions, next_idx))


def go_prev(session_questions: List[Dict], idx: int):
    if not session_questions:
        return idx, *render_current(session_questions, 0)
    new_idx = max(0, idx - 1)
    return new_idx, *render_current(session_questions, new_idx)


def skip(session_questions: List[Dict], idx: int):
    if not session_questions:
        return idx, *render_current(session_questions, 0)
    new_idx = min(len(session_questions) - 1, idx + 1)
    return new_idx, *render_current(session_questions, new_idx)


# === Build Gradio UI ===

with gr.Blocks(title="Human Scoring for Questions") as demo:
    gr.Markdown("# Human Scoring for Questions")
    with gr.Row():
        run_path = gr.Textbox(label="Run path", placeholder="/path/to/run_dir", scale=3)
        bias_attrs = gr.Textbox(
            label="(Optional) bias_attributes override (comma-separated)",
            placeholder="e.g., gender, race",
            scale=2,
        )

    # Deduplication controls
    with gr.Row():
        enable_dedup = gr.Checkbox(
            label="Enable Deduplication",
            value=False,
            info="Remove similar questions based on similarity threshold",
        )
        similarity_threshold = gr.Slider(
            label="Similarity Threshold",
            minimum=0.5,
            maximum=1.0,
            value=0.95,
            step=0.05,
            info="Questions above this similarity will be considered duplicates",
        )
        similarity_method = gr.Dropdown(
            label="Similarity Method",
            choices=["embedding", "fuzzy"],
            value="embedding",
            info="Method to compute text similarity",
        )

    load_btn = gr.Button("Load Questions", variant="primary")

    status = gr.Markdown(visible=False)

    # Session state
    session_questions = gr.State([])  # list of dicts (question with id)
    idx = gr.State(0)
    output_dir_state = gr.State("")

    # Current question view
    with gr.Group():
        progress = gr.Markdown("")  # "x / N"

        meta_md = gr.Markdown(label="Meta")
        superdomain_tb = gr.Textbox(label="Superdomain")
        domain_tb = gr.Textbox(label="Domain")
        topic_tb = gr.Textbox(label="Topic")

        example_tb = gr.Textbox(
            label="Example (editable)", lines=8, placeholder="Question example text here"
        )

        with gr.Row():
            prev_btn = gr.Button("← Previous")
            skip_btn = gr.Button("Skip →")

        with gr.Row():
            very_good_btn = gr.Button("⭐ Very Good", variant="primary")
            good_btn = gr.Button("✅ Good", variant="secondary")
            too_specific_btn = gr.Button("🎯 Too Specific")

        with gr.Row():
            wrong_grammar_btn = gr.Button("📝 Wrong Grammar")
            last_sentence_btn = gr.Button("🔚 Last Sentence Issue")
            repetition_btn = gr.Button("🔄 Repetition")

        with gr.Row():
            discuss_btn = gr.Button("💬 Discuss")
            multiple_choice_btn = gr.Button("🅰️🅱️🅾️ Multiple Choice")
            other_btn = gr.Button("❓ Other")

    # Wire up
    load_btn.click(
        start_session,
        inputs=[run_path, bias_attrs, enable_dedup, similarity_threshold, similarity_method],
        outputs=[
            status,
            session_questions,
            idx,
            output_dir_state,
            example_tb,
            meta_md,
            progress,
            superdomain_tb,
            domain_tb,
            topic_tb,
        ],
    )

    # Navigation
    prev_btn.click(
        go_prev,
        inputs=[session_questions, idx],
        outputs=[idx, meta_md, example_tb, superdomain_tb, domain_tb, topic_tb, progress],
    )
    skip_btn.click(
        skip,
        inputs=[session_questions, idx],
        outputs=[idx, meta_md, example_tb, superdomain_tb, domain_tb, topic_tb, progress],
    )

    # Label actions
    very_good_btn.click(
        save_label,
        inputs=[
            gr.State("very_good"),
            session_questions,
            idx,
            output_dir_state,
            example_tb,
            superdomain_tb,
            domain_tb,
            topic_tb,
        ],
        outputs=[status, idx, meta_md, example_tb, superdomain_tb, domain_tb, topic_tb, progress],
    )
    good_btn.click(
        save_label,
        inputs=[
            gr.State("good"),
            session_questions,
            idx,
            output_dir_state,
            example_tb,
            superdomain_tb,
            domain_tb,
            topic_tb,
        ],
        outputs=[status, idx, meta_md, example_tb, superdomain_tb, domain_tb, topic_tb, progress],
    )
    too_specific_btn.click(
        save_label,
        inputs=[
            gr.State("too_specific"),
            session_questions,
            idx,
            output_dir_state,
            example_tb,
            superdomain_tb,
            domain_tb,
            topic_tb,
        ],
        outputs=[status, idx, meta_md, example_tb, superdomain_tb, domain_tb, topic_tb, progress],
    )
    wrong_grammar_btn.click(
        save_label,
        inputs=[
            gr.State("wrong_grammar"),
            session_questions,
            idx,
            output_dir_state,
            example_tb,
            superdomain_tb,
            domain_tb,
            topic_tb,
        ],
        outputs=[status, idx, meta_md, example_tb, superdomain_tb, domain_tb, topic_tb, progress],
    )
    last_sentence_btn.click(
        save_label,
        inputs=[
            gr.State("last_sentence_issue"),
            session_questions,
            idx,
            output_dir_state,
            example_tb,
            superdomain_tb,
            domain_tb,
            topic_tb,
        ],
        outputs=[status, idx, meta_md, example_tb, superdomain_tb, domain_tb, topic_tb, progress],
    )
    repetition_btn.click(
        save_label,
        inputs=[
            gr.State("repetition"),
            session_questions,
            idx,
            output_dir_state,
            example_tb,
            superdomain_tb,
            domain_tb,
            topic_tb,
        ],
        outputs=[status, idx, meta_md, example_tb, superdomain_tb, domain_tb, topic_tb, progress],
    )
    discuss_btn.click(
        save_label,
        inputs=[
            gr.State("discuss"),
            session_questions,
            idx,
            output_dir_state,
            example_tb,
            superdomain_tb,
            domain_tb,
            topic_tb,
        ],
        outputs=[status, idx, meta_md, example_tb, superdomain_tb, domain_tb, topic_tb, progress],
    )
    multiple_choice_btn.click(
        save_label,
        inputs=[
            gr.State("multiple_choice"),
            session_questions,
            idx,
            output_dir_state,
            example_tb,
            superdomain_tb,
            domain_tb,
            topic_tb,
        ],
        outputs=[status, idx, meta_md, example_tb, superdomain_tb, domain_tb, topic_tb, progress],
    )
    other_btn.click(
        save_label,
        inputs=[
            gr.State("other"),
            session_questions,
            idx,
            output_dir_state,
            example_tb,
            superdomain_tb,
            domain_tb,
            topic_tb,
        ],
        outputs=[status, idx, meta_md, example_tb, superdomain_tb, domain_tb, topic_tb, progress],
    )


if __name__ == "__main__":
    demo.launch()
