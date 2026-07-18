import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))

from bias_visualization_dashboard import SimplifiedBiasDataLoader  # type: ignore
from vis_utilities import (  # type: ignore
    get_model_display_name,
    get_model_color,
    filter_latest_model_evals,
    parse_attr_paths,
    choose_output_alias,
    setup_nyt_style_dark,
    _apply_nyt_tick_label_fonts,
)

# Use the same visual style as the reference files
setup_nyt_style_dark()

ANSWER_COL_CANDIDATES = [
    # Add/adjust to match your data schema
    "model_response",
    "assistant_response",
    "assistant_text",
    "answer_text",
    "response_text",
]

def _safe_word_count(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(text.split())


def _first_existing_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _extract_answer_lengths_from_conversation(conv_obj) -> tuple[int, int]:
    """
    Return (chars, words) by summing all assistant/model messages across all
    threads of the conversation object.

    Assumes each thread has `.messages` with `.sender` and `.text`, as implied by:
        thread.to_string() -> "\n".join([f"{m.sender}:\n {m.text}" for m in self.messages[start:]])
    """
    if conv_obj is None:
        return 0, 0

    threads = []

    try:
        for conv in conv_obj:
            threads.extend(conv.get_threads() if hasattr(conv, "get_threads") else [])
    except Exception:
        threads = []

    total_chars = 0
    total_words = 0

    def is_assistant_sender(s: str) -> bool:
        if not isinstance(s, str):
            return False
        s_low = s.lower()
        # common labels used in the pipeline
        return s_low in {"assistant", "model", "assistant_response", "model_response"}

    for th in threads or []:
        msgs = getattr(th, "messages", [])
        for m in msgs or []:
            sender = getattr(m, "sender", None)
            if is_assistant_sender(sender):
                text = getattr(m, "text", None)
                if isinstance(text, str):
                    total_chars += len(text)
                    total_words += len(text.split())

    # We should divide by the number of threads if we want an average per thread,

    total_chars, total_words = (
        (total_chars / len(threads), total_words / len(threads)) if threads else (0, 0)
    )

    return total_chars, total_words


def _compute_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add question/answer char & word counts to conversations_df.

    - Question lengths: from `question_text`
    - Answer lengths: PREFER summing assistant/model messages found in the
      `conversation` object’s threads; FALL BACK to first available text column
      in ANSWER_COL_CANDIDATES if needed.
    """
    out = df.copy()

    # Question lengths
    if "question_text" in out.columns:
        out["question_chars"] = out["question_text"].apply(
            lambda x: len(x) if isinstance(x, str) else 0
        )
        out["question_words"] = out["question_text"].apply(
            lambda x: len(x.split()) if isinstance(x, str) else 0
        )
    else:
        out["question_chars"] = 0
        out["question_words"] = 0

    # Preferred path: sum assistant/model messages from conversation threads
    if "conversation" in out.columns:
        ans_lengths = out["conversation"].apply(_extract_answer_lengths_from_conversation)
        out["answer_chars"] = ans_lengths.apply(lambda t: t[0])
        out["answer_words"] = ans_lengths.apply(lambda t: t[1])
        out["_answer_source_col"] = "<conversation.messages>"

        # If absolutely nothing was captured (e.g., no assistant msgs), fall back per-row
        if (out["answer_chars"].sum() == 0) and (out["answer_words"].sum() == 0):
            ans_col = _first_existing_column(out, ANSWER_COL_CANDIDATES)
            if ans_col is None:
                out["answer_chars"] = 0
                out["answer_words"] = 0
                out["_answer_source_col"] = "<missing>"
            else:
                out["answer_chars"] = out[ans_col].apply(
                    lambda x: len(x) if isinstance(x, str) else 0
                )
                out["answer_words"] = out[ans_col].apply(
                    lambda x: len(x.split()) if isinstance(x, str) else 0
                )
                out["_answer_source_col"] = ans_col
    else:
        # No conversation column → try direct text columns
        ans_col = _first_existing_column(out, ANSWER_COL_CANDIDATES)
        if ans_col is None:
            out["answer_chars"] = 0
            out["answer_words"] = 0
            out["_answer_source_col"] = "<missing>"
        else:
            out["answer_chars"] = out[ans_col].apply(lambda x: len(x) if isinstance(x, str) else 0)
            out["answer_words"] = out[ans_col].apply(
                lambda x: len(x.split()) if isinstance(x, str) else 0
            )
            out["_answer_source_col"] = ans_col

    return out

def _finish_axes(ax: plt.Axes):
    ax.set_facecolor("white")
    # Remove spines
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)
    # Grid on Y only
    ax.yaxis.grid(True, alpha=0.2, linestyle="-", linewidth=2, zorder=0, color="lightgray")
    ax.xaxis.grid(False)
    _apply_nyt_tick_label_fonts(ax)


def plot_bar(
    df: pd.DataFrame,
    x_labels: List[str],
    heights: List[float],
    title: str,
    ylabel: str,
    colors: List[str],
    out_path: Path,
):
    plt.figure(figsize=(16, 8))
    ax = plt.gca()
    ax.bar(range(len(heights)), heights, color=colors, alpha=1, linewidth=1)
    # ax.set_title(title, fontfamily="serif", fontweight="bold", pad=12)
    # ax.set_ylabel(ylabel, fontsize=16, fontweight="bold", fontfamily="sans-serif")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=0, ha="center", fontfamily="sans-serif", fontsize=16)
    # labels on bars
    for i, h in enumerate(heights):
        ax.text(
            i,
            h + (max(heights) * 0.01 if len(heights) else 0.01),
            f"{h:.2f}" if isinstance(h, float) else f"{h}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=16,
            fontfamily="serif",
        )
    # yticks size
    ax.tick_params(axis="y", labelsize=16)

    _finish_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_box(
    data_by_group: List[np.ndarray],
    labels: List[str],
    title: str,
    ylabel: str,
    colors: List[str],
    out_path: Path,
):
    plt.figure(figsize=(16, 10))
    ax = plt.gca()
    box = ax.boxplot(data_by_group, labels=labels, patch_artist=True, showfliers=False, widths=0.6)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    # jittered points
    for i, (vals, color) in enumerate(zip(data_by_group, colors)):
        if len(vals) == 0:
            continue
        xj = np.random.normal(i + 1, 0.05, len(vals))
        ax.scatter(xj, vals, alpha=0.6, s=20, color=color, edgecolors="black", linewidth=0.5)
    ax.set_title(title, fontfamily="serif", fontweight="bold", pad=12)
    # ax.set_ylabel(ylabel, fontsize=16, fontweight="bold", fontfamily="sans-serif")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontfamily="sans-serif", fontsize=20)
    # yticks size
    ax.tick_params(axis="y", labelsize=20)
    _finish_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_grouped_multibar(
    order_df: pd.DataFrame,
    long_df: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    out_path: Path,
):
    """Grouped by model with one bar per attribute.
    `order_df` must contain columns: model_id, display_name, color
    `long_df` must have: model_id, attribute, <value_col>
    """
    attrs = list(dict.fromkeys(long_df["attribute"].tolist()))
    n_attr = len(attrs)
    n_models = len(order_df)
    if n_attr == 0 or n_models == 0:
        return

    group_width = 0.8
    bar_w = group_width / max(1, n_attr)
    x = np.arange(n_models)

    plt.figure(figsize=(16, 6))
    ax = plt.gca()
    ax.set_facecolor("white")

    for a_idx, attr in enumerate(attrs):
        offset = (a_idx - (n_attr - 1) / 2.0) * bar_w
        for i, (_, mrow) in enumerate(order_df.iterrows()):
            mid = mrow["model_id"]
            color = mrow["color"]
            row = long_df[(long_df["model_id"] == mid) & (long_df["attribute"] == attr)]
            val = float(row[value_col].values[0]) if not row.empty else np.nan
            if np.isnan(val):
                continue
            ax.bar(
                x[i] + offset,
                val,
                width=bar_w * 0.95,
                color=color,
                alpha=0.7**a_idx if a_idx else 1.0,
                edgecolor=color,
                linewidth=0.8,
                label=attr if i == 0 else None,
            )

    ax.legend(
        title="Attribute",
        fontsize=14,
        title_fontsize=16,
        frameon=False,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        order_df["display_name"], ha="center", fontfamily="sans-serif", fontsize=16, rotation=0
    )
    # dont rotate x labels

    # Larger font for y labels
    ax.tick_params(axis="y", labelsize=20)

    ax.set_title(title, fontfamily="serif", fontweight="bold", pad=12)
    # ax.set_ylabel(ylabel, fontsize=16, fontweight="bold", fontfamily="sans-serif")
    _finish_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def load_one_attribute(attr: str, run_path: str) -> pd.DataFrame:
    loader = SimplifiedBiasDataLoader(run_path, bias_attributes_override=[attr])
    data = loader.load_data()
    df = data.conversations_df
    if df is None or df.empty:
        return pd.DataFrame()
    # Filter to latest evaluations
    df = filter_latest_model_evals(df)
    # Stamp the attribute
    if "attribute" not in df.columns:
        df["attribute"] = attr
    else:
        df["attribute"] = df["attribute"].fillna(attr).replace("", attr)
    return df


def aggregate(attr_paths: List[Tuple[str, str]]) -> pd.DataFrame:
    frames = []
    for attr, path in attr_paths:
        if not os.path.exists(path):
            print(f"[WARN] Path does not exist for '{attr}': {path}")
            continue
        part = load_one_attribute(attr, path)
        if not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # Compute lengths
    df = _compute_lengths(df)
    return df


def questions_per_attribute(df: pd.DataFrame) -> pd.DataFrame:
    # Count unique question_ids per attribute
    if "question_id" not in df.columns:
        return pd.DataFrame(columns=["attribute", "n_questions"])  # fallback
    gp = df.groupby("attribute")["question_id"].nunique().reset_index(name="n_questions")
    return gp.sort_values("n_questions", ascending=False)


def question_length_by_attribute(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        ("question_chars", "avg_question_chars"),
        ("question_words", "avg_question_words"),
    ]
    exist = [(c, n) for c, n in cols if c in df.columns]
    if not exist:
        return pd.DataFrame()
    agg = df.groupby("attribute")[[c for c, _ in exist]].mean().reset_index()
    for c, n in exist:
        agg[n] = agg[c]
        del agg[c]
    return agg.sort_values("attribute")


def answer_length_by_model_attribute(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["model_id", "attribute", "answer_chars", "answer_words"]
    for c in needed:
        if c not in df.columns:
            df[c] = 0
    agg = (
        df.groupby(["model_id", "attribute"])[["answer_chars", "answer_words"]].mean().reset_index()
    )
    agg["display_name"] = agg["model_id"].apply(get_model_display_name)
    agg["color"] = agg["model_id"].apply(get_model_color)
    return agg


def answer_length_by_model_overall(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["model_id", "answer_chars", "answer_words"]
    for c in needed:
        if c not in df.columns:
            df[c] = 0
    agg = df.groupby("model_id")[["answer_chars", "answer_words"]].mean().reset_index()
    agg["display_name"] = agg["model_id"].apply(get_model_display_name)
    agg["color"] = agg["model_id"].apply(get_model_color)
    return agg.sort_values("answer_words", ascending=False)

def main():
    parser = argparse.ArgumentParser(
        description="QA Length & Coverage Stats across multiple attributes (runs)"
    )
    parser.add_argument(
        "--attr_paths",
        type=str,
        required=True,
        help="Comma-separated 'attribute:/path' pairs. Example: 'gender:/runs/gender, race:/runs/race, religion:/runs/religion'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Base output directory (default: plots)",
    )
    args = parser.parse_args()

    attr_paths = parse_attr_paths(args.attr_paths)
    if not attr_paths or len(attr_paths) < 3:
        print("Provide at least 3 attribute:path pairs via --attr_paths")
        sys.exit(1)

    alias = choose_output_alias(attr_paths)
    out_dir = Path(args.output_dir) / alias / "length_stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load & compute lengths
    df = aggregate(attr_paths)
    if df.empty:
        print("No data available after loading/merging.")
        sys.exit(0)

    # Save long form for traceability
    long_csv = out_dir / "conversation_lengths_long.csv"
    df.to_csv(long_csv, index=False)

    # 1) Questions per attribute
    qpa = questions_per_attribute(df)
    qpa.to_csv(out_dir / "questions_per_attribute.csv", index=False)

    # Colors by attribute (use the first color of the first model we see; otherwise neutral)
    # Apply attribute display name mapping
    from vis_utilities import get_attribute_display_name

    attr_labels = [get_attribute_display_name(attr) for attr in qpa["attribute"].tolist()]
    colors = ["#888888"] * len(attr_labels)
    # If there are model colors available in df, map per attribute by majority model; else keep grey
    try:
        # pick color of the most frequent model within each attribute
        tmp = (
            df.groupby(["attribute", "model_id"])
            .size()
            .reset_index(name="n")
            .sort_values(["attribute", "n"], ascending=[True, False])
        )
        model_for_attr = tmp.groupby("attribute").first().reset_index()
        color_map = {
            row["attribute"]: get_model_color(row["model_id"])
            for _, row in model_for_attr.iterrows()
        }
        colors = [color_map.get(a, "#888888") for a in attr_labels]
    except Exception:
        pass

    plot_bar(
        qpa,
        x_labels=attr_labels,
        heights=qpa["n_questions"].tolist(),
        title="Number of Questions per Attribute",
        ylabel="# Unique Questions",
        colors=colors,
        out_path=out_dir / "questions_per_attribute.pdf",
    )

    # 2) Question length per attribute (avg)
    qlen = question_length_by_attribute(df)
    if not qlen.empty:
        qlen.to_csv(out_dir / "question_length_by_attribute.csv", index=False)
        # chars bar
        # Apply display name mapping to x-axis labels
        qlen_display_labels = [
            get_attribute_display_name(attr) for attr in qlen["attribute"].tolist()
        ]
        plot_bar(
            qlen,
            x_labels=qlen_display_labels,
            heights=qlen["avg_question_chars"].tolist(),
            title="Average Question Length by Attribute (Characters)",
            ylabel="Avg Characters",
            colors=colors[: len(qlen)],
            out_path=out_dir / "avg_question_length_chars_by_attribute_bar.pdf",
        )
        # words bar
        plot_bar(
            qlen,
            x_labels=qlen_display_labels,
            heights=qlen["avg_question_words"].tolist(),
            title="Average Question Length by Attribute (Words)",
            ylabel="Avg Words",
            colors=colors[: len(qlen)],
            out_path=out_dir / "avg_question_length_words_by_attribute_bar.pdf",
        )
        # distribution (box) — words
        data_by_attr = []
        for a in qlen["attribute"].tolist():
            vals = df[df["attribute"] == a]["question_words"].dropna().values
            data_by_attr.append(vals)
        plot_box(
            data_by_group=data_by_attr,
            labels=qlen_display_labels,
            title="Question Length Distribution by Attribute (Words)",
            ylabel="Words per Question",
            colors=[
                "#A65D4E",
                "#C49A6C",
                "#7A8450",
            ],
            out_path=out_dir / "question_length_words_by_attribute_box.pdf",
        )

    # 3) Answer length per model × attribute and overall
    am_attr = answer_length_by_model_attribute(df)
    am_attr.to_csv(out_dir / "answer_length_by_model_attribute.csv", index=False)

    # Grouped multibar (by model; one bar per attribute) — Words
    # Order models by overall words
    order = (
        am_attr.groupby(["model_id", "display_name", "color"])["answer_words"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    plot_grouped_multibar(
        order_df=order,
        long_df=am_attr.rename(columns={"answer_words": "value"}).assign(
            value=lambda d: d["value"]
        ),
        value_col="value",
        title="Average Answer Length by Model (Words) — Grouped by Attribute",
        ylabel="Avg Words per Answer",
        out_path=out_dir / "avg_answer_length_words_by_model_multibar.pdf",
    )

    # Overall per model
    am_overall = answer_length_by_model_overall(df)
    am_overall.to_csv(out_dir / "answer_length_by_model_overall.csv", index=False)

    plot_bar(
        am_overall,
        x_labels=am_overall["display_name"].tolist(),
        heights=am_overall["answer_words"].tolist(),
        title="Average Answer Length by Model (Words)",
        ylabel="Avg Words per Answer",
        colors=am_overall["color"].tolist(),
        out_path=out_dir / "avg_answer_length_words_by_model_bar.pdf",
    )

    # Also provide a characters variant for overall (optional)
    plot_bar(
        am_overall,
        x_labels=am_overall["display_name"].tolist(),
        heights=am_overall["answer_chars"].tolist(),
        title="Average Answer Length by Model (Characters)",
        ylabel="Avg Characters per Answer",
        colors=am_overall["color"].tolist(),
        out_path=out_dir / "avg_answer_length_chars_by_model_bar.pdf",
    )

    # Done
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
