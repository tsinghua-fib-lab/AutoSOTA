"""Core component."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional


def summarize_hc_soinn_classifier(hc_soinn) -> Optional[Dict[str, Any]]:
    """Handle summarize hc soinn classifier."""
    if hc_soinn is None:
        return None
    if not hasattr(hc_soinn, "class_mu") or not hc_soinn.class_mu:
        return None

    classes: List[int] = sorted(hc_soinn.class_mu.keys())
    node_counts: List[int] = []
    for cls in classes:
        clusters = hc_soinn.class_clusters.get(cls, [])
        node_counts.append(len(clusters))

    total_classes = len(classes)
    if total_classes == 0:
        return None

    total_nodes = sum(node_counts)
    avg_nodes_per_class_with_data = total_nodes / len(node_counts)
    avg_nodes_per_class = (
        total_nodes / total_classes if total_classes > 0 else avg_nodes_per_class_with_data
    )

    return {
        "total_classes": total_classes,
        "classes_with_nodes": len(node_counts),
        "total_nodes": total_nodes,
        "avg_nodes_per_class_with_data": avg_nodes_per_class_with_data,
        "avg_nodes_per_class": avg_nodes_per_class,
        "min_nodes": min(node_counts),
        "max_nodes": max(node_counts),
        "node_counts": node_counts,
    }


def log_hc_soinn_dataset_end_summary(learner) -> None:
    """Handle log hc soinn dataset end summary."""
    if not getattr(learner, "use_hc_soinn", False):
        return
    dm = getattr(learner, "data_manager", None)
    if dm is None:
        return
    if learner._cur_task != dm.nb_tasks - 1:
        return

    hc = getattr(learner, "hc_soinn", None)
    stats = summarize_hc_soinn_classifier(hc)
    if not stats:
        logging.warning("[HC-SOINN Node Summary] No classifier state to summarize.")
        return

    args = getattr(learner, "args", {}) or {}
    dataset = args.get("dataset", "")

    logging.info("=" * 60)
    logging.info(
        "[HC-SOINN Node Summary | dataset end] dataset=%s tasks=%s (last task index=%s)",
        dataset,
        dm.nb_tasks,
        learner._cur_task,
    )
    logging.info("  total_classes: %s", stats["total_classes"])
    logging.info("  classes_with_nodes: %s", stats["classes_with_nodes"])
    logging.info("  total_nodes: %s", stats["total_nodes"])
    logging.info(
        "  avg_nodes_per_class (over all classes): %.4f",
        stats["avg_nodes_per_class"],
    )
    logging.info(
        "  avg_nodes_per_class (over listed classes): %.4f",
        stats["avg_nodes_per_class_with_data"],
    )
    logging.info("  min_nodes / max_nodes per class: %s / %s", stats["min_nodes"], stats["max_nodes"])
    logging.info("  node_counts per class (sorted by class id): %s", stats["node_counts"])
    logging.info("=" * 60)
