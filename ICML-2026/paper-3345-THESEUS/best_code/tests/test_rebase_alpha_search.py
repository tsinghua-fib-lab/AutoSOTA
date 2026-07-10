from __future__ import annotations

from merge_and_rebase.alpha_search import PerTaskAlphaTracker, average_scores



def test_average_scores_returns_mean() -> None:
    assert average_scores([1.0, 2.0, 5.0]) == 8.0 / 3.0
    assert average_scores([]) == 0.0



def test_per_task_alpha_tracker_updates_and_early_stops() -> None:
    tracker = PerTaskAlphaTracker(task_names=['Cars', 'DTD'], initial_alpha=0.0)

    stopped = tracker.update(alpha=0.0, indices=[0, 1], baseline_accs=[0.10, 0.20], rebase_accs=[0.15, 0.25])
    assert stopped == []
    assert tracker.active_indices() == [0, 1]
    assert tracker.best_alpha == [0.0, 0.0]

    stopped = tracker.update(alpha=0.1, indices=[0, 1], baseline_accs=[0.10, 0.20], rebase_accs=[0.18, 0.25])
    assert stopped == []
    assert tracker.active_indices() == [0, 1]
    assert tracker.best_alpha == [0.1, 0.0]
    assert tracker.best_rebase_acc == [0.18, 0.25]

    stopped = tracker.update(alpha=0.2, indices=[0, 1], baseline_accs=[0.10, 0.20], rebase_accs=[0.17, 0.24])
    assert stopped == [0, 1]
    assert tracker.active_indices() == []
    assert tracker.best_alpha == [0.1, 0.0]
    assert tracker.best_rebase_acc == [0.18, 0.25]


def test_per_task_alpha_tracker_respects_patience() -> None:
    tracker = PerTaskAlphaTracker(task_names=['Cars'], initial_alpha=0.0, patience=1)

    stopped = tracker.update(alpha=0.0, indices=[0], baseline_accs=[0.10], rebase_accs=[0.20])
    assert stopped == []
    assert tracker.active_indices() == [0]

    stopped = tracker.update(alpha=0.1, indices=[0], baseline_accs=[0.10], rebase_accs=[0.19])
    assert stopped == []
    assert tracker.active_indices() == [0]

    stopped = tracker.update(alpha=0.2, indices=[0], baseline_accs=[0.10], rebase_accs=[0.18])
    assert stopped == [0]
    assert tracker.active_indices() == []


def test_per_task_alpha_tracker_resets_bad_steps_on_tie_or_improvement() -> None:
    tracker = PerTaskAlphaTracker(task_names=['Cars'], initial_alpha=0.0, patience=1)

    tracker.update(alpha=0.0, indices=[0], baseline_accs=[0.10], rebase_accs=[0.20])
    stopped = tracker.update(alpha=0.1, indices=[0], baseline_accs=[0.10], rebase_accs=[0.19])
    assert stopped == []
    assert tracker.bad_steps == [1]

    stopped = tracker.update(alpha=0.2, indices=[0], baseline_accs=[0.10], rebase_accs=[0.20])
    assert stopped == []
    assert tracker.bad_steps == [0]
    assert tracker.active_indices() == [0]
