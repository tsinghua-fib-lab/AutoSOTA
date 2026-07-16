""" Tools used for summarization scripts """

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial._qhull import QhullError

from .classes import Hull, Robustness


def mad(l):
    """Median Absolute Deviation"""
    l = list(l)
    if not len(l):
        return math.inf

    median = np.median(l)
    if median == math.inf:
        return math.inf

    return np.median([abs(v - median) for v in l])


def prepare_for_hull(d, bounds, baselines, direction=None, add_points=True):
    N = len(bounds)
    if not direction:
        direction = [1 for _ in range(N)]

    bounds = np.array(bounds, dtype=float)
    direction = np.array(direction, dtype=float)
    if type(d) == dict:
        d = {k: tuple(d[k][:N]) for k in d}
        values = np.clip(
            np.array([list(v) for v in d.values()]), 0, max(bounds)
        )
        keys = list(d.keys())
    else:
        d = np.array([v[:N] for v in d])
        values = np.clip(d, 0, max(bounds))
        keys = list(range(len(values)))

    # Normalize values
    try:
        values = np.clip(values.dot(np.diag(np.reciprocal(bounds))), 0, 1)
    except ValueError:
        values = None

    # Normalize baselines
    baselines = [
        [b / bounds[i] for b in baselines[i]] for i in range(len(bounds))
    ]
    baselines = np.array(baselines)

    if not add_points:
        if values is None:
            return [], keys
        return values, keys

    # Add achievable extermes
    if len(direction) == 2:
        max_quality, max_watermarking = baselines[0, 1], baselines[1, 1]
        min_quality, min_watermarking = baselines[0, 0], baselines[1, 0]
        # 0 quality, perfect watermarking
        values = (
            np.vstack([values, [min_quality, max_watermarking]])
            if values is not None
            else np.array([[min_quality, max_watermarking]])
        )
        # Ideal quality, no watermarking
        values = np.vstack([values, [max_quality, min_watermarking]])
        # 0 everywhere
        values = np.vstack([values, [min_quality, min_watermarking]])
        keys += ["E1", "E2", "E3"]
    else:
        max_quality, max_watermarking, max_robustness = (
            baselines[0, 1],
            baselines[1, 1],
            baselines[2, 1],
        )
        min_quality, min_watermarking, min_robustness = (
            baselines[0, 0],
            baselines[1, 0],
            baselines[2, 0],
        )
        # 0 quality, perfect watermarking, perfect robustness
        values = (
            np.vstack([values, [min_quality, max_watermarking, max_robustness]])
            if values is not None
            else np.array([[min_quality, max_watermarking, max_robustness]])
        )
        # 0 quality, perfect watermarking, 0 robustness
        values = np.vstack(
            [values, [min_quality, max_watermarking, min_robustness]]
        )
        # 0 quality, 0 watermarking, perfect robustness
        values = np.vstack(
            [values, [min_quality, min_watermarking, max_robustness]]
        )
        # Ideal quality, no watermarking, no robustness
        values = np.vstack(
            [values, [max_quality, min_watermarking, min_robustness]]
        )
        # Ideal quality, no watermarking, perfect robustness
        values = np.vstack(
            [values, [max_quality, min_watermarking, max_robustness]]
        )
        # 0 everywhere
        values = np.vstack(
            [values, [min_quality, min_watermarking, min_robustness]]
        )
        keys += ["E1", "E2", "E3", "E4", "E5", "E6"]

    return values, keys


def get_normal(face, normalize=True):
    face = np.array(face)
    norm = None

    if len(face) == 2:
        norm = np.array([face[0][1] - face[1][1], face[1][0] - face[0][0]])
    else:
        norm = np.cross(face[1] - face[0], face[2] - face[0])

    if np.linalg.norm(norm) > 0 and normalize:
        norm = norm / np.linalg.norm(norm)

    return norm


def get_area(face, projection_direction=None):
    face = np.array(face)
    if projection_direction != None:
        face[:, projection_direction] = 0
    if len(face) == 2:
        return np.linalg.norm(face[1] - face[0])
    else:
        return 0.5 * np.linalg.norm(get_normal(face, normalize=False))


def convex_hull(d, bounds, baseline_quality, direction=None, add_points=True):
    values, keys = prepare_for_hull(
        d, bounds, baseline_quality, direction, add_points=add_points
    )

    # Get convex hull
    ch = ConvexHull(values)
    volume = ch.volume
    faces = ch.simplices
    normals = [v[:-1] for v in ch.equations]

    # Orient faces
    oriented_faces = []
    up = np.array([1, 0, 0]) if len(bounds) == 3 else np.array([1, 0])
    for face, normal in zip(faces, normals):
        norm = get_normal(values[face])
        if np.dot(norm, normal) < 0:
            face[-1], face[-2] = face[-2], face[-1]
        oriented_faces.append(face)

    # Add extermum points
    added = {k for k in keys if k not in d}
    d = {keys[idx]: values[idx] for idx in range(len(keys))}
    selected = {keys[index] for face in faces for index in face}

    return Hull(volume, keys, oriented_faces, d, selected, added)


def hull_volume(hull, points, bounds, baseline_quality, direction=None):
    values, _ = prepare_for_hull(points, bounds, baseline_quality, direction)
    up = np.array([1, 0, 0]) if len(bounds) == 3 else np.array([1, 0])

    # Compute volume

    volume = 0
    for face in hull.faces:
        orientation = np.dot(get_normal(values[face]), up)
        surface_area = get_area(values[face])
        volume += (
            sum([np.dot(up, v) for v in values[face]])
            * surface_area
            * orientation
            / len(bounds)
        )

    # Update points
    hull = replace(
        hull,
        points={k: values[i] for i, k in enumerate(hull.keys)},
        volume=volume,
    )

    return hull


def summarize_robustness(stats, threshold=0.8, existing_hull=None):
    summary = {}
    for model in stats:
        if model == "None":
            continue

        base_q = stats[model][None].quality
        base_w = stats[model][None].percent_watermarked

        if base_w == 0:
            summary[model] = Robustness(Hull(1), math.inf, 0, None, None)
            continue

        metric_a = {
            attack: (
                stats[model][attack].quality,
                stats[model][attack].percent_watermarked,
            )
            for attack in stats[model]
            if attack is not None
            and "dipper" not in attack
            and "gpt" not in attack
        }
        metric_b = {
            attack: (
                stats[model][attack].quality,
                stats[model][attack].efficiency,
            )
            for attack in stats[model]
            if attack is not None
            and "dipper" not in attack
            and "gpt" not in attack
        }

        threshold_metric_a = {
            k: metric_a[k]
            for k in metric_a
            if metric_a[k][0] >= threshold * base_q
        }
        threshold_metric_b = {
            k: metric_b[k]
            for k in metric_b
            if metric_b[k][0] >= threshold * base_q
        }

        try:
            best_attack_a, score_a = sorted(
                threshold_metric_a.items(), key=lambda x: (x[1][1], -x[1][0])
            )[0]
            score_a = score_a[1]

            best_attack_b, score_b = sorted(
                threshold_metric_b.items(), key=lambda x: (-x[1][1], -x[1][0])
            )[0]
            score_b = score_b[1]

        except IndexError:
            score_a, best_attack_a, score_b, best_attack_b = 1, None, 0, None

        if existing_hull is None:
            hull = convex_hull(
                metric_a,
                [base_q, 1],
                ((0, base_q), (base_w, 0)),
                direction=[1, 1],
            )
        else:
            points = [
                metric_a[k]
                for k in existing_hull[model].hull.keys
                if k not in existing_hull[model].hull.added
            ]
            hull = hull_volume(
                existing_hull[model].hull,
                points,
                [base_q, 1],
                ((0, base_q), (base_w, 0)),
                direction=[1, 1],
            )

        hull = replace(hull, volume=max(0, (base_w) - hull.volume))

        summary[model] = Robustness(
            hull=hull,
            efficiency_threshold=score_b,
            percentage_threshold=score_a,
            best_attack_efficiency=best_attack_b,
            best_attack_percentage=best_attack_a,
        )

    return summary


def find_convex_hull(data, baselines, max_seq_len, ignore_robustness=False):
    rtn = {}
    for (model, temp), points in data.items():
        local_baselines = (
            [(0, baselines[temp]), (max_seq_len, 0), (0, 0.5)]
            if not ignore_robustness
            else [(0, baselines[temp]), (max_seq_len, 0)]
        )
        rtn[(model, temp)] = convex_hull(
            points,
            (
                [1, max_seq_len, 0.5]
                if not ignore_robustness
                else [1, max_seq_len]
            ),
            local_baselines,
            direction=[1, -1, 1] if not ignore_robustness else [1, -1],
        )

    return rtn


def convex_hull_validation(
    orig_hull, data, baselines, max_seq_len, ignore_robustness=False
):
    rtn = {}
    for (model, temp), points in data.items():
        if (model, temp) not in orig_hull:
            continue

        hull = orig_hull[(model, temp)]

        all_points = [
            (0, 0) if ignore_robustness else (0, 0, 0)
            for k in hull.keys
            if k not in hull.added
        ]
        key_to_idx = {k: i for i, k in enumerate(hull.keys)}
        for key, pt in points.items():
            if key not in key_to_idx:
                continue
            all_points[key_to_idx[key]] = pt

        local_baselines = (
            [(0, baselines[temp]), (max_seq_len, 0), (0, 0.5)]
            if not ignore_robustness
            else [(0, baselines[temp]), (max_seq_len, 0)]
        )
        hull = hull_volume(
            hull,
            all_points,
            (
                [1, max_seq_len, 0.5]
                if not ignore_robustness
                else [1, max_seq_len]
            ),
            local_baselines,
            direction=[1, -1, 1] if not ignore_robustness else [1, -1],
        )
        rtn[(model, temp)] = hull

    return rtn


def find_threshold(data, thresholds, baselines):
    rtn = {}
    for (model, temp), points in data.items():
        q_baseline, r_baseline = baselines[temp], 0.5 * baselines[temp]
        if thresholds[1] >= 0:
            q_t, r_t = q_baseline * (1 - float(thresholds[0])), r_baseline * (
                1 - float(thresholds[1])
            )
            eligible = [
                item
                for item in points.items()
                if item[1][0] >= q_t and item[1][2] >= r_t
            ]
            if not len(eligible):
                best_label, best_values = None, (0, math.inf, 0)
            else:
                best_label, best_values = min(
                    eligible,
                    key=lambda x: x[1][1],
                )
        else:
            q_t = q_baseline * (1 - float(thresholds[0]))
            eligible = [item for item in points.items() if item[1][0] >= q_t]
            if not len(eligible):
                best_label, best_values = None, (0, math.inf, 0)
            else:
                best_label, best_values = max(
                    eligible,
                    key=lambda x: x[1][2],
                )

        rtn[(model, temp)] = (best_label, best_values)

    return rtn


def verify_threshold(data, orig):
    rtn = {}
    for (model, temp), points in data.items():
        if (model, temp) not in orig:
            continue

        key = orig[(model, temp)][0]
        if key is None:
            rtn[(model, temp)] = (key, (0, math.inf, 0))
        elif key in points:
            rtn[(model, temp)] = (key, points[key])
        else:
            print("Missing {}".format((model, temp)))
            rtn[(model, temp)] = (key, (orig[(model, temp)][1]))

    return rtn
