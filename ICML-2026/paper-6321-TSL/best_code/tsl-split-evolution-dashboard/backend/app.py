#!/usr/bin/env python3
"""FastAPI backend for the TSL split-event visualizer (tslviz).

Usage:
  Recommended:
    tslviz --db path/to/your.sqlite
    tslviz --db path/to/your.sqlite --port 8080 --reload

  As a module:
    python -m tslviz.backend.app --db path/to/your.sqlite

  Or with uvicorn (env-driven):
    export DATABASE_PATH=path/to/your.sqlite
    uvicorn tslviz.backend.app:app --reload --port 8051
"""

import json
import os
import sqlite3
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from fastapi.staticfiles import StaticFiles


def _resolve_frontend_dir() -> str:
    """Resolve frontend directory, handling both development and installed package scenarios."""
    here = os.path.dirname(__file__)
    # Try development path first (when running from source)
    dev_path = os.path.abspath(os.path.join(here, "..", "frontend"))
    if os.path.exists(dev_path) and os.path.isdir(dev_path):
        return dev_path
    
    # Try installed package path (when installed via pip)
    try:
        import importlib.resources
        package = importlib.resources.files("tslviz")
        frontend_path = package / "frontend"
        if frontend_path.is_dir():
            return str(frontend_path)
    except (ImportError, AttributeError):
        # Fallback for older Python versions or if importlib.resources not available
        try:
            import pkg_resources
            frontend_path = pkg_resources.resource_filename("tslviz", "frontend")
            if os.path.exists(frontend_path):
                return frontend_path
        except Exception:
            pass
    
    # Final fallback to development path
    return dev_path


@lru_cache(maxsize=1)
def get_db_path() -> str:
    """Get database path from environment variable."""
    db_path = os.environ.get("DATABASE_PATH", "")
    if not db_path:
        raise RuntimeError(
            "DATABASE_PATH not set. Either:\n"
            "  1. Export DATABASE_PATH environment variable, or\n"
            "  2. Run with: tslviz --db path/to/db.sqlite"
        )
    if not os.path.exists(db_path):
        raise RuntimeError(f"DATABASE_PATH not found: {db_path}")
    return db_path


def get_connection() -> sqlite3.Connection:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Read-optimized pragmas (safe for read-heavy workloads)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA cache_size=-32000;")  # ~32MB
    except Exception:
        pass
    return conn


def ensure_perf_indexes() -> None:
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_run_epoch_tree_col ON events(run_id, epoch, tree_id, col)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_run_epoch_tree_iter ON events(run_id, epoch, tree_id, iter_no)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_run_iter ON events(run_id, iter_no)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_component_states_loc ON component_states(run_id, epoch, tree_id, col, iter_no)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_component_states_iter ON component_states(run_id, epoch, tree_id, iter_no)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_f_component_stats_loc ON f_component_stats(run_id, epoch, tree_id, iter_no, component)"
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Validate DB and build helpful indexes
    _ = get_db_path()
    ensure_perf_indexes()
    yield
    # Shutdown: cleanup if needed (currently nothing to do)


app = FastAPI(
    default_response_class=ORJSONResponse,
    title="tslviz API",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


def _dicts(rows: List[sqlite3.Row]) -> List[Dict[str, Any]]:
    return [dict(r) for r in rows]


def decode_f64_array(data: bytes) -> List[float]:
    """Decode binary blob to f64 array."""
    import struct

    return list(struct.unpack(f"{len(data) // 8}d", data))


def _parse_tree_ids_csv(tree_ids: str, max_trees: int = 200) -> List[int]:
    if not tree_ids:
        return []
    out: List[int] = []
    for part in str(tree_ids).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            continue
        if len(out) >= max_trees:
            break
    seen = set()
    uniq: List[int] = []
    for t in out:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    return uniq


def _decode_intervals_from_component_blob(
    data: Optional[bytes], intervals_count: int
) -> List[List[Optional[float]]]:
    """Decode interval boundaries from component_states.data.

    Format: [starts (N), ends (N), values (N)] as f64.
    Some initial/legacy snapshots store an empty blob; synthesize open bounds.
    """
    if intervals_count <= 0:
        return []
    if not data:
        if intervals_count == 1:
            return [[-float("inf"), float("inf")]]
        return [[None, None] for _ in range(intervals_count)]

    import struct

    n = intervals_count * 3
    if len(data) != n * 8:
        if intervals_count == 1:
            return [[-float("inf"), float("inf")]]
        return [[None, None] for _ in range(intervals_count)]

    vals = struct.unpack(f"{n}d", data)
    starts = vals[0:intervals_count]
    ends = vals[intervals_count : 2 * intervals_count]
    return [[float(a), float(b)] for a, b in zip(starts, ends)]


def _finite_or_none(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except Exception:
        return None
    # `orjson` will serialize non-finite floats as null, but we also want to avoid
    # computing summary stats on infinities in Python.
    if f != f:  # NaN
        return None
    if f == float("inf") or f == -float("inf"):
        return None
    return f


def _safe_exp(x: float) -> Optional[float]:
    import math

    try:
        v = math.exp(x)
    except OverflowError:
        return None
    return _finite_or_none(v)


@app.get("/api/ping")
def ping() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/runs")
def list_runs() -> List[Dict[str, Any]]:
    conn = get_connection()
    q = (
        "SELECT r.id as run_id, r.created_at, r.n_rows, r.n_cols, r.params_json, "
        "COUNT(e.id) as n_events, COUNT(DISTINCT e.iter_no) as n_iterations, "
        "COUNT(DISTINCT e.col) as n_columns_split, "
        "SUM(CASE WHEN LOWER(e.action) = 'split' THEN 1 ELSE 0 END) as n_splits, "
        "SUM(CASE WHEN LOWER(e.action) = 'resplit' THEN 1 ELSE 0 END) as n_resplits "
        "FROM runs r LEFT JOIN events e ON r.id = e.run_id "
        "GROUP BY r.id, r.created_at, r.n_rows, r.n_cols, r.params_json "
        "ORDER BY r.created_at DESC"
    )
    rows = conn.execute(q).fetchall()
    conn.close()
    return _dicts(rows)


@app.get("/api/run/{run_id}/timeline")
def split_timeline(run_id: int) -> List[Dict[str, Any]]:
    conn = get_connection()
    q = (
        "SELECT epoch, tree_id, iter_no, action, col, left_interval_idx, split_value, "
        "update_a, update_b, left_count, right_count, gain, n_cells_after "
        "FROM events WHERE run_id = ? ORDER BY iter_no, col"
    )
    rows = conn.execute(q, (run_id,)).fetchall()
    conn.close()
    return _dicts(rows)


@app.get("/api/run/{run_id}/columns")
def column_stats(run_id: int) -> List[Dict[str, Any]]:
    conn = get_connection()
    q = (
        "SELECT col, COUNT(*) as n_events, "
        "SUM(CASE WHEN LOWER(action) = 'split' THEN 1 ELSE 0 END) as n_splits, "
        "SUM(CASE WHEN LOWER(action) = 'resplit' THEN 1 ELSE 0 END) as n_resplits, "
        "SUM(CASE WHEN LOWER(action) = 'merge' THEN 1 ELSE 0 END) as n_merges, "
        "AVG(gain) as avg_gain, MAX(gain) as max_gain, "
        "AVG(left_count + right_count) as avg_samples_affected "
        "FROM events WHERE run_id = ? AND gain IS NOT NULL GROUP BY col ORDER BY col"
    )
    rows = conn.execute(q, (run_id,)).fetchall()
    conn.close()
    return _dicts(rows)


@app.get("/api/run/{run_id}/columns_summary")
def column_stats_series(run_id: int) -> Dict[str, Any]:
    conn = get_connection()
    q = (
        "SELECT col, "
        "SUM(CASE WHEN LOWER(action) = 'split' THEN 1 ELSE 0 END) as n_splits, "
        "SUM(CASE WHEN LOWER(action) = 'resplit' THEN 1 ELSE 0 END) as n_resplits, "
        "SUM(CASE WHEN LOWER(action) = 'merge' THEN 1 ELSE 0 END) as n_merges, "
        "AVG(gain) as avg_gain, "
        "AVG(left_count + right_count) as avg_samples_affected "
        "FROM events WHERE run_id = ? AND gain IS NOT NULL GROUP BY col ORDER BY col"
    )
    rows = conn.execute(q, (run_id,)).fetchall()
    conn.close()
    events = [
        {
            "col": int(r["col"]) if r["col"] is not None else 0,
            "splits": int(r["n_splits"]) if r["n_splits"] is not None else 0,
            "resplits": int(r["n_resplits"]) if r["n_resplits"] is not None else 0,
            "merges": int(r["n_merges"]) if r["n_merges"] is not None else 0,
        }
        for r in rows
    ]
    avg_gain = [
        {
            "col": int(r["col"]) if r["col"] is not None else 0,
            "avg": float(r["avg_gain"]) if r["avg_gain"] is not None else 0.0,
        }
        for r in rows
    ]
    samples = [
        {
            "col": int(r["col"]) if r["col"] is not None else 0,
            "val": (
                float(r["avg_samples_affected"])
                if r["avg_samples_affected"] is not None
                else 0.0
            ),
        }
        for r in rows
    ]
    return {"events": events, "avg_gain": avg_gain, "samples": samples}


@app.get("/api/run/{run_id}/learning")
def learning_by_epoch(run_id: int) -> List[Dict[str, Any]]:
    conn = get_connection()
    q = (
        "SELECT epoch, AVG(gain) as avg_gain_per_split, SUM(gain) as total_gain_per_epoch, "
        "COUNT(*) as num_splits, MIN(gain) as min_gain, MAX(gain) as max_gain "
        "FROM events WHERE run_id = ? AND gain IS NOT NULL AND epoch IS NOT NULL "
        "GROUP BY epoch ORDER BY epoch"
    )
    rows = conn.execute(q, (run_id,)).fetchall()
    conn.close()
    return _dicts(rows)


@app.get("/api/run/{run_id}/convergence")
def convergence_by_epoch(run_id: int) -> Dict[str, Any]:
    conn = get_connection()
    q = (
        "SELECT epoch, "
        "AVG(gain) as avg_gain_per_split, "
        "SUM(gain) as total_gain_per_epoch, "
        "COUNT(*) as num_events, "
        "COUNT(CASE WHEN LOWER(action) = 'split' THEN 1 END) as new_splits, "
        "COUNT(CASE WHEN LOWER(action) = 'resplit' THEN 1 END) as resplits, "
        "COUNT(CASE WHEN LOWER(action) = 'merge' THEN 1 END) as merges, "
        "AVG(n_cells_after) as avg_model_complexity "
        "FROM events WHERE run_id = ? AND epoch IS NOT NULL GROUP BY epoch ORDER BY epoch"
    )
    rows = conn.execute(q, (run_id,)).fetchall()
    conn.close()
    epochs: List[int] = []
    total: List[float] = []
    avg: List[float] = []
    new_splits: List[int] = []
    resplits: List[int] = []
    merges: List[int] = []
    complexity: List[float] = []
    for r in rows:
        epochs.append(int(r["epoch"]) if r["epoch"] is not None else 0)
        avg.append(
            float(r["avg_gain_per_split"])
            if r["avg_gain_per_split"] is not None
            else 0.0
        )
        total.append(
            float(r["total_gain_per_epoch"])
            if r["total_gain_per_epoch"] is not None
            else 0.0
        )
        new_splits.append(int(r["new_splits"]) if r["new_splits"] is not None else 0)
        resplits.append(int(r["resplits"]) if r["resplits"] is not None else 0)
        merges.append(int(r["merges"]) if r["merges"] is not None else 0)
        complexity.append(
            float(r["avg_model_complexity"])
            if r["avg_model_complexity"] is not None
            else 0.0
        )
    return {
        "epochs": epochs,
        "total": total,
        "avg": avg,
        "new_splits": new_splits,
        "resplits": resplits,
        "merges": merges,
        "complexity": complexity,
    }


@app.get("/api/run/{run_id}/epochs_trees")
def epochs_trees(run_id: int) -> Dict[str, List[int]]:
    conn = get_connection()
    q = (
        "SELECT DISTINCT epoch, tree_id FROM events WHERE run_id = ? AND epoch IS NOT NULL AND tree_id IS NOT NULL "
        "ORDER BY epoch, tree_id"
    )
    rows = conn.execute(q, (run_id,)).fetchall()
    conn.close()
    out: Dict[str, List[int]] = {}
    for r in rows:
        epoch = int(r["epoch"]) if r["epoch"] is not None else 0
        tree = int(r["tree_id"]) if r["tree_id"] is not None else 0
        out.setdefault(str(epoch), []).append(tree)
    return out


@app.get("/api/run/{run_id}/f_component_epochs_trees")
def f_component_epochs_trees(run_id: int) -> Dict[str, List[int]]:
    """Get epochs and trees that have f_component_stats data."""
    conn = get_connection()
    try:
        # Check if table exists first
        table_check = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='f_component_stats'"
        ).fetchone()
        if not table_check:
            return {}

        q = (
            "SELECT DISTINCT epoch, tree_id FROM f_component_stats WHERE run_id = ? "
            "ORDER BY epoch, tree_id"
        )
        rows = conn.execute(q, (run_id,)).fetchall()
        out: Dict[str, List[int]] = {}
        for r in rows:
            epoch = int(r["epoch"]) if r["epoch"] is not None else 0
            tree = int(r["tree_id"]) if r["tree_id"] is not None else 0
            out.setdefault(str(epoch), []).append(tree)
        return out
    except Exception as e:
        # Log the error for debugging
        import traceback

        print(f"Error in f_component_epochs_trees: {e}")
        print(traceback.format_exc())
        # Return empty dict instead of raising to allow graceful handling
        return {}
    finally:
        conn.close()


def _reconstruct_tree_state(
    run_id: int, epoch: int, tree_id: int, iteration: int
) -> Dict[str, Any]:
    conn = get_connection()
    q = (
        "SELECT iter_no, action, col, gain, split_value, update_a, update_b, err_before, err_after FROM events "
        "WHERE run_id = ? AND epoch = ? AND tree_id = ? AND iter_no <= ? ORDER BY iter_no"
    )
    rows = conn.execute(q, (run_id, epoch, tree_id, iteration)).fetchall()
    if not rows:
        conn.close()
        return {"history": [], "n_dims": 0, "bounds": {}}

    q2 = (
        "SELECT col, MIN(split_value) AS mn, MAX(split_value) AS mx FROM events "
        "WHERE run_id = ? AND epoch = ? AND tree_id = ? AND split_value IS NOT NULL GROUP BY col"
    )
    bounds_rows = conn.execute(q2, (run_id, epoch, tree_id)).fetchall()
    conn.close()

    n_dims = 0
    bounds: Dict[int, Dict[str, float]] = {}
    for br in bounds_rows:
        dim = int(br["col"]) if br["col"] is not None else 0
        mn = br["mn"]
        mx = br["mx"]
        if mn is None or mx is None:
            lo, hi = -1.0, 1.0
        else:
            mn = float(mn)
            mx = float(mx)
            margin = (mx - mn) * 0.1 if (mx - mn) > 0 else 1.0
            lo, hi = mn - margin, mx + margin
        bounds[dim] = {"min": lo, "max": hi}
        n_dims = max(n_dims, dim + 1)

    if n_dims == 0:
        # Fall back to at least one component
        n_dims = 1
        bounds[0] = {"min": -1.0, "max": 1.0}

    import math

    estimators: List[List[List[float]]] = [
        [[-float("inf"), float("inf"), 0.0]] for _ in range(n_dims)
    ]
    last_event: Optional[Dict[str, Any]] = None

    for r in rows:
        action = (r["action"] or "").lower()
        col = int(r["col"]) if r["col"] is not None else 0
        col = min(col, n_dims - 1)
        split_value = r["split_value"]
        a = r["update_a"]
        b = r["update_b"]
        gain = r["gain"]
        dim_est = estimators[col]
        dim_est.sort(key=lambda iv: (iv[0], iv[1]))

        if (
            action == "split"
            and split_value is not None
            and a is not None
            and b is not None
        ):
            s = float(split_value)
            target_idx = None
            for j, (start, end, _v) in enumerate(dim_est):
                if start < s < end:
                    target_idx = j
                    break
            if target_idx is not None:
                start, end, _old = dim_est[target_idx]
                if start < s < end:
                    dim_est[target_idx : target_idx + 1] = [
                        [start, s, float(a)],
                        [s, end, float(b)],
                    ]
        elif action == "resplit" and a is not None and b is not None:
            # Adjust values at the boundary if the boundary exists
            if split_value is not None:
                s = float(split_value)
                left_idx = right_idx = None
                for j, (start, end, _v) in enumerate(dim_est):
                    if math.isclose(end, s, rel_tol=1e-9, abs_tol=1e-12):
                        left_idx = j
                    if math.isclose(start, s, rel_tol=1e-9, abs_tol=1e-12):
                        right_idx = j
                if left_idx is not None and right_idx is not None:
                    ls, le, _lv = dim_est[left_idx]
                    rs, re, _rv = dim_est[right_idx]
                    dim_est[left_idx] = [ls, le, float(a)]
                    dim_est[right_idx] = [rs, re, float(b)]
        elif action == "merge" and split_value is not None and a is not None:
            s = float(split_value)
            left_idx = right_idx = None
            for j, (start, end, _v) in enumerate(dim_est):
                if math.isclose(end, s, rel_tol=1e-9, abs_tol=1e-12):
                    left_idx = j
                if math.isclose(start, s, rel_tol=1e-9, abs_tol=1e-12):
                    right_idx = j
            if left_idx is not None and right_idx is not None:
                ls, le, _lv = dim_est[left_idx]
                rs, re, _rv = dim_est[right_idx]
                dim_est[left_idx : right_idx + 1] = [[ls, re, float(a)]]

        last_event = {
            "action": r["action"],
            "col": col,
            "split_value": split_value,
            "gain": gain,
            "update_a": a,
            "update_b": b,
            "iter_no": r["iter_no"],
            "err_before": r["err_before"],
            "err_after": r["err_after"],
        }

    # Calculate current error - use err_after from last event if available, otherwise compute from current state
    current_error = None
    if last_event and last_event.get("err_after") is not None:
        current_error = last_event["err_after"]

    return {
        "estimators": estimators,
        "n_dims": n_dims,
        "bounds": bounds,
        "split_event": last_event,
        "current_error": current_error,
    }


@app.get("/api/run/{run_id}/tree_evolution")
def tree_evolution(
    run_id: int,
    epoch: int = Query(...),
    tree_id: int = Query(...),
    iteration: int = Query(0, ge=0),
) -> Dict[str, Any]:
    return _reconstruct_tree_state(run_id, epoch, tree_id, iteration)


@app.get("/api/run/{run_id}/identified_components")
def identified_components(
    run_id: int,
    epoch: int = Query(0),
    tree_id: int = Query(0),
) -> Dict[str, Any]:
    conn = get_connection()
    try:
        rows = conn.execute(
            (
                "SELECT col, intervals_count, data FROM component_states "
                "WHERE run_id=? AND epoch=? AND tree_id=? AND iter_no=(SELECT MAX(iter_no) FROM component_states WHERE run_id=? AND epoch=? AND tree_id=?)"
            ),
            (run_id, epoch, tree_id, run_id, epoch, tree_id),
        ).fetchall()
    finally:
        conn.close()
    components: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        col = int(r["col"]) if "col" in r.keys() else int(r[0])
        intervals_count = (
            int(r["intervals_count"]) if "intervals_count" in r.keys() else int(r[1])
        )
        blob = r["data"] if "data" in r.keys() else r[2]
        import numpy as np  # type: ignore

        arr = np.frombuffer(blob, dtype=np.float64)
        if arr.size != intervals_count * 3:
            continue
        starts = arr[0:intervals_count]
        ends = arr[intervals_count : 2 * intervals_count]
        vals = arr[2 * intervals_count : 3 * intervals_count]
        components[col] = {
            "intervals": list(
                map(
                    lambda ab: [float(ab[0]), float(ab[1])],
                    zip(starts.tolist(), ends.tolist()),
                )
            ),
            "values": vals.tolist(),
        }
    n_dims = (max(components.keys()) + 1) if components else 0
    return {"n_dims": n_dims, "components": components}


@app.get("/api/run/{run_id}/identified_components_all")
def identified_components_all(
    run_id: int,
    epoch: int = Query(...),
    max_trees: int = Query(50, ge=1),
) -> Dict[str, Any]:
    """Return final identified components for all trees at an epoch.

    Output: { n_dims, bounds: {dim:{min,max}}, trees: [ { tree_id, components: { dim: [[a,b,val], ...] } } ] }
    """
    conn = get_connection()
    try:
        t_rows = conn.execute(
            "SELECT DISTINCT tree_id FROM component_states WHERE run_id=? AND epoch=? ORDER BY tree_id",
            (run_id, epoch),
        ).fetchall()
        tree_ids = [int(r["tree_id"]) for r in t_rows if r["tree_id"] is not None][
            :max_trees
        ]
        # Pull all (tree_id, col) latest snapshots in one query
        rows = conn.execute(
            (
                "SELECT tree_id, col, intervals_count, data, MAX(iter_no) as max_iter "
                "FROM component_states WHERE run_id=? AND epoch=? GROUP BY tree_id, col"
            ),
            (run_id, epoch),
        ).fetchall()
    finally:
        conn.close()

    # Organize by tree -> col -> intervals/values
    import numpy as np  # type: ignore

    tree_to_components: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for r in rows:
        tid = int(r["tree_id"]) if "tree_id" in r.keys() else int(r[0])
        if tid not in tree_ids:
            continue
        col = int(r["col"]) if "col" in r.keys() else int(r[1])
        intervals_count = (
            int(r["intervals_count"]) if "intervals_count" in r.keys() else int(r[2])
        )
        blob = r["data"] if "data" in r.keys() else r[3]
        arr = np.frombuffer(blob, dtype=np.float64)
        if arr.size != intervals_count * 3:
            continue
        starts = arr[0:intervals_count]
        ends = arr[intervals_count : 2 * intervals_count]
        vals = arr[2 * intervals_count : 3 * intervals_count]
        tree_to_components.setdefault(tid, {})[col] = {
            "intervals": list(
                map(
                    lambda ab: [float(ab[0]), float(ab[1])],
                    zip(starts.tolist(), ends.tolist()),
                )
            ),
            "values": vals.tolist(),
        }

    # Determine n_dims and bounds across all trees
    n_dims = 0
    bounds: Dict[int, Dict[str, float]] = {}
    for tid, comp_map in tree_to_components.items():
        for col, comp in comp_map.items():
            n_dims = max(n_dims, col + 1)
            finite_vals = [v for ab in comp["intervals"] for v in ab if np.isfinite(v)]
            if finite_vals:
                mn = float(min(finite_vals))
                mx = float(max(finite_vals))
                margin = (mx - mn) * 0.1 if (mx - mn) > 0 else 1.0
                lo, hi = mn - margin, mx + margin
            else:
                lo, hi = -1.0, 1.0
            if col not in bounds:
                bounds[col] = {"min": lo, "max": hi}
            else:
                bounds[col]["min"] = min(bounds[col]["min"], lo)
                bounds[col]["max"] = max(bounds[col]["max"], hi)

    trees_out: List[Dict[str, Any]] = []
    for tid in tree_ids:
        comp_map = tree_to_components.get(tid, {})
        comp_out: Dict[int, List[List[float]]] = {}
        for dim, comp in comp_map.items():
            comp_out[dim] = [
                [float(a), float(b), float(v)]
                for (a, b), v in zip(comp["intervals"], comp["values"])
            ]
        trees_out.append({"tree_id": tid, "components": comp_out})

    if n_dims == 0 and bounds:
        n_dims = len(bounds)
    if n_dims == 0:
        n_dims = 1
        bounds[0] = {"min": -1.0, "max": 1.0}

    return {"n_dims": n_dims, "bounds": bounds, "trees": trees_out}


@app.get("/api/run/{run_id}/grid_errors")
def grid_errors(run_id: int) -> Dict[str, Any]:
    """Return per-epoch grid errors pre-bucketed for faster frontend rendering.

    Output:
    {
      epochs: [e0, e1, ...],
      family: { train: [err_by_epoch], test: [err_by_epoch] },
      trees:  { train: [{epoch, err, tree_id}], test: [{epoch, err, tree_id}] }
    }
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT epoch, tree_id, err, variant FROM training_errors WHERE run_id=? ORDER BY epoch, tree_id, variant",
            (run_id,),
        ).fetchall()
    finally:
        conn.close()
    # Aggregate
    epochs_set = set()
    fam_train_map: Dict[int, float] = {}
    fam_test_map: Dict[int, float] = {}
    tree_train: List[Dict[str, Any]] = []
    tree_test: List[Dict[str, Any]] = []
    for r in rows:
        epoch = int(r["epoch"]) if r["epoch"] is not None else 0
        tree_id = r["tree_id"]
        err = float(r["err"]) if r["err"] is not None else 0.0
        variant = str(r["variant"]).lower()
        epochs_set.add(epoch)
        if tree_id is None:
            if variant == "train":
                fam_train_map[epoch] = err
            else:
                fam_test_map[epoch] = err
        else:
            if variant == "train":
                tree_train.append({"epoch": epoch, "err": err, "tree_id": tree_id})
            else:
                tree_test.append({"epoch": epoch, "err": err, "tree_id": tree_id})
    epochs = sorted(list(epochs_set))
    fam_train = [fam_train_map.get(e, None) for e in epochs]
    fam_test = [fam_test_map.get(e, None) for e in epochs]
    return {
        "epochs": epochs,
        "family": {"train": fam_train, "test": fam_test},
        "trees": {"train": tree_train, "test": tree_test},
    }


@app.get("/api/run/{run_id}/unified_tree_components")
def unified_tree_components(
    run_id: int,
    epoch: int = Query(...),
    iteration: int = Query(0, ge=0),
    identified: bool = Query(False),
    selected_trees: Optional[str] = Query(
        None,
        description="Comma-separated list of tree IDs to include. Use '-1' for all trees.",
    ),
) -> Dict[str, Any]:
    """Return unified tree components format for both identified and regular components.

    This endpoint provides a consistent data structure regardless of whether you want
    identified components or regular tree evolution components.

    Args:
        run_id: The run ID
        epoch: The epoch to fetch
        iteration: The iteration number (for regular components)
        identified: If True, return identified components; if False, return regular components
        selected_trees: Comma-separated list of tree IDs, or '-1' for all trees

    Returns:
        {
            "n_dims": int,
            "bounds": {dim: {"min": float, "max": float}},
            "trees": [
                {
                    "tree_id": int,
                    "components": {dim: [[a, b, val], ...]},
                    "split_event": {...}  # only for regular components
                }
            ]
        }
    """
    if identified:
        # Use identified components logic (supports selected_trees filtering)
        conn = get_connection()
        try:
            # Determine which trees to include
            if selected_trees == "-1" or not (
                selected_trees and selected_trees.strip()
            ):
                t_rows = conn.execute(
                    "SELECT DISTINCT tree_id FROM component_states WHERE run_id=? AND epoch=? ORDER BY tree_id",
                    (run_id, epoch),
                ).fetchall()
                tree_ids = [
                    int(r["tree_id"]) for r in t_rows if r["tree_id"] is not None
                ]
            else:
                # Parse requested IDs, verify they exist for this run/epoch; if none verified, fall back to requested
                requested_tree_ids = [
                    int(tid.strip())
                    for tid in selected_trees.split(",")
                    if tid is not None and tid.strip().lstrip("+-").isdigit()
                ]
                verified_ids: List[int] = []
                if requested_tree_ids:
                    placeholders = ",".join(["?"] * len(requested_tree_ids))
                    q = (
                        f"SELECT DISTINCT tree_id FROM component_states WHERE run_id=? AND epoch=? AND tree_id IN ({placeholders}) "
                        "ORDER BY tree_id"
                    )
                    v_rows = conn.execute(
                        q, (run_id, epoch, *requested_tree_ids)
                    ).fetchall()
                    verified_ids = [
                        int(r["tree_id"]) for r in v_rows if r["tree_id"] is not None
                    ]
                tree_ids = (
                    verified_ids
                    if verified_ids
                    else list(dict.fromkeys(requested_tree_ids))
                )

            # Pull all (tree_id, col) latest snapshots in one query
            rows = conn.execute(
                (
                    "SELECT tree_id, col, intervals_count, data, MAX(iter_no) as max_iter "
                    "FROM component_states WHERE run_id=? AND epoch=? GROUP BY tree_id, col"
                ),
                (run_id, epoch),
            ).fetchall()
        finally:
            conn.close()

        # Organize by tree -> col -> intervals/values
        import numpy as np  # type: ignore

        tree_to_components: Dict[int, Dict[int, Dict[str, Any]]] = {}
        for r in rows:
            tid = int(r["tree_id"]) if "tree_id" in r.keys() else int(r[0])
            if tid not in tree_ids:
                continue
            col = int(r["col"]) if "col" in r.keys() else int(r[1])
            intervals_count = (
                int(r["intervals_count"])
                if "intervals_count" in r.keys()
                else int(r[2])
            )
            blob = r["data"] if "data" in r.keys() else r[3]
            arr = np.frombuffer(blob, dtype=np.float64)
            if arr.size != intervals_count * 3:
                continue
            starts = arr[0:intervals_count]
            ends = arr[intervals_count : 2 * intervals_count]
            vals = arr[2 * intervals_count : 3 * intervals_count]
            tree_to_components.setdefault(tid, {})[col] = {
                "intervals": list(
                    map(
                        lambda ab: [float(ab[0]), float(ab[1])],
                        zip(starts.tolist(), ends.tolist()),
                    )
                ),
                "values": vals.tolist(),
            }

        # Determine n_dims and bounds across all selected trees
        n_dims = 0
        # Track global finite min/max per dimension first; compute margin once from the global span
        col_global_min: Dict[int, float] = {}
        col_global_max: Dict[int, float] = {}
        seen_cols: set[int] = set()
        for _tid, comp_map in tree_to_components.items():
            for col, comp in comp_map.items():
                seen_cols.add(col)
                n_dims = max(n_dims, col + 1)
                finite_vals = [
                    v for ab in comp["intervals"] for v in ab if np.isfinite(v)
                ]
                if finite_vals:
                    mn_local = float(min(finite_vals))
                    mx_local = float(max(finite_vals))
                    if col in col_global_min:
                        col_global_min[col] = min(col_global_min[col], mn_local)
                        col_global_max[col] = max(col_global_max[col], mx_local)
                    else:
                        col_global_min[col] = mn_local
                        col_global_max[col] = mx_local
        bounds: Dict[int, Dict[str, float]] = {}
        for col in seen_cols:
            if col in col_global_min and col in col_global_max:
                span = col_global_max[col] - col_global_min[col]
                margin = span * 0.1 if span > 0 else 1.0
                lo, hi = col_global_min[col] - margin, col_global_max[col] + margin
            else:
                lo, hi = -1.0, 1.0
            bounds[col] = {"min": lo, "max": hi}

        trees_out: List[Dict[str, Any]] = []
        for tid in tree_ids:
            comp_map = tree_to_components.get(tid, {})
            comp_out: Dict[int, List[List[float]]] = {}
            for dim, comp in comp_map.items():
                comp_out[dim] = [
                    [float(a), float(b), float(v)]
                    for (a, b), v in zip(comp["intervals"], comp["values"])
                ]
            trees_out.append({"tree_id": tid, "components": comp_out})

        if n_dims == 0 and bounds:
            n_dims = len(bounds)
        if n_dims == 0:
            n_dims = 1
            bounds[0] = {"min": -1.0, "max": 1.0}

        return {"n_dims": n_dims, "bounds": bounds, "trees": trees_out}
    else:
        # Use regular tree evolution logic
        conn = get_connection()

        # Handle tree selection: -1 means all trees, otherwise parse specific tree IDs
        if selected_trees == "-1":
            # Get all available trees
            tree_rows = conn.execute(
                "SELECT DISTINCT tree_id FROM events WHERE run_id=? AND epoch=? ORDER BY tree_id",
                (run_id, epoch),
            ).fetchall()
            tree_ids = [
                int(r["tree_id"]) for r in tree_rows if r["tree_id"] is not None
            ]
        elif selected_trees and selected_trees.strip():
            # Parse comma-separated tree IDs
            requested_tree_ids = [
                int(tid.strip())
                for tid in selected_trees.split(",")
                if tid is not None and tid.strip().lstrip("+-").isdigit()
            ]
            verified_ids: List[int] = []
            if requested_tree_ids:
                # Verify these trees exist for this run/epoch
                placeholders = ",".join(["?"] * len(requested_tree_ids))
                q = f"SELECT DISTINCT tree_id FROM events WHERE run_id=? AND epoch=? AND tree_id IN ({placeholders}) ORDER BY tree_id"
                tree_rows = conn.execute(
                    q, (run_id, epoch, *requested_tree_ids)
                ).fetchall()
                verified_ids = [
                    int(r["tree_id"]) for r in tree_rows if r["tree_id"] is not None
                ]
            # Fall back to requested IDs if verification found none (keeps intent)
            tree_ids = (
                verified_ids
                if verified_ids
                else list(dict.fromkeys(requested_tree_ids))
            )
        else:
            # Default: no trees selected
            tree_ids = []
        row = conn.execute(
            "SELECT MAX(col) AS mx FROM events WHERE run_id=? AND epoch=?",
            (run_id, epoch),
        ).fetchone()
        n_dims = 0 if row is None or row["mx"] is None else int(row["mx"]) + 1
        # Compute bounds over selected trees only (if any specified), using a single margin from the global span
        if tree_ids:
            placeholders = ",".join(["?"] * len(tree_ids))
            b_rows = conn.execute(
                (
                    f"SELECT col, MIN(split_value) AS mn, MAX(split_value) AS mx FROM events "
                    f"WHERE run_id=? AND epoch=? AND split_value IS NOT NULL AND tree_id IN ({placeholders}) GROUP BY col"
                ),
                (run_id, epoch, *tree_ids),
            ).fetchall()
        else:
            b_rows = conn.execute(
                (
                    "SELECT col, MIN(split_value) AS mn, MAX(split_value) AS mx FROM events "
                    "WHERE run_id=? AND epoch=? AND split_value IS NOT NULL GROUP BY col"
                ),
                (run_id, epoch),
            ).fetchall()
        conn.close()
        bounds: Dict[int, Dict[str, float]] = {}
        for br in b_rows:
            dim = int(br["col"]) if br["col"] is not None else 0
            mn = br["mn"]
            mx = br["mx"]
            if mn is None or mx is None:
                lo, hi = -1.0, 1.0
            else:
                mn = float(mn)
                mx = float(mx)
                margin = (mx - mn) * 0.1 if (mx - mn) > 0 else 1.0
                lo, hi = mn - margin, mx + margin
            bounds[dim] = {"min": lo, "max": hi}
        if n_dims == 0:
            n_dims = len(bounds) if bounds else 1

        trees_out: List[Dict[str, Any]] = []
        for t_id in tree_ids:
            state = _reconstruct_tree_state(run_id, epoch, t_id, iteration)
            ests = state.get("estimators") or []
            comp: Dict[int, List[List[float]]] = {}
            for dim, intervals in enumerate(ests):
                comp[dim] = [[float(a), float(b), float(v)] for a, b, v in intervals]
            trees_out.append(
                {
                    "tree_id": t_id,
                    "components": comp,
                    "split_event": state.get("split_event"),
                    "current_error": state.get("current_error"),
                }
            )

        return {"n_dims": n_dims, "bounds": bounds, "trees": trees_out}


@app.get("/api/epochs")
def distinct_combined_epochs() -> List[int]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT DISTINCT epoch FROM combined_grids ORDER BY epoch"
        ).fetchall()
    except Exception as e:
        conn.close()
        raise HTTPException(
            status_code=404, detail=f"No combined_grids table or data: {e}"
        )
    conn.close()
    return [int(r["epoch"]) for r in rows if r["epoch"] is not None]


@app.get("/api/run/{run_id}/energy")
def epoch_energy(run_id: int) -> Dict[str, Any]:
    """Return energy history for a run.

    Returns:
    {
        "epochs": [e0, e1, ...],
        "energy": [energy0, energy1, ...]
    }
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT epoch, energy FROM combined_grids WHERE run_id = ? AND energy IS NOT NULL ORDER BY epoch",
            (run_id,),
        ).fetchall()
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=404, detail=f"No combined_grids data: {e}")
    conn.close()

    epochs = []
    energy = []
    for r in rows:
        epochs.append(int(r["epoch"]))
        energy.append(float(r["energy"]))

    return {
        "epochs": epochs,
        "energy": energy,
    }


@app.get("/api/run/{run_id}/backbone_tilt_evolution")
def backbone_tilt_evolution(
    run_id: int,
    epoch: int = Query(...),
    tree_id: int = Query(...),
    col: int = Query(...),
    start_iter: int = Query(0),
    end_iter: Optional[int] = Query(None),
) -> Dict[str, Any]:
    """Get backbone and tilt evolution for a specific axis over iterations."""
    conn = get_connection()
    try:
        # Build query with optional end_iter filter
        if end_iter is not None:
            q = (
                "SELECT iter_no, backbone_data, tilt_data, lambda_plus, lambda_minus "
                "FROM component_states "
                "WHERE run_id=? AND epoch=? AND tree_id=? AND col=? AND iter_no>=? AND iter_no<=? "
                "ORDER BY iter_no"
            )
            rows = conn.execute(
                q, (run_id, epoch, tree_id, col, start_iter, end_iter)
            ).fetchall()
        else:
            q = (
                "SELECT iter_no, backbone_data, tilt_data, lambda_plus, lambda_minus "
                "FROM component_states "
                "WHERE run_id=? AND epoch=? AND tree_id=? AND col=? AND iter_no>=? "
                "ORDER BY iter_no"
            )
            rows = conn.execute(q, (run_id, epoch, tree_id, col, start_iter)).fetchall()

        if not rows:
            conn.close()
            return {
                "col": col,
                "iterations": [],
                "intervals": [],
                "lambda_plus": [],
                "lambda_minus": [],
            }

        # Extract iterations and lambda values
        iterations = []
        lambda_plus_list = []
        lambda_minus_list = []
        interval_data: Dict[
            int, Dict[str, List[float]]
        ] = {}  # interval_idx -> {backbone: [], tilt: []}

        for r in rows:
            iter_no = int(r["iter_no"]) if r["iter_no"] is not None else 0
            iterations.append(iter_no)

            # Extract lambda values (same for all intervals in this iteration)
            if r["lambda_plus"] is not None:
                lambda_plus_list.append(float(r["lambda_plus"]))
            else:
                lambda_plus_list.append(None)
            if r["lambda_minus"] is not None:
                lambda_minus_list.append(float(r["lambda_minus"]))
            else:
                lambda_minus_list.append(None)

            # Decode backbone and tilt arrays
            if r["backbone_data"] and r["tilt_data"]:
                backbone_vals = decode_f64_array(r["backbone_data"])
                tilt_vals = decode_f64_array(r["tilt_data"])

                # Group by interval index
                for interval_idx, (b_val, d_val) in enumerate(
                    zip(backbone_vals, tilt_vals)
                ):
                    if interval_idx not in interval_data:
                        interval_data[interval_idx] = {"backbone": [], "tilt": []}
                    interval_data[interval_idx]["backbone"].append(b_val)
                    interval_data[interval_idx]["tilt"].append(d_val)

        # Convert interval_data to list format
        intervals = [
            {
                "interval_idx": idx,
                "backbone": data["backbone"],
                "tilt": data["tilt"],
            }
            for idx, data in sorted(interval_data.items())
        ]

    finally:
        conn.close()

    return {
        "col": col,
        "iterations": iterations,
        "intervals": intervals,
        "lambda_plus": lambda_plus_list,
        "lambda_minus": lambda_minus_list,
    }


@app.get("/api/run/{run_id}/backbone_tilt_evolution_all_columns")
def backbone_tilt_evolution_all_columns(
    run_id: int,
    epoch: int = Query(...),
    tree_id: int = Query(...),
    start_iter: int = Query(0),
    end_iter: Optional[int] = Query(None),
) -> Dict[str, Any]:
    """Get backbone and tilt evolution for ALL columns simultaneously."""
    conn = get_connection()
    try:
        # Build query with optional end_iter filter
        if end_iter is not None:
            q = (
                "SELECT col, iter_no, backbone_data, tilt_data, lambda_plus, lambda_minus "
                "FROM component_states "
                "WHERE run_id=? AND epoch=? AND tree_id=? AND iter_no>=? AND iter_no<=? "
                "ORDER BY col, iter_no"
            )
            rows = conn.execute(
                q, (run_id, epoch, tree_id, start_iter, end_iter)
            ).fetchall()
        else:
            q = (
                "SELECT col, iter_no, backbone_data, tilt_data, lambda_plus, lambda_minus "
                "FROM component_states "
                "WHERE run_id=? AND epoch=? AND tree_id=? AND iter_no>=? "
                "ORDER BY col, iter_no"
            )
            rows = conn.execute(q, (run_id, epoch, tree_id, start_iter)).fetchall()

        if not rows:
            conn.close()
            return {
                "epoch": epoch,
                "tree_id": tree_id,
                "iterations": [],
                "lambda_plus": [],
                "lambda_minus": [],
                "columns": [],
            }

        # Extract unique iterations and lambda values (from first row per iteration)
        iterations_set = set()
        lambda_by_iter: Dict[
            int, Dict[str, Optional[float]]
        ] = {}  # iter_no -> {lambda_plus, lambda_minus}
        col_data: Dict[
            int, Dict[int, Dict[str, List[float]]]
        ] = {}  # col -> interval_idx -> {backbone: [], tilt: []}

        for r in rows:
            col = int(r["col"]) if r["col"] is not None else 0
            iter_no = int(r["iter_no"]) if r["iter_no"] is not None else 0
            iterations_set.add(iter_no)

            # Store lambda values (same for all columns in this iteration)
            if iter_no not in lambda_by_iter:
                lambda_by_iter[iter_no] = {
                    "lambda_plus": float(r["lambda_plus"])
                    if r["lambda_plus"] is not None
                    else None,
                    "lambda_minus": float(r["lambda_minus"])
                    if r["lambda_minus"] is not None
                    else None,
                }

            # Decode backbone and tilt arrays
            if r["backbone_data"] and r["tilt_data"]:
                backbone_vals = decode_f64_array(r["backbone_data"])
                tilt_vals = decode_f64_array(r["tilt_data"])

                if col not in col_data:
                    col_data[col] = {}

                # Group by interval index
                for interval_idx, (b_val, d_val) in enumerate(
                    zip(backbone_vals, tilt_vals)
                ):
                    if interval_idx not in col_data[col]:
                        col_data[col][interval_idx] = {"backbone": [], "tilt": []}
                    col_data[col][interval_idx]["backbone"].append(b_val)
                    col_data[col][interval_idx]["tilt"].append(d_val)

        iterations = sorted(list(iterations_set))
        lambda_plus_list = [
            lambda_by_iter.get(iter, {}).get("lambda_plus") for iter in iterations
        ]
        lambda_minus_list = [
            lambda_by_iter.get(iter, {}).get("lambda_minus") for iter in iterations
        ]

        # Convert col_data to list format
        columns = []
        for col in sorted(col_data.keys()):
            intervals = [
                {
                    "interval_idx": idx,
                    "backbone": data["backbone"],
                    "tilt": data["tilt"],
                }
                for idx, data in sorted(col_data[col].items())
            ]
            columns.append({"col": col, "intervals": intervals})

    finally:
        conn.close()

    return {
        "epoch": epoch,
        "tree_id": tree_id,
        "iterations": iterations,
        "lambda_plus": lambda_plus_list,
        "lambda_minus": lambda_minus_list,
        "columns": columns,
    }


@app.get("/api/run/{run_id}/f_component_per_axis_evolution")
def f_component_per_axis_evolution(
    run_id: int,
    epoch: int = Query(...),
    tree_id: int = Query(...),
    start_iter: int = Query(0),
    end_iter: Optional[int] = Query(None),
) -> Dict[str, Any]:
    """Get per-axis f+ and f- factor contributions evolution over iterations.

    Returns:
      {
        "iterations": [iter0, iter1, ...],
        "axes": [
           {
             "col": int,
             "intervals": [[a,b], ...],
             "intervals_plus": [ { "interval_idx": i, "values": [v_per_iter...] }, ... ],
             "intervals_minus": [ { "interval_idx": i, "values": [v_per_iter...] }, ... ]
           }
        ]
      }
    """
    conn = get_connection()
    try:
        if end_iter is not None:
            q = (
                "SELECT iter_no, col, backbone_data, tilt_data, data, intervals_count "
                "FROM component_states "
                "WHERE run_id=? AND epoch=? AND tree_id=? AND iter_no>=? AND iter_no<=? "
                "ORDER BY iter_no"
            )
            rows = conn.execute(
                q, (run_id, epoch, tree_id, start_iter, end_iter)
            ).fetchall()
        else:
            q = (
                "SELECT iter_no, col, backbone_data, tilt_data, data, intervals_count "
                "FROM component_states "
                "WHERE run_id=? AND epoch=? AND tree_id=? AND iter_no>=? "
                "ORDER BY iter_no"
            )
            rows = conn.execute(q, (run_id, epoch, tree_id, start_iter)).fetchall()

        if not rows:
            conn.close()
            return {"iterations": [], "axes": []}

        iterations_set = []
        # axes_map: col -> { intervals: [[a,b],...], interval_plus_map: idx -> [vals...], interval_minus_map: idx -> [vals...] }
        axes_map: Dict[int, Dict[str, Any]] = {}

        for r in rows:
            iter_no = int(r["iter_no"]) if r["iter_no"] is not None else 0
            if iter_no not in iterations_set:
                iterations_set.append(iter_no)

            col = int(r["col"]) if r["col"] is not None else 0

            # Decode intervals boundaries from data blob (backward compat) if we don't have them yet
            intervals = None
            if r["data"]:
                import numpy as np

                arr = np.frombuffer(r["data"], dtype=np.float64)
                intervals_count = (
                    r["intervals_count"] if r["intervals_count"] is not None else 0
                )
                if arr.size == intervals_count * 3:
                    starts = arr[0:intervals_count]
                    ends = arr[intervals_count : 2 * intervals_count]
                    intervals = [
                        [float(a), float(b)]
                        for a, b in zip(starts.tolist(), ends.tolist())
                    ]

            # Decode backbone/tilt arrays and compute factor_plus/factor_minus for this iteration
            if r["backbone_data"] and r["tilt_data"]:
                backbone_vals = decode_f64_array(r["backbone_data"])
                tilt_vals = decode_f64_array(r["tilt_data"])
                import math

                factor_plus = [
                    b * math.exp(d) for b, d in zip(backbone_vals, tilt_vals)
                ]
                factor_minus = [
                    b * math.exp(-d) for b, d in zip(backbone_vals, tilt_vals)
                ]
            else:
                factor_plus = []
                factor_minus = []

            # Initialize axis entry if needed
            if col not in axes_map:
                axes_map[col] = {
                    "intervals": intervals if intervals is not None else [],
                    "intervals_plus_map": {},  # idx -> [vals...]
                    "intervals_minus_map": {},
                }

            axis_entry = axes_map[col]
            # If we discovered intervals this row and axis_entry has none, set it
            if (not axis_entry["intervals"]) and intervals is not None:
                axis_entry["intervals"] = intervals

            # Append factor values per interval index
            for idx in range(max(len(factor_plus), len(factor_minus))):
                # Ensure lists exist
                if idx not in axis_entry["intervals_plus_map"]:
                    axis_entry["intervals_plus_map"][idx] = []
                if idx not in axis_entry["intervals_minus_map"]:
                    axis_entry["intervals_minus_map"][idx] = []

                axis_entry["intervals_plus_map"][idx].append(
                    factor_plus[idx] if idx < len(factor_plus) else None
                )
                axis_entry["intervals_minus_map"][idx].append(
                    factor_minus[idx] if idx < len(factor_minus) else None
                )

        iterations = sorted(iterations_set)

        axes_out: List[Dict[str, Any]] = []
        for col, entry in sorted(axes_map.items()):
            intervals = entry.get("intervals", [])
            intervals_plus = []
            intervals_minus = []
            for idx in sorted(entry["intervals_plus_map"].keys()):
                intervals_plus.append(
                    {"interval_idx": idx, "values": entry["intervals_plus_map"][idx]}
                )
            for idx in sorted(entry["intervals_minus_map"].keys()):
                intervals_minus.append(
                    {"interval_idx": idx, "values": entry["intervals_minus_map"][idx]}
                )

            axes_out.append(
                {
                    "col": col,
                    "intervals": intervals,
                    "intervals_plus": intervals_plus,
                    "intervals_minus": intervals_minus,
                }
            )

        return {"iterations": iterations, "axes": axes_out}
    finally:
        conn.close()


@app.get("/api/run/{run_id}/f_component_evolution")
def f_component_evolution(
    run_id: int,
    epoch: int = Query(...),
    tree_id: int = Query(...),
    start_iter: int = Query(0),
    end_iter: Optional[int] = Query(None),
) -> Dict[str, Any]:
    """Get f+ and f- statistics evolution over iterations."""
    conn = get_connection()
    try:
        # Build query with optional end_iter filter
        if end_iter is not None:
            q = (
                "SELECT iter_no, component, min_val, max_val, mean_val, std_val, p25, p50, p75, p95, p99, n_samples "
                "FROM f_component_stats "
                "WHERE run_id=? AND epoch=? AND tree_id=? AND iter_no>=? AND iter_no<=? "
                "ORDER BY iter_no, component"
            )
            rows = conn.execute(
                q, (run_id, epoch, tree_id, start_iter, end_iter)
            ).fetchall()
        else:
            q = (
                "SELECT iter_no, component, min_val, max_val, mean_val, std_val, p25, p50, p75, p95, p99, n_samples "
                "FROM f_component_stats "
                "WHERE run_id=? AND epoch=? AND tree_id=? AND iter_no>=? "
                "ORDER BY iter_no, component"
            )
            rows = conn.execute(q, (run_id, epoch, tree_id, start_iter)).fetchall()

        if not rows:
            return {
                "iterations": [],
                "f_plus": {},
                "f_minus": {},
            }

        # Group by iteration and component
        iterations_set = set()
        f_plus_data: Dict[int, Dict[str, Any]] = {}  # iter_no -> {min, max, mean, ...}
        f_minus_data: Dict[int, Dict[str, Any]] = {}  # iter_no -> {min, max, mean, ...}

        for r in rows:
            iter_no = int(r["iter_no"]) if r["iter_no"] is not None else 0
            component = str(r["component"]).lower()
            iterations_set.add(iter_no)

            stats = {
                "min": float(r["min_val"]) if r["min_val"] is not None else 0.0,
                "max": float(r["max_val"]) if r["max_val"] is not None else 0.0,
                "mean": float(r["mean_val"]) if r["mean_val"] is not None else 0.0,
                "std": float(r["std_val"]) if r["std_val"] is not None else None,
                "p25": float(r["p25"]) if r["p25"] is not None else None,
                "p50": float(r["p50"]) if r["p50"] is not None else None,
                "p75": float(r["p75"]) if r["p75"] is not None else None,
                "p95": float(r["p95"]) if r["p95"] is not None else None,
                "p99": float(r["p99"]) if r["p99"] is not None else None,
                "n_samples": int(r["n_samples"]) if r["n_samples"] is not None else 0,
            }

            if component == "f_plus":
                f_plus_data[iter_no] = stats
            elif component == "f_minus":
                f_minus_data[iter_no] = stats

        iterations = sorted(list(iterations_set))

        # Build time-series arrays
        f_plus = {
            "min": [f_plus_data.get(iter, {}).get("min", 0.0) for iter in iterations],
            "max": [f_plus_data.get(iter, {}).get("max", 0.0) for iter in iterations],
            "mean": [f_plus_data.get(iter, {}).get("mean", 0.0) for iter in iterations],
            "std": [f_plus_data.get(iter, {}).get("std") for iter in iterations],
            "p25": [f_plus_data.get(iter, {}).get("p25") for iter in iterations],
            "p50": [f_plus_data.get(iter, {}).get("p50") for iter in iterations],
            "p75": [f_plus_data.get(iter, {}).get("p75") for iter in iterations],
            "p95": [f_plus_data.get(iter, {}).get("p95") for iter in iterations],
            "p99": [f_plus_data.get(iter, {}).get("p99") for iter in iterations],
        }

        f_minus = {
            "min": [f_minus_data.get(iter, {}).get("min", 0.0) for iter in iterations],
            "max": [f_minus_data.get(iter, {}).get("max", 0.0) for iter in iterations],
            "mean": [
                f_minus_data.get(iter, {}).get("mean", 0.0) for iter in iterations
            ],
            "std": [f_minus_data.get(iter, {}).get("std") for iter in iterations],
            "p25": [f_minus_data.get(iter, {}).get("p25") for iter in iterations],
            "p50": [f_minus_data.get(iter, {}).get("p50") for iter in iterations],
            "p75": [f_minus_data.get(iter, {}).get("p75") for iter in iterations],
            "p95": [f_minus_data.get(iter, {}).get("p95") for iter in iterations],
            "p99": [f_minus_data.get(iter, {}).get("p99") for iter in iterations],
        }

        return {
            "iterations": iterations,
            "f_plus": f_plus,
            "f_minus": f_minus,
        }
    except Exception as e:
        import traceback

        print(f"Error in f_component_evolution: {e}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch f+/f- component evolution: {str(e)}",
        )
    finally:
        conn.close()


@app.get("/api/run/{run_id}/component_decomposition")
def component_decomposition(
    run_id: int,
    epoch: int = Query(...),
    tree_id: int = Query(...),
    iter_no: int = Query(...),
) -> Dict[str, Any]:
    """Get full component decomposition (f+, f-, backbone, tilt) for a specific iteration."""
    conn = get_connection()
    try:
        # Get component states (backbone/tilt) for all columns
        cs_rows = conn.execute(
            (
                "SELECT col, data, intervals_count, backbone_data, tilt_data, lambda_plus, lambda_minus "
                "FROM component_states "
                "WHERE run_id=? AND epoch=? AND tree_id=? AND iter_no=?"
            ),
            (run_id, epoch, tree_id, iter_no),
        ).fetchall()

        # Get f+/f- statistics
        f_stats_rows = conn.execute(
            (
                "SELECT component, min_val, max_val, mean_val, std_val, p25, p50, p75, p95, p99, n_samples "
                "FROM f_component_stats "
                "WHERE run_id=? AND epoch=? AND tree_id=? AND iter_no=?"
            ),
            (run_id, epoch, tree_id, iter_no),
        ).fetchall()

        if not cs_rows:
            conn.close()
            return {
                "iter_no": iter_no,
                "lambda_plus": None,
                "lambda_minus": None,
                "components": [],
            }

        # Extract lambda values (from first row, should be same for all)
        lambda_plus = None
        lambda_minus = None
        if cs_rows[0]["lambda_plus"] is not None:
            lambda_plus = float(cs_rows[0]["lambda_plus"])
        if cs_rows[0]["lambda_minus"] is not None:
            lambda_minus = float(cs_rows[0]["lambda_minus"])

        # Process component states
        components = []
        for r in cs_rows:
            col = int(r["col"]) if r["col"] is not None else 0

            # Decode backbone and tilt
            backbone: List[float] = []
            tilt: List[float] = []
            if r["backbone_data"] and r["tilt_data"]:
                backbone = decode_f64_array(r["backbone_data"])
                tilt = decode_f64_array(r["tilt_data"])

            intervals_count = (
                int(r["intervals_count"])
                if r["intervals_count"] is not None
                else (len(backbone) if backbone else 0)
            )
            intervals = _decode_intervals_from_component_blob(
                r["data"], intervals_count
            )

            # Get f+/f- stats for this iteration
            f_plus_stats = None
            f_minus_stats = None
            for f_row in f_stats_rows:
                component = str(f_row["component"]).lower()
                stats = {
                    "min": float(f_row["min_val"])
                    if f_row["min_val"] is not None
                    else 0.0,
                    "max": float(f_row["max_val"])
                    if f_row["max_val"] is not None
                    else 0.0,
                    "mean": float(f_row["mean_val"])
                    if f_row["mean_val"] is not None
                    else 0.0,
                    "std": float(f_row["std_val"])
                    if f_row["std_val"] is not None
                    else None,
                    "p25": float(f_row["p25"]) if f_row["p25"] is not None else None,
                    "p50": float(f_row["p50"]) if f_row["p50"] is not None else None,
                    "p75": float(f_row["p75"]) if f_row["p75"] is not None else None,
                    "p95": float(f_row["p95"]) if f_row["p95"] is not None else None,
                    "p99": float(f_row["p99"]) if f_row["p99"] is not None else None,
                    "n_samples": int(f_row["n_samples"])
                    if f_row["n_samples"] is not None
                    else 0,
                }
                if component == "f_plus":
                    f_plus_stats = stats
                elif component == "f_minus":
                    f_minus_stats = stats

            components.append(
                {
                    "col": col,
                    "intervals": intervals,
                    "backbone": backbone,
                    "tilt": tilt,
                    "f_plus_stats": f_plus_stats,
                    "f_minus_stats": f_minus_stats,
                }
            )

    finally:
        conn.close()

    return {
        "iter_no": iter_no,
        "lambda_plus": lambda_plus,
        "lambda_minus": lambda_minus,
        "components": components,
    }


@app.get("/api/run/{run_id}/component_decomposition_multi")
def component_decomposition_multi(
    run_id: int,
    epoch: int = Query(...),
    iter_no: int = Query(...),
    tree_ids: str = Query(""),
    max_trees: int = Query(200),
) -> Dict[str, Any]:
    """Multi-tree version of /component_decomposition for merged dashboards.

    For each requested tree, returns the latest component_states snapshot at or before `iter_no`.
    """
    trees = _parse_tree_ids_csv(tree_ids, max_trees=max_trees)
    if not trees:
        return {"epoch": epoch, "iter_no": iter_no, "trees": []}

    conn = get_connection()
    try:
        placeholders = ",".join(["?"] * len(trees))
        rows = conn.execute(
            (
                f"SELECT tree_id, MAX(iter_no) AS iter_no "
                f"FROM component_states "
                f"WHERE run_id=? AND epoch=? AND tree_id IN ({placeholders}) AND iter_no<=? "
                f"GROUP BY tree_id"
            ),
            (run_id, epoch, *trees, iter_no),
        ).fetchall()
        tree_iter_map = {
            int(r["tree_id"]): int(r["iter_no"])
            for r in rows
            if r["iter_no"] is not None
        }
        if not tree_iter_map:
            return {"epoch": epoch, "iter_no": iter_no, "trees": []}

        # CTE for (tree_id, iter_no) pairs to fetch in one query.
        pairs = [(t, tree_iter_map[t]) for t in trees if t in tree_iter_map]
        values_sql = ",".join(["(?,?)"] * len(pairs))
        params: List[Any] = []
        for t, it in pairs:
            params.extend([t, it])
        params.extend([run_id, epoch])

        cs_rows = conn.execute(
            (
                f"WITH tree_iters(tree_id, iter_no) AS (VALUES {values_sql}) "
                f"SELECT cs.tree_id, cs.iter_no, cs.col, cs.data, cs.intervals_count, "
                f"cs.backbone_data, cs.tilt_data, cs.lambda_plus, cs.lambda_minus "
                f"FROM component_states cs "
                f"JOIN tree_iters ti ON cs.tree_id=ti.tree_id AND cs.iter_no=ti.iter_no "
                f"WHERE cs.run_id=? AND cs.epoch=? "
                f"ORDER BY cs.tree_id, cs.col"
            ),
            params,
        ).fetchall()

        out_by_tree: Dict[int, Dict[str, Any]] = {}
        for r in cs_rows:
            t_id = int(r["tree_id"]) if r["tree_id"] is not None else 0
            t_entry = out_by_tree.get(t_id)
            if not t_entry:
                t_entry = {
                    "tree_id": t_id,
                    "iter_no": int(r["iter_no"])
                    if r["iter_no"] is not None
                    else tree_iter_map.get(t_id, iter_no),
                    "lambda_plus": float(r["lambda_plus"])
                    if r["lambda_plus"] is not None
                    else None,
                    "lambda_minus": float(r["lambda_minus"])
                    if r["lambda_minus"] is not None
                    else None,
                    "components": [],
                }
                out_by_tree[t_id] = t_entry

            col = int(r["col"]) if r["col"] is not None else 0
            backbone = (
                decode_f64_array(r["backbone_data"]) if r["backbone_data"] else []
            )
            tilt = decode_f64_array(r["tilt_data"]) if r["tilt_data"] else []
            intervals_count = (
                int(r["intervals_count"])
                if r["intervals_count"] is not None
                else (len(backbone) if backbone else 0)
            )
            intervals = _decode_intervals_from_component_blob(
                r["data"], intervals_count
            )

            t_entry["components"].append(
                {
                    "col": col,
                    "intervals": intervals,
                    "backbone": backbone,
                    "tilt": tilt,
                }
            )

        # Preserve caller order where possible
        ordered = [out_by_tree[t] for t in trees if t in out_by_tree]
        return {"epoch": epoch, "iter_no": iter_no, "trees": ordered}
    finally:
        conn.close()


@app.get("/api/run/{run_id}/f_component_per_axis_multi")
def f_component_per_axis_multi(
    run_id: int,
    epoch: int = Query(...),
    iter_no: int = Query(...),
    tree_ids: str = Query(""),
    max_trees: int = Query(200),
) -> Dict[str, Any]:
    """Multi-tree version of /f_component_per_axis for merged dashboards.

    For each requested tree, returns the latest component_states snapshot at or before `iter_no`
    and computes per-interval f+ and f- factors from backbone/tilt.
    """
    trees = _parse_tree_ids_csv(tree_ids, max_trees=max_trees)
    if not trees:
        return {"epoch": epoch, "iter_no": iter_no, "trees": []}

    conn = get_connection()
    try:
        placeholders = ",".join(["?"] * len(trees))
        rows = conn.execute(
            (
                f"SELECT tree_id, MAX(iter_no) AS iter_no "
                f"FROM component_states "
                f"WHERE run_id=? AND epoch=? AND tree_id IN ({placeholders}) AND iter_no<=? "
                f"GROUP BY tree_id"
            ),
            (run_id, epoch, *trees, iter_no),
        ).fetchall()
        tree_iter_map = {
            int(r["tree_id"]): int(r["iter_no"])
            for r in rows
            if r["iter_no"] is not None
        }
        if not tree_iter_map:
            return {"epoch": epoch, "iter_no": iter_no, "trees": []}

        pairs = [(t, tree_iter_map[t]) for t in trees if t in tree_iter_map]
        values_sql = ",".join(["(?,?)"] * len(pairs))
        params: List[Any] = []
        for t, it in pairs:
            params.extend([t, it])
        params.extend([run_id, epoch])

        cs_rows = conn.execute(
            (
                f"WITH tree_iters(tree_id, iter_no) AS (VALUES {values_sql}) "
                f"SELECT cs.tree_id, cs.iter_no, cs.col, cs.data, cs.intervals_count, "
                f"cs.backbone_data, cs.tilt_data, cs.lambda_plus, cs.lambda_minus "
                f"FROM component_states cs "
                f"JOIN tree_iters ti ON cs.tree_id=ti.tree_id AND cs.iter_no=ti.iter_no "
                f"WHERE cs.run_id=? AND cs.epoch=? "
                f"ORDER BY cs.tree_id, cs.col"
            ),
            params,
        ).fetchall()

        out_by_tree: Dict[int, Dict[str, Any]] = {}
        for r in cs_rows:
            t_id = int(r["tree_id"]) if r["tree_id"] is not None else 0
            t_entry = out_by_tree.get(t_id)
            if not t_entry:
                t_entry = {
                    "tree_id": t_id,
                    "iter_no": int(r["iter_no"])
                    if r["iter_no"] is not None
                    else tree_iter_map.get(t_id, iter_no),
                    "lambda_plus": float(r["lambda_plus"])
                    if r["lambda_plus"] is not None
                    else None,
                    "lambda_minus": float(r["lambda_minus"])
                    if r["lambda_minus"] is not None
                    else None,
                    "axes": [],
                }
                out_by_tree[t_id] = t_entry

            col = int(r["col"]) if r["col"] is not None else 0
            backbone = (
                decode_f64_array(r["backbone_data"]) if r["backbone_data"] else []
            )
            tilt = decode_f64_array(r["tilt_data"]) if r["tilt_data"] else []
            intervals_count = (
                int(r["intervals_count"])
                if r["intervals_count"] is not None
                else (len(backbone) if backbone else 0)
            )
            intervals = _decode_intervals_from_component_blob(
                r["data"], intervals_count
            )

            intervals_plus: List[List[Any]] = []
            intervals_minus: List[List[Any]] = []
            for (a, b), bb, dd in zip(intervals, backbone, tilt):
                b_f = _finite_or_none(bb)
                ep = _safe_exp(float(dd))
                em = _safe_exp(float(-dd))
                fp = (b_f * ep) if (b_f is not None and ep is not None) else None
                fm = (b_f * em) if (b_f is not None and em is not None) else None
                intervals_plus.append([a, b, fp])
                intervals_minus.append([a, b, fm])

            t_entry["axes"].append(
                {
                    "col": col,
                    "intervals_plus": intervals_plus,
                    "intervals_minus": intervals_minus,
                }
            )

        ordered = [out_by_tree[t] for t in trees if t in out_by_tree]
        return {"epoch": epoch, "iter_no": iter_no, "trees": ordered}
    finally:
        conn.close()


@app.get("/api/run/{run_id}/f_component_per_axis")
def f_component_per_axis(
    run_id: int,
    epoch: int = Query(...),
    tree_id: int = Query(...),
    iter_no: int = Query(...),
) -> Dict[str, Any]:
    """Get f+ and f- factor contributions per axis at a specific iteration.

    Returns per-axis factor contributions:
    - factor_plus_j = b_j * exp(d_j) for each axis j
    - factor_minus_j = b_j * exp(-d_j) for each axis j
    - Also includes aggregated f+ and f- statistics
    """
    conn = get_connection()
    try:
        # Get component states (backbone/tilt) for all columns
        cs_rows = conn.execute(
            (
                "SELECT col, data, backbone_data, tilt_data, intervals_count, lambda_plus, lambda_minus "
                "FROM component_states "
                "WHERE run_id=? AND epoch=? AND tree_id=? AND iter_no=? "
                "ORDER BY col"
            ),
            (run_id, epoch, tree_id, iter_no),
        ).fetchall()

        # Get aggregated f+/f- statistics
        f_stats_rows = conn.execute(
            (
                "SELECT component, min_val, max_val, mean_val, std_val, p25, p50, p75, p95, p99, n_samples "
                "FROM f_component_stats "
                "WHERE run_id=? AND epoch=? AND tree_id=? AND iter_no=?"
            ),
            (run_id, epoch, tree_id, iter_no),
        ).fetchall()

        if not cs_rows:
            conn.close()
            return {
                "iter_no": iter_no,
                "lambda_plus": None,
                "lambda_minus": None,
                "f_plus_stats": None,
                "f_minus_stats": None,
                "axes": [],
            }

        # Extract lambda values
        lambda_plus = None
        lambda_minus = None
        if cs_rows[0]["lambda_plus"] is not None:
            lambda_plus = float(cs_rows[0]["lambda_plus"])
        if cs_rows[0]["lambda_minus"] is not None:
            lambda_minus = float(cs_rows[0]["lambda_minus"])

        # Get aggregated f+/f- stats
        f_plus_stats = None
        f_minus_stats = None
        for f_row in f_stats_rows:
            component = str(f_row["component"]).lower()
            stats = {
                "min": float(f_row["min_val"]) if f_row["min_val"] is not None else 0.0,
                "max": float(f_row["max_val"]) if f_row["max_val"] is not None else 0.0,
                "mean": float(f_row["mean_val"])
                if f_row["mean_val"] is not None
                else 0.0,
                "std": float(f_row["std_val"])
                if f_row["std_val"] is not None
                else None,
                "p25": float(f_row["p25"]) if f_row["p25"] is not None else None,
                "p50": float(f_row["p50"]) if f_row["p50"] is not None else None,
                "p75": float(f_row["p75"]) if f_row["p75"] is not None else None,
                "p95": float(f_row["p95"]) if f_row["p95"] is not None else None,
                "p99": float(f_row["p99"]) if f_row["p99"] is not None else None,
                "n_samples": int(f_row["n_samples"])
                if f_row["n_samples"] is not None
                else 0,
            }
            if component == "f_plus":
                f_plus_stats = stats
            elif component == "f_minus":
                f_minus_stats = stats

        # Process per-axis data
        axes = []
        for r in cs_rows:
            col = int(r["col"]) if r["col"] is not None else 0

            # Decode backbone and tilt
            backbone: List[float] = []
            tilt: List[float] = []
            intervals_count = (
                int(r["intervals_count"]) if r["intervals_count"] is not None else 0
            )
            intervals = _decode_intervals_from_component_blob(
                r["data"], intervals_count
            )

            if r["backbone_data"] and r["tilt_data"]:
                backbone = decode_f64_array(r["backbone_data"])
                tilt = decode_f64_array(r["tilt_data"])

                # Compute per-interval factor contributions
                factor_plus: List[Optional[float]] = []
                factor_minus: List[Optional[float]] = []
                for b, d in zip(backbone, tilt):
                    bb = _finite_or_none(b)
                    edp = _safe_exp(float(d))
                    edm = _safe_exp(float(-d))
                    factor_plus.append(
                        (bb * edp) if (bb is not None and edp is not None) else None
                    )
                    factor_minus.append(
                        (bb * edm) if (bb is not None and edm is not None) else None
                    )

                # Create intervals with values: [[a, b, val], ...]
                intervals_plus = []
                intervals_minus = []

                for i, (interval, fp, fm) in enumerate(
                    zip(intervals, factor_plus, factor_minus)
                ):
                    if len(interval) >= 2:
                        intervals_plus.append([interval[0], interval[1], fp])
                        intervals_minus.append([interval[0], interval[1], fm])

                # Compute statistics for this axis (finite-only, nullable)
                fp_fin = [v for v in factor_plus if v is not None]
                fm_fin = [v for v in factor_minus if v is not None]
                f_plus_stats_axis = None
                f_minus_stats_axis = None
                if fp_fin:
                    fp_sorted = sorted(fp_fin)
                    f_plus_stats_axis = {
                        "min": min(fp_fin),
                        "max": max(fp_fin),
                        "mean": sum(fp_fin) / len(fp_fin),
                        "p50": fp_sorted[len(fp_sorted) // 2],
                    }
                if fm_fin:
                    fm_sorted = sorted(fm_fin)
                    f_minus_stats_axis = {
                        "min": min(fm_fin),
                        "max": max(fm_fin),
                        "mean": sum(fm_fin) / len(fm_fin),
                        "p50": fm_sorted[len(fm_sorted) // 2],
                    }

                axes.append(
                    {
                        "col": col,
                        "backbone": backbone,
                        "tilt": tilt,
                        "intervals": intervals,
                        "intervals_plus": intervals_plus,
                        "intervals_minus": intervals_minus,
                        "factor_plus": factor_plus,
                        "factor_minus": factor_minus,
                        "f_plus_stats": f_plus_stats_axis,
                        "f_minus_stats": f_minus_stats_axis,
                    }
                )

    except Exception as e:
        import traceback

        print(f"Error in f_component_per_axis: {e}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch f+/- per axis: {str(e)}"
        )
    finally:
        conn.close()

    return {
        "iter_no": iter_no,
        "lambda_plus": lambda_plus,
        "lambda_minus": lambda_minus,
        "f_plus_stats": f_plus_stats,
        "f_minus_stats": f_minus_stats,
        "axes": axes,
    }


@app.get("/api/run/{run_id}/tensor_lambdas")
def tensor_lambdas(
    run_id: int,
    epoch: int = Query(...),
    max_trees: int = Query(500, ge=1),
) -> Dict[str, Any]:
    """Return lambda+/lambda- for the latest snapshot of each tree in the epoch."""

    conn = get_connection()
    try:
        table_check = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='component_states'"
        ).fetchone()
        if not table_check:
            return {"epoch": epoch, "trees": []}

        rows = conn.execute(
            (
                "WITH latest AS ("
                "    SELECT tree_id, MAX(iter_no) AS iter_no "
                "    FROM component_states "
                "    WHERE run_id=? AND epoch=? AND tree_id IS NOT NULL "
                "    GROUP BY tree_id"
                ") "
                "SELECT cs.tree_id, cs.iter_no, "
                "       MAX(cs.lambda_plus) AS lambda_plus, "
                "       MAX(cs.lambda_minus) AS lambda_minus "
                "FROM component_states cs "
                "JOIN latest lt ON cs.tree_id = lt.tree_id AND cs.iter_no = lt.iter_no "
                "WHERE cs.run_id=? AND cs.epoch=? AND cs.tree_id IS NOT NULL "
                "GROUP BY cs.tree_id, cs.iter_no "
                "ORDER BY cs.tree_id "
                "LIMIT ?"
            ),
            (run_id, epoch, run_id, epoch, max_trees),
        ).fetchall()

        trees: List[Dict[str, Any]] = []
        for r in rows:
            tree_id = int(r["tree_id"]) if r["tree_id"] is not None else None
            if tree_id is None:
                continue
            trees.append(
                {
                    "tree_id": tree_id,
                    "iter_no": int(r["iter_no"]) if r["iter_no"] is not None else None,
                    "lambda_plus": float(r["lambda_plus"])
                    if r["lambda_plus"] is not None
                    else None,
                    "lambda_minus": float(r["lambda_minus"])
                    if r["lambda_minus"] is not None
                    else None,
                }
            )

        return {"epoch": epoch, "trees": trees}
    finally:
        conn.close()


@app.get("/api/run/{run_id}/combination_choice")
def combination_choice(
    run_id: int, epoch: int = Query(...)
) -> Optional[Dict[str, Any]]:
    """Return combination choice (best index + scored candidates) for a run/epoch if available."""
    conn = get_connection()
    try:
        # Query the summary table for this run/epoch (pick latest method if multiple)
        row = conn.execute(
            "SELECT method, best_index, candidates_json FROM combination_choices WHERE run_id=? AND epoch=? LIMIT 1",
            (run_id, epoch),
        ).fetchone()
        if not row:
            return None
        method = row["method"]
        best_index = row["best_index"]
        candidates_json = row["candidates_json"]
        try:
            candidates = json.loads(candidates_json) if candidates_json else []
        except Exception:
            candidates = []
        # Normalize types
        out_candidates = []
        for c in candidates:
            try:
                tid = int(
                    c.get("tree_id", c.get("tree") if isinstance(c, dict) else None)
                )
                score = float(c.get("score", 0.0))
                out_candidates.append({"tree_id": tid, "score": score})
            except Exception:
                continue
        return {
            "method": method,
            "best_index": best_index,
            "candidates": out_candidates,
        }
    finally:
        conn.close()


@app.get("/api/run/{run_id}/scalings")
def epoch_scalings(run_id: int) -> Dict[str, Any]:
    """Return scaling history for a run.

    Returns:
    {
        "epochs": [e0, e1, ...],
        "scalings": [{epoch: e, scaling: s, optimization_epoch: opt_e}, ...],
        "latest": {epoch: scaling, ...}  # Latest scaling for each epoch
    }
    """
    conn = get_connection()
    try:
        # Get all scalings ordered by epoch and optimization_epoch
        rows = conn.execute(
            """SELECT epoch, scaling, optimization_epoch 
               FROM epoch_scalings 
               WHERE run_id = ? 
               ORDER BY epoch, optimization_epoch""",
            (run_id,),
        ).fetchall()

        # Get latest scaling for each epoch
        latest_rows = conn.execute(
            """SELECT epoch, scaling 
               FROM epoch_scalings 
               WHERE (run_id, epoch, optimization_epoch) IN (
                   SELECT run_id, epoch, MAX(optimization_epoch) 
                   FROM epoch_scalings 
                   WHERE run_id = ? 
                   GROUP BY run_id, epoch
               )
               ORDER BY epoch""",
            (run_id,),
        ).fetchall()
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=404, detail=f"No epoch_scalings data: {e}")
    conn.close()

    epochs_set = set()
    scalings = []
    for r in rows:
        epoch = int(r["epoch"])
        epochs_set.add(epoch)
        scalings.append(
            {
                "epoch": epoch,
                "scaling": float(r["scaling"]),
                "optimization_epoch": int(r["optimization_epoch"]),
            }
        )

    epochs = sorted(list(epochs_set))
    latest = {int(r["epoch"]): float(r["scaling"]) for r in latest_rows}

    return {
        "epochs": epochs,
        "scalings": scalings,
        "latest": latest,
    }


@app.get("/api/combined/epoch/{epoch}")
def combined_grids_for_epoch(epoch: int) -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT run_id, epoch, energy, scaling, scaling_plus, scaling_minus, grid_json, f_plus_json, f_minus_json FROM combined_grids WHERE epoch = ? ORDER BY run_id",
            (epoch,),
        ).fetchall()
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=404, detail=f"No combined_grids data: {e}")

    # Get scalings for this epoch (latest optimization_epoch for each run_id)
    # This is for backward compatibility - prefer scaling_plus/scaling_minus from combined_grids
    scaling_rows = conn.execute(
        """SELECT run_id, epoch, scaling, optimization_epoch 
           FROM epoch_scalings 
           WHERE epoch = ? 
           AND (run_id, epoch, optimization_epoch) IN (
               SELECT run_id, epoch, MAX(optimization_epoch) 
               FROM epoch_scalings 
               WHERE epoch = ? 
               GROUP BY run_id, epoch
           )
           ORDER BY run_id""",
        (epoch, epoch),
    ).fetchall()

    # Create a mapping of run_id -> scaling (for backward compatibility)
    scaling_map = {int(r["run_id"]): float(r["scaling"]) for r in scaling_rows}

    conn.close()
    result: List[Dict[str, Any]] = []
    for r in rows:
        cell = r["grid_json"]
        try:
            parsed = json.loads(cell) if isinstance(cell, str) else cell
        except Exception:
            parsed = None

        # Parse f_plus and f_minus JSON arrays if available
        f_plus = None
        f_minus = None
        if r["f_plus_json"]:
            try:
                f_plus = json.loads(r["f_plus_json"])
            except Exception:
                pass
        if r["f_minus_json"]:
            try:
                f_minus = json.loads(r["f_minus_json"])
            except Exception:
                pass

        run_id = int(r["run_id"])
        result.append(
            {
                "run_id": run_id,
                "epoch": int(r["epoch"]) if r["epoch"] is not None else None,
                "energy": r["energy"],
                "scaling": r["scaling"]
                if r["scaling"] is not None
                else scaling_map.get(
                    run_id
                ),  # Use combined_grids.scaling if available, fallback to epoch_scalings
                "scaling_plus": r["scaling_plus"],
                "scaling_minus": r["scaling_minus"],
                "f_plus": f_plus,
                "f_minus": f_minus,
                "snapshot": parsed,
            }
        )
    return result


# Serve the frontend (modern, minimal single-page)
frontend_dir = _resolve_frontend_dir()
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


def main():
    """Entry point for the `tslviz` CLI command."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(
        prog="tslviz",
        description="Run the TSL split-event visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db",
        "--database",
        dest="database",
        required=False,
        help="Path to the SQLite database file (alternative to DATABASE_PATH env var)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8051,
        help="Port to bind to (default: 8051)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes",
    )

    args = parser.parse_args()

    # Set database path: if provided via CLI, set it as env var (so it persists to uvicorn subprocess)
    if args.database:
        db_path = os.path.abspath(args.database)
        if not os.path.exists(db_path):
            parser.error(f"Database file not found: {db_path}")
        os.environ["DATABASE_PATH"] = db_path
        print(f"Using database: {db_path}")
    elif not os.environ.get("DATABASE_PATH"):
        parser.error(
            "No database path provided. Either:\n"
            "  1. Use --db argument, or\n"
            "  2. Set DATABASE_PATH environment variable"
        )
    else:
        print(f"Using database from DATABASE_PATH: {os.environ['DATABASE_PATH']}")

    print(f"Starting server on {args.host}:{args.port}")

    uvicorn.run(
        "tslviz.backend.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
