import time
import veritas
import pandas as pd

import numpy as np
import zarr
import fsspec

from numba import njit
from tqdm import tqdm
from index import index


from veritas import (
    smt,
    Interval,
    Verifier,
    VerifierTimeout,
)
try:
    from veritas import Z3Backend as Backend
except ImportError:
    try:
        from veritas.smt import VerifierBackend as Backend
    except ImportError:
        Backend = None

#### HELPER FUNCTIONS ####
def zarr_metadata(base_path, storage_options=None):
    storage_options = storage_options or {}

    mapper = fsspec.get_mapper(base_path, **storage_options)
    root = zarr.open_group(store=mapper, mode="r")

    feat_map = root.attrs.get("feat_map", None)
    feat_map = np.array(feat_map) if feat_map is not None else None
    num_solutions = root.attrs.get("num_solutions", 0)

    return feat_map, num_solutions

def iter_zarr_shards(base_path, storage_options=None):
    """
    Yields (boxes, outvalues, metadata) for each shard in a sharded Zarr dataset.

    Parameters
    ----------
    base_path : str
        e.g. "run.zarr" or "sftp://user@host/path/run.zarr"
    storage_options : dict
        fsspec options (username/password/etc. for SFTP)

    Yields
    ------
    boxes : np.ndarray
    outvalues : np.ndarray
    meta : dict
        shard-level metadata (e.g. num_solutions)
    """

    storage_options = storage_options or {}

    # --- filesystem for listing shards ---
    fs, path = fsspec.url_to_fs(base_path, **storage_options)
    shard_dirs = sorted(fs.ls(path))

    join = lambda p, s: f"{p}/{s}" if not p.endswith("/") else f"{p}{s}"
    # --- iterate shards ---
    for shard in shard_dirs:
        if not shard.endswith(".zarr"):
            continue

        shard_path = join(base_path, shard.split("/")[-1])

        store = fsspec.get_mapper(shard_path, **storage_options)
        root = zarr.open_group(store, mode="r")

        boxes = root["boxes"][:]
        outvalues = root["outvalues"][:]


        yield boxes, outvalues

@njit(cache=True, fastmath=True)
def dist_to_boxes(example, boxes, feat_map):
    dist = np.zeros((boxes.shape[0],), dtype=np.float64)
    for s in range(boxes.shape[0]):
        max_dist = 0.0
        los = boxes[s, 0]
        his = boxes[s, 1]
        for idx, i in enumerate(feat_map):
            x = example[i]
            lo, hi = los[idx], his[idx]

            d = 0
            t = lo - x
            if t > 0.0:
                d = t
            t = x - hi
            if t > d:
                d = t

            if d > max_dist:
                max_dist = d
        dist[s] = max_dist
    return dist

@njit(cache=True, fastmath=True)
def min_dist_to_solutions(example, boxes, feat_map):
    min_dist = 1e18
    for s in range(boxes.shape[0]):
        max_dist = 0.0
        los = boxes[s, 0]
        his = boxes[s, 1]
        for idx, i in enumerate(feat_map):
            x = example[i]
            lo, hi = los[idx], his[idx]

            d = 0
            t = lo - x
            if t > 0.0:
                d = t
            t = x - hi
            if t > d:
                d = t

            if d > max_dist:
                max_dist = d
                if max_dist >= min_dist:
                    break
        if max_dist < min_dist:
            min_dist = max_dist
    return min_dist


#### NEAREST ADVERSARIAL EXAMPLE SEARCH METHODS ####

def approx_emp_robustness(at, max_delta, x, y, time_limit):
    t0 = time.time()
    
    delta_lo = np.zeros(x.shape[0])
    verification_times = np.zeros(x.shape[0])
    count = 0
    
    timeout = False
    for i, (example, target_label) in enumerate(zip(x, y)):
        t = time.time()
        if target_label:
            source_at, target_at = None, at
        else:
            source_at, target_at = at, None
        
        start_delta = max_delta
        rob = veritas.VeritasRobustnessSearch(
            example, start_delta, source_at, target_at, silent=True
        )
        _, _delta_lo, _ = rob.search()
        
        delta_lo[i] = _delta_lo
        verification_times[i] = time.time() - t
        count += 1

        if time.time() - t0 >= time_limit:
            timeout = True
            break

    return {'emp_rob': np.mean(delta_lo), 'emp_rob_n': count, 'emp_rob_time': time.time() - t0, 'individual_verification_times': verification_times.tolist(), 'individual_robustness_values': delta_lo.tolist(), 'timeout': timeout}

def exact_emp_robustness(at, x, y, time_limit):
    from gurobipy import GRB

    t0 = time.time()
    delta_lo = np.zeros(x.shape[0], np.float64)
    verification_times = np.zeros(x.shape[0])
    count = 0

    timeout = False
    for i, (example, target_label) in enumerate(zip(x, y)):
        t = time.time()
        kan = veritas.KantchelianAttack(at, target_label, example)
        kan.model.setParam(GRB.Param.TimeLimit, time_limit - (time.time() - t))
        kan.model.setParam(GRB.Param.Threads, 1)
        kan.optimize()
        
        try:
            delta_lo[i] = kan.bounds[-1][0]
        except IndexError:
            delta_lo[i] = 1e18
        count += 1
        verification_times[i] = time.time() - t
        if np.sum(verification_times) >= time_limit:
            timeout = True
            break

    return {'emp_rob': np.mean(delta_lo), 'emp_rob_n': count, 'emp_rob_time': time.time() - t0, 'individual_verification_times': verification_times.tolist(), 'individual_robustness_values': delta_lo.tolist(), 'timeout': timeout}

def emp_robustness_linear_scan(at, oc_file, storage_options, x, y, time_limit):
    t0 = time.time()

    shards = iter_zarr_shards(oc_file, storage_options=storage_options)
    feat_map, _ = zarr_metadata(oc_file, storage_options=storage_options)
    
    pni = index.PosNegIndex(at, feat_map)
    for boxes_batch, preds_batch in shards:
        pni.store(boxes_batch, preds_batch)
    pni.to_arrays()

    index_time = time.time() - t0
    t0 = time.time()
    
    delta_lo = np.zeros(x.shape[0], dtype=np.float64)
    verification_times = np.zeros(x.shape[0])
    count = 0

    timeout = False
    for i, (example, target_label) in enumerate(zip(x, y)):
        t = time.time()

        delta_lo[i] = pni.find_closest_adversarial_example(example, target_label)
        verification_times[i] = time.time() - t
        
        count += 1

        if time.time() - t0 >= time_limit:
            timeout = True
            break
    return {'emp_rob': np.mean(delta_lo), 'emp_rob_n': count, 'emp_rob_time': time.time() - t0, 'individual_verification_times': verification_times.tolist(), 'individual_robustness_values': delta_lo.tolist(), 'index_building_time': index_time, 'timeout': timeout}

def emp_robustness_rootbox_index(at, oc_file, storage_options, x, y, time_limit):
    t0 = time.time()

    shards = iter_zarr_shards(oc_file, storage_options=storage_options)
    feat_map, _ = zarr_metadata(oc_file, storage_options=storage_options)
    
    rbi = index.RootboxIndex(at, feat_map)
    for boxes_batch, preds_batch in shards:
        rbi.store(boxes_batch, preds_batch)
    rbi.to_arrays()
    
    index_time = time.time() - t0
    t0 = time.time()

    delta_lo = np.zeros(x.shape[0], dtype=np.float64)
    verification_times = np.zeros(x.shape[0])
    count = 0

    timeout = False
    for i, (example, target_label) in enumerate(zip(x, y)):
        t = time.time()

        delta_lo[i] = rbi.find_closest_adversarial_example(example, target_label)
        verification_times[i] = time.time() - t
        count += 1

        if time.time() - t0 >= time_limit:
            timeout = True
            break

    return {'emp_rob': np.mean(delta_lo), 'emp_rob_n': count, 'emp_rob_time': time.time() - t0, 'individual_verification_times': verification_times.tolist(), 'individual_robustness_values': delta_lo.tolist(), 'index_building_time': index_time, 'timeout': timeout}

def emp_robustness_octree_index(at, oc_file, storage_options, x, y, time_limit):
    t0 = time.time()
    shards = iter_zarr_shards(oc_file, storage_options=storage_options)
    feat_map, num_solutions = zarr_metadata(oc_file, storage_options=storage_options)

    oci = index.OCTreeIndex(at, feat_map)
    for boxes_batch, preds_batch in shards:
        oci.store(boxes_batch, preds_batch)
    oci.to_arrays()

    index_time = time.time() - t0
    t0 = time.time()

    delta_lo = np.zeros(x.shape[0], dtype=np.float64)
    verification_times = np.zeros(x.shape[0])
    count = 0

    timeout = False
    for i, (example, target_label) in enumerate(zip(x, y)):
        t = time.time()
        delta_lo[i] = oci.find_closest_adversarial_example(example, target_label)
        
        verification_times[i] = time.time() - t
        count += 1

        if time.time() - t0 >= time_limit:
            timeout = True
            break

    return {'emp_rob': np.mean(delta_lo), 'emp_rob_n': count, 'emp_rob_time': time.time() - t0, 'individual_verification_times': verification_times.tolist(), 'individual_robustness_values': delta_lo.tolist(), 'index_building_time': index_time, 'timeout': timeout}

def emp_robustness(at, x, y, n, method, time_limit, oc_file=None, storage_options=None):
    
    x_correct = []
    y_correct = []
    count = 0
    for i in x.index:
        target_label = not (y.loc[i] > 0.0)
        example = x.loc[i, :].to_numpy()
        pred_label = at.eval(example)[0, 0] > 0.0

        if pred_label != target_label: # Only consider examples that are correctly classified by the model, since for misclassified examples the robustness is 0 and they are not interesting for our evaluation
            x_correct.append(example)
            y_correct.append(target_label)

            count += 1
        if count >= n:
            break
    x_correct = np.array(x_correct)
    y_correct = np.array(y_correct)
    if method == 'exact':
        result = exact_emp_robustness(at, x_correct, y_correct, time_limit)
    elif method == 'approx':
        result = approx_emp_robustness(at, 1.0, x_correct, y_correct, time_limit)
    elif method == 'linear_scan':
        result = emp_robustness_linear_scan(at, oc_file, storage_options, x_correct, y_correct, time_limit)
    elif method == 'rootbox_index':
        result = emp_robustness_rootbox_index(at, oc_file, storage_options, x_correct, y_correct, time_limit)
    elif method == 'octree_index':
        result = emp_robustness_octree_index(at, oc_file, storage_options, x_correct, y_correct, time_limit)

    result['count'] = count
    return result


def run_adversarial_robustness_tasks(at, x, y, oc_file, storage_options, timeout, n):
    result = {}

    result['octree_index'] = emp_robustness( # Only use the OCTree index if we have more than 2 trees, otherwise it's just a linear scan.
        at, x, y, n, method='octree_index', time_limit=timeout, oc_file=oc_file, storage_options=storage_options
    ) if len(at) > 2 else emp_robustness(at, x, y, n, method='linear_scan', time_limit=timeout, oc_file=oc_file, storage_options=storage_options)

    result['rootbox_index'] = emp_robustness(
        at, x, y, n, method='rootbox_index', time_limit=timeout, oc_file=oc_file, storage_options=storage_options
    )

    result['linear_scan'] = emp_robustness(
        at, x, y, n, method='linear_scan', time_limit=timeout, oc_file=oc_file, storage_options=storage_options
    )
    
    result['kantchelian'] = emp_robustness(
        at, x, y, n, method='exact', time_limit=timeout
    )
    result['approx'] = emp_robustness(
        at, x, y, n, method='approx', time_limit=timeout
    )
    return result

#### FAIRNESS TESTS ####

def constrast_two_examples(at, columns, nonfixed_columns, silent=True):
    """Create a `veritas.AddTree` that contrast two instances.
    
    The new AddTree outputs difference between the original AddTree's outputs
    for instances 0 and 1.

    Essentially, we duplicate the original AddTree, but with a different feature index for the protected feature(s). 
    Then we concatenate the original and the duplicated AddTree, but with the duplicated one negated so that the 
    output is the difference between the two AddTrees (=T0 - T1). By having a different feature index for the protected 
    feature, we can check the output of the new AddTree for two instances that are identical except for the protected 
    feature. If the output is always zero, then we know that the ensemble is fair. So we let Veritas look for positive solutions 
    in this combined AddTree. A positive solution is an unfair region.

    at: The original veritas.AddTree tree ensemble model
    columns: array with column names
    nonfixed_columns: columns that are allowed to change between the two instances
    """
    feat_map = veritas.FeatMap(columns)
    for column in columns:
        if column not in nonfixed_columns:
            index_for_instance0 = feat_map.get_index(column, 0)
            index_for_instance1 = feat_map.get_index(column, 1)
            feat_map.use_same_id_for(index_for_instance0, index_for_instance1)

    at_for_instance1 = feat_map.transform(at, 1)
    at_for_instance1.set_base_score(0, at.get_base_score(0)) # <--- (!) BUG in feat_map.transform: base score not copied
    at_contrast = at.concat_negated(at_for_instance1)
    

    if not silent:
        print("  Feature IDs used by instance 0\n   and instance 1 respectively:")
        print("-"*(25+4+4))
        for column in columns:
            mark = "*" if column in nonfixed_columns else ""
            feat_id_instance0 = feat_map.get_feat_id(column, 0)
            feat_id_instance1 = feat_map.get_feat_id(column, 1)
            print(f"{column:25s} {feat_id_instance0:3d} {feat_id_instance1:3d}", mark)
    
    return at_contrast, feat_map

def contrasting_examples_from_solutions(feat_map, sol, columns):
    nb_features = len(feat_map)
    two_examples = np.zeros((2, nb_features))
    
    if isinstance(sol, veritas.Solution):
        box = sol.box()
    elif isinstance(sol, dict):
        box = sol

    for instance in (0, 1):
        for column in columns:
            feat_id = feat_map.get_feat_id(column, instance)
            feat_id_untrasformed = feat_map.get_feat_id(column, 0)
            if feat_id in box:
                interval = box[feat_id]
                if interval.lo_is_unbound():
                    assert not interval.hi_is_unbound()
                    value = interval.hi - 1e-4  # not inclusive
                else:
                    value = interval.lo
                # if feat_id_untrasformed in ordinal_feature_indexes:  # ordinal feature
                # value = np.floor(value)
                two_examples[instance, feat_id_untrasformed] = value

    return pd.DataFrame(two_examples, columns=columns, index=["instance 0", "instance 1"])

def veritas_fairness(at, columns, non_fixed_columns, timeout):
    t = time.time()
    at_contrast, feat_map = constrast_two_examples(at, columns=columns, nonfixed_columns=non_fixed_columns)
    config = veritas.Config(veritas.HeuristicType.MAX_OUTPUT)

    config.ignore_state_when_worse_than = 0.0
    config.focal_eps = 0.95
    config.max_focal_size = 100

    # try enumerating ALL violating boxes, not only the one with max output 
    config.stop_when_optimal = False
    config.stop_when_num_solutions_exceeds = int(1e10) # dummy large number 
    config.stop_when_num_new_solutions_exceeds = int(1e10) # dummy large number 
    
    config.max_memory = 100*1024*1024*1024

    search = config.get_search(at_contrast)

    t = time.time()
    bounds = []
    num_search_steps_per_iteration = 100
    stop_reason = veritas.StopReason.NONE
    with tqdm() as pbar:
        while stop_reason != veritas.StopReason.NO_MORE_OPEN: # search until all unfair pairs are found
            stop_reason = search.steps(num_search_steps_per_iteration)
            bound_lh = search.current_bounds()
            bounds.append((bound_lh.atleast, bound_lh.top_of_open))
            pbar.update(1)
            pbar.set_description(f"lower {bound_lh.atleast:.3f}, "
                                f"upper {bound_lh.top_of_open:.3f}, "
                                f"#sols {search.num_solutions():<4d}")
            
            if time.time() - t > timeout:
                print(f"Reached maximum time limit of {timeout} seconds: stopping.")
                stop_reason = veritas.StopReason.OUT_OF_TIME
                break
        
    tsearch = time.time() - t

    # HACK remove solutions with:
    # 1) identical instances
    # 2) both instances belonging to the same class
    violating_regions = []
    same_class, identical_instances = 0, 0
    for i in range(search.num_solutions()):
        sol = search.get_solution(i)
        two_examples = contrasting_examples_from_solutions(feat_map, sol, columns)
        instance0 = two_examples.iloc[0, :].to_numpy()
        instance1 = two_examples.iloc[1, :].to_numpy()
        pred0 = at.predict(instance0.reshape(1, -1))[0]
        pred1 = at.predict(instance1.reshape(1, -1))[0]
        class0 = 1 if pred0 > 0.5 else 0
        class1 = 1 if pred1 > 0.5 else 0
        if np.allclose(instance0, instance1):
            identical_instances += 1
        elif class0 == class1:
            same_class += 1
        else:
            violating_regions.append(sol)


    return {"isfair": len(violating_regions) == 0, "time": tsearch, "num_violating_regions": len(violating_regions)}

def index_fairness(at, oc_file, storage_options, columns, non_fixed_columns, timeout):
    t0 = time.time()

    shards = iter_zarr_shards(oc_file, storage_options=storage_options)
    feat_map, _ = zarr_metadata(oc_file, storage_options=storage_options)
    
    oci = index.OCTreeIndex(at, feat_map)
    for boxes_batch, preds_batch in shards:
        oci.store(boxes_batch, preds_batch)
    oci.to_arrays()
    
    index_time = time.time() - t0
    t0 = time.time()

    protected_idx = columns.get_loc(non_fixed_columns[0])
    violating_regions = oci.find_unfair_boxes(protected_idx, timeout)
    
    return {"isfair": len(violating_regions) == 0, "time": time.time() - t0, "num_violating_regions": len(violating_regions), "index_time": index_time}


def smt_fairness(at, columns, non_fixed_columns, timeout):
    t0 = time.time()
    
    at_contrast, feat_map = constrast_two_examples(at, columns=columns, nonfixed_columns=non_fixed_columns)
    box = {}
    
    v = Verifier(at_contrast, box, Backend())
    v.set_timeout(timeout)
    v.add_all_trees()
    v.add_constraint(v.fvar() > 0.0)

    all_solutions = []
    while v.check() == Verifier.Result.SAT:
        sol = v.model_family(v.model())
        all_solutions.append(sol)
        v.add_constraint(encode_found_sol(v, sol))
        v.set_timeout(timeout - (time.time() - t0))

    
    # HACK remove solutions with:
    # 1) identical instances
    # 2) both instances belonging to the same class
    violating_regions = []
    same_class, identical_instances = 0, 0
    for sol in all_solutions:
        two_examples = contrasting_examples_from_solutions(feat_map, sol, columns)
        instance0 = two_examples.iloc[0, :].to_numpy()
        instance1 = two_examples.iloc[1, :].to_numpy()
        pred0 = at.predict(instance0.reshape(1, -1))[0]
        pred1 = at.predict(instance1.reshape(1, -1))[0]
        class0 = 1 if pred0 > 0.5 else 0
        class1 = 1 if pred1 > 0.5 else 0
        if np.allclose(instance0, instance1):
            identical_instances += 1
        elif class0 == class1:
            same_class += 1
        else:
            violating_regions.append(sol)

    return {"isfair": len(violating_regions) == 0, "time": time.time() - t0, "num_violating_regions": len(violating_regions)}


def encode_found_sol(verifier, sol):
    # This generates constraints such that when added to the verifier, the verifier will not return the same solution again. 
    # This is used to find multiple solutions with the SMT-based approach.
    cs = []
    for feat_id, interval in sol.items():
        var = verifier.xvar(feat_id)
        if interval.lo_is_unbound(): cs.append(var >=  interval.hi)
        elif interval.hi_is_unbound(): cs.append(var <  interval.lo)
        else:
            cs.append((var < interval.lo) | (var >= interval.hi))
    return smt.VerifierOrExpr(*cs)

# def double_loop_fairness(at, oc_file, columns, non_fixed_columns, timeout):
#     t = time.time()
#     boxes = hdf5_generator(oc_file, 'boxes', 100*8194)
#     preds = hdf5_generator(oc_file, 'outvalues', 100*8194)
#     with h5py.File(oc_file, "r") as f:
#         feat_map = f.attrs['feat_map']

#     pni = index.PosNegIndex(at, feat_map)

#     protected_idx = columns.get_loc(non_fixed_columns[0])

#     for boxes_batch, preds_batch in zip(boxes, preds):
#         pni.store(boxes_batch, preds_batch)
#         # r += boxes_batch.shape[0]
#         # print(r)
#     # print('to arrays')
#     pni.to_arrays()

#     index_time = time.time() - t
#     t0 = time.time()

#     pos_boxes = pni.get_boxes_at_index(1)
#     neg_boxes = pni.get_boxes_at_index(0)

#     if pos_boxes.shape[0] == 0 or neg_boxes.shape[0] == 0:
#         return {"isfair": True, "time": time.time() - t0, "num_unfair_regions": 0, "index_time": index_time}
    
#     if pos_boxes.shape[2] <= protected_idx:
#         return {"isfair": True, "time": time.time() - t0, "num_unfair_regions": 0, "index_time": index_time}
#     # print('hi')
#     unfair_regions = []
#     for box1 in pos_boxes:
#         if box1[0, feat_map[protected_idx]] == float('-inf') and box1[1, feat_map[protected_idx]] == float('inf'):
#             continue # if the box doesn't allow changing the protected feature, skip it
#         # print('hi')
#         relaxed_box1 = box1.copy()
#         relaxed_box1[0, feat_map[protected_idx]] = float('-inf')
#         relaxed_box1[1, feat_map[protected_idx]] = float('inf')

#         for box2 in neg_boxes:
#             if index.overlaps(relaxed_box1, box2):
#                 relaxed_box2 = box2.copy()
#                 relaxed_box2[0, feat_map[protected_idx]] = float('-inf')
#                 relaxed_box2[1, feat_map[protected_idx]] = float('inf')

#                 unfair_regions.append(util.intersect(relaxed_box1, relaxed_box2))

#                 # print(len(unfair_regions))

#         if time.time() - t0 >= timeout:
#             break
    
#     # print(type(len(unfair_regions)))
#     return {"isfair": len(unfair_regions) == 0, "time": time.time() - t0, "num_unfair_regions": len(unfair_regions), "index_time": index_time}


def run_fairness_tasks(at, oc_file, storage_options, columns, non_fixed_columns, timeout):
    result = {}
    result['veritas_search'] = veritas_fairness(at, columns, non_fixed_columns, timeout)
    result['index_based'] = index_fairness(at, oc_file, storage_options,columns, non_fixed_columns, timeout)
    # result['smt_based'] = smt_fairness(at, columns, non_fixed_columns, timeout) # SMT-based approach is too slow for most models
    return result

def index_adversarial_robustness(at, oc_file, storage_options, x, y, l_inf, l_1, timeout):
    t0 = time.time()

    shards = iter_zarr_shards(oc_file, storage_options=storage_options)
    feat_map, _ = zarr_metadata(oc_file, storage_options=storage_options)
    
    rbi = index.RootboxIndex(at, feat_map)
    for boxes_batch, preds_batch in shards:
        rbi.store(boxes_batch, preds_batch)
    rbi.to_arrays()
    
    index_time = time.time() - t0
    t0 = time.time()

    count = 0
    verification_times = np.zeros(x.shape[0])
    robust = np.zeros(x.shape[0])

    
    for i, (example, target_label) in enumerate(zip(x, y)):
        t = time.time()
        robust[i] = rbi.is_robust(example, target_label, l_inf, l_1)
        count += 1
        verification_times[i] = time.time() - t

        if time.time() - t0 >= timeout:
            break
    
    print(robust)
    return {'robustness_count': np.sum(robust), 'n': count, 'time': time.time() - t0, 'index_time': index_time, 'individual_verification_times': verification_times.tolist()}

def smt_adversarial_robustness(at, x, y, l_inf, l_1, timeout):
    t0 = time.time()
    
    robust = np.zeros(x.shape[0])
    verification_times = np.zeros(x.shape[0])
    count = 0

    
    for i, (example, target) in enumerate(zip(x, y)):
        t = time.time()
        box = {f: Interval(example[f]-l_inf, example[f]+l_inf) for f in range(len(example))}
        v = Verifier(at, box, Backend(), budget=l_1, instance=example)
        v.set_timeout(timeout - (time.time() - t0))
        v.add_all_trees()
        v.add_constraint(v.fvar() > 0.0) if target else v.add_constraint(v.fvar() <= 0.0)

        try:
            r = v.check()
        except VerifierTimeout as e:
            print("SMT verifier timeout.")
            break
        
        robust[i] = 0 if r == Verifier.Result.SAT else 1
        verification_times[i] = time.time() - t
        count += 1

    print(robust)
    return {'robustness_count': np.sum(robust), 'n': count, 'time': time.time() - t0, 'individual_verification_times': verification_times.tolist()}

def run_complex_norm_robustness_tasks(at, x, y, n, l_inf, l_1, oc_file, storage_options, timeout, ):
    result = {}

    x_correct = []
    y_target = []
    count = 0
    for i in x.index:
        target_label = not (y.loc[i] > 0.0)
        example = x.loc[i, :].to_numpy()
        pred_label = at.eval(example)[0, 0] > 0.0

        if pred_label != target_label:
            x_correct.append(example)
            y_target.append(target_label)

            count += 1
        if count >= n:
            break

    x_correct = np.array(x_correct)
    y_target = np.array(y_target)

    result['index_based'] = index_adversarial_robustness(at, oc_file, storage_options, x_correct, y_target, l_inf, l_1, timeout)
    result['smt_based'] = smt_adversarial_robustness(at, x_correct, y_target, l_inf, l_1, timeout)

    return result

def run_lipschitz_task(at, oc_file, storage_options, minDx,timeout):
    t0 = time.time()

    shards = iter_zarr_shards(oc_file, storage_options=storage_options)
    feat_map, _ = zarr_metadata(oc_file, storage_options=storage_options)
    
    oci = index.OCTreeIndex(at, feat_map)
    for boxes_batch, preds_batch in shards:
        oci.store(boxes_batch, preds_batch)
    oci.to_arrays()
    
    index_time = time.time() - t0
    t0 = time.time()

    c, dx, dy = oci.find_lipschitz_constant(minDx, timeout)

    return {'lipschitz_constant': c, 'time': time.time() - t0, 'dx': dx, 'dy': dy, 'index_time': index_time}
