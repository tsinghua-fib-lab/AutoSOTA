
import os

import numpy as np

from rl4co.data.utils import check_extension
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


VARIANT_FEATURES = {
    "CVRP": {"O": False, "TW": False, "L": False, "B": False, "M": False},
    "OVRP": {"O": True, "TW": False, "L": False, "B": False, "M": False},
    "VRPB": {"O": False, "TW": False, "L": False, "B": True, "M": False},
    "VRPL": {"O": False, "TW": False, "L": True, "B": False, "M": False},
    "VRPTW": {"O": False, "TW": True, "L": False, "B": False, "M": False},
    "OVRPTW": {"O": True, "TW": True, "L": False, "B": False, "M": False},
    "OVRPB": {"O": True, "TW": False, "L": False, "B": True, "M": False},
    "OVRPL": {"O": True, "TW": False, "L": True, "B": False, "M": False},
    "VRPBL": {"O": False, "TW": False, "L": True, "B": True, "M": False},
    "VRPBTW": {"O": False, "TW": True, "L": False, "B": True, "M": False},
    "VRPLTW": {"O": False, "TW": True, "L": True, "B": False, "M": False},
    "OVRPBL": {"O": True, "TW": False, "L": True, "B": True, "M": False},
    "OVRPBTW": {"O": True, "TW": True, "L": False, "B": True, "M": False},
    "OVRPLTW": {"O": True, "TW": True, "L": True, "B": False, "M": False},
    "VRPBLTW": {"O": False, "TW": True, "L": True, "B": True, "M": False},
    "OVRPBLTW": {"O": True, "TW": True, "L": True, "B": True, "M": False},
    #"VRPMB": {"O": False, "TW": False, "L": False, "B": True, "M": True},
    #"OVRPMB": {"O": True, "TW": False, "L": False, "B": True, "M": True},
    #"VRPMBL": {"O": False, "TW": False, "L": True, "B": True, "M": True},
    #"VRPMBTW": {"O": False, "TW": True, "L": False, "B": True, "M": True},
    #"OVRPMBL": {"O": True, "TW": False, "L": True, "B": True, "M": True},
    #"OVRPMBTW": {"O": True, "TW": True, "L": False, "B": True, "M": True},
    #"VRPMBLTW": {"O": False, "TW": True, "L": True, "B": True, "M": True},
    #"OVRPMBLTW": {"O": True, "TW": True, "L": True, "B": True, "M": True},
}


def get_vehicle_capacity(num_loc):
    if num_loc > 1000:
        extra_cap = 1000 // 5 + (num_loc - 1000) // 33.3
    elif num_loc > 20:
        extra_cap = num_loc // 5
    else:
        extra_cap = 0
    return 30 + extra_cap


def generate_mtvrp_data(
    dataset_size,
    num_loc=20,
    min_loc=0.0,
    max_loc=1.0,
    capacity=None,
    min_demand=1,
    max_demand=9,
    scale_demand=True,
    max_time=4.6,
    max_distance_limit=2.8,  # 2sqrt(2) ~= 2.8
    speed=1.0,
    num_depots=3,  # number of depots, only used for multi-depot problems
    variant="CVRP",
    time_window_a=0.15,
    time_window_b=0.18,
    time_window_c=0.20,
):
    """Generate MTVRP data using NumPy for a specific variant.
    NOTE: for MD (multi-depot) variants, we generate in the same way
    as single depot (considering the first one), but set the number of depots to the desired value.
    """
    # if first two letters are "md", strip and set multi-depot flag
    if variant[:2].lower() == "md":
        variant = variant[2:]
        num_depots = num_depots
    else:
        num_depots = 1  # single depot
    variant = variant.upper()
    if variant not in VARIANT_FEATURES:
        raise ValueError(f"Unknown variant: {variant}")

    features = VARIANT_FEATURES[variant]

    if capacity is None:
        capacity = get_vehicle_capacity(num_loc)

    # Generate locations
    locs = np.random.uniform(min_loc, max_loc, (dataset_size, num_depots + num_loc, 2))

    # Generate demands
    def generate_demand(size):
        return (
            np.random.randint(min_demand, max_demand + 1, size).astype(np.float32)
            / capacity
        )

    demand_linehaul = generate_demand((dataset_size, num_loc))
    demand_backhaul = None

    if features["B"]:
        demand_backhaul = np.zeros((dataset_size, num_loc))
        backhaul_mask = (
            np.random.rand(dataset_size, num_loc) < 0.2
        )  # 20% of nodes are backhaul
        demand_backhaul[backhaul_mask] = generate_demand(backhaul_mask.sum())
        demand_linehaul[backhaul_mask] = 0

    # Generate backhaul class
    backhaul_class = (
        np.full((dataset_size, 1), 2 if features["M"] else 1) if features["B"] else None
    )

    # Generate open route
    open_route = np.full((dataset_size, 1), features["O"]) if features["O"] else None

    # Generate time windows and service time
    time_windows = None
    service_time = None
    if features["TW"]:
        a, b, c = time_window_a, time_window_b, time_window_c
        service_time = a + (b - a) * np.random.rand(dataset_size, num_loc)
        tw_length = b + (c - b) * np.random.rand(dataset_size, num_loc)
        # note: we assume that the first depot only for this
        d_0i = np.linalg.norm(locs[:, 0:1] - locs[:, num_depots:], axis=2)
        h_max = (max_time - service_time - tw_length) / d_0i * speed - 1
        tw_start = (
            (1 + (h_max - 1) * np.random.rand(dataset_size, num_loc)) * d_0i / speed
        )
        tw_end = tw_start + tw_length

        time_windows = np.concatenate(
            [
                np.zeros((dataset_size, num_depots, 2)),
                np.stack([tw_start, tw_end], axis=-1),
            ],
            axis=1,
        )
        time_windows[:, :num_depots, 1] = max_time
        # pad service time until the number of depots
        service_time = np.pad(service_time, ((0, 0), (num_depots, 0)))

    # Generate distance limits: dist_lower_bound = 2 * max(depot_to_location_distance),
    # max = min(dist_lower_bound, max_distance_limit). Ensures feasible yet challenging
    # constraints, with each instance having a unique, meaningful limit.
    if features["L"]:
        # Calculate the maximum distance from depot to any location
        max_dist = np.max(np.linalg.norm(locs[:, 1:] - locs[:, 0:1], axis=2), axis=1)

        # Calculate the minimum distance limit (2 * max_distance)
        distance_lower_bound = 2 * max_dist + 1e-6  # Add epsilon to avoid zero distance

        # Ensure max_distance_limit is not exceeded
        max_distance_limit = np.maximum(max_distance_limit, distance_lower_bound + 1e-6)

        # Generate distance limits between min_distance_limits and max_distance_limit
        distance_limit = np.random.uniform(
            distance_lower_bound,
            np.full_like(distance_lower_bound, max_distance_limit),
            (dataset_size,),
        )[:, None]
    else:
        distance_limit = None

    # Generate speed
    speed = np.full((dataset_size, 1), speed)

    # Scale demand if needed
    if scale_demand:
        vehicle_capacity = np.full((dataset_size, 1), 1.0)
    else:
        vehicle_capacity = np.full((dataset_size, 1), capacity)
        if demand_backhaul is not None:
            demand_backhaul *= capacity
        demand_linehaul *= capacity

    data = {
        "locs": locs.astype(np.float32),
        "demand_linehaul": demand_linehaul.astype(np.float32),
        "vehicle_capacity": vehicle_capacity.astype(np.float32),
        "speed": speed.astype(np.float32),
        "num_depots": np.full((dataset_size, 1), num_depots).astype(np.int32),
    }

    # Only include features that are used in the variant
    if features["B"]:
        data["demand_backhaul"] = demand_backhaul.astype(np.float32)
        data["backhaul_class"] = backhaul_class.astype(np.float32)
    if features["O"]:
        data["open_route"] = open_route
    if features["TW"]:
        data["time_windows"] = time_windows.astype(np.float32)
        data["service_time"] = service_time.astype(np.float32)
    if features["L"]:
        data["distance_limit"] = distance_limit.astype(np.float32)

    return data


def generate_dataset(
    filename=None,
    data_dir="data",
    name=None,
    problem="cvrp",
    dataset_size=1000,
    graph_sizes=[100],
    overwrite=True,
    seed=1234,
    disable_warning=False,
    **kwargs,
):
    """We keep a similar structure as in Kool et al. 2019 but save and load the data as npz
    This is way faster and more memory efficient than pickle and also allows for easy transfer to TensorDict
    """

    fname = filename
    if isinstance(graph_sizes, int):
        graph_sizes = [graph_sizes]
    for graph_size in graph_sizes:
        datadir = os.path.join(data_dir, problem)
        os.makedirs(datadir, exist_ok=True)

        if filename is None:
            fname = os.path.join(
                datadir,
                "{}{}_seed{}.npz".format(
                    graph_size,
                    "_{}".format(name) if name is not None else "",
                    seed,
                ),
            )
        else:
            fname = check_extension(filename, extension=".npz")

        # Generate any needed directories
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        if not overwrite and os.path.isfile(check_extension(fname, extension=".npz")):
            if not disable_warning:
                log.info(
                    "File {} already exists! Run with -f option to overwrite. Skipping...".format(
                        fname
                    )
                )
            continue

        # Set seed
        np.random.seed(seed)

        # Automatically generate dataset
        dataset = generate_mtvrp_data(
            dataset_size=dataset_size, num_loc=graph_size, variant=problem, **kwargs
        )

        # A function can return None in case of an error or a skip
        if dataset is not None:
            # Save to disk as dict
            log.info("Saving {} data to {}".format(problem.upper(), fname))
            np.savez(fname, **dataset)

