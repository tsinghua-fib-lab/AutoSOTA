import argparse
import logging

from data.generate import VARIANT_FEATURES, generate_dataset

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--val_data_size", type=int, default=128)  # to make validation faster use 128
    parser.add_argument("--test_data_size", type=int, default=1000)
    parser.add_argument("--num_nodes", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--val_seed", type=int, default=4321)
    parser.add_argument("--test_seed", type=int, default=1234)
    parser.add_argument("--generate_multi_depot", type=bool, default=False)

    parser.add_argument("--capacity", type=float, default=None)
    parser.add_argument("--max_distance_limit", type=float, default=2.8)
    parser.add_argument("--time_window_a", type=float, default=0.15)
    parser.add_argument("--time_window_b", type=float, default=0.18)
    parser.add_argument("--time_window_c", type=float, default=0.20)
    parser.add_argument("--variants", type=str, nargs="+", default=['all'])

    args = parser.parse_args()

    # Print config
    print("Config:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")

    #seeds = {"val": args.val_seed, "test": args.test_seed}
    seeds = dict()
    if args.val_data_size > 0:
        seeds['val'] = args.val_seed
    if args.test_data_size > 0:
        seeds['test'] = args.test_seed

    # Add multi-depot problems if needed for each variant
    if args.variants[0] == 'all':
        variants = list(VARIANT_FEATURES.keys())
    else:
        variants = args.variants

    if args.generate_multi_depot:
        variants += ["MD" + problem for problem in VARIANT_FEATURES]

    for problem in variants:
        problem = problem.lower()
        for phase, seed in seeds.items():
            for size in args.num_nodes:
                generate_dataset(
                    problem=problem,
                    data_dir=args.data_dir,
                    filename=args.data_dir + f"/{problem}/{phase}/{size}.npz",
                    dataset_size=(
                        args.val_data_size if phase == "val" else args.test_data_size
                    ),
                    graph_sizes=size,
                    seed=seed,
                    capacity=args.capacity,
                    max_distance_limit=args.max_distance_limit,
                    time_window_a=args.time_window_a,
                    time_window_b=args.time_window_b,
                    time_window_c=args.time_window_c,
                )