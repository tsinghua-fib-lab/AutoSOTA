import os
import argparse
from data_processing.wind import create_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root', 
        type=str, 
        required=True,
        help='Root directory where dataset is saved'
    )
    parser.add_argument(
        '--save_dir', 
        type=str, 
        required=True,
        help='Directory where dataset is saved (will be created if it does not exist)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=674829,
        help='Random number generator seed'
    )
    parser.add_argument(
        '--knn_k', 
        type=int, 
        default=10,
        help='Number of nearest neighbors for k-NN graph'
    )
    parser.add_argument(
        '--sample_n', 
        type=int, 
        default=400,
        help='Number of samples'
    )
    parser.add_argument(
        '--mask_prop', 
        type=float, 
        default=0.1,
        help='Proportion of samples to mask'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = create_dataset(
        root=args.root,
        seed=args.seed,
        knn_k=args.knn_k,
        sample_n=args.sample_n,
        mask_prop=args.mask_prop,
    )

    # Save data object
    os.makedirs(args.save_dir, exist_ok=True)
    mask_prop_str = int(args.mask_prop * 100)
    filename = f'wind_{args.sample_n}_{mask_prop_str}_k{args.knn_k}.pkl'
    save_path = os.path.join(args.save_dir, filename)
    data.save(save_path)
    print(f"Data saved to '{save_path}'")

