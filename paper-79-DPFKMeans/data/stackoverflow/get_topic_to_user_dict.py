import argparse
import pickle
import h5py
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id', type=int, default=0)
    parser.add_argument(
        '--num_jobs_per_split', type=int, default=1,
        help='How many jobs per split in [train, test, val]')

    args = parser.parse_args()

    split = ['train', 'val', 'test'][args.job_id // args.num_jobs_per_split]
    filename = f'embedded_stackoverflow_{split}_{args.job_id % args.num_jobs_per_split}.hdf5'

    data_dir = 'embedded_data'
    mapping = dict()

    with h5py.File(os.path.join(data_dir, filename), 'r') as f:
        d = f[split]
        user_ids = d.keys()
        for user_id in user_ids:
            print(f'Running user {user_id}')
            tag_array = f[split][user_id]['tags'][()].astype(str)
            for tag_str in tag_array:
                for tag in tag_str.split('|'):
                    try:
                        mapping[tag].add(user_id)
                    except KeyError:
                        mapping[tag] = {user_id}

    with open(f'{data_dir}/tag_to_user_{split}_{args.job_id % args.num_jobs_per_split}.pkl', 'wb') as pickle_f:
        pickle.dump(mapping, pickle_f)


if __name__ == '__main__':
    main()
