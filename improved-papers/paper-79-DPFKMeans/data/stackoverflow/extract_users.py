import argparse
import h5py
import multiprocessing as mp
import numpy as np
import os
import pickle


def extract_users(topics, data_dir, split, idx):
    topic_to_label = dict(zip(topics, range(len(topics))))
    topics_set = set(topics)
    dict_filename = f'tag_to_user_{split}_{idx}.pkl'
    with open(os.path.join(data_dir, dict_filename), 'rb') as f:
        tag_to_user = pickle.load(f)

    user_ids = []
    for topic in topics:
        user_ids += tag_to_user[topic]

    user_ids = list(set(user_ids))

    filename = f'embedded_stackoverflow_{split}_{idx}.hdf5'
    user_data = dict()
    with h5py.File(os.path.join(data_dir, filename), 'r') as f:
        for i, user_id in enumerate(user_ids):
            d = f[split][user_id]
            tags = d['tags'][()].astype(str)
            embeddings = d['embeddings'][()]
            extracted_embeddings = []
            extracted_labels = []
            for embedding, tag in zip(embeddings, tags):
                tag_list = tag.split('|')
                if tag_list[0] in topics_set:
                    extracted_embeddings.append(embedding)
                    extracted_labels.append(topic_to_label[tag_list[0]])

            if extracted_embeddings:
                extracted_embeddings = np.vstack(extracted_embeddings)
                extracted_labels = np.hstack(extracted_labels)
                user_data[user_id] = (extracted_embeddings, extracted_labels)

    return user_data


def _make_worker_fn(topics, save_dir, data_dir, lock):

    def _extract_users(args):
        split, idx = args
        user_data = extract_users(topics, data_dir, split, idx)
        with lock:
            save_filename = f'{split}_users.hdf5'
            with h5py.File(os.path.join(save_dir, save_filename), 'a') as h5:
                for user_id, (x, y) in user_data.items():
                    h5.create_dataset(f'/{split}/{user_id}/embeddings', data=x)
                    h5.create_dataset(f'/{split}/{user_id}/labels', data=y)

        return split, idx

    def _worker_fn(work_queue, result_queue):
        # Continuously query for user_ids to process,
        # process them and send results.
        while True:
            args = work_queue.get()
            if args is None:
                result_queue.put((None, None))
                break
            result_queue.put(_extract_users(args))

    return _worker_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_processes', type=int, default=3)
    parser.add_argument(
        '--num_jobs_per_split', type=int, default=1,
        help='How many jobs per split in [train, test, val]')

    args = parser.parse_args()

    all_topics = [
        ['machine-learning', 'math'],
        ['facebook', 'hibernate'],
        ['github', 'pdf'],
        ['plot', 'cookies']

    ]

    topic_abreviations = {'machine-learning': 'ml',
                          'math': 'math',
                          'facebook': 'fb',
                          'hibernate': 'hb',
                          'github': 'gith',
                          'pdf': 'pdf',
                          'plot': 'plt',
                          'cookies': 'cook'
                          }

    local_dir_name = 'embedded_data'

    for topics in all_topics:
        print(f'Extracting users for topics: {topics}')
        topics_code = '-'.join([topic_abreviations[topic] for topic in topics])
        save_dir = os.path.join('topic_extracted_data', topics_code)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        lock = mp.Lock()

        work_queue = mp.Queue()
        results_queue = mp.Queue()
        processes = [
            mp.Process(target=_make_worker_fn(topics, save_dir, local_dir_name, lock),
                       args=(work_queue, results_queue))
            for _ in range(args.num_processes)
        ]

        for p in processes:
            p.start()

        for split in ['train', 'val', 'test']:
            for idx in range(args.num_jobs_per_split):
                work_queue.put((split, idx))

        for _ in processes:
            work_queue.put(None)

        for p in processes:
            p.join()


if __name__ == '__main__':
    main()
