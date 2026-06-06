# Some of the code in this file is adapted from:
#
# pfl-research:
# Copyright 2024, Apple Inc.
# Licensed under the Apache License, Version 2.0 (the "License").

import argparse
import json
import os
import sqlite3
import time
from collections import OrderedDict
from typing import Dict, Iterator, Optional

import h5py
from sentence_transformers import SentenceTransformer
import tensorflow as tf


def load_word_counts(cache_dir=None,
                     vocab_size: Optional[int] = None) -> Dict[str, int]:
    """Loads the word counts for the Stack Overflow dataset.

    :param: cache_dir:
        (Optional) directory to cache the downloaded file. If `None`,
        caches in Keras' default cache directory.
    :param: vocab_size:
        (Optional) when specified, only load the first `vocab_size`
        unique words in the vocab file (i.e. the most frequent `vocab_size`
        words).

    :returns:
      A collections.OrderedDict where the keys are string tokens, and the values
      are the counts of unique users who have at least one example in the
      training set containing that token in the body text.
    """
    if vocab_size is not None:
        if not isinstance(vocab_size, int):
            raise TypeError(
                f'vocab_size should be None or int, got {type(vocab_size)}.')
        if vocab_size <= 0:
            raise ValueError(f'vocab_size must be positive, got {vocab_size}.')

    path = tf.keras.utils.get_file(
        'stackoverflow.word_count.tar.bz2',
        origin='https://storage.googleapis.com/tff-datasets-public/'
        'stackoverflow.word_count.tar.bz2',
        file_hash=(
            '1dc00256d6e527c54b9756d968118378ae14e6692c0b3b6cad470cdd3f0c519c'
        ),
        hash_algorithm='sha256',
        extract=True,
        archive_format='tar',
        cache_dir=cache_dir,
    )

    word_counts = OrderedDict()
    dir_path = os.path.dirname(path)
    file_path = os.path.join(dir_path, 'stackoverflow.word_count')
    with open(file_path) as f:
        for line in f:
            word, count = line.split()
            word_counts[word] = int(count)
            if vocab_size is not None and len(word_counts) >= vocab_size:
                break
    return word_counts


def fetch_client_ids(database_filepath: str,
                     split_name: Optional[str] = None) -> Iterator[str]:
    """Fetches the list of client_ids.

    :param database_filepath:
        A path to a SQL database.
    :param split_name:
        An optional split name to filter on. If `None`, all client ids
         are returned.
    :returns:
      An iterator of string client ids.
    """
    if split_name == "val":
        # heldout is used in the raw sqlite database
        split_name = "heldout"
    connection = sqlite3.connect(database_filepath)
    query = "SELECT DISTINCT client_id FROM client_metadata"
    if split_name is not None:
        query += f" WHERE split_name = '{split_name}'"
    query += ";"
    result = connection.execute(query)
    return (x[0] for x in result)


def query_client_dataset(database_filepath: str,
                         client_id: str,
                         split_name: Optional[str] = None) -> tf.data.Dataset:

    def add_proto_parsing(dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Add parsing of the tf.Example proto to the dataset pipeline."""

        def parse_proto(tensor_proto):
            parse_spec = OrderedDict(
                creation_date=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
                score=tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
                tags=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
                title=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
                tokens=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
                type=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
            )
            parsed_features = tf.io.parse_example(tensor_proto, parse_spec)
            return OrderedDict(
                (key, parsed_features[key]) for key in parse_spec)

        return dataset.map(parse_proto, num_parallel_calls=tf.data.AUTOTUNE)

    query_parts = [
        "SELECT serialized_example_proto FROM examples WHERE client_id = '",
        client_id,
        "'",
    ]
    if split_name is not None:
        if split_name == "val":
            # heldout is used in the raw sqlite database
            split_name = "heldout"
        query_parts.extend([" and split_name ='", split_name, "'"])
    return add_proto_parsing(
        tf.data.experimental.SqlDataset(
            driver_name="sqlite",
            data_source_name=database_filepath,
            query=tf.strings.join(query_parts),
            output_types=(tf.string, ),
        ))


def process_user(user_id, model, database_filepath, partition, h5_path):
    # tf Dataset with sentences from user.
    tfdata = query_client_dataset(database_filepath, user_id, partition)

    sentences = []
    tags = []
    for sentence_data in tfdata:
        sentences.append(sentence_data['tokens'].numpy().decode('UTF-8'))
        tags.append(sentence_data['tags'].numpy().decode('UTF-8'))

    embeddings = model.encode(sentences)
    with h5py.File(h5_path, 'a') as h5:
        # Store encoded inputs.
        h5.create_dataset(f'/{partition}/{user_id}/embeddings', data=embeddings)
        h5.create_dataset(f'/{partition}/{user_id}/tags', data=tags)

    return user_id, len(embeddings)


def dl_preprocess_and_dump_h5(output_dir: str, job_id: int, num_jobs_per_split: int):
    """
    Preprocess StackOverflow dataset.

    :param output_dir:
        Directory for all output files, both raw and processed data.
    :param job_id:
        Which job is being run.
    """
    partition = ['train', 'val', 'test'][job_id // num_jobs_per_split]
    partition_job_id = job_id % num_jobs_per_split

    h5_path = os.path.join(output_dir, f'embedded_stackoverflow_{partition}_{partition_job_id}.hdf5')
    database_filepath = os.path.join(output_dir, "stackoverflow.sqlite")

    print(f'Processing users for partition {partition}')
    client_ids = list(fetch_client_ids(database_filepath, partition))
    block_size = (len(client_ids) // num_jobs_per_split) + 1
    start_idx = partition_job_id * block_size
    end_idx = min(len(client_ids), (partition_job_id + 1) * block_size)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    client_ids = client_ids[start_idx: end_idx]
    start = time.time()
    user_num_datapoints = dict()
    for i, client_id in enumerate(client_ids):
        client_id, n = process_user(client_id, model, database_filepath, partition, h5_path)
        user_num_datapoints[client_id] = n
        if (i + 1) % 100 == 0:
            print(f'Completed client {i + 1} / {len(client_ids)} in {time.time() - start:.2f}s')

    if len(user_num_datapoints):
        with h5py.File(h5_path, 'a') as h5:
            h5[f'/metadata/user_num_datapoints/{partition}'] = json.dumps(
                user_num_datapoints)


if __name__ == '__main__':

    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        '--output_dir',
        help=('Output directory for the original sqlite '
              'data and the processed hdf5 files.'),
        default='embedded_data')


    argument_parser.add_argument(
        '--job_id',
        type=int,
        default=0,
        help='Integer in range [0, 3 * num_jobs_per_split], specifiying which portion of the data to process.')

    argument_parser.add_argument(
        '--num_jobs_per_split',
        type=int,
        default=1,
        help='How many jobs per split in [train, test, val]')

    arguments = argument_parser.parse_args()

    dl_preprocess_and_dump_h5(arguments.output_dir,
                              arguments.job_id,
                              arguments.num_jobs_per_split)