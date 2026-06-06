# Some of the code in this file is adapted from:
#
# pfl-research:
# Copyright 2024, Apple Inc.
# Licensed under the Apache License, Version 2.0 (the "License").

import argparse
import lzma
import os
from tqdm import tqdm
import urllib.parse
import urllib.request

def fetch_lzma_file(origin: str, filename: str):
    """Fetches a LZMA compressed file and decompresses on the fly."""

    # Read and decompress in approximately megabyte chunks.
    def url_basename(origin: str) -> str:
        origin_path = urllib.parse.urlparse(origin).path
        return origin_path.rsplit('/', maxsplit=1)[-1]

    chunk_size = 2**20
    decompressor = lzma.LZMADecompressor()
    with urllib.request.urlopen(origin) as in_stream, open(filename,
                                                           'wb') as out_stream:
        length = in_stream.headers.get('content-length')
        total_size = int(length) if length is not None else None
        download_chunk = in_stream.read(chunk_size)
        with tqdm(total=total_size,
                  desc=f'Downloading {url_basename(origin)}') as progbar:
            while download_chunk:
                progbar.update(len(download_chunk))
                out_stream.write(decompressor.decompress(download_chunk))
                download_chunk = in_stream.read(chunk_size)

def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        '--output_dir',
        help=('Output directory for the original sqlite '
              'data and the processed hdf5 files.'),
        default='embedded_data')

    arguments = argument_parser.parse_args()

    os.makedirs(arguments.output_dir, exist_ok=True)

    database_filepath = os.path.join(arguments.output_dir, "stackoverflow.sqlite")
    if not os.path.exists(database_filepath):
        print(f'Downloading StackOverflow data to {arguments.output_dir}')
        database_origin = (
            "https://storage.googleapis.com/tff-datasets-public/"
            "stackoverflow.sqlite.lzma")
        fetch_lzma_file(origin=database_origin, filename=database_filepath)


if __name__ == '__main__':
    main()
