"""Download the official archive using checked HTTP byte ranges.

Uses the configured Kaggle SDK credentials. Signed URLs remain in memory and
are never logged. Each range is checked against object identity and length.
"""
import argparse
import concurrent.futures
import json
import os
import threading
import time
from pathlib import Path

import requests
from kaggle import api
from kaggle.api.kaggle_api_extended import ApiDownloadDataFilesRequest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--part-mib', type=int, default=256)
    args = parser.parse_args()
    destination = ROOT / 'data/raw/g2net-gravitational-wave-detection.zip'
    partial = destination.with_suffix('.ranges.partial')
    manifest = destination.with_suffix('.ranges.json')
    with api.build_kaggle_client() as client:
        request = ApiDownloadDataFilesRequest()
        request.competition_name = 'g2net-gravitational-wave-detection'
        response = client.competitions.competition_api_client.download_data_files(request)
        url = response.url
        total = int(response.headers['Content-Length'])
        etag = response.headers['ETag']
        response.close()
    part_size = args.part_mib * 1024**2
    identity = dict(size=total, etag=etag, part_size=part_size)
    state = dict(identity, completed=[])
    if manifest.exists() and partial.exists():
        existing = json.loads(manifest.read_text())
        if all(existing.get(k) == v for k, v in identity.items()):
            state = existing
        else:
            raise RuntimeError('Remote identity changed; previous ranges cannot be reused')
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(partial, os.O_RDWR | os.O_CREAT, 0o644)
    os.ftruncate(descriptor, total)
    lock = threading.Lock()
    transferred = 0
    started = time.monotonic()
    completed = set(state['completed'])
    ranges = [(i, start, min(start + part_size, total))
              for i, start in enumerate(range(0, total, part_size)) if i not in completed]
    reused = sum(min(part_size, total-i*part_size) for i in completed)

    def fetch(part):
        nonlocal transferred
        index, start, end = part
        for attempt in range(5):
            position = start
            try:
                with requests.get(url, headers={'Range': f'bytes={start}-{end-1}', 'If-Range': etag},
                                  stream=True, timeout=(30, 120)) as result:
                    if (result.status_code != 206 or result.headers.get('ETag') != etag or
                            result.headers.get('Content-Range') != f'bytes {start}-{end-1}/{total}'):
                        raise RuntimeError('Byte-range response or object identity mismatch')
                    for chunk in result.iter_content(4 * 1024**2):
                        if position + len(chunk) > end:
                            raise RuntimeError('Range response exceeds requested length')
                        written = os.pwrite(descriptor, chunk, position)
                        if written != len(chunk):
                            raise RuntimeError('Incomplete local write')
                        position += written
                        with lock:
                            transferred += written
                    if position != end:
                        raise RuntimeError('Range response shorter than requested')
                with lock:
                    completed.add(index)
                    state['completed'] = sorted(completed)
                    temporary = manifest.with_suffix('.tmp')
                    temporary.write_text(json.dumps(state))
                    temporary.replace(manifest)
                return
            except Exception as error:
                # Exception strings may contain a signed URL, so print only type.
                print(f'part={index} attempt={attempt+1} error={type(error).__name__}', flush=True)
                if attempt == 4:
                    raise RuntimeError(f'Part {index} failed after retries') from None
                time.sleep(min(2**attempt, 15))

    print(json.dumps(dict(identity, workers=args.workers, reused_bytes=reused)), flush=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(fetch, part) for part in ranges]
            while futures:
                done, pending = concurrent.futures.wait(futures, timeout=10,
                                                       return_when=concurrent.futures.FIRST_EXCEPTION)
                for future in done:
                    future.result()
                futures = list(pending)
                elapsed = time.monotonic() - started
                finished = sum(min(part_size, total-i*part_size) for i in completed)
                print(f'completed={finished/total:.1%} parts={len(completed)} '
                      f'transferred_GiB={transferred/1024**3:.2f} '
                      f'MiB_per_sec={transferred/1024**2/max(elapsed,1):.1f}', flush=True)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    partial.replace(destination)
    manifest.unlink()
    marker = destination.with_name(destination.name + '.kaggle-partial')
    marker.unlink(missing_ok=True)
    print(f'Official archive downloaded: {total} bytes', flush=True)


if __name__ == '__main__':
    main()
