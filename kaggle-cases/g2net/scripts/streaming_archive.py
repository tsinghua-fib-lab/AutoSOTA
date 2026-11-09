"""Read a checked, concurrently downloaded official archive.

Fetch the ZIP directory first, then wait for completed range markers before any
waveform bytes are read. This overlaps packing with the network transfer.
"""
import io
import json
import os
import struct
import time
from pathlib import Path

import requests
from kaggle import api
from kaggle.api.kaggle_api_extended import ApiDownloadDataFilesRequest


class CheckedRangeReader(io.RawIOBase):
    def __init__(self, path):
        super().__init__()
        self.path = Path(path)
        self.manifest = self.path.with_name(self.path.name.replace('.ranges.partial', '.ranges.json'))
        self.destination = self.path.with_name(self.path.name.replace('.ranges.partial', '.zip'))
        identity = json.loads(self.manifest.read_text())
        self.size = identity['size']
        self.part_size = identity['part_size']
        with api.build_kaggle_client() as client:
            request = ApiDownloadDataFilesRequest()
            request.competition_name = 'g2net-gravitational-wave-detection'
            response = client.competitions.competition_api_client.download_data_files(request)
            url = response.url
            if int(response.headers['Content-Length']) != self.size or response.headers['ETag'] != identity['etag']:
                response.close()
                raise RuntimeError('The archive identity changed')
            response.close()

        def fetch(start, end):
            try:
                with requests.get(url, headers={'Range': f'bytes={start}-{end-1}', 'If-Range': identity['etag']},
                                  stream=True, timeout=(30, 120)) as result:
                    if (result.status_code != 206 or result.headers.get('ETag') != identity['etag'] or
                            result.headers.get('Content-Range') != f'bytes {start}-{end-1}/{self.size}'):
                        raise RuntimeError('Invalid directory byte range')
                    content = result.content
                    if len(content) != end-start:
                        raise RuntimeError('Truncated directory range')
                    return content
            except Exception as error:
                raise RuntimeError(f'Directory download failed ({type(error).__name__})') from None

        footer = fetch(self.size-65558, self.size)
        position = footer.rfind(b'PK\x05\x06')
        if position < 0:
            raise RuntimeError('Missing ZIP end record')
        locator = footer[position-20:position]
        if locator[:4] != b'PK\x06\x07':
            raise RuntimeError('Expected ZIP64 locator for the official archive')
        _, _, record_offset, _ = struct.unpack('<IIQI', locator)
        record = fetch(record_offset, record_offset+56)
        fields = struct.unpack('<IQHHIIQQQQ', record)
        self.tail_start = fields[-1]
        tail = fetch(self.tail_start, self.size)
        # These bytes are the identical remote object. The download process may
        # later write the same range; its completion manifest remains untouched.
        descriptor = os.open(self.path, os.O_RDWR)
        try:
            written = os.pwrite(descriptor, tail, self.tail_start)
            if written != len(tail):
                raise RuntimeError('Incomplete directory write')
        finally:
            os.close(descriptor)
        self.file = self.path.open('rb')
        self.completed = set()
        self.last_refresh = 0
        print(f'ZIP directory available: {len(tail)} bytes; packing can overlap downloading', flush=True)

    def seek(self, offset, whence=0):
        return self.file.seek(offset, whence)

    def tell(self):
        return self.file.tell()

    def readable(self):
        return True

    def seekable(self):
        return True

    def read(self, size=-1):
        position = self.tell()
        end = self.size if size < 0 else min(position+size, self.size)
        if position < self.tail_start and end > position:
            needed = set(range(position//self.part_size, (min(end, self.tail_start)-1)//self.part_size+1))
            started = time.monotonic()
            while not needed.issubset(self.completed):
                if not self.manifest.exists() and self.destination.exists() and not self.path.exists():
                    break
                now = time.monotonic()
                if now-self.last_refresh > 0.2:
                    self.completed = set(json.loads(self.manifest.read_text())['completed'])
                    self.last_refresh = now
                if now-started > 3600:
                    raise TimeoutError('Archive download has not completed required ranges for one hour')
                time.sleep(0.05)
        return self.file.read(size)

    def close(self):
        if hasattr(self, 'file'):
            self.file.close()
        super().close()
