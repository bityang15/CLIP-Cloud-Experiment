import concurrent.futures
import hashlib
import math
import shutil
import sys
import time
from pathlib import Path

import requests


url, destination_name, expected_digest = sys.argv[1:]
destination = Path(destination_name)
algorithm = 'sha256' if len(expected_digest) == 64 else 'md5'


def checksum(path):
    value = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


if destination.exists():
    assert checksum(destination) == expected_digest, f'Existing file checksum mismatch: {destination}'
    print('Already verified', destination, flush=True)
    sys.exit(0)

with requests.get(url, headers={'Range': 'bytes=0-0'}, stream=True, timeout=(30, 60)) as response:
    response.raise_for_status()
    assert response.status_code == 206, 'The source does not support byte ranges'
    total_size = int(response.headers['Content-Range'].split('/')[-1])

parts_dir = destination.with_name(destination.name + '.parts')
parts_dir.mkdir(parents=True, exist_ok=True)
part_size = math.ceil(total_size / 16)


def fetch_part(index):
    start = index * part_size
    end = min(total_size, start + part_size) - 1
    path = parts_dir / f'{index:02d}'
    for attempt in range(4):
        existing_size = path.stat().st_size if path.exists() else 0
        expected_size = end - start + 1
        assert existing_size <= expected_size
        if existing_size == expected_size:
            return path
        offset = start + existing_size
        try:
            with requests.get(url, headers={'Range': f'bytes={offset}-{end}'},
                              stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                assert response.status_code == 206
                assert response.headers['Content-Range'] == f'bytes {offset}-{end}/{total_size}'
                with path.open('ab') as stream:
                    for block in response.iter_content(1024 * 1024):
                        stream.write(block)
            assert path.stat().st_size == expected_size
            print(f'{destination.name} part {index + 1}/16 complete', flush=True)
            return path
        except (requests.RequestException, AssertionError):
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


print('Downloading', url, 'bytes', total_size, 'connections', 16, flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    parts = list(executor.map(fetch_part, range(16)))
assembled = destination.with_name(destination.name + '.assembling')
with assembled.open('wb') as output:
    for part in parts:
        with part.open('rb') as stream:
            shutil.copyfileobj(stream, output, 1024 * 1024)
assert assembled.stat().st_size == total_size
assert checksum(assembled) == expected_digest, f'Checksum mismatch: {destination}'
assembled.replace(destination)
for part in parts:
    part.unlink()
parts_dir.rmdir()
print('Verified', algorithm, expected_digest, destination, flush=True)
