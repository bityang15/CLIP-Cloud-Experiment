import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verify(include_checkpoints):
    failures = []
    checked = {}
    skipped = set()
    for scope in ['B', 'E']:
        manifest = json.loads((ROOT / 'records' / f'{scope}_file_manifest.json').read_text(encoding='utf-8'))
        for name, expected in manifest['files'].items():
            if expected.get('checkpoint') and not include_checkpoints:
                skipped.add(name)
                continue
            checked[name] = (expected['size'], expected['sha256'])
    for scope, directory, filename in [('B', 'PromptSRC_expB', 'implementation_files.json'),
                                       ('E', 'DPC_expE', 'E_implementation_files.json')]:
        manifest = json.loads((ROOT / 'records' / filename).read_text(encoding='utf-8'))
        for name, expected in manifest['files'].items():
            relative = f'code/{directory}/{name}'
            checked[relative] = (None, expected)
    initial_sources = ROOT / 'records/source_sha256_before.txt'
    for line in initial_sources.read_text(encoding='utf-8').splitlines():
        expected, original = line.split(None, 1)
        relative = original.strip().removeprefix('/root/autodl-tmp/clip-exp/')
        checked[relative] = (None, expected)
    for name, (size, expected) in checked.items():
        path = ROOT / name
        if not path.is_file():
            failures.append(f'Missing file {name}')
        elif size is not None and path.stat().st_size != size:
            failures.append(f'Size mismatch {name}')
        elif sha256(path) != expected:
            failures.append(f'SHA256 mismatch {name}')
    print(json.dumps({'passed': not failures, 'unique_files_checked': len(checked),
                      'checkpoints_skipped': len(skipped), 'failures': failures}, indent=2))
    return not failures


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify the archived cloud experiment files.')
    parser.add_argument('--with-checkpoints', action='store_true', help='Also verify all downloaded Git LFS checkpoints.')
    raise SystemExit(0 if verify(parser.parse_args().with_checkpoints) else 1)
