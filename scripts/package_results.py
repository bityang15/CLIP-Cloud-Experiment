import hashlib
import json
import tarfile
from datetime import datetime
from pathlib import Path

ROOT = Path('/root/autodl-tmp/clip-exp')
OUTPUT = ROOT / 'outputs' / 'B'
RECORDS = ROOT / 'records'


def checksum(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


progress = json.loads((RECORDS / 'B_progress.json').read_text())
analysis = json.loads((RECORDS / 'analysis_checks.json').read_text())
assert progress['completed'] == 18 and analysis['passed']
checkpoints = []
for directory in progress['selected_runs'].values():
    directory = Path(directory)
    status = json.loads((directory / 'run_status.json').read_text())
    assert status['status'] == 'complete' and status['cross_process_probe_exact']
    candidates = list(directory.rglob('model.pth.tar-*'))
    assert len(candidates) == 1, directory
    checkpoints.extend(candidates)
assert len(checkpoints) == 18
assert len(list(OUTPUT.rglob('predictions.npz'))) == 58

excluded_names = {'B_file_manifest.json', 'B_packaging.log', 'B_packaging.exit', 'B_package_checks.json'}
record_files = [path for path in RECORDS.rglob('*') if path.is_file()
                and not path.name.endswith(('.tar', '.tar.gz', '.tgz')) and path.name not in excluded_names]
output_files = [path for path in OUTPUT.rglob('*') if path.is_file()]
all_files = sorted(record_files + output_files)
manifest = dict(created_at=datetime.now().astimezone().isoformat(),
                formal_checkpoints=18, prediction_exports=58, files={})
for path in all_files:
    manifest['files'][path.relative_to(ROOT).as_posix()] = dict(
        size=path.stat().st_size, sha256=checksum(path), checkpoint=path in checkpoints)
manifest_path = RECORDS / 'B_file_manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')

evidence_path = RECORDS / 'B_evidence.tar.gz'
with tarfile.open(evidence_path, 'w:gz', compresslevel=3) as archive:
    for path in all_files:
        if path not in checkpoints:
            archive.add(path, arcname=path.relative_to(ROOT).as_posix())
    archive.add(manifest_path, arcname=manifest_path.relative_to(ROOT).as_posix())
checkpoint_path = RECORDS / 'B_checkpoints.tar'
with tarfile.open(checkpoint_path, 'w') as archive:
    for path in checkpoints:
        archive.add(path, arcname=path.relative_to(ROOT).as_posix())
report = dict(passed=True, files=len(all_files), checkpoints=len(checkpoints), archives={})
for path in [evidence_path, checkpoint_path]:
    report['archives'][path.name] = dict(size=path.stat().st_size, sha256=checksum(path))
(RECORDS / 'B_package_checks.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, indent=2), flush=True)
