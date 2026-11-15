import hashlib
import json
import tarfile
from datetime import datetime
from pathlib import Path

ROOT = Path('/root/autodl-tmp/clip-exp')
RECORDS = ROOT / 'records'


def checksum(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


progress = json.loads((RECORDS / 'E_progress.json').read_text())
analysis = json.loads((RECORDS / 'E_analysis_checks.json').read_text())
assert progress['completed'] == 12 and analysis['passed']
checkpoints = []
for directory in progress['selected_runs'].values():
    directory = Path(directory)
    status = json.loads((directory / 'run_status.json').read_text())
    assert status['status'] == 'complete' and status['cross_process_probe_exact']
    candidates = list(directory.rglob('model.pth.tar-*'))
    assert len(candidates) == 1
    checkpoints.extend(candidates)
assert len(checkpoints) == 12
assert sum(len(list((Path(directory) / 'predictions').rglob('predictions.npz')))
           for directory in progress['selected_runs'].values()) == 36

excluded = {'E_file_manifest.json', 'E_package_checks.json', 'E_packaging.log', 'E_packaging.exit'}
files = [path for path in RECORDS.glob('E_*') if path.is_file()
         and path.name not in excluded and not path.name.endswith(('.tar', '.tar.gz'))]
files += [path for path in (ROOT / 'outputs/E').rglob('*') if path.is_file()]
files += [path for path in (ROOT / 'code/DPC_expE').rglob('*')
          if path.is_file() and '__pycache__' not in path.parts]
files += [path for path in (ROOT / 'datasets/SPLE_Database').rglob('*') if path.is_file()]
files += [ROOT / 'downloads/SPLE_database.rar']
files += [ROOT / 'code/PromptSRC_expB' / name
          for name in ['experiment.py', 'experiment_metrics.py', 'run_matrix.py']]
files += [ROOT / 'scripts' / name for name in ['analyze_e.py', 'package_results_e.py']]
files = sorted(set(files))
manifest = dict(created_at=datetime.now().astimezone().isoformat(), formal_checkpoints=12,
                prediction_exports=36, reused_B_runs=18, files={})
for path in files:
    manifest['files'][path.relative_to(ROOT).as_posix()] = dict(
        size=path.stat().st_size, sha256=checksum(path), checkpoint=path in checkpoints)
manifest_path = RECORDS / 'E_file_manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
archive_path = RECORDS / 'E_evidence.tar.gz'
with tarfile.open(archive_path, 'w:gz', compresslevel=3) as archive:
    for path in files:
        archive.add(path, arcname=path.relative_to(ROOT).as_posix())
    archive.add(manifest_path, arcname=manifest_path.relative_to(ROOT).as_posix())
report = dict(passed=True, files=len(files), checkpoints=12, archives={
    archive_path.name: dict(size=archive_path.stat().st_size, sha256=checksum(archive_path))})
(RECORDS / 'E_package_checks.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, indent=2), flush=True)
