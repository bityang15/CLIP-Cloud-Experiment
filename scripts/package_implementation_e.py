import difflib
import hashlib
import json
import tarfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
EXPERIMENT = PROJECT / '实验复现/方案E'
SOURCE = PROJECT / '综述成稿/最终版/代码提交包/official/DPC'
WORK = EXPERIMENT / 'DPC_expE'
RECORDS = EXPERIMENT / 'records'

files = {}
patch = []
for path in sorted(WORK.rglob('*')):
    if not path.is_file() or '__pycache__' in path.parts:
        continue
    relative = path.relative_to(WORK).as_posix()
    files[relative] = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    original = SOURCE / relative
    if original.exists() and path.read_bytes() == original.read_bytes():
        continue
    try:
        before = original.read_text(encoding='utf-8-sig').splitlines(keepends=True) if original.exists() else []
        after = path.read_text(encoding='utf-8-sig').splitlines(keepends=True)
    except UnicodeDecodeError:
        continue
    patch.extend(difflib.unified_diff(before, after, fromfile='a/' + relative, tofile='b/' + relative))
RECORDS.mkdir(exist_ok=True)
(RECORDS / 'E_implementation.patch').write_text(''.join(patch), encoding='utf-8')
manifest = dict(dpc_commit='01e6b599d6515c38f4d723a8ea844ac4d1abfadd',
                dept_commit='d4ff41684bac42da30ebb6898496772dec3157f2',
                shared_B_modules=['experiment.py', 'experiment_metrics.py', 'run_matrix.py'], files=files)
(RECORDS / 'E_implementation_files.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
bundle = EXPERIMENT / 'E_implementation.tar.gz'
with tarfile.open(bundle, 'w:gz', compresslevel=3) as archive:
    archive.add(WORK, arcname='code/DPC_expE', filter=lambda item: None if '__pycache__' in item.name else item)
    for name in ['E_implementation.patch', 'E_implementation_files.json']:
        archive.add(RECORDS / name, arcname='records/' + name)
    archive.add(EXPERIMENT / 'annotation_checks.json', arcname='records/E_annotation_checks.json')
    for path in (EXPERIMENT / 'inputs/restricted_annotations').glob('*.json'):
        archive.add(path, arcname='datasets/SPLE_Database/' + path.name)
    for path in (EXPERIMENT / 'inputs/official_annotations').glob('*.json'):
        archive.add(path, arcname='datasets/SPLE_Database/official/' + path.name)
    archive.add(EXPERIMENT / 'inputs/SPLE_database.rar', arcname='downloads/SPLE_database.rar')
print(json.dumps(dict(source_files=len(files), bundle_size=bundle.stat().st_size,
                     sha256=hashlib.file_digest(bundle.open('rb'), 'sha256').hexdigest())))
