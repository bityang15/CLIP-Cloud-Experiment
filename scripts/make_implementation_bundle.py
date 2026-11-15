import difflib
import hashlib
import json
import tarfile
from pathlib import Path

workspace = Path(__file__).resolve().parents[3]
project = workspace / '实验复现' / '方案B'
original = workspace / '综述成稿' / '最终版' / '代码提交包' / 'official' / 'PromptSRC'
working = project / 'PromptSRC_expB'
records = project / 'records'
records.mkdir(exist_ok=True)


def source_files(directory):
    return {path.relative_to(directory).as_posix(): path for path in directory.rglob('*')
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'}


before = source_files(original)
after = source_files(working)
changes = []
for name in sorted(before.keys() | after.keys()):
    old_bytes = before[name].read_bytes() if name in before else b''
    new_bytes = after[name].read_bytes() if name in after else b''
    if old_bytes != new_bytes:
        changes.extend(difflib.unified_diff(
            old_bytes.decode('utf-8').splitlines(keepends=True),
            new_bytes.decode('utf-8').splitlines(keepends=True),
            fromfile='a/' + name if name in before else '/dev/null',
            tofile='b/' + name if name in after else '/dev/null'))
patch_path = records / 'implementation.patch'
patch_path.write_text(''.join(changes), encoding='utf-8')
manifest = dict(source_commit='bb95c77b634d63488f2cad81ff4a72d53bdd06d5',
                patch_sha256=hashlib.sha256(patch_path.read_bytes()).hexdigest(),
                files={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in after.items()})
(records / 'implementation_files.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
bundle = workspace / '实验复现' / '.b-implementation.tar.gz'
with tarfile.open(bundle, 'w:gz') as archive:
    for name, path in after.items():
        archive.add(path, arcname='code/PromptSRC_expB/' + name)
    archive.add(patch_path, arcname='records/implementation.patch')
    archive.add(records / 'implementation_files.json', arcname='records/implementation_files.json')
print(json.dumps(dict(bundle=str(bundle), size=bundle.stat().st_size,
                      sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
                      patch_sha256=manifest['patch_sha256'], source_files=len(after))))
