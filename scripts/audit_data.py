import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image
from scipy.io import loadmat

ROOT = Path('/root/autodl-tmp/clip-exp')
DATA = ROOT / 'datasets'
RECORDS = ROOT / 'records'
sys.path.insert(0, str(ROOT / 'code' / 'PromptSRC'))

from train import extend_cfg
from dassl.config import get_cfg_default
from dassl.utils import set_random_seed
from datasets.dtd import DescribableTextures
from datasets.oxford_flowers import OxfordFlowers
from datasets.oxford_pets import OxfordPets


def digest(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def rows(items):
    return [dict(image_id=Path(item.impath).relative_to(DATA).as_posix(),
                 label=item.label, classname=item.classname) for item in items]


dataset_specs = [
    ('dtd', DescribableTextures, 'images', 'split_zhou_DescribableTextures.json', 47, 5640),
    ('oxford_flowers', OxfordFlowers, 'jpg', 'split_zhou_OxfordFlowers.json', 102, 8189),
]
source_urls = {
    'split_zhou_DescribableTextures.json': 'https://drive.google.com/file/d/1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x/view',
    'cat_to_name.json': 'https://drive.google.com/file/d/1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0/view',
    'split_zhou_OxfordFlowers.json': 'https://drive.google.com/file/d/1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT/view',
    'imagelabels.mat': 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat',
}
manifest = {'created_at': datetime.now().astimezone().isoformat(), 'datasets': {}}
checks = {'passed': False, 'datasets': {}}
for name, dataset_type, image_folder, split_name, expected_classes, expected_images in dataset_specs:
    directory = DATA / name
    split_path = directory / split_name
    split = json.loads(split_path.read_text(encoding='utf-8'))
    identities = {}
    classnames = {}
    image_records = {}
    for split_group in ['train', 'val', 'test']:
        group_rows = split[split_group]
        image_ids = []
        for relative_name, label, classname in group_rows:
            path = (directory / image_folder / relative_name).resolve()
            assert path.is_relative_to(directory), f'Unexpected path {path}'
            assert path.is_file(), str(path)
            assert classnames.setdefault(label, classname) == classname
            image_id = path.relative_to(DATA).as_posix()
            image_ids.append(image_id)
            if image_id not in image_records:
                with Image.open(path) as image:
                    image.verify()
                image_records[image_id] = dict(size=path.stat().st_size, sha256=digest(path))
        assert len(image_ids) == len(set(image_ids)), f'Duplicate images in {name}/{split_group}'
        identities[split_group] = set(image_ids)
    assert len(classnames) == expected_classes
    assert sorted(classnames) == list(range(expected_classes))
    assert len(image_records) == expected_images
    intersections = {
        'train_val': len(identities['train'] & identities['val']),
        'train_test': len(identities['train'] & identities['test']),
        'val_test': len(identities['val'] & identities['test']),
    }
    assert not any(intersections.values()), intersections
    class_map = {'all': classnames}
    base_count = (expected_classes + 1) // 2
    for group, start, end in [('base', 0, base_count), ('new', base_count, expected_classes)]:
        class_map[group] = [dict(label=label - start, original_label=label, classname=classnames[label])
                            for label in range(start, end)]
    save_json(RECORDS / 'manifests' / name / 'classes.json', class_map)
    seed_checks = {}
    fixed_test = None
    for seed in [1, 2, 3]:
        cfg = get_cfg_default()
        extend_cfg(cfg)
        cfg.DATASET.ROOT = str(DATA)
        cfg.DATASET.NUM_SHOTS = 16
        cfg.DATASET.SUBSAMPLE_CLASSES = 'all'
        cfg.SEED = seed
        set_random_seed(seed)
        dataset = dataset_type(cfg)
        base = OxfordPets.subsample_classes(dataset.train_x, dataset.val, dataset.test, subsample='base')
        new = OxfordPets.subsample_classes(dataset.train_x, dataset.val, dataset.test, subsample='new')
        train_counts = Counter(item.label for item in base[0])
        val_counts = Counter(item.label for item in base[1])
        assert len(train_counts) == base_count and set(train_counts.values()) == {16}
        assert len(val_counts) == base_count and set(val_counts.values()) == {4}
        sample_groups = dict(train_base=base[0], val_base=base[1], test_base=base[2], test_new=new[2])
        sample_hashes = {}
        for group, items in sample_groups.items():
            output = RECORDS / 'manifests' / name / f'seed{seed}' / f'{group}.json'
            save_json(output, rows(items))
            sample_hashes[group] = digest(output)
        current_test = [sample_hashes['test_base'], sample_hashes['test_new']]
        if fixed_test is not None:
            assert current_test == fixed_test, 'Test identities changed between seeds'
        fixed_test = current_test
        seed_checks[str(seed)] = dict(counts={group: len(items) for group, items in sample_groups.items()},
                                     manifest_sha256=sample_hashes,
                                     shared_methods=['maple', 'tcpt', 'promptsrc'],
                                     cache_sha256=digest(directory / 'split_fewshot' / f'shot_16-seed_{seed}.pkl'))
    metadata_files = [split_path]
    if name == 'oxford_flowers':
        category_path = directory / 'cat_to_name.json'
        categories = json.loads(category_path.read_text())
        assert len(categories) == expected_classes
        assert all(categories[str(label + 1)].replace('_', ' ') == classname.replace('_', ' ')
                   for label, classname in classnames.items())
        original_labels = loadmat(directory / 'imagelabels.mat')['labels'].reshape(-1)
        assert all(int(original_labels[int(Path(image_name).stem.split('_')[-1]) - 1]) - 1 == label
                   for group_rows in split.values() for image_name, label, _ in group_rows)
        metadata_files += [category_path, directory / 'imagelabels.mat']
    manifest['datasets'][name] = dict(
        images=image_records,
        metadata={path.relative_to(DATA).as_posix(): dict(size=path.stat().st_size, sha256=digest(path),
                                                       source_url=source_urls[path.name])
                  for path in metadata_files},
        class_count=expected_classes, base_classes=base_count, new_classes=expected_classes - base_count,
        raw_split_sizes={group: len(items) for group, items in split.items()},
    )
    checks['datasets'][name] = dict(passed=True, image_count=len(image_records),
                                  split_intersections=intersections, seeds=seed_checks)
    print(name, checks['datasets'][name], flush=True)
checks['passed'] = True
manifest['archives'] = {
    'dtd-r1.0.1.tar.gz': dict(source_url='https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz',
                            torchvision_md5='fff73e5086ae6bdbea199a49dfb8a4c1'),
    '102flowers.tgz': dict(source_url='https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz',
                          torchvision_md5='52808999861908f626f3c1f4e79d11fa'),
}
for filename, archive_info in manifest['archives'].items():
    path = ROOT / 'downloads' / filename
    archive_info.update(size=path.stat().st_size, sha256=digest(path))
weight_path = ROOT / 'weights' / 'ViT-B-16.pt'
weight_hash = digest(weight_path)
assert weight_hash == '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'
manifest['weight'] = dict(path=str(weight_path), size=weight_path.stat().st_size, sha256=weight_hash)
save_json(RECORDS / 'data_manifest.json', manifest)
save_json(RECORDS / 'data_checks.json', checks)
print('All data checks passed.', flush=True)
