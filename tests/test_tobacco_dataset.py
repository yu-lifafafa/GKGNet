import csv
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


class _Registry:
    def __init__(self, name, parent=None):
        self.name = name
        self.module_dict = {}

    def register_module(self):
        def decorator(cls):
            self.module_dict[cls.__name__] = cls
            return cls

        return decorator

    def build(self, cfg, default_args=None):
        return _build_from_cfg(cfg, self, default_args)


def _build_from_cfg(cfg, registry, default_args=None):
    args = dict(default_args or {})
    args.update(cfg)
    cls = registry.module_dict[args.pop('type')]
    return cls(**args)


class _Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            if isinstance(transform, dict):
                raise AssertionError('Dict pipelines require a real MMCV runtime')
            data = transform(data)
        return data


def _load_source(module_name, relative_path):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def tobacco_stack():
    """Load the real dataset sources with a minimal MMCV compatibility shim."""
    prefixes = ('mmcv', 'mmcls')
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name in prefixes or name.startswith(('mmcv.', 'mmcls.'))
    }
    for name in list(sys.modules):
        if name in prefixes or name.startswith(('mmcv.', 'mmcls.')):
            sys.modules.pop(name)

    mmcv = types.ModuleType('mmcv')
    mmcv.__version__ = '1.5.0'
    mmcv.list_from_file = lambda path: Path(path).read_text().splitlines()
    mmcv_parallel = types.ModuleType('mmcv.parallel')
    mmcv_parallel.collate = lambda batch, **kwargs: batch
    mmcv_runner = types.ModuleType('mmcv.runner')
    mmcv_runner.get_dist_info = lambda: (0, 1)
    mmcv_utils = types.ModuleType('mmcv.utils')
    mmcv_utils.Registry = _Registry
    mmcv_utils.build_from_cfg = _build_from_cfg
    mmcv_utils.digit_version = lambda version: tuple(
        int(part) for part in version.split('+')[0].split('.') if part.isdigit())
    mmcv.parallel = mmcv_parallel
    mmcv.runner = mmcv_runner
    mmcv.utils = mmcv_utils
    sys.modules.update({
        'mmcv': mmcv,
        'mmcv.parallel': mmcv_parallel,
        'mmcv.runner': mmcv_runner,
        'mmcv.utils': mmcv_utils,
    })

    mmcls = types.ModuleType('mmcls')
    mmcls.__path__ = [str(REPO_ROOT / 'mmcls')]
    datasets_pkg = types.ModuleType('mmcls.datasets')
    datasets_pkg.__path__ = [str(REPO_ROOT / 'mmcls' / 'datasets')]
    sys.modules['mmcls'] = mmcls
    sys.modules['mmcls.datasets'] = datasets_pkg

    builder = _load_source('mmcls.datasets.builder', 'mmcls/datasets/builder.py')

    pipelines = types.ModuleType('mmcls.datasets.pipelines')
    pipelines.Compose = _Compose
    sys.modules['mmcls.datasets.pipelines'] = pipelines

    core = types.ModuleType('mmcls.core')
    core.average_performance = lambda *args, **kwargs: ()
    core.mAP = lambda *args, **kwargs: 0.0
    core_eval = types.ModuleType('mmcls.core.evaluation')
    core_eval.precision_recall_f1 = lambda *args, **kwargs: ()
    core_eval.support = lambda *args, **kwargs: 0
    sys.modules['mmcls.core'] = core
    sys.modules['mmcls.core.evaluation'] = core_eval

    models = types.ModuleType('mmcls.models')
    losses = types.ModuleType('mmcls.models.losses')
    losses.accuracy = lambda *args, **kwargs: ()
    sys.modules['mmcls.models'] = models
    sys.modules['mmcls.models.losses'] = losses

    _load_source('mmcls.datasets.base_dataset', 'mmcls/datasets/base_dataset.py')
    multi_label = _load_source(
        'mmcls.datasets.multi_label', 'mmcls/datasets/multi_label.py')
    wrappers = _load_source(
        'mmcls.datasets.dataset_wrappers',
        'mmcls/datasets/dataset_wrappers.py')
    tobacco = _load_source('mmcls.datasets.tobacco', 'mmcls/datasets/tobacco.py')

    yield types.SimpleNamespace(
        builder=builder,
        multi_label=multi_label,
        wrappers=wrappers,
        tobacco=tobacco,
    )

    for name in list(sys.modules):
        if name in prefixes or name.startswith(('mmcv.', 'mmcls.')):
            sys.modules.pop(name)
    sys.modules.update(saved)


def _row(source_columns, file_name, split, positives=(), **extra):
    row = {column: '0' for column in source_columns}
    for index in positives:
        row[source_columns[index]] = '1'
    row.update(file_name=file_name, split=split, **extra)
    return row


def _write_manifest(path, source_columns, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = ['file_name', 'split', *source_columns, 'metadata']
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_dataset(stack, tmp_path, rows, split='train', source_columns=None):
    source_columns = source_columns or stack.tobacco.SOURCE_LABEL_COLUMNS
    manifest = tmp_path / 'image_manifest_updated.csv'
    _write_manifest(manifest, source_columns, rows)
    return stack.tobacco.TobaccoMultiLabelDataset(
        data_prefix=str(tmp_path / 'images'),
        ann_file=str(manifest),
        split=split,
        pipeline=[])


def test_fixed_label_protocol(tobacco_stack):
    tobacco = tobacco_stack.tobacco
    assert len(tobacco.SOURCE_LABEL_COLUMNS) == 19
    assert len(tobacco.MODEL_LABEL_COLUMNS) == 18
    assert len(tobacco.CLASSES) == 18
    assert tobacco.EXCLUDED_SOURCE_LABEL in tobacco.SOURCE_LABEL_COLUMNS
    assert tobacco.EXCLUDED_SOURCE_LABEL not in tobacco.MODEL_LABEL_COLUMNS
    assert tobacco.MODEL_LABEL_COLUMNS == (
        tobacco.SOURCE_LABEL_COLUMNS[:13]
        + tobacco.SOURCE_LABEL_COLUMNS[14:])


def test_single_and_multiple_retained_labels(tobacco_stack, tmp_path):
    source = tobacco_stack.tobacco.SOURCE_LABEL_COLUMNS
    rows = [
        _row(source, 'single.jpg', 'train', positives=(2,), metadata='ok'),
        _row(source, 'multi.jpg', 'train', positives=(1, 8), metadata='ok'),
    ]
    dataset = _build_dataset(tobacco_stack, tmp_path, rows)
    assert dataset.data_infos[0]['gt_label'].tolist() == [0, 0, 1] + [0] * 15
    assert np.flatnonzero(dataset.data_infos[1]['gt_label']).tolist() == [1, 8]
    assert dataset.data_infos[1]['gt_label'].dtype == np.int8
    assert dataset.data_infos[1]['gt_label'].shape == (18,)


def test_root_knot_is_removed_without_removing_image(tobacco_stack, tmp_path):
    source = tobacco_stack.tobacco.SOURCE_LABEL_COLUMNS
    rows = [
        _row(source, 'mixed.jpg', 'train', positives=(4, 13)),
        _row(source, 'root_only.jpg', 'train', positives=(13,)),
    ]
    dataset = _build_dataset(tobacco_stack, tmp_path, rows)
    assert len(dataset) == 2
    assert np.flatnonzero(dataset.data_infos[0]['gt_label']).tolist() == [4]
    assert dataset.data_infos[1]['gt_label'].tolist() == [0] * 18


def test_post_exclusion_indices_do_not_shift_incorrectly(
        tobacco_stack, tmp_path):
    source = tobacco_stack.tobacco.SOURCE_LABEL_COLUMNS
    rows = [
        _row(source, 'source14.jpg', 'train', positives=(14,)),
        _row(source, 'source18.jpg', 'train', positives=(18,)),
    ]
    dataset = _build_dataset(tobacco_stack, tmp_path, rows)
    assert np.flatnonzero(dataset.data_infos[0]['gt_label']).tolist() == [13]
    assert np.flatnonzero(dataset.data_infos[1]['gt_label']).tolist() == [17]


def test_missing_source_column_fails_fast(tobacco_stack, tmp_path):
    source = tobacco_stack.tobacco.SOURCE_LABEL_COLUMNS
    missing = source[7]
    columns = [column for column in source if column != missing]
    manifest = tmp_path / 'manifest.csv'
    rows = [_row(source, 'image.jpg', 'train')]
    fieldnames = ['file_name', 'split', *columns]
    with manifest.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match=missing):
        tobacco_stack.tobacco.TobaccoMultiLabelDataset(
            data_prefix=str(tmp_path),
            ann_file=str(manifest),
            split='train',
            pipeline=[])


@pytest.mark.parametrize('invalid_value', ['', '2', '-1', 'yes', 'nan'])
def test_invalid_label_value_fails_fast(
        tobacco_stack, tmp_path, invalid_value):
    source = tobacco_stack.tobacco.SOURCE_LABEL_COLUMNS
    row = _row(source, 'bad.jpg', 'train')
    row[source[5]] = invalid_value
    manifest = tmp_path / 'manifest.csv'
    _write_manifest(manifest, source, [row])

    with pytest.raises(ValueError, match=source[5]):
        tobacco_stack.tobacco.TobaccoMultiLabelDataset(
            data_prefix=str(tmp_path),
            ann_file=str(manifest),
            split='train',
            pipeline=[])


def test_split_filtering_and_empty_split_error(tobacco_stack, tmp_path):
    source = tobacco_stack.tobacco.SOURCE_LABEL_COLUMNS
    rows = [
        _row(source, 'train.jpg', 'train'),
        _row(source, 'val.jpg', 'val'),
        _row(source, 'test.jpg', 'test'),
    ]
    manifest = tmp_path / 'manifest.csv'
    _write_manifest(manifest, source, rows)

    for split in ('train', 'val', 'test'):
        dataset = tobacco_stack.tobacco.TobaccoMultiLabelDataset(
            data_prefix=str(tmp_path / 'images'),
            ann_file=str(manifest),
            split=split,
            pipeline=[])
        assert len(dataset) == 1
        assert dataset.data_infos[0]['img_info']['filename'] == f'{split}.jpg'
        assert Path(dataset.data_infos[0]['img_prefix']) == (
            tmp_path / 'images' / split)

    with pytest.raises(ValueError, match='dev'):
        tobacco_stack.tobacco.TobaccoMultiLabelDataset(
            data_prefix=str(tmp_path),
            ann_file=str(manifest),
            split='dev',
            pipeline=[])


def test_image_path_is_data_root_plus_split_plus_file_name(
        tobacco_stack, tmp_path):
    source = tobacco_stack.tobacco.SOURCE_LABEL_COLUMNS
    image_root = tmp_path / 'images'
    image_path = image_root / 'train' / 'nested' / 'sample.jpg'
    image_path.parent.mkdir(parents=True)
    Image.new('RGB', (8, 6), color=(10, 20, 30)).save(image_path)

    dataset = _build_dataset(
        tobacco_stack,
        tmp_path,
        [_row(source, 'nested/sample.jpg', 'train', positives=(0, ))])
    info = dataset.data_infos[0]
    resolved_path = Path(info['img_prefix']) / info['img_info']['filename']
    assert resolved_path == image_path
    assert resolved_path.is_file()


def test_get_cat_ids_for_multilabel_and_zero_label_samples(
        tobacco_stack, tmp_path):
    source = tobacco_stack.tobacco.SOURCE_LABEL_COLUMNS
    rows = [
        _row(source, 'multi.jpg', 'train', positives=(0, 14, 18)),
        _row(source, 'zero.jpg', 'train', positives=(13,)),
    ]
    dataset = _build_dataset(tobacco_stack, tmp_path, rows)
    assert dataset.get_cat_ids(0) == [0, 13, 17]
    assert dataset.get_cat_ids(1) == []


def test_class_balanced_dataset_accepts_tobacco_dataset(
        tobacco_stack, tmp_path):
    source = tobacco_stack.tobacco.SOURCE_LABEL_COLUMNS
    rows = [
        _row(source, 'positive.jpg', 'train', positives=(3,)),
        _row(source, 'zero.jpg', 'train', positives=(13,)),
    ]
    dataset = _build_dataset(tobacco_stack, tmp_path, rows)
    wrapped = tobacco_stack.wrappers.ClassBalancedDataset(
        dataset, oversample_thr=0.01)
    assert len(wrapped) == 2
    assert wrapped.dataset is dataset


def test_registry_builds_tobacco_and_class_balanced_wrapper(
        tobacco_stack, tmp_path):
    source = tobacco_stack.tobacco.SOURCE_LABEL_COLUMNS
    manifest = tmp_path / 'manifest.csv'
    _write_manifest(
        manifest, source,
        [_row(source, 'image.jpg', 'train', positives=(1, 2))])
    cfg = dict(
        type='ClassBalancedDataset',
        oversample_thr=0.01,
        dataset=dict(
            type='TobaccoMultiLabelDataset',
            data_prefix=str(tmp_path / 'images'),
            ann_file=str(manifest),
            split='train',
            pipeline=[]))
    dataset = tobacco_stack.builder.build_dataset(cfg)
    assert isinstance(dataset, tobacco_stack.wrappers.ClassBalancedDataset)
    assert isinstance(
        dataset.dataset, tobacco_stack.tobacco.TobaccoMultiLabelDataset)


def test_dataset_is_exported_from_package_init():
    init_source = (REPO_ROOT / 'mmcls' / 'datasets' / '__init__.py').read_text(
        encoding='utf-8')
    assert 'from .tobacco import TobaccoMultiLabelDataset' in init_source
    assert "'TobaccoMultiLabelDataset'" in init_source
