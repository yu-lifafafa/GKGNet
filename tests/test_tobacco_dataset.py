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
    source_columns = source_columns or stack.tobacco.MODEL_LABEL_COLUMNS
    manifest = tmp_path / 'tobacco_clean_16cls_721.csv'
    _write_manifest(manifest, source_columns, rows)
    return stack.tobacco.TobaccoMultiLabelDataset(
        data_prefix=str(tmp_path / 'images'),
        ann_file=str(manifest),
        split=split,
        pipeline=[])


def test_fixed_label_protocol(tobacco_stack):
    tobacco = tobacco_stack.tobacco
    assert len(tobacco.MODEL_LABEL_COLUMNS) == 16
    assert len(tobacco.CLASSES) == 16
    codes = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 16, 17, 18, 19, 20]
    assert tobacco.MODEL_LABEL_COLUMNS == [
        f'y{i:02d}_c{c:02d}_{name}'
        for i, (c, name) in enumerate(zip(codes, tobacco.CLASSES))]


def test_single_and_multiple_retained_labels(tobacco_stack, tmp_path):
    source = tobacco_stack.tobacco.MODEL_LABEL_COLUMNS
    rows = [
        _row(source, 'single.jpg', 'train', positives=(2,), metadata='ok'),
        _row(source, 'multi.jpg', 'train', positives=(1, 8), metadata='ok'),
    ]
    dataset = _build_dataset(tobacco_stack, tmp_path, rows)
    assert dataset.data_infos[0]['gt_label'].tolist() == [0, 0, 1] + [0] * 13
    assert np.flatnonzero(dataset.data_infos[1]['gt_label']).tolist() == [1, 8]
    assert dataset.data_infos[1]['gt_label'].dtype == np.int8
    assert dataset.data_infos[1]['gt_label'].shape == (16,)


def test_all_indices_and_healthy(tobacco_stack, tmp_path):
    source = tobacco_stack.tobacco.MODEL_LABEL_COLUMNS
    rows = [_row(source, f'{i}.jpg', 'train', positives=(i,))
            for i in range(16)] + [_row(source, 'healthy.jpg', 'train')]
    dataset = _build_dataset(tobacco_stack, tmp_path, rows)
    np.testing.assert_array_equal(dataset.get_gt_labels()[:16], np.eye(16))
    assert dataset.get_cat_ids(16) == []


def test_missing_source_column_fails_fast(tobacco_stack, tmp_path):
    source = tobacco_stack.tobacco.MODEL_LABEL_COLUMNS
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
    source = tobacco_stack.tobacco.MODEL_LABEL_COLUMNS
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
    source = tobacco_stack.tobacco.MODEL_LABEL_COLUMNS
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
    source = tobacco_stack.tobacco.MODEL_LABEL_COLUMNS
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
    source = tobacco_stack.tobacco.MODEL_LABEL_COLUMNS
    rows = [
        _row(source, 'multi.jpg', 'train', positives=(0, 13, 15)),
        _row(source, 'zero.jpg', 'train', positives=()),
    ]
    dataset = _build_dataset(tobacco_stack, tmp_path, rows)
    assert dataset.get_cat_ids(0) == [0, 13, 15]
    assert dataset.get_cat_ids(1) == []


def test_class_balanced_dataset_accepts_tobacco_dataset(
        tobacco_stack, tmp_path):
    source = tobacco_stack.tobacco.MODEL_LABEL_COLUMNS
    rows = [
        _row(source, 'positive.jpg', 'train', positives=(3,)),
        _row(source, 'zero.jpg', 'train', positives=()),
    ]
    dataset = _build_dataset(tobacco_stack, tmp_path, rows)
    wrapped = tobacco_stack.wrappers.ClassBalancedDataset(
        dataset, oversample_thr=0.01)
    assert len(wrapped) == 2
    assert wrapped.dataset is dataset


def test_registry_builds_tobacco_and_class_balanced_wrapper(
        tobacco_stack, tmp_path):
    source = tobacco_stack.tobacco.MODEL_LABEL_COLUMNS
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


def test_real_clean16_csv(tobacco_stack):
    import hashlib
    manifest = REPO_ROOT / 'tobacco_clean_16cls_721.csv'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == (
        '60fee2bccbca44ed6447ec11ea6c4a8ed97434703d95aa0d905ff8f75d74db42')
    healthy = 0
    for split, size in [('train', 7805), ('val', 2230), ('test', 1115)]:
        dataset = tobacco_stack.tobacco.TobaccoMultiLabelDataset(
            ann_file=str(manifest), data_prefix='DATA_ROOT', split=split,
            pipeline=[])
        labels = dataset.get_gt_labels()
        assert labels.shape == (size, 16)
        assert (labels.sum(axis=0) > 0).all()
        healthy += (labels.sum(axis=1) == 0).sum()
        info = dataset.data_infos[0]
        assert info['img_info']['filename'].startswith('images/')
        assert Path(info['img_prefix']) == Path('DATA_ROOT') / split
        if split == 'train':
            wrapped = tobacco_stack.wrappers.ClassBalancedDataset(
                dataset, oversample_thr=0.01)
            print(f'Clean16 balanced train length: {len(wrapped)}')
            assert len(wrapped) == sum(
                int(np.ceil(r)) for r in wrapped._get_repeat_factors(dataset, 0.01))
    assert healthy == 231


def test_reordered_columns(tobacco_stack, tmp_path):
    source = tobacco_stack.tobacco.MODEL_LABEL_COLUMNS
    manifest = tmp_path / 'manifest.csv'
    _write_manifest(manifest, source,
                    [_row(source, 'images/a.jpg', 'train', (1, 15))],
                    ['split', *reversed(source), 'file_name'])
    ds = tobacco_stack.tobacco.TobaccoMultiLabelDataset(
        ann_file=str(manifest), data_prefix='root', split='train', pipeline=[])
    assert ds.get_cat_ids(0) == [1, 15]


@pytest.mark.parametrize('duplicate', ['file_name', 'split', 'y00_c02_weather_fleck'])
def test_duplicate_columns_rejected(tobacco_stack, tmp_path, duplicate):
    source = tobacco_stack.tobacco.MODEL_LABEL_COLUMNS
    manifest = tmp_path / 'manifest.csv'
    _write_manifest(manifest, source, [_row(source, 'a.jpg', 'train')],
                    ['file_name', 'split', *source, duplicate])
    with pytest.raises(ValueError, match='uplicate|exactly once'):
        tobacco_stack.tobacco.TobaccoMultiLabelDataset(
            ann_file=str(manifest), data_prefix='root', split='train', pipeline=[])


@pytest.mark.parametrize('extra', [
    {'split': 'dev'}, {'file_name': ''},
    {'positive_label_count': '1'}, {'is_healthy': '0'}])
def test_invalid_row_metadata(tobacco_stack, tmp_path, extra):
    source = tobacco_stack.tobacco.MODEL_LABEL_COLUMNS
    row = _row(source, 'a.jpg', 'train')
    row.update(extra)
    manifest = tmp_path / 'manifest.csv'
    _write_manifest(manifest, source, [row], list(row))
    with pytest.raises(ValueError):
        tobacco_stack.tobacco.TobaccoMultiLabelDataset(
            ann_file=str(manifest), data_prefix='root', split='train', pipeline=[])


@pytest.mark.parametrize('size', [18, 19])
def test_legacy_manifest_rejected(tobacco_stack, tmp_path, size):
    # Old source order, including removed diseases; no compatibility fallback.
    names = ['weather_fleck', 'potato_virus_y', 'tobacco_mosaic_virus',
             'cucumber_mosaic_virus', 'alternaria_leaf_spot', 'black_shank',
             'bacterial_wilt', 'hollow_stalk', 'black_root_rot', 'wildfire',
             'potassium_deficiency', 'magnesium_deficiency', 'powdery_mildew',
             'root_knot_nematodes', 'anthracnose', 'frogeye_leaf_spot',
             'angular_leaf_spot', 'sunburn', 'herbicide_phytotoxicity']
    columns = [f'y{i:02d}_c{i+2:02d}_{n}' for i, n in enumerate(names)]
    if size == 18:
        columns.pop(13)
    with pytest.raises(ValueError):
        _build_dataset(tobacco_stack, tmp_path,
                       [_row(columns, 'a.jpg', 'train')], source_columns=columns)


def test_custom_class_order_rejected(tobacco_stack, tmp_path):
    source = tobacco_stack.tobacco.MODEL_LABEL_COLUMNS
    manifest = tmp_path / 'manifest.csv'
    _write_manifest(manifest, source, [_row(source, 'a.jpg', 'train')])
    with pytest.raises(ValueError, match='class'):
        tobacco_stack.tobacco.TobaccoMultiLabelDataset(
            data_prefix='root', ann_file=str(manifest), split='train',
            classes=tobacco_stack.tobacco.CLASSES[::-1], pipeline=[])
