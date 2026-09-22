import copy
import runpy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / 'configs' / 'gkgnet' / 'gkgnet_tobacco_448.py'
REFERENCE_CONFIG_PATH = (
    REPO_ROOT / 'configs' / 'gkgnet' / 'gkgnet_tobacco_576.py')
COCO_CONFIG_PATH = REPO_ROOT / 'configs' / 'gkgnet' / 'gkgnet_coco_576.py'


def _load_config():
    return runpy.run_path(str(CONFIG_PATH))


def test_tobacco_448_config_exists():
    assert CONFIG_PATH.is_file()


def test_tobacco_config_model_contract():
    cfg = _load_config()
    assert cfg['model']['backbone']['n_classes'] == 18
    assert cfg['model']['head']['num_classes'] == 18
    assert cfg['model']['backbone']['size'] == 448
    assert cfg['model']['backbone']['choice'] == 's'
    assert cfg['model']['backbone']['k'] == 9
    assert cfg['model']['backbone']['k_label_gcn'] == 9
    assert cfg['model']['head']['softmax'] is False
    assert cfg['model']['head']['loss'] == dict(
        type='AsymmetricLoss', gamma_pos=0.0, gamma_neg=2.0, clip=0.05)


def test_tobacco_config_dataset_contract():
    cfg = _load_config()
    data = cfg['data']
    train = data['train']
    assert train['type'] == 'ClassBalancedDataset'
    assert train['oversample_thr'] == 0.01
    assert train['dataset']['type'] == 'TobaccoMultiLabelDataset'
    assert train['dataset']['split'] == 'train'
    assert data['val']['type'] == 'TobaccoMultiLabelDataset'
    assert data['val']['split'] == 'val'
    assert data['test']['type'] == 'TobaccoMultiLabelDataset'
    assert data['test']['split'] == 'test'


def test_tobacco_config_resolution_and_pipeline_contract():
    cfg = _load_config()
    assert cfg['crop_size'] == 448
    crop_mixup = next(
        transform for transform in cfg['train_pipeline']
        if transform['type'] == 'CropMixup')
    test_resize = next(
        transform for transform in cfg['test_pipeline']
        if transform['type'] == 'Resize')
    assert crop_mixup['size'] == 448
    assert test_resize['size'] == 448
    assert cfg['data']['val']['pipeline'] is cfg['test_pipeline']
    assert cfg['data']['test']['pipeline'] is cfg['test_pipeline']
    assert [item['type'] for item in cfg['test_pipeline']] == [
        'LoadImageFromFile', 'Resize', 'Normalize', 'ImageToTensor', 'Collect'
    ]


def test_tobacco_config_keeps_official_training_recipe():
    cfg = _load_config()
    assert cfg['data']['samples_per_gpu'] == 16
    assert cfg['runner'] == dict(type='EpochBasedRunner', max_epochs=80)
    assert cfg['optimizer']['type'] == 'AdamW'
    assert cfg['optimizer']['lr'] == 1e-4
    assert cfg['optimizer']['weight_decay'] == 0.05
    assert cfg['optimizer_config'] == dict(grad_clip=dict(max_norm=5.0))
    assert cfg['lr_config']['step'] == [10, 50]
    assert cfg['lr_config']['warmup'] == 'linear'
    assert cfg['lr_config']['warmup_iters'] == 5
    assert cfg['lr_config']['warmup_by_epoch'] is True
    assert cfg['fp16'] == dict(loss_scale='dynamic')


def test_tobacco_config_does_not_drift_from_official_recipe():
    cfg = _load_config()
    reference = runpy.run_path(str(REFERENCE_CONFIG_PATH))

    for key in (
            'img_norm_cfg', 'scale_size', 'sampler', 'runner',
            'paramwise_cfg', 'optimizer', 'optimizer_config', 'lr_config',
            'log_config', 'dist_params', 'log_level', 'workflow', 'fp16',
            'evaluation'):
        assert cfg[key] == reference[key]

    model_448 = copy.deepcopy(cfg['model'])
    model_576 = copy.deepcopy(reference['model'])
    assert model_448['backbone'].pop('size') == 448
    assert model_576['backbone'].pop('size') == 576
    assert model_448 == model_576

    train_448 = copy.deepcopy(cfg['train_pipeline'])
    train_576 = copy.deepcopy(reference['train_pipeline'])
    crop_mixup_448 = next(
        item for item in train_448 if item['type'] == 'CropMixup')
    crop_mixup_576 = next(
        item for item in train_576 if item['type'] == 'CropMixup')
    assert crop_mixup_448.pop('size') == 448
    assert crop_mixup_576.pop('size') == 576
    assert train_448 == train_576

    test_448 = copy.deepcopy(cfg['test_pipeline'])
    test_576 = copy.deepcopy(reference['test_pipeline'])
    resize_448 = next(item for item in test_448 if item['type'] == 'Resize')
    resize_576 = next(item for item in test_576 if item['type'] == 'Resize')
    assert resize_448.pop('size') == 448
    assert resize_576.pop('size') == 576
    assert test_448 == test_576


def test_tobacco_paths_are_centralized():
    cfg = _load_config()
    assert cfg['tobacco_root'] == '../0data/tobacco'
    assert cfg['data_root'] == '../0data/tobacco/data'
    assert cfg['manifest_file'] == (
        '../0data/tobacco/image_manifest_updated.csv')
    assert 'ASL' not in cfg['manifest_file']
    assert isinstance(cfg['data_root'], str)
    assert isinstance(cfg['manifest_file'], str)
    assert Path(cfg['data_root']).name != 'images'
    assert Path(cfg['manifest_file']).name == 'image_manifest_updated.csv'
    train_dataset = cfg['data']['train']['dataset']
    assert train_dataset['data_prefix'] == cfg['data_root']
    assert train_dataset['ann_file'] == cfg['manifest_file']
    for split in ('val', 'test'):
        assert cfg['data'][split]['data_prefix'] == cfg['data_root']
        assert cfg['data'][split]['ann_file'] == cfg['manifest_file']


def test_checkpoint_selection_remains_validation_map_only():
    cfg = _load_config()
    assert cfg['evaluation'] == dict(
        interval=1, metric='mAP', save_best='mAP')


def test_tobacco_448_work_dir_and_static_stage_geometry():
    cfg = _load_config()
    assert cfg['work_dir'] == './work_dirs/gkgnet_tobacco_448'
    input_size = cfg['model']['backbone']['size']
    stage_spatial = [input_size // divisor for divisor in (4, 8, 16, 32)]
    assert stage_spatial == [112, 56, 28, 14]
    assert [side * side for side in stage_spatial] == [12544, 3136, 784, 196]
    assert [(channels, side, side) for channels, side in zip(
        (80, 160, 400, 640), stage_spatial)] == [
            (80, 112, 112),
            (160, 56, 56),
            (400, 28, 28),
            (640, 14, 14),
        ]


def test_tobacco_576_reference_config_is_preserved():
    cfg = runpy.run_path(str(REFERENCE_CONFIG_PATH))
    assert cfg['model']['backbone']['size'] == 576
    assert cfg['crop_size'] == 576
    assert cfg['work_dir'] == './work_dirs/gkgnet_tobacco_576'
    crop_mixup = next(
        item for item in cfg['train_pipeline'] if item['type'] == 'CropMixup')
    resize = next(
        item for item in cfg['test_pipeline'] if item['type'] == 'Resize')
    assert crop_mixup['size'] == 576
    assert resize['size'] == 576
