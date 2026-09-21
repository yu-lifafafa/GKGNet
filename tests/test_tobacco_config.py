import runpy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / 'configs' / 'gkgnet' / 'gkgnet_tobacco_576.py'
COCO_CONFIG_PATH = REPO_ROOT / 'configs' / 'gkgnet' / 'gkgnet_coco_576.py'


def _load_config():
    return runpy.run_path(str(CONFIG_PATH))


def test_tobacco_config_model_contract():
    cfg = _load_config()
    assert cfg['model']['backbone']['n_classes'] == 18
    assert cfg['model']['head']['num_classes'] == 18
    assert cfg['model']['backbone']['size'] == 576
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
    assert cfg['crop_size'] == 576
    crop_mixup = next(
        transform for transform in cfg['train_pipeline']
        if transform['type'] == 'CropMixup')
    test_resize = next(
        transform for transform in cfg['test_pipeline']
        if transform['type'] == 'Resize')
    assert crop_mixup['size'] == 576
    assert test_resize['size'] == 576
    assert cfg['data']['val']['pipeline'] is cfg['test_pipeline']
    assert cfg['data']['test']['pipeline'] is cfg['test_pipeline']


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
    coco = runpy.run_path(str(COCO_CONFIG_PATH))

    for key in (
            'img_norm_cfg', 'scale_size', 'crop_size', 'train_pipeline',
            'test_pipeline', 'sampler', 'runner', 'paramwise_cfg', 'optimizer',
            'optimizer_config', 'lr_config', 'log_config', 'dist_params',
            'log_level', 'workflow', 'fp16'):
        assert cfg[key] == coco[key]

    tobacco_backbone = dict(cfg['model']['backbone'])
    coco_backbone = dict(coco['model']['backbone'])
    assert tobacco_backbone.pop('n_classes') == 18
    assert coco_backbone.pop('n_classes') == 80
    assert tobacco_backbone == coco_backbone

    tobacco_head = dict(cfg['model']['head'])
    coco_head = dict(coco['model']['head'])
    assert tobacco_head.pop('num_classes') == 18
    assert coco_head.pop('num_classes') == 80
    assert tobacco_head == coco_head


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
