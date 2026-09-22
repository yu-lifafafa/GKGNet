import importlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_server_tools_import_without_mmcv_runtime():
    verify = importlib.import_module('tools.verify_pvig_checkpoint')
    smoke = importlib.import_module('tools.smoke_tobacco_gkgnet')
    inference = importlib.import_module('tools.infer_tobacco')
    calibration = importlib.import_module('tools.calibrate_tobacco')
    evaluation = importlib.import_module('tools.evaluate_tobacco')
    best_info = importlib.import_module('tools.create_best_checkpoint_info')
    assert callable(verify.main)
    assert callable(smoke.main)
    assert callable(inference.main)
    assert callable(calibration.main)
    assert callable(evaluation.main)
    assert callable(best_info.main)


def test_gpu_smoke_is_real_dataset_single_batch_contract():
    source = (REPO_ROOT / 'tools' / 'smoke_tobacco_gkgnet.py').read_text(
        encoding='utf-8')
    assert 'build_dataset' in source
    assert 'build_dataloader' in source
    assert 'batch_size = 1' in source
    assert '(1, 3, 448, 448)' in source
    assert '(1, 3, 576, 576)' not in source
    assert '(1, 18)' in source
    assert 'torch.rand' not in source
    assert '--pretrained-checkpoint' in source
    assert 'model.init_weights()' in source
    assert 'backward()' in source


def test_server_runbook_freezes_448_and_artifact_order():
    runbook = (REPO_ROOT / 'docs' / 'tobacco_gkgnet_server_runbook.md')
    text = runbook.read_text(encoding='utf-8')
    assert 'configs/gkgnet/gkgnet_tobacco_448.py' in text
    assert 'gkgnet_tobacco_576.py' not in text
    assert '[1, 3, 448, 448]' in text
    assert text.index('## 7. validation inference') < text.index(
        '## 8. validation threshold calibration')
    assert text.index('## 8. validation threshold calibration') < text.index(
        '## 9. test inference')
    for placeholder in ('<DATA_ROOT>', '<MANIFEST>', '<OUTPUT_DIR>',
                        '<BEST_CHECKPOINT>'):
        assert placeholder in text
    for artifact in (
            'best_checkpoint_info.json', 'thresholds.json',
            'val_predictions.csv', 'val_predictions.meta.json',
            'test_predictions.csv', 'test_predictions.meta.json',
            'test_metrics.json', 'test_per_class.csv', 'train.log'):
        assert artifact in text


def test_gitignore_excludes_large_and_runtime_artifacts():
    lines = (REPO_ROOT / '.gitignore').read_text(encoding='utf-8').splitlines()
    assert 'checkpoint/pvig_s_82.1.pth.tar' in lines
    assert 'work_dirs/' in lines
    assert 'test_tmp*/' in lines
