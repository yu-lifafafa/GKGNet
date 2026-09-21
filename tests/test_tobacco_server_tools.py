import importlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_server_tools_import_without_mmcv_runtime():
    verify = importlib.import_module('tools.verify_pvig_checkpoint')
    smoke = importlib.import_module('tools.smoke_tobacco_gkgnet')
    assert callable(verify.main)
    assert callable(smoke.main)


def test_gpu_smoke_is_real_dataset_single_batch_contract():
    source = (REPO_ROOT / 'tools' / 'smoke_tobacco_gkgnet.py').read_text(
        encoding='utf-8')
    assert 'build_dataset' in source
    assert 'build_dataloader' in source
    assert 'batch_size = 1' in source
    assert '(1, 3, 576, 576)' in source
    assert '(1, 18)' in source
    assert 'torch.rand' not in source
    assert '--pretrained-checkpoint' in source
    assert 'model.init_weights()' in source
    assert 'backward()' in source
