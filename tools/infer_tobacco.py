"""Run full-split Tobacco inference with one strictly restored checkpoint."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

try:
    from tools.tobacco_artifacts import (
        CHECKPOINT_TYPES, FORMAL_CONFIG, build_prediction_metadata,
        checkpoint_identity, write_prediction_csv)
    from tools.tobacco_metrics import NUM_CLASSES, validate_probability_inputs
except ModuleNotFoundError:  # Direct execution as ``python tools/...``.
    from tobacco_artifacts import (
        CHECKPOINT_TYPES, FORMAL_CONFIG, build_prediction_metadata,
        checkpoint_identity, write_prediction_csv)
    from tobacco_metrics import NUM_CLASSES, validate_probability_inputs


def validate_inference_request(config_path, split, thresholds_path=None):
    """Enforce the frozen 448 identity and val/test-only workflow."""
    if split not in ('val', 'test'):
        raise ValueError('Inference split must be val or test; train is forbidden.')
    repo_root = Path(__file__).resolve().parents[1]
    expected = (repo_root / FORMAL_CONFIG).resolve()
    if Path(config_path).resolve() != expected:
        raise ValueError(
            f'Formal Tobacco inference must use the 448 config: {FORMAL_CONFIG}.')
    if split == 'test' and thresholds_path is None:
        raise ValueError(
            'Test inference requires a frozen validation threshold artifact.')
    if split == 'val' and thresholds_path is not None:
        raise ValueError(
            'validation inference cannot use thresholds or calibrated labels.')
    return split


def validate_probability_batch(probabilities):
    """Validate probabilities already emitted by LabelQueryHead.simple_test."""
    score = np.asarray(probabilities, dtype=np.float64)
    dummy = np.zeros(score.shape, dtype=np.int8)
    _, score = validate_probability_inputs(dummy, score)
    return score


def _extract_trained_state_dict(checkpoint, checkpoint_type):
    if checkpoint_type not in CHECKPOINT_TYPES:
        raise ValueError(f'checkpoint_type must be one of {CHECKPOINT_TYPES}.')
    if not isinstance(checkpoint, dict):
        raise ValueError('Trained checkpoint top level must be a dictionary.')
    if checkpoint_type == 'ordinary':
        state = checkpoint.get('state_dict')
        if state is None and checkpoint and all(
                torch.is_tensor(value) for value in checkpoint.values()):
            state = checkpoint
        location = 'state_dict' if 'state_dict' in checkpoint else 'raw'
    else:
        location = next((
            key for key in ('state_dict_ema', 'ema_state_dict')
            if key in checkpoint
        ), None)
        state = checkpoint.get(location) if location else None
    if not isinstance(state, dict) or not state:
        raise ValueError(
            f'Checkpoint does not contain a {checkpoint_type} model state_dict.')
    if not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError('Trained state_dict must contain tensors only.')
    return location, state


def strict_restore_trained_checkpoint(model,
                                      checkpoint_path,
                                      checkpoint_type='ordinary'):
    """Restore a complete trained model without key filtering or rewriting."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f'Trained checkpoint does not exist: {path}')
    checkpoint = torch.load(str(path), map_location='cpu')
    location, state = _extract_trained_state_dict(checkpoint, checkpoint_type)
    model.load_state_dict(state, strict=True)
    return {
        'checkpoint': checkpoint_identity(path),
        'checkpoint_type': checkpoint_type,
        'state_dict_location': location,
        'state_dict_keys': len(state),
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Infer Tobacco val/test probabilities from a trained model.')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--split', required=True, choices=('val', 'test'))
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--thresholds')
    parser.add_argument(
        '--checkpoint-type', choices=CHECKPOINT_TYPES, default='ordinary')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)
    return parser.parse_args(args)


def main():
    args = parse_args()
    validate_inference_request(args.config, args.split, args.thresholds)
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Formal GKGNet inference requires CUDA.')
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError('batch-size must be positive and workers non-negative.')

    from mmcv import Config
    from mmcv.parallel import scatter
    from mmcls.datasets import build_dataloader, build_dataset
    from mmcls.datasets.tobacco import CLASSES
    from mmcls.models import build_classifier

    cfg = Config.fromfile(args.config)
    if cfg.model.backbone.size != 448:
        raise ValueError('Formal GKGNet backbone size must remain 448.')
    dataset_cfg = copy.deepcopy(cfg.data[args.split])
    dataset_cfg.data_prefix = args.data_root
    dataset_cfg.ann_file = args.manifest
    dataset_cfg.split = args.split
    dataset_cfg.test_mode = True
    dataset = build_dataset(dataset_cfg)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=args.workers,
        num_gpus=1,
        dist=False,
        shuffle=False,
        round_up=False,
        pin_memory=False,
        sampler_cfg=None)

    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.backbone.init_cfg = None
    model = build_classifier(model_cfg)
    restored = strict_restore_trained_checkpoint(
        model, args.checkpoint, checkpoint_type=args.checkpoint_type)
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    batches = []
    with torch.inference_mode():
        for data in loader:
            data = scatter(data, [device])[0]
            batches.append(validate_probability_batch(
                model(return_loss=False, **data)))
    y_score = np.vstack(batches)
    y_true = np.asarray(dataset.get_gt_labels(), dtype=np.int8)
    y_true, y_score = validate_probability_inputs(y_true, y_score)
    if y_score.shape[0] != len(dataset):
        raise RuntimeError('Inference did not return exactly one row per sample.')
    file_names = [
        data_info['img_info']['filename'] for data_info in dataset.data_infos
    ]

    thresholds = None
    threshold_identity = None
    if args.split == 'test':
        try:
            from tools.evaluate_tobacco import validate_threshold_artifact
        except ModuleNotFoundError:
            from evaluate_tobacco import validate_threshold_artifact

        threshold_path = Path(args.thresholds)
        artifact = json.loads(threshold_path.read_text(encoding='utf-8'))
        thresholds = validate_threshold_artifact(
            artifact, CLASSES, checkpoint_path=args.checkpoint,
            checkpoint_type=args.checkpoint_type)
        threshold_identity = checkpoint_identity(threshold_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = 'val_predictions' if args.split == 'val' else 'test_predictions'
    csv_path = output_dir / f'{stem}.csv'
    meta_path = output_dir / f'{stem}.meta.json'
    write_prediction_csv(
        csv_path,
        args.split,
        file_names,
        y_true,
        y_score,
        CLASSES,
        thresholds=thresholds)
    metadata = build_prediction_metadata(
        split=args.split,
        num_samples=len(dataset),
        class_names=CLASSES,
        checkpoint=restored['checkpoint'],
        checkpoint_type=args.checkpoint_type,
        threshold_artifact=threshold_identity)
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    print(json.dumps({
        'predictions': str(csv_path.resolve()),
        'metadata': str(meta_path.resolve()),
        'num_samples': len(dataset),
        'checkpoint_sha256': restored['checkpoint']['sha256'],
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
