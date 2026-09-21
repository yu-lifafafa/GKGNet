"""Diagnose Pyramid ViG checkpoint compatibility without loading weights."""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch


def extract_state_dict(checkpoint):
    """Extract a tensor state dict from common checkpoint layouts."""
    if not isinstance(checkpoint, dict):
        raise ValueError('Checkpoint top level must be a dictionary.')
    for key in ('state_dict', 'model'):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict) and candidate:
            if all(torch.is_tensor(value) for value in candidate.values()):
                return candidate
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise ValueError(
        'Could not find a tensor state_dict at top level, "state_dict", '
        'or "model".')


def _key_candidates(source_key):
    candidates = [source_key]
    current = source_key
    for prefix in ('module.', 'model.'):
        if current.startswith(prefix):
            current = current[len(prefix):]
            candidates.append(current)
    for candidate in tuple(candidates):
        if candidate.startswith('backbone.'):
            candidates.append(candidate[len('backbone.'):])
    return list(dict.fromkeys(candidates))


def _normalized_unexpected_key(source_key):
    key = source_key
    changed = True
    while changed:
        changed = False
        for prefix in ('module.', 'model.'):
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def _category(target_key):
    if target_key.startswith('stem.'):
        return 'stem'
    if target_key == 'pos_embed':
        return 'pos_embed'
    if target_key.startswith('label_lt.'):
        return 'label_embedding'
    if target_key.startswith(('gcn_label.', 'ffn_label.')):
        return 'label_gcn'
    if target_key.startswith('backbone.') and '.conv.' in target_key:
        return 'downsample'
    if target_key.startswith('backbone.'):
        return 'visual_grapher'
    return 'other'


def compare_state_dicts(source_state, target_state, parameter_keys=None):
    """Compare keys and shapes without mutating or loading the model."""
    target_keys = set(target_state)
    matched = OrderedDict()
    mismatches = []
    unexpected = []
    for source_key, source_value in source_state.items():
        target_key = next(
            (candidate for candidate in _key_candidates(source_key)
             if candidate in target_keys), None)
        if target_key is None:
            unexpected.append(_normalized_unexpected_key(source_key))
            continue
        source_shape = tuple(source_value.shape)
        target_shape = tuple(target_state[target_key].shape)
        if source_shape == target_shape:
            matched[target_key] = source_key
        else:
            mismatches.append({
                'checkpoint_key': source_key,
                'target_key': target_key,
                'checkpoint_shape': list(source_shape),
                'target_shape': list(target_shape),
            })

    mismatch_targets = {item['target_key'] for item in mismatches}
    missing = sorted(target_keys - set(matched) - mismatch_targets)
    matched_keys = sorted(matched)
    mismatches.sort(key=lambda item: item['target_key'])
    unexpected = sorted(set(unexpected))

    category_names = (
        'stem', 'visual_grapher', 'downsample', 'pos_embed',
        'label_embedding', 'label_gcn', 'other')
    categories = {}
    for name in category_names:
        category_targets = {
            key for key in target_keys if _category(key) == name
        }
        categories[name] = {
            'target_keys': len(category_targets),
            'matched_keys': len(category_targets & set(matched_keys)),
            'shape_mismatch_keys': len(category_targets & mismatch_targets),
            'missing_keys': len(category_targets & set(missing)),
        }

    if parameter_keys is None:
        parameter_keys = set(target_state)
    else:
        parameter_keys = set(parameter_keys)
    target_parameter_count = sum(
        int(target_state[key].numel()) for key in parameter_keys)
    matched_parameter_count = sum(
        int(target_state[key].numel()) for key in matched_keys
        if key in parameter_keys)
    return {
        'matched_keys': matched_keys,
        'missing_keys': missing,
        'unexpected_keys': unexpected,
        'shape_mismatches': mismatches,
        'matched_key_count': len(matched_keys),
        'target_key_count': len(target_state),
        'matched_parameter_count': matched_parameter_count,
        'target_parameter_count': target_parameter_count,
        'matched_parameter_ratio': (
            matched_parameter_count / target_parameter_count
            if target_parameter_count else 0.0),
        'categories': categories,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inspect pvig_s_82.1.pth.tar against GKGNet backbone.')
    parser.add_argument('config', help='GKGNet Tobacco config path.')
    parser.add_argument('checkpoint', help='Downloaded Pyramid ViG checkpoint.')
    parser.add_argument('--output', help='Optional JSON report path.')
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f'Checkpoint does not exist: {checkpoint_path}')

    from mmcv import Config
    from mmcls.models import build_backbone

    cfg = Config.fromfile(args.config)
    backbone_cfg = cfg.model.backbone.copy()
    backbone_cfg['init_cfg'] = None
    backbone = build_backbone(backbone_cfg)
    target_state = backbone.state_dict()

    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False)
    source_state = extract_state_dict(checkpoint)
    report = compare_state_dicts(
        source_state,
        target_state,
        parameter_keys=dict(backbone.named_parameters()))
    report.update({
        'checkpoint_path': str(checkpoint_path.resolve()),
        'checkpoint_top_level_type': type(checkpoint).__name__,
        'checkpoint_top_level_keys': (
            sorted(checkpoint.keys()) if isinstance(checkpoint, dict) else []),
        'checkpoint_state_key_count': len(source_state),
        'target_backbone': type(backbone).__name__,
        'target_num_classes': cfg.model.backbone.n_classes,
        'target_input_size': cfg.model.backbone.size,
        'expected_class_agnostic_categories': [
            'stem', 'visual_grapher', 'downsample', 'pos_embed'
        ],
        'expected_label_specific_random_init_categories': [
            'label_embedding', 'label_gcn'
        ],
    })
    pos_mismatch = next(
        (item for item in report['shape_mismatches']
         if item['target_key'] == 'pos_embed'), None)
    report['pos_embed'] = {
        'matched': 'pos_embed' in report['matched_keys'],
        'shape_mismatch': pos_mismatch,
        'target_shape': list(target_state['pos_embed'].shape),
        'automatic_interpolation': False,
    }

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
