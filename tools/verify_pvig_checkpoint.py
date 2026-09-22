"""Read-only Pyramid ViG checkpoint compatibility diagnostics."""

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import re
import runpy
import sys
import types
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


NORMALIZATION_RULES = [
    'exact key first',
    'strip leading module.',
    'strip leading model.',
    'strip one outer backbone. only if it matches a target key',
]


@dataclass(frozen=True)
class TensorMetadata:
    """Shape-only tensor metadata used to avoid allocating spatial state."""

    shape: tuple

    def __init__(self, shape):
        object.__setattr__(self, 'shape', tuple(int(value) for value in shape))

    def numel(self):
        return math.prod(self.shape)


def _locate_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError('Checkpoint top level must be a dictionary.')
    for key in ('state_dict', 'model', 'model_state_dict'):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict) and candidate:
            if all(torch.is_tensor(value) for value in candidate.values()):
                return key, candidate
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return 'raw', checkpoint
    raise ValueError(
        'Could not find a tensor state_dict at top level, "state_dict", '
        '"model", or "model_state_dict".')


def extract_state_dict(checkpoint):
    """Extract a tensor state dict from common checkpoint layouts."""
    return _locate_state_dict(checkpoint)[1]


def inspect_checkpoint_structure(checkpoint):
    """Report the original checkpoint structure without normalizing keys."""
    location, state = _locate_state_dict(checkpoint)
    keys = list(state)
    top_level_keys = list(checkpoint) if isinstance(checkpoint, dict) else []
    prefix_counts = Counter(key.split('.', 1)[0] for key in keys)
    return {
        'top_level_type': type(checkpoint).__name__,
        'top_level_keys': top_level_keys,
        'contains_state_dict': 'state_dict' in top_level_keys,
        'contains_model': 'model' in top_level_keys,
        'contains_model_state_dict': 'model_state_dict' in top_level_keys,
        'state_dict_location': location,
        'checkpoint_key_count': len(keys),
        'first_parameter_keys': keys[:20],
        'raw_prefix_counts': dict(prefix_counts),
        'raw_has_module_prefix': any(key.startswith('module.') for key in keys),
        'raw_has_model_prefix': any(key.startswith('model.') for key in keys),
        'raw_has_backbone_prefix': any(
            key.startswith('backbone.') for key in keys),
        'normalization_rules': NORMALIZATION_RULES,
    }


def _key_candidates(source_key):
    candidates = [(source_key, 'exact')]
    current = source_key
    applied = []
    for prefix, description in (
            ('module.', 'strip leading module.'),
            ('model.', 'strip leading model.')):
        if current.startswith(prefix):
            current = current[len(prefix):]
            applied.append(description)
            candidates.append((current, ' + '.join(applied)))
    for candidate, rule in tuple(candidates):
        if candidate.startswith('backbone.'):
            description = 'strip one outer backbone.'
            combined = description if rule == 'exact' else f'{rule} + {description}'
            candidates.append((candidate[len('backbone.'):], combined))
    unique = []
    seen = set()
    for key, rule in candidates:
        if key not in seen:
            unique.append((key, rule))
            seen.add(key)
    return unique


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
    if re.match(r'^backbone\.\d+\.conv\.', target_key):
        return 'downsample'
    if target_key.startswith('backbone.'):
        return 'visual_grapher'
    return 'other'


def _module_predicates():
    return {
        'stem': lambda key: key.startswith('stem.'),
        'visual_grapher': lambda key: bool(
            re.match(r'^backbone\.\d+\.0\.', key)),
        'downsample': lambda key: bool(
            re.match(r'^backbone\.\d+\.conv\.', key)),
        'visual_ffn': lambda key: bool(
            re.match(r'^backbone\.\d+\.1\.', key)),
        'label_lt': lambda key: key.startswith('label_lt.'),
        'gcn_label': lambda key: key.startswith('gcn_label.'),
        'ffn_label': lambda key: key.startswith('ffn_label.'),
        'pos_embed': lambda key: key == 'pos_embed',
        'relative_pos': lambda key: 'relative_pos' in key,
    }


def compare_state_dicts(source_state,
                        target_state,
                        parameter_keys=None,
                        trainable_parameter_keys=None):
    """Compare original checkpoint keys and shapes without loading weights."""
    target_keys = set(target_state)
    matched = {}
    mismatches = []
    unexpected = []
    mappings = []

    for source_key, source_value in source_state.items():
        target_key = None
        normalization = None
        for candidate, rule in _key_candidates(source_key):
            if candidate in target_keys:
                target_key = candidate
                normalization = rule
                break
        if target_key is None:
            unexpected.append({
                'checkpoint_key': source_key,
                'normalized_key': _normalized_unexpected_key(source_key),
            })
            continue
        source_shape = tuple(source_value.shape)
        target_shape = tuple(target_state[target_key].shape)
        mapping = {
            'checkpoint_key': source_key,
            'target_key': target_key,
            'normalization': normalization,
        }
        if source_shape == target_shape:
            matched[target_key] = source_key
            mappings.append(mapping)
        else:
            mismatches.append({
                **mapping,
                'checkpoint_shape': list(source_shape),
                'target_shape': list(target_shape),
            })

    mismatch_targets = {item['target_key'] for item in mismatches}
    missing = sorted(target_keys - set(matched) - mismatch_targets)
    matched_keys = sorted(matched)
    mismatches.sort(key=lambda item: item['target_key'])
    unexpected.sort(key=lambda item: item['checkpoint_key'])

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

    normalized_target_keys = {
        item['target_key'] for item in mappings
        if item['normalization'] != 'exact'
    }
    module_breakdown = {}
    for name, predicate in _module_predicates().items():
        selected = {key for key in target_keys if predicate(key)}
        module_breakdown[name] = {
            'target_keys': len(selected),
            'matched_keys': len(selected & set(matched_keys)),
            'missing_keys': len(selected & set(missing)),
            'shape_mismatch_keys': len(selected & mismatch_targets),
            'normalized_name_matches': len(selected & normalized_target_keys),
            'obvious_naming_mismatch': bool(selected & normalized_target_keys),
        }

    parameter_keys = (set(target_state) if parameter_keys is None
                      else set(parameter_keys))
    target_parameter_count = sum(
        int(target_state[key].numel()) for key in parameter_keys)
    matched_parameter_count = sum(
        int(target_state[key].numel()) for key in matched_keys
        if key in parameter_keys)
    if trainable_parameter_keys is None:
        trainable_parameter_keys = parameter_keys
    else:
        trainable_parameter_keys = set(trainable_parameter_keys)
    target_trainable_count = sum(
        int(target_state[key].numel()) for key in trainable_parameter_keys)
    matched_trainable_count = sum(
        int(target_state[key].numel()) for key in matched_keys
        if key in trainable_parameter_keys)
    exact_matched = sum(
        item['checkpoint_key'] == item['target_key'] for item in mappings)

    return {
        'matched_keys': matched_keys,
        'missing_keys': missing,
        'unexpected_keys': unexpected,
        'shape_mismatches': mismatches,
        'key_mappings': mappings,
        'exact_matched_key_count': exact_matched,
        'matched_key_count': len(matched_keys),
        'target_key_count': len(target_state),
        'matched_key_ratio': (
            len(matched_keys) / len(target_state) if target_state else 0.0),
        'matched_parameter_count': matched_parameter_count,
        'target_parameter_count': target_parameter_count,
        'matched_parameter_ratio': (
            matched_parameter_count / target_parameter_count
            if target_parameter_count else 0.0),
        'matched_trainable_parameter_count': matched_trainable_count,
        'target_trainable_parameter_count': target_trainable_count,
        'matched_trainable_parameter_ratio': (
            matched_trainable_count / target_trainable_count
            if target_trainable_count else 0.0),
        'categories': categories,
        'module_breakdown': module_breakdown,
    }


class _Registry:
    def __init__(self, name, parent=None):
        self.name = name
        self.module_dict = {}

    def register_module(self):
        def decorator(cls):
            self.module_dict[cls.__name__] = cls
            return cls
        return decorator

    def build(self, cfg):
        values = dict(cfg)
        cls = self.module_dict[values.pop('type')]
        return cls(**values)


class _BaseModule(nn.Module):
    def __init__(self, init_cfg=None):
        super().__init__()
        self.init_cfg = init_cfg

    def init_weights(self):
        return None


class _DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, value):
        return value


class _EasyDict(dict):
    def __init__(self, values=None, **kwargs):
        values = {} if values is None else dict(values)
        values.update(kwargs)
        super().__init__({
            key: (_EasyDict(value) if isinstance(value, dict) else value)
            for key, value in values.items()
        })

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def _build_norm_layer(cfg, num_features, postfix=''):
    layer = nn.SyncBatchNorm(num_features)
    requires_grad = cfg.get('requires_grad', True)
    for parameter in layer.parameters():
        parameter.requires_grad = requires_grad
    return f'{cfg.get("type", "norm")}{postfix}', layer


def _install_dependency_shims():
    """Install state-compatible import shims; no forward path uses them."""
    mmcv = types.ModuleType('mmcv')
    mmcv_cnn = types.ModuleType('mmcv.cnn')
    mmcv_bricks = types.ModuleType('mmcv.cnn.bricks')
    mmcv_registry = types.ModuleType('mmcv.cnn.bricks.registry')
    mmcv_runner = types.ModuleType('mmcv.runner')
    mmcv_utils = types.ModuleType('mmcv.utils')
    mmcv_cnn.MODELS = _Registry('mmcv_models')
    mmcv_cnn.ConvModule = nn.Module
    mmcv_cnn.build_conv_layer = lambda cfg, *args, **kwargs: nn.Conv2d(
        *args, **kwargs)
    mmcv_cnn.build_norm_layer = _build_norm_layer
    mmcv_cnn.constant_init = lambda *args, **kwargs: None
    mmcv_bricks.DropPath = _DropPath
    mmcv_registry.ATTENTION = _Registry('attention')
    mmcv_runner.BaseModule = _BaseModule
    mmcv_utils.Registry = _Registry
    sys.modules.update({
        'mmcv': mmcv,
        'mmcv.cnn': mmcv_cnn,
        'mmcv.cnn.bricks': mmcv_bricks,
        'mmcv.cnn.bricks.registry': mmcv_registry,
        'mmcv.runner': mmcv_runner,
        'mmcv.utils': mmcv_utils,
    })

    timm = types.ModuleType('timm')
    timm_data = types.ModuleType('timm.data')
    timm_models = types.ModuleType('timm.models')
    timm_layers = types.ModuleType('timm.models.layers')
    timm_data.IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    timm_data.IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    timm_layers.DropPath = _DropPath
    sys.modules.update({
        'timm': timm,
        'timm.data': timm_data,
        'timm.models': timm_models,
        'timm.models.layers': timm_layers,
    })
    easydict = types.ModuleType('easydict')
    easydict.EasyDict = _EasyDict
    sys.modules['easydict'] = easydict


def _package(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


def _load_source(name, path, package_path=None):
    kwargs = ({'submodule_search_locations': [str(package_path)]}
              if package_path is not None else {})
    spec = importlib.util.spec_from_file_location(name, path, **kwargs)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_gkgnet_with_shims(repo_root):
    _install_dependency_shims()
    if not hasattr(np, 'float'):
        np.float = float
    mmcls_root = repo_root / 'mmcls'
    models_root = mmcls_root / 'models'
    backbones_root = models_root / 'backbones'
    _package('mmcls', mmcls_root)
    _package('mmcls.models', models_root)
    _package('mmcls.models.utils', models_root / 'utils')
    _package('mmcls.models.backbones', backbones_root)
    _load_source('mmcls.models.builder', models_root / 'builder.py')
    _load_source(
        'mmcls.models.utils.differentiable_topk',
        models_root / 'utils' / 'differentiable_topk.py')
    _load_source(
        'mmcls.models.backbones.base_backbone',
        backbones_root / 'base_backbone.py')
    vig_root = backbones_root / 'vig_model'
    _load_source(
        'mmcls.models.backbones.vig_model',
        vig_root / '__init__.py',
        package_path=vig_root)
    return _load_source(
        'mmcls.models.backbones.gkgnet', backbones_root / 'gkgnet.py')


def _config_backbone(config_path):
    config = runpy.run_path(str(config_path))
    backbone = dict(config['model']['backbone'])
    required = {'choice': 's', 'n_classes': 18}
    for key, expected in required.items():
        if backbone.get(key) != expected:
            raise ValueError(
                f'Target config must keep {key}={expected!r}, got '
                f'{backbone.get(key)!r}.')
    size = backbone.get('size')
    if not isinstance(size, int) or size <= 0 or size % 32:
        raise ValueError(
            'Target config backbone.size must be a positive multiple of 32, '
            f'got {size!r}.')
    return backbone


def _metadata_from_model(model, target_size, config_values, build_mode):
    state = {
        key: TensorMetadata(value.shape)
        for key, value in model.state_dict().items()
    }
    channels = state['pos_embed'].shape[1]
    spatial = target_size // 4
    state['pos_embed'] = TensorMetadata((1, channels, spatial, spatial))

    current_nodes = spatial * spatial
    relative_shapes = {}
    for index, module in enumerate(model.backbone):
        if module.__class__.__name__ == 'Downsample':
            current_nodes //= 4
            continue
        if not isinstance(module, nn.Sequential) or not len(module):
            continue
        grapher = module[0]
        if getattr(grapher, 'relative_pos', None) is None:
            continue
        reduced_nodes = current_nodes // (grapher.r * grapher.r)
        key = f'backbone.{index}.0.relative_pos'
        shape = (1, current_nodes, reduced_nodes)
        state[key] = TensorMetadata(shape)
        relative_shapes[key] = list(shape)

    parameter_keys = dict(model.named_parameters())
    trainable_keys = {
        key for key, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    stage_channels = [channels]
    for module in model.backbone:
        if module.__class__.__name__ == 'Downsample':
            stage_channels.append(module.conv[0].out_channels)
    stage_spatial = [target_size // (4 * (2 ** index))
                     for index in range(len(stage_channels))]
    metadata = {
        'build_mode': build_mode,
        'choice': config_values['choice'],
        'size': target_size,
        'n_classes': config_values['n_classes'],
        'target_backbone': type(model).__name__,
        'stage_feature_shapes': [
            [channel, spatial, spatial]
            for channel, spatial in zip(stage_channels, stage_spatial)
        ],
        'stage_node_counts': [spatial ** 2 for spatial in stage_spatial],
        'relative_pos_shapes': relative_shapes,
        'dependency_shims_affect_state_names': False,
        'forward_executed': False,
    }
    return state, parameter_keys, trainable_keys, metadata


def build_target_backbone_metadata(config_path, force_metadata_only=False):
    """Build exact state names/shapes without allocating full spatial state."""
    config_path = Path(config_path)
    backbone_cfg = _config_backbone(config_path)
    repo_root = Path(__file__).resolve().parents[1]

    if not force_metadata_only:
        try:
            from mmcv import Config
            from mmcls.models import build_backbone
        except ModuleNotFoundError:
            pass
        else:
            cfg = Config.fromfile(str(config_path))
            values = cfg.model.backbone.copy()
            values['init_cfg'] = None
            model = build_backbone(values)
            return _metadata_from_model(
                model, backbone_cfg['size'], backbone_cfg,
                build_mode='native_mmcv')

    gkgnet_module = _load_gkgnet_with_shims(repo_root)
    signature = inspect.signature(gkgnet_module.GKGNet.__init__)
    constructor_values = {
        key: value for key, value in backbone_cfg.items()
        if key in signature.parameters and key not in {'size', 'init_cfg'}
    }
    diagnostic_size = 32
    model = gkgnet_module.GKGNet(
        **constructor_values, size=diagnostic_size, init_cfg=None)
    return _metadata_from_model(
        model, backbone_cfg['size'], backbone_cfg,
        build_mode='metadata_only_dependency_shim')


def _file_identity(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return {
        'absolute_path': str(path.resolve()),
        'size_bytes': path.stat().st_size,
        'sha256': digest.hexdigest(),
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
    if checkpoint_path.stat().st_size == 0:
        raise ValueError(f'Checkpoint is empty: {checkpoint_path}')

    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False)
    structure = inspect_checkpoint_structure(checkpoint)
    source_state = extract_state_dict(checkpoint)
    target_state, parameter_keys, trainable_keys, target_metadata = (
        build_target_backbone_metadata(args.config))
    report = compare_state_dicts(
        source_state,
        target_state,
        parameter_keys=parameter_keys,
        trainable_parameter_keys=trainable_keys)
    report.update({
        'checkpoint': _file_identity(checkpoint_path),
        'checkpoint_structure': structure,
        'checkpoint_key_count': len(source_state),
        'target_key_count': len(target_state),
        'target': target_metadata,
        'normalization': {
            'rules': NORMALIZATION_RULES,
            'changed_mapping_count': sum(
                item['normalization'] != 'exact'
                for item in report['key_mappings']),
            'changed_mappings': [
                item for item in report['key_mappings']
                if item['normalization'] != 'exact'
            ],
        },
    })

    pos_source = next((
        (key, value) for key, value in source_state.items()
        if any(candidate == 'pos_embed'
               for candidate, _ in _key_candidates(key))), None)
    pos_mismatch = next((
        item for item in report['shape_mismatches']
        if item['target_key'] == 'pos_embed'), None)
    report['pos_embed'] = {
        'checkpoint_key': pos_source[0] if pos_source else None,
        'checkpoint_shape': (
            list(pos_source[1].shape) if pos_source else None),
        'target_shape': list(target_state['pos_embed'].shape),
        'exact_shape_match': pos_mismatch is None and pos_source is not None,
        'shape_mismatch': pos_mismatch,
        'automatic_interpolation': False,
    }
    relative_source = [
        {'checkpoint_key': key, 'shape': list(value.shape)}
        for key, value in source_state.items() if 'relative_pos' in key
    ]
    relative_mismatches = [
        item for item in report['shape_mismatches']
        if 'relative_pos' in item['target_key']
    ]
    report['relative_pos'] = {
        'checkpoint_entries': relative_source,
        'target_entries': [
            {'target_key': key, 'shape': list(value.shape)}
            for key, value in target_state.items() if 'relative_pos' in key
        ],
        'matched_keys': report['module_breakdown']['relative_pos'][
            'matched_keys'],
        'shape_mismatch_keys': len(relative_mismatches),
        'shape_mismatches': relative_mismatches,
    }
    classifier_tokens = ('prediction', 'classifier', 'head')
    report['imagenet_classifier'] = [
        {'checkpoint_key': key, 'shape': list(value.shape)}
        for key, value in source_state.items()
        if any(token in key.lower() for token in classifier_tokens)
    ]
    report['load_policy'] = {
        'weights_were_loaded': False,
        'checkpoint_was_modified': False,
        'keys_were_deleted': False,
        'positional_state_was_interpolated': False,
    }

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
