from pathlib import Path

import torch

from tools.verify_pvig_checkpoint import (TensorMetadata,
                                          build_target_backbone_metadata,
                                          compare_state_dicts,
                                          extract_state_dict,
                                          inspect_checkpoint_structure)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_extract_state_dict_supports_common_checkpoint_layouts():
    tensor = torch.zeros(1)
    assert extract_state_dict({'state_dict': {'module.stem.weight': tensor}})[
        'module.stem.weight'] is tensor
    assert extract_state_dict({'model': {'stem.weight': tensor}})[
        'stem.weight'] is tensor
    assert extract_state_dict({'stem.weight': tensor})['stem.weight'] is tensor


def test_checkpoint_structure_preserves_raw_keys_and_location():
    tensor = torch.zeros(1)
    checkpoint = {
        'model_state_dict': {
            'module.backbone.stem.weight': tensor,
            'module.prediction.weight': tensor,
        },
        'epoch': 3,
    }
    structure = inspect_checkpoint_structure(checkpoint)
    assert structure['top_level_type'] == 'dict'
    assert structure['top_level_keys'] == ['model_state_dict', 'epoch']
    assert structure['state_dict_location'] == 'model_state_dict'
    assert structure['checkpoint_key_count'] == 2
    assert structure['first_parameter_keys'] == [
        'module.backbone.stem.weight', 'module.prediction.weight'
    ]
    assert structure['raw_prefix_counts'] == {'module': 2}
    assert structure['normalization_rules'] == [
        'exact key first', 'strip leading module.', 'strip leading model.',
        'strip one outer backbone. only if it matches a target key'
    ]


def test_checkpoint_report_separates_matches_mismatches_and_label_parts():
    target = {
        'stem.weight': torch.zeros(4, 3, 3, 3),
        'pos_embed': torch.zeros(1, 80, 144, 144),
        'backbone.0.graph.weight': torch.zeros(4, 4),
        'backbone.2.conv.0.weight': torch.zeros(8, 4, 3, 3),
        'label_lt.weight': torch.zeros(18, 80),
        'gcn_label.0.graph.weight': torch.zeros(4, 4),
    }
    source = {
        'module.stem.weight': torch.ones(4, 3, 3, 3),
        'module.pos_embed': torch.zeros(1, 80, 56, 56),
        'module.backbone.0.graph.weight': torch.ones(4, 4),
        'module.prediction.weight': torch.zeros(1000, 640),
    }
    report = compare_state_dicts(source, target)
    assert report['matched_keys'] == [
        'backbone.0.graph.weight', 'stem.weight'
    ]
    assert report['shape_mismatches'][0]['target_key'] == 'pos_embed'
    assert report['shape_mismatches'][0]['checkpoint_shape'] == [1, 80, 56, 56]
    assert 'label_lt.weight' in report['missing_keys']
    assert 'gcn_label.0.graph.weight' in report['missing_keys']
    assert report['unexpected_keys'] == [{
        'checkpoint_key': 'module.prediction.weight',
        'normalized_key': 'prediction.weight',
    }]
    assert report['categories']['stem']['matched_keys'] == 1
    assert report['categories']['visual_grapher']['matched_keys'] == 1
    assert report['categories']['pos_embed']['shape_mismatch_keys'] == 1
    assert report['categories']['label_embedding']['missing_keys'] == 1
    assert report['categories']['label_gcn']['missing_keys'] == 1
    assert report['exact_matched_key_count'] == 0
    assert report['matched_key_ratio'] == 2 / 6
    assert report['matched_parameter_ratio'] == (
        report['matched_parameter_count'] / report['target_parameter_count'])
    assert report['key_mappings'][0] == {
        'checkpoint_key': 'module.stem.weight',
        'target_key': 'stem.weight',
        'normalization': 'strip leading module.',
    }


def test_module_breakdown_separates_all_requested_backbone_parts():
    target = {
        'stem.convs.0.weight': TensorMetadata((40, 3, 3, 3)),
        'backbone.0.0.relative_pos': TensorMetadata((1, 20736, 1296)),
        'backbone.0.0.graph_conv.gconv.nn.0.weight': TensorMetadata((40, 40, 1, 1)),
        'backbone.0.1.fc1.0.weight': TensorMetadata((320, 80, 1, 1)),
        'backbone.2.conv.0.weight': TensorMetadata((160, 80, 3, 3)),
        'label_lt.weight': TensorMetadata((18, 80)),
        'gcn_label.0.0.fc1.0.weight': TensorMetadata((80, 80, 1, 1)),
        'ffn_label.0.0.weight': TensorMetadata((160, 80)),
        'pos_embed': TensorMetadata((1, 80, 144, 144)),
    }
    source = {
        'stem.convs.0.weight': torch.zeros(40, 3, 3, 3),
        'backbone.0.0.relative_pos': torch.zeros(1, 3136, 196),
        'backbone.0.0.graph_conv.gconv.nn.0.weight': torch.zeros(40, 40, 1, 1),
        'backbone.0.1.fc1.0.weight': torch.zeros(320, 80, 1, 1),
        'backbone.2.conv.0.weight': torch.zeros(160, 80, 3, 3),
        'pos_embed': torch.zeros(1, 80, 56, 56),
    }
    report = compare_state_dicts(source, target)
    modules = report['module_breakdown']
    assert modules['stem']['matched_keys'] == 1
    assert modules['visual_grapher']['matched_keys'] == 1
    assert modules['visual_ffn']['matched_keys'] == 1
    assert modules['downsample']['matched_keys'] == 1
    assert modules['relative_pos']['shape_mismatch_keys'] == 1
    assert modules['pos_embed']['shape_mismatch_keys'] == 1
    assert modules['label_lt']['missing_keys'] == 1
    assert modules['gcn_label']['missing_keys'] == 1
    assert modules['ffn_label']['missing_keys'] == 1


def test_metadata_only_target_is_exact_tobacco_576_contract():
    config = REPO_ROOT / 'configs' / 'gkgnet' / 'gkgnet_tobacco_576.py'
    target, parameter_keys, trainable_keys, metadata = (
        build_target_backbone_metadata(config, force_metadata_only=True))
    assert metadata['choice'] == 's'
    assert metadata['size'] == 576
    assert metadata['n_classes'] == 18
    assert metadata['build_mode'] == 'metadata_only_dependency_shim'
    assert target['pos_embed'].shape == (1, 80, 144, 144)
    relative = {
        key: value.shape for key, value in target.items()
        if 'relative_pos' in key
    }
    assert len(relative) == 12
    assert relative['backbone.0.0.relative_pos'] == (1, 20736, 1296)
    assert relative['backbone.3.0.relative_pos'] == (1, 5184, 1296)
    assert relative['backbone.6.0.relative_pos'] == (1, 1296, 1296)
    assert relative['backbone.13.0.relative_pos'] == (1, 324, 324)
    assert 'label_lt.weight' in target
    assert any(key.startswith('gcn_label.') for key in target)
    assert any(key.startswith('ffn_label.') for key in target)
    assert set(trainable_keys).issubset(parameter_keys)


def test_metadata_only_target_is_exact_tobacco_448_contract():
    config = REPO_ROOT / 'configs' / 'gkgnet' / 'gkgnet_tobacco_448.py'
    target, parameter_keys, trainable_keys, metadata = (
        build_target_backbone_metadata(config, force_metadata_only=True))
    assert metadata['choice'] == 's'
    assert metadata['size'] == 448
    assert metadata['n_classes'] == 18
    assert metadata['stage_feature_shapes'] == [
        [80, 112, 112],
        [160, 56, 56],
        [400, 28, 28],
        [640, 14, 14],
    ]
    assert metadata['stage_node_counts'] == [12544, 3136, 784, 196]
    assert target['pos_embed'].shape == (1, 80, 112, 112)
    relative = {
        key: value.shape for key, value in target.items()
        if 'relative_pos' in key
    }
    assert len(relative) == 12
    assert relative['backbone.0.0.relative_pos'] == (1, 12544, 784)
    assert relative['backbone.3.0.relative_pos'] == (1, 3136, 784)
    assert relative['backbone.6.0.relative_pos'] == (1, 784, 784)
    assert relative['backbone.13.0.relative_pos'] == (1, 196, 196)
    assert set(trainable_keys).issubset(parameter_keys)
