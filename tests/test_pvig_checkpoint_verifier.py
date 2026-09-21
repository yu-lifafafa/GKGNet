import torch

from tools.verify_pvig_checkpoint import (compare_state_dicts,
                                          extract_state_dict)


def test_extract_state_dict_supports_common_checkpoint_layouts():
    tensor = torch.zeros(1)
    assert extract_state_dict({'state_dict': {'module.stem.weight': tensor}})[
        'module.stem.weight'] is tensor
    assert extract_state_dict({'model': {'stem.weight': tensor}})[
        'stem.weight'] is tensor
    assert extract_state_dict({'stem.weight': tensor})['stem.weight'] is tensor


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
    assert report['unexpected_keys'] == ['prediction.weight']
    assert report['categories']['stem']['matched_keys'] == 1
    assert report['categories']['visual_grapher']['matched_keys'] == 1
    assert report['categories']['pos_embed']['shape_mismatch_keys'] == 1
    assert report['categories']['label_embedding']['missing_keys'] == 1
    assert report['categories']['label_gcn']['missing_keys'] == 1
