import csv
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.calibrate_tobacco import (
    DEFAULT_THRESHOLD_GRID,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    THRESHOLD_STEP,
    build_calibration_artifact,
    calibrate_thresholds,
)
from tools.create_best_checkpoint_info import build_best_checkpoint_info
from tools.evaluate_tobacco import evaluate_prediction_artifacts
from tools.infer_tobacco import (
    strict_restore_trained_checkpoint,
    validate_inference_request,
    validate_probability_batch,
)
from tools.tobacco_artifacts import (
    ARCHITECTURE,
    FORMAL_CONFIG,
    INPUT_SIZE,
    build_prediction_metadata,
    checkpoint_identity,
    prediction_fieldnames,
    read_prediction_csv,
    validate_best_checkpoint_info,
    write_prediction_csv,
)
from tools.tobacco_metrics import NUM_CLASSES, TASK_PROTOCOL, evaluate_scores


REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = (
    'weather_fleck',
    'potato_virus_y',
    'tobacco_mosaic_virus',
    'cucumber_mosaic_virus',
    'alternaria_leaf_spot',
    'black_shank',
    'bacterial_wilt',
    'hollow_stalk',
    'black_root_rot',
    'wildfire',
    'potassium_deficiency',
    'magnesium_deficiency',
    'powdery_mildew',
    'anthracnose',
    'frogeye_leaf_spot',
    'angular_leaf_spot',
    'sunburn',
    'herbicide_phytotoxicity',
)


def _perfect_data():
    true = np.vstack([
        np.ones((1, NUM_CLASSES), dtype=np.int8),
        np.zeros((1, NUM_CLASSES), dtype=np.int8),
    ])
    score = np.vstack([
        np.full((1, NUM_CLASSES), 0.9),
        np.full((1, NUM_CLASSES), 0.1),
    ])
    return true, score


def _identity_artifacts(tmp_path):
    checkpoint = tmp_path / 'best.pth'
    checkpoint.write_bytes(b'full trained checkpoint fixture')
    best = build_best_checkpoint_info(
        checkpoint_path=checkpoint,
        epoch=12,
        validation_map=0.75,
        class_names=CLASS_NAMES,
        checkpoint_type='ordinary')
    metadata = build_prediction_metadata(
        split='val',
        num_samples=2,
        class_names=CLASS_NAMES,
        checkpoint=best['checkpoint'],
        checkpoint_type='ordinary')
    return checkpoint, best, metadata


def test_formal_threshold_grid_is_exact_999_candidates():
    assert THRESHOLD_MIN == pytest.approx(0.001)
    assert THRESHOLD_MAX == pytest.approx(0.999)
    assert THRESHOLD_STEP == pytest.approx(0.001)
    assert DEFAULT_THRESHOLD_GRID.shape == (999, )
    assert DEFAULT_THRESHOLD_GRID[0] == pytest.approx(0.001)
    assert DEFAULT_THRESHOLD_GRID[-1] == pytest.approx(0.999)
    assert np.diff(DEFAULT_THRESHOLD_GRID).tolist() == pytest.approx(
        [0.001] * 998)


def test_f1_search_and_deterministic_tie_break_rules():
    true = np.vstack([
        np.ones((1, NUM_CLASSES), dtype=np.int8),
        np.zeros((1, NUM_CLASSES), dtype=np.int8),
    ])
    score = np.vstack([
        np.full((1, NUM_CLASSES), 0.2),
        np.full((1, NUM_CLASSES), 0.1),
    ])
    calibrated = calibrate_thresholds(true, score)
    assert calibrated['thresholds'].tolist() == pytest.approx([0.2] * 18)
    equal_distance = calibrate_thresholds(
        true, score, threshold_grid=np.array([0.499, 0.501]))
    assert equal_distance['thresholds'].tolist() == pytest.approx([0.499] * 18)


def test_support_and_all_standard_aggregate_metrics_are_exact():
    true = np.zeros((4, NUM_CLASSES), dtype=np.int8)
    score = np.zeros((4, NUM_CLASSES), dtype=np.float64)
    true[[0, 1], 0] = 1
    score[[0, 2], 0] = 1
    true[3, 1] = 1
    score[3, 1] = 1
    result = evaluate_scores(true, score, np.full(NUM_CLASSES, 0.5))
    assert np.array_equal(result['support'], result['tp'] + result['fn'])
    assert result['micro_precision'] == pytest.approx(2 / 3)
    assert result['micro_recall'] == pytest.approx(2 / 3)
    assert result['micro_f1'] == pytest.approx(2 / 3)
    assert result['macro_precision'] == pytest.approx((0.5 + 1) / 18)
    assert result['macro_recall'] == pytest.approx((0.5 + 1) / 18)
    assert result['macro_f1'] == pytest.approx((0.5 + 1) / 18)


def test_map_is_threshold_free_and_mean_of_ordered_ap():
    true, score = _perfect_data()
    low = evaluate_scores(true, score, np.full(NUM_CLASSES, 0.001))
    high = evaluate_scores(true, score, np.full(NUM_CLASSES, 0.999))
    assert low['mAP'] == pytest.approx(high['mAP'])
    assert low['mAP'] == pytest.approx(low['per_class_ap'].mean())
    assert low['micro_f1'] != high['micro_f1']


def test_validation_zero_support_and_non_18_inputs_fail_fast():
    true, score = _perfect_data()
    true[:, 7] = 0
    with pytest.raises(ValueError, match='class 7'):
        calibrate_thresholds(true, score)
    with pytest.raises(ValueError, match='18'):
        calibrate_thresholds(true[:, :17], score[:, :17])


def test_inference_allows_only_val_test_and_formal_448_config():
    formal = REPO_ROOT / FORMAL_CONFIG
    assert validate_inference_request(formal, 'val') == 'val'
    assert validate_inference_request(formal, 'test', thresholds_path='x.json') == 'test'
    with pytest.raises(ValueError, match='val or test'):
        validate_inference_request(formal, 'train')
    with pytest.raises(ValueError, match='448'):
        validate_inference_request(
            REPO_ROOT / 'configs/gkgnet/gkgnet_tobacco_576.py', 'val')
    with pytest.raises(ValueError, match='threshold'):
        validate_inference_request(formal, 'test')
    with pytest.raises(ValueError, match='validation'):
        validate_inference_request(formal, 'val', thresholds_path='x.json')


def test_inference_consumes_head_probabilities_without_second_sigmoid():
    scores = validate_probability_batch(
        np.full((2, NUM_CLASSES), 0.75, dtype=np.float64))
    assert scores.shape == (2, 18)
    source = (REPO_ROOT / 'tools/infer_tobacco.py').read_text(encoding='utf-8')
    assert 'torch.sigmoid' not in source
    head_source = (REPO_ROOT / 'mmcls/models/heads/label_query_head.py').read_text(
        encoding='utf-8')
    assert 'torch.sigmoid(cls_score)' in head_source
    with pytest.raises(ValueError, match=r'\[0, 1\]'):
        validate_probability_batch(np.full((1, NUM_CLASSES), 2.0))


def test_full_trained_checkpoint_restore_is_strict(tmp_path):
    model = torch.nn.Linear(3, 2)
    valid = tmp_path / 'valid.pth'
    torch.save({'state_dict': model.state_dict()}, valid)
    strict_restore_trained_checkpoint(model, valid, checkpoint_type='ordinary')

    invalid_states = {
        'missing': {'weight': torch.zeros(2, 3)},
        'unexpected': {
            **model.state_dict(),
            'extra': torch.zeros(1),
        },
        'shape': {
            'weight': torch.zeros(3, 3),
            'bias': torch.zeros(2),
        },
    }
    for name, state in invalid_states.items():
        invalid = tmp_path / f'{name}.pth'
        torch.save({'state_dict': state}, invalid)
        with pytest.raises(RuntimeError):
            strict_restore_trained_checkpoint(
                torch.nn.Linear(3, 2), invalid, checkpoint_type='ordinary')


def test_best_checkpoint_info_schema_and_identity(tmp_path):
    checkpoint, best, metadata = _identity_artifacts(tmp_path)
    assert best == {
        'schema_version': 1,
        'task_protocol': TASK_PROTOCOL,
        'architecture': ARCHITECTURE,
        'input_size': INPUT_SIZE,
        'config': FORMAL_CONFIG,
        'num_classes': NUM_CLASSES,
        'class_names': list(CLASS_NAMES),
        'checkpoint_path': str(checkpoint.resolve()),
        'checkpoint_sha256': checkpoint_identity(checkpoint)['sha256'],
        'checkpoint_type': 'ordinary',
        'epoch': 12,
        'validation_mAP': 0.75,
        'checkpoint': checkpoint_identity(checkpoint),
    }
    assert metadata['split'] == 'val'
    assert metadata['input_size'] == 448
    assert metadata['config'] == FORMAL_CONFIG
    assert metadata['class_names'] == list(CLASS_NAMES)
    assert metadata['checkpoint_sha256'] == best['checkpoint_sha256']
    assert metadata['score_semantics'] == (
        'sigmoid_probability_from_LabelQueryHead.simple_test')
    validate_best_checkpoint_info(best, CLASS_NAMES, verify_checkpoint=True)
    best['config'] = 'configs/gkgnet/gkgnet_tobacco_576.py'
    with pytest.raises(ValueError, match='config'):
        validate_best_checkpoint_info(best, CLASS_NAMES)


def test_val_and_test_prediction_csv_schemas(tmp_path):
    true, score = _perfect_data()
    files = ['images/a.jpg', 'images/b.jpg']
    val_path = tmp_path / 'val_predictions.csv'
    write_prediction_csv(
        val_path, 'val', files, true, score, CLASS_NAMES)
    assert prediction_fieldnames(CLASS_NAMES, include_predictions=False) == [
        'sample_index', 'split', 'file_name',
        *[f'gt_{name}' for name in CLASS_NAMES],
        *[f'prob_{name}' for name in CLASS_NAMES],
    ]
    val = read_prediction_csv(val_path, CLASS_NAMES, 'val')
    assert val['y_pred'] is None

    test_path = tmp_path / 'test_predictions.csv'
    thresholds = np.full(NUM_CLASSES, 0.5)
    write_prediction_csv(
        test_path, 'test', files, true, score, CLASS_NAMES,
        thresholds=thresholds)
    assert prediction_fieldnames(CLASS_NAMES, include_predictions=True)[-18:] == [
        f'pred_{name}' for name in CLASS_NAMES
    ]
    test = read_prediction_csv(
        test_path, CLASS_NAMES, 'test', require_predictions=True)
    assert np.array_equal(test['y_pred'], score >= thresholds)


def test_calibration_artifact_schema_and_three_way_identity(tmp_path):
    checkpoint, best, metadata = _identity_artifacts(tmp_path)
    true, score = _perfect_data()
    artifact = build_calibration_artifact(
        true, score, CLASS_NAMES, best, metadata)
    assert artifact['architecture'] == ARCHITECTURE
    assert artifact['input_size'] == 448
    assert artifact['checkpoint']['sha256'] == checkpoint_identity(checkpoint)['sha256']
    assert artifact['calibration']['threshold_min'] == pytest.approx(0.001)
    assert artifact['calibration']['threshold_max'] == pytest.approx(0.999)
    assert artifact['calibration']['threshold_step'] == pytest.approx(0.001)
    assert artifact['calibration']['candidate_count'] == 999
    assert artifact['calibration']['tie_break'] == 'closest_to_0.5_then_lower'
    assert len(artifact['thresholds']) == 18
    assert len(artifact['per_class_validation']) == 18
    metadata['checkpoint']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='checkpoint'):
        build_calibration_artifact(true, score, CLASS_NAMES, best, metadata)


def test_final_test_artifact_schemas_and_fixed_half_reference(tmp_path):
    checkpoint, best, val_metadata = _identity_artifacts(tmp_path)
    true, score = _perfect_data()
    thresholds = build_calibration_artifact(
        true, score, CLASS_NAMES, best, val_metadata)
    threshold_path = tmp_path / 'thresholds.json'
    threshold_path.write_text(json.dumps(thresholds), encoding='utf-8')

    test_csv = tmp_path / 'test_predictions.csv'
    write_prediction_csv(
        test_csv, 'test', ['images/a.jpg', 'images/b.jpg'], true, score,
        CLASS_NAMES, thresholds=np.asarray(thresholds['thresholds']))
    test_meta = build_prediction_metadata(
        split='test',
        num_samples=2,
        class_names=CLASS_NAMES,
        checkpoint=best['checkpoint'],
        checkpoint_type='ordinary',
        threshold_artifact=checkpoint_identity(threshold_path))
    test_meta_path = tmp_path / 'test_predictions.meta.json'
    test_meta_path.write_text(json.dumps(test_meta), encoding='utf-8')
    best_path = tmp_path / 'best_checkpoint_info.json'
    best_path.write_text(json.dumps(best), encoding='utf-8')
    output_dir = tmp_path / 'output'

    result = evaluate_prediction_artifacts(
        test_predictions=test_csv,
        test_metadata=test_meta_path,
        threshold_artifact=threshold_path,
        best_checkpoint_info=best_path,
        output_dir=output_dir,
        class_names=CLASS_NAMES)
    metrics = json.loads((output_dir / 'test_metrics.json').read_text())
    assert metrics['micro_f1'] == result['calibrated']['micro_f1']
    assert metrics['macro_f1'] == result['calibrated']['macro_f1']
    assert metrics['fixed_0_5_reference']['micro_f1'] == pytest.approx(1.0)
    assert metrics['checkpoint']['sha256'] == best['checkpoint_sha256']
    with (output_dir / 'test_per_class.csv').open(
            encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    assert list(rows[0]) == [
        'class_id', 'class_name', 'support', 'AP', 'threshold', 'precision',
        'recall', 'F1', 'TP', 'FP', 'FN'
    ]
    assert [int(row['class_id']) for row in rows] == list(range(18))
    assert [row['class_name'] for row in rows] == list(CLASS_NAMES)


def test_test_evaluator_has_no_calibration_call():
    source = (REPO_ROOT / 'tools/evaluate_tobacco.py').read_text(
        encoding='utf-8')
    assert 'calibrate_thresholds' not in source
    assert 'argmax' not in source
    assert 'softmax' not in source
    assert 'gkgnet_tobacco_576.py' not in source
