import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest

from tools.calibrate_tobacco import (build_calibration_artifact,
                                     calibrate_thresholds)
from tools.evaluate_tobacco import validate_threshold_artifact
from tools.tobacco_artifacts import (build_best_checkpoint_info,
                                     build_prediction_metadata)
from tools.tobacco_metrics import (NUM_CLASSES, TASK_PROTOCOL,
                                   checkpoint_identity, evaluate_scores,
                                   per_class_rows)


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
    y_true = np.vstack([
        np.ones((1, NUM_CLASSES), dtype=np.int8),
        np.zeros((1, NUM_CLASSES), dtype=np.int8),
    ])
    y_score = np.vstack([
        np.full((1, NUM_CLASSES), 0.9),
        np.full((1, NUM_CLASSES), 0.1),
    ])
    return y_true, y_score


def _artifact(tmp_path):
    checkpoint = tmp_path / 'best.pth'
    checkpoint.write_bytes(b'checkpoint identity fixture')
    y_true, y_score = _perfect_data()
    best = build_best_checkpoint_info(
        checkpoint_path=checkpoint,
        epoch=1,
        validation_map=1.0,
        class_names=CLASS_NAMES)
    metadata = build_prediction_metadata(
        split='val',
        num_samples=len(y_true),
        class_names=CLASS_NAMES,
        checkpoint=best['checkpoint'],
        checkpoint_type='ordinary')
    artifact = build_calibration_artifact(
        y_true, y_score, CLASS_NAMES, best, metadata)
    return artifact, checkpoint


def test_perfect_multilabel_prediction_has_unit_metrics():
    y_true, y_score = _perfect_data()
    result = evaluate_scores(y_true, y_score, np.full(NUM_CLASSES, 0.5))
    assert result['mAP'] == pytest.approx(1.0)
    assert result['Macro-F1'] == pytest.approx(1.0)
    assert result['Micro-F1'] == pytest.approx(1.0)
    assert result['per_class_f1'].tolist() == pytest.approx([1.0] * 18)


def test_same_sample_can_have_two_correct_positive_labels():
    y_true, y_score = _perfect_data()
    y_true[0] = 0
    y_score[0] = 0.1
    y_true[0, [2, 15]] = 1
    y_score[0, [2, 15]] = 0.9
    result = evaluate_scores(y_true, y_score, np.full(NUM_CLASSES, 0.5))
    assert result['y_pred'][0, [2, 15]].tolist() == [1, 1]
    assert result['tp'][[2, 15]].tolist() == [1, 1]


def test_macro_f1_is_mean_per_class_f1_not_official_cf1():
    y_true = np.zeros((4, NUM_CLASSES), dtype=np.int8)
    y_score = np.zeros((4, NUM_CLASSES), dtype=np.float64)
    y_true[[0, 1], 0] = 1
    y_score[0, 0] = 1
    y_true[2, 1] = 1
    y_score[[2, 3], 1] = 1
    result = evaluate_scores(y_true, y_score, np.full(NUM_CLASSES, 0.5))
    official_cf1 = (
        2 * result['per_class_precision'].mean()
        * result['per_class_recall'].mean()
        / (result['per_class_precision'].mean()
           + result['per_class_recall'].mean()))
    assert result['Macro-F1'] == pytest.approx(
        result['per_class_f1'].mean())
    assert result['Macro-F1'] != pytest.approx(official_cf1)


def test_micro_f1_matches_hand_calculation():
    y_true = np.zeros((4, NUM_CLASSES), dtype=np.int8)
    y_score = np.zeros((4, NUM_CLASSES), dtype=np.float64)
    y_true[[0, 1], 0] = 1
    y_score[[0, 2], 0] = 1
    y_true[3, 1] = 1
    y_score[3, 1] = 1
    result = evaluate_scores(y_true, y_score, np.full(NUM_CLASSES, 0.5))
    assert result['tp'].sum() == 2
    assert result['fp'].sum() == 1
    assert result['fn'].sum() == 1
    assert result['Micro-F1'] == pytest.approx(4 / 6)


def test_per_class_ap_order_and_map_mean_are_locked():
    y_true, y_score = _perfect_data()
    y_score[:, 5] = [0.1, 0.9]
    result = evaluate_scores(y_true, y_score, np.full(NUM_CLASSES, 0.5))
    rows = per_class_rows(result, CLASS_NAMES)
    assert [row['class_name'] for row in rows] == list(CLASS_NAMES)
    assert rows[5]['AP'] == pytest.approx(0.5)
    assert result['mAP'] == pytest.approx(result['per_class_ap'].mean())


def test_ranking_ap_matches_repository_definition_without_percent_scale():
    module_path = (Path(__file__).resolve().parents[1] / 'mmcls' / 'core'
                   / 'evaluation' / 'mean_ap.py')
    spec = importlib.util.spec_from_file_location('repository_mean_ap', module_path)
    repository_mean_ap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repository_mean_ap)
    y_true, y_score = _perfect_data()
    y_score[:, 3] = [0.2, 0.8]
    result = evaluate_scores(y_true, y_score, np.full(NUM_CLASSES, 0.5))
    assert result['mAP'] * 100 == pytest.approx(
        repository_mean_ap.mAP(y_score, y_true))


def test_calibration_api_has_validation_inputs_only():
    parameters = inspect.signature(calibrate_thresholds).parameters
    assert 'y_true' in parameters
    assert 'y_score' in parameters
    assert all('test' not in name for name in parameters)


def test_threshold_tie_breaks_nearest_half_then_lower():
    y_true, y_score = _perfect_data()
    result = calibrate_thresholds(
        y_true, y_score, threshold_grid=np.array([0.49, 0.51]))
    assert result['thresholds'].tolist() == pytest.approx([0.49] * 18)


@pytest.mark.parametrize(
    ('field', 'replacement', 'message'),
    [
        ('class_names', list(reversed(CLASS_NAMES)), 'class_names'),
        ('thresholds', [0.5] * 17, 'threshold'),
        ('task_protocol', 'wrong_protocol', 'task_protocol'),
    ])
def test_invalid_threshold_artifact_is_rejected(
        tmp_path, field, replacement, message):
    artifact, checkpoint = _artifact(tmp_path)
    artifact[field] = replacement
    with pytest.raises(ValueError, match=message):
        validate_threshold_artifact(
            artifact, CLASS_NAMES, checkpoint_path=checkpoint)


def test_artifact_checkpoint_sha256_is_enforced(tmp_path):
    artifact, checkpoint = _artifact(tmp_path)
    other = tmp_path / 'other.pth'
    other.write_bytes(b'different checkpoint')
    with pytest.raises(ValueError, match='SHA-256'):
        validate_threshold_artifact(
            artifact, CLASS_NAMES, checkpoint_path=other)
    validate_threshold_artifact(
        artifact, CLASS_NAMES, checkpoint_path=checkpoint)


@pytest.mark.parametrize(
    ('y_true_change', 'y_score_change', 'message'),
    [
        (None, lambda score: score.__setitem__((0, 0), np.nan), 'finite'),
        (lambda true: true.__setitem__((0, 0), 2), None, 'binary'),
    ])
def test_invalid_probability_inputs_fail_fast(
        y_true_change, y_score_change, message):
    y_true, y_score = _perfect_data()
    if y_true_change:
        y_true_change(y_true)
    if y_score_change:
        y_score_change(y_score)
    with pytest.raises(ValueError, match=message):
        evaluate_scores(y_true, y_score, np.full(NUM_CLASSES, 0.5))


def test_shape_mismatch_and_wrong_class_count_fail_fast():
    y_true, y_score = _perfect_data()
    with pytest.raises(ValueError, match='same shape'):
        evaluate_scores(y_true[:1], y_score, np.full(NUM_CLASSES, 0.5))
    with pytest.raises(ValueError, match='18'):
        evaluate_scores(y_true[:, :17], y_score[:, :17], np.full(17, 0.5))


def test_out_of_range_probability_fails_fast():
    y_true, y_score = _perfect_data()
    y_score[0, 0] = 1.01
    with pytest.raises(ValueError, match=r'\[0, 1\]'):
        evaluate_scores(y_true, y_score, np.full(NUM_CLASSES, 0.5))


def test_all_zero_predictions_have_defined_zero_f1():
    y_true, _ = _perfect_data()
    y_score = np.zeros_like(y_true, dtype=np.float64)
    result = evaluate_scores(y_true, y_score, np.full(NUM_CLASSES, 0.5))
    assert result['Macro-F1'] == 0
    assert result['Micro-F1'] == 0
    assert np.isfinite(result['per_class_f1']).all()


def test_calibration_rejects_class_without_validation_positive():
    y_true, y_score = _perfect_data()
    y_true[:, 7] = 0
    with pytest.raises(ValueError, match='class 7'):
        calibrate_thresholds(y_true, y_score)


def test_calibration_artifact_has_fixed_schema(tmp_path):
    artifact, checkpoint = _artifact(tmp_path)
    assert artifact['task_protocol'] == TASK_PROTOCOL
    assert artifact['calibration']['validation_mAP'] == pytest.approx(1.0)
    assert artifact['calibration']['validation_macro_f1'] == pytest.approx(1.0)
    assert artifact['calibration']['validation_micro_f1'] == pytest.approx(1.0)
    assert artifact['calibration']['num_samples'] == 2
    assert artifact['num_classes'] == 18
    assert len(artifact['per_class_validation']) == 18
    assert artifact['checkpoint']['sha256'] == checkpoint_identity(
        checkpoint)['sha256']
