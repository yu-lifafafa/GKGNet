"""Calibrate frozen per-class thresholds from validation CSV artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from tools.tobacco_artifacts import (
        ARCHITECTURE, INPUT_SIZE, SCHEMA_VERSION, THRESHOLD_CANDIDATE_COUNT,
        THRESHOLD_MAX, THRESHOLD_MIN, THRESHOLD_STEP, THRESHOLD_TIE_BREAK,
        read_prediction_csv, validate_best_checkpoint_info,
        validate_prediction_metadata)
    from tools.tobacco_metrics import (
        NUM_CLASSES, TASK_PROTOCOL, evaluate_scores,
        validate_probability_inputs)
except ModuleNotFoundError:  # Direct execution as ``python tools/...``.
    from tobacco_artifacts import (
        ARCHITECTURE, INPUT_SIZE, SCHEMA_VERSION, THRESHOLD_CANDIDATE_COUNT,
        THRESHOLD_MAX, THRESHOLD_MIN, THRESHOLD_STEP, THRESHOLD_TIE_BREAK,
        read_prediction_csv, validate_best_checkpoint_info,
        validate_prediction_metadata)
    from tobacco_metrics import (
        NUM_CLASSES, TASK_PROTOCOL, evaluate_scores,
        validate_probability_inputs)


DEFAULT_THRESHOLD_GRID = np.arange(1, 1000, dtype=np.float64) / 1000.0


def calibrate_thresholds(y_true,
                         y_score,
                         threshold_grid=DEFAULT_THRESHOLD_GRID,
                         require_positive_per_class=True):
    """Independently maximize validation F1 for every model class."""
    true, score = validate_probability_inputs(y_true, y_score)
    grid = np.asarray(threshold_grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError('threshold_grid must be a non-empty 1-D array.')
    if not np.isfinite(grid).all() or np.any(grid < 0) or np.any(grid > 1):
        raise ValueError('threshold_grid values must be finite and in [0, 1].')
    grid = np.unique(grid)

    support = true.sum(axis=0)
    if require_positive_per_class:
        missing = np.flatnonzero(support == 0)
        if missing.size:
            raise ValueError(
                'Validation threshold calibration requires positives for '
                f'every class; class {int(missing[0])} has none.')

    thresholds = np.zeros(NUM_CLASSES, dtype=np.float64)
    best_f1 = np.zeros(NUM_CLASSES, dtype=np.float64)
    for class_index in range(NUM_CLASSES):
        target = true[:, class_index]
        scores = score[:, class_index]
        class_f1 = np.zeros(grid.size, dtype=np.float64)
        for index, threshold in enumerate(grid):
            prediction = scores >= threshold
            tp = np.count_nonzero(prediction & (target == 1))
            fp = np.count_nonzero(prediction & (target == 0))
            fn = np.count_nonzero(~prediction & (target == 1))
            denominator = 2 * tp + fp + fn
            class_f1[index] = 2 * tp / denominator if denominator else 0.0
        maximum = class_f1.max()
        tied_thresholds = grid[class_f1 == maximum]
        distances = np.abs(tied_thresholds - 0.5)
        nearest = tied_thresholds[np.isclose(
            distances, distances.min(), rtol=0, atol=1e-12)]
        thresholds[class_index] = nearest.min()
        best_f1[class_index] = maximum

    return {
        'thresholds': thresholds,
        'validation_per_class_f1': best_f1,
        'metrics': evaluate_scores(true, score, thresholds),
    }


def build_calibration_artifact(y_true,
                               y_score,
                               class_names,
                               best_checkpoint_info,
                               validation_metadata):
    """Bind validation calibration to one selected trained checkpoint."""
    names = tuple(class_names)
    best = validate_best_checkpoint_info(best_checkpoint_info, names)
    true, score = validate_probability_inputs(y_true, y_score)
    validate_prediction_metadata(
        validation_metadata,
        best,
        expected_split='val',
        class_names=names,
        expected_num_samples=true.shape[0])
    calibrated = calibrate_thresholds(true, score)
    metrics = calibrated['metrics']

    per_class = []
    for class_id, class_name in enumerate(names):
        per_class.append({
            'class_id': class_id,
            'class_name': class_name,
            'support': int(metrics['support'][class_id]),
            'threshold': float(calibrated['thresholds'][class_id]),
            'precision': float(metrics['per_class_precision'][class_id]),
            'recall': float(metrics['per_class_recall'][class_id]),
            'F1': float(metrics['per_class_f1'][class_id]),
        })

    return {
        'schema_version': SCHEMA_VERSION,
        'task_protocol': TASK_PROTOCOL,
        'architecture': ARCHITECTURE,
        'input_size': INPUT_SIZE,
        'num_classes': NUM_CLASSES,
        'class_names': list(names),
        'checkpoint': {
            'path': best['checkpoint_path'],
            'sha256': best['checkpoint_sha256'],
            'checkpoint_type': best['checkpoint_type'],
            'epoch': best['epoch'],
        },
        'calibration': {
            'split': 'validation',
            'objective': 'per_class_f1',
            'threshold_min': THRESHOLD_MIN,
            'threshold_max': THRESHOLD_MAX,
            'threshold_step': THRESHOLD_STEP,
            'candidate_count': THRESHOLD_CANDIDATE_COUNT,
            'tie_break': THRESHOLD_TIE_BREAK,
            'num_samples': int(true.shape[0]),
            'validation_mAP': metrics['mAP'],
            'validation_micro_f1': metrics['micro_f1'],
            'validation_macro_f1': metrics['macro_f1'],
            'validation_micro_precision': metrics['micro_precision'],
            'validation_micro_recall': metrics['micro_recall'],
            'validation_macro_precision': metrics['macro_precision'],
            'validation_macro_recall': metrics['macro_recall'],
        },
        'thresholds': calibrated['thresholds'].tolist(),
        'per_class_validation': per_class,
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Calibrate Tobacco thresholds from validation artifacts.')
    parser.add_argument('--validation-predictions', required=True)
    parser.add_argument('--validation-metadata', required=True)
    parser.add_argument('--best-checkpoint-info', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args(args)


def main():
    args = parse_args()
    from mmcls.datasets.tobacco import CLASSES

    predictions = read_prediction_csv(
        args.validation_predictions, CLASSES, expected_split='val')
    metadata = json.loads(
        Path(args.validation_metadata).read_text(encoding='utf-8'))
    best = json.loads(
        Path(args.best_checkpoint_info).read_text(encoding='utf-8'))
    validate_best_checkpoint_info(best, CLASSES, verify_checkpoint=True)
    artifact = build_calibration_artifact(
        predictions['y_true'], predictions['y_score'], CLASSES, best, metadata)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    print(f'Wrote validation thresholds: {output}')


if __name__ == '__main__':
    main()
