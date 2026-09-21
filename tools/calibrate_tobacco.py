"""Calibrate Tobacco thresholds using validation predictions only."""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from tools.tobacco_metrics import (NUM_CLASSES, TASK_PROTOCOL,
                                       checkpoint_identity, evaluate_scores,
                                       load_prediction_npz,
                                       validate_probability_inputs)
except ModuleNotFoundError:  # Direct execution as ``python tools/...``.
    from tobacco_metrics import (NUM_CLASSES, TASK_PROTOCOL,
                                 checkpoint_identity, evaluate_scores,
                                 load_prediction_npz,
                                 validate_probability_inputs)


DEFAULT_THRESHOLD_GRID = np.arange(101, dtype=np.float64) / 100.0


def calibrate_thresholds(y_true,
                         y_score,
                         threshold_grid=DEFAULT_THRESHOLD_GRID,
                         require_positive_per_class=True):
    """Independently maximize validation F1 for each model class."""
    true, score = validate_probability_inputs(y_true, y_score)
    grid = np.asarray(threshold_grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError('threshold_grid must be a non-empty 1-D array.')
    if not np.isfinite(grid).all() or np.any(grid < 0) or np.any(grid > 1):
        raise ValueError('threshold_grid values must be finite and in [0, 1].')
    grid = np.unique(grid)

    positive_counts = true.sum(axis=0)
    if require_positive_per_class:
        missing = np.flatnonzero(positive_counts == 0)
        if missing.size:
            raise ValueError(
                'Validation threshold calibration requires positives for '
                f'every class; class {int(missing[0])} has none.')

    thresholds = np.zeros(NUM_CLASSES, dtype=np.float64)
    best_f1 = np.zeros(NUM_CLASSES, dtype=np.float64)
    for class_index in range(NUM_CLASSES):
        target = true[:, class_index]
        scores = score[:, class_index]
        class_f1 = []
        for threshold in grid:
            prediction = scores >= threshold
            tp = np.count_nonzero(prediction & (target == 1))
            fp = np.count_nonzero(prediction & (target == 0))
            fn = np.count_nonzero(~prediction & (target == 1))
            denominator = 2 * tp + fp + fn
            class_f1.append(2 * tp / denominator if denominator else 0.0)
        class_f1 = np.asarray(class_f1, dtype=np.float64)
        maximum = class_f1.max()
        tied = np.flatnonzero(np.isclose(class_f1, maximum, rtol=0, atol=1e-12))
        tied_thresholds = grid[tied]
        distances = np.abs(tied_thresholds - 0.5)
        nearest = tied_thresholds[np.isclose(
            distances, distances.min(), rtol=0, atol=1e-12)]
        thresholds[class_index] = nearest.min()
        best_f1[class_index] = maximum

    metrics = evaluate_scores(true, score, thresholds)
    return {
        'thresholds': thresholds,
        'validation_per_class_f1': best_f1,
        'metrics': metrics,
    }


def build_calibration_artifact(y_true,
                               y_score,
                               class_names,
                               checkpoint_path,
                               threshold_grid=DEFAULT_THRESHOLD_GRID):
    """Build the complete, checkpoint-bound validation artifact."""
    names = tuple(class_names)
    if len(names) != NUM_CLASSES:
        raise ValueError('class_names must contain exactly 18 ordered names.')
    true, score = validate_probability_inputs(y_true, y_score)
    calibrated = calibrate_thresholds(true, score, threshold_grid)
    metrics = calibrated['metrics']
    grid = np.asarray(threshold_grid, dtype=np.float64)
    return {
        'schema_version': 1,
        'task_protocol': TASK_PROTOCOL,
        'num_classes': NUM_CLASSES,
        'class_names': list(names),
        'checkpoint': checkpoint_identity(checkpoint_path),
        'thresholds': calibrated['thresholds'].tolist(),
        'calibration': {
            'split': 'validation',
            'objective': 'per_class_f1',
            'threshold_grid': grid.tolist(),
            'tie_break': 'nearest_to_0.5_then_lower',
            'num_samples': int(true.shape[0]),
            'per_class_f1': metrics['per_class_f1'].tolist(),
            'Macro-F1': metrics['Macro-F1'],
            'Micro-F1': metrics['Micro-F1'],
            'mAP': metrics['mAP'],
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Calibrate 18 Tobacco thresholds on validation only.')
    parser.add_argument(
        '--validation-predictions', required=True,
        help='NPZ containing validation y_true and sigmoid y_score arrays.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True, help='Threshold JSON path.')
    return parser.parse_args()


def main():
    args = parse_args()
    from mmcls.datasets.tobacco import CLASSES

    y_true, y_score = load_prediction_npz(args.validation_predictions)
    artifact = build_calibration_artifact(
        y_true, y_score, CLASSES, args.checkpoint)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    print(f'Wrote validation thresholds: {output}')


if __name__ == '__main__':
    main()
