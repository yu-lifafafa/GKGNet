"""Evaluate test artifacts with frozen validation-derived thresholds."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from tools.tobacco_artifacts import (
        ARCHITECTURE, INPUT_SIZE, SCHEMA_VERSION, checkpoint_identity,
        THRESHOLD_CANDIDATE_COUNT, THRESHOLD_MAX, THRESHOLD_MIN,
        THRESHOLD_STEP, THRESHOLD_TIE_BREAK, read_prediction_csv,
        validate_best_checkpoint_info, validate_prediction_metadata)
    from tools.tobacco_metrics import (
        NUM_CLASSES, TASK_PROTOCOL, evaluate_scores, test_per_class_rows,
        validate_thresholds)
except ModuleNotFoundError:  # Direct execution as ``python tools/...``.
    from tobacco_artifacts import (
        ARCHITECTURE, INPUT_SIZE, SCHEMA_VERSION, checkpoint_identity,
        THRESHOLD_CANDIDATE_COUNT, THRESHOLD_MAX, THRESHOLD_MIN,
        THRESHOLD_STEP, THRESHOLD_TIE_BREAK, read_prediction_csv,
        validate_best_checkpoint_info, validate_prediction_metadata)
    from tobacco_metrics import (
        NUM_CLASSES, TASK_PROTOCOL, evaluate_scores, test_per_class_rows,
        validate_thresholds)


PER_CLASS_FIELDS = [
    'class_id', 'class_name', 'support', 'AP', 'threshold', 'precision',
    'recall', 'F1', 'TP', 'FP', 'FN'
]
AGGREGATE_FIELDS = (
    'micro_f1', 'macro_f1', 'micro_precision', 'micro_recall',
    'macro_precision', 'macro_recall')


def validate_threshold_artifact(artifact,
                                expected_class_names,
                                checkpoint_path=None,
                                checkpoint_type=None):
    """Validate frozen threshold protocol and optional checkpoint file."""
    names = list(expected_class_names)
    required = {
        'schema_version': SCHEMA_VERSION,
        'task_protocol': TASK_PROTOCOL,
        'architecture': ARCHITECTURE,
        'input_size': INPUT_SIZE,
        'num_classes': NUM_CLASSES,
        'class_names': names,
    }
    if not isinstance(artifact, dict):
        raise ValueError('Threshold artifact must be a JSON object.')
    for key, expected in required.items():
        if artifact.get(key) != expected:
            raise ValueError(f'Threshold artifact {key} mismatch.')
    thresholds = validate_thresholds(artifact.get('thresholds', []))
    calibration = artifact.get('calibration')
    expected_calibration = {
        'split': 'validation',
        'objective': 'per_class_f1',
        'threshold_min': THRESHOLD_MIN,
        'threshold_max': THRESHOLD_MAX,
        'threshold_step': THRESHOLD_STEP,
        'candidate_count': THRESHOLD_CANDIDATE_COUNT,
        'tie_break': THRESHOLD_TIE_BREAK,
    }
    if not isinstance(calibration, dict):
        raise ValueError('Threshold artifact has no calibration metadata.')
    for key, expected in expected_calibration.items():
        if calibration.get(key) != expected:
            raise ValueError(f'Threshold calibration {key} mismatch.')
    per_class = artifact.get('per_class_validation')
    if not isinstance(per_class, list) or len(per_class) != NUM_CLASSES:
        raise ValueError('Threshold per_class_validation must have 18 rows.')
    for class_id, (row, class_name) in enumerate(zip(per_class, names)):
        if (row.get('class_id') != class_id
                or row.get('class_name') != class_name):
            raise ValueError('Threshold per-class order mismatch.')
    checkpoint = artifact.get('checkpoint')
    if not isinstance(checkpoint, dict) or not checkpoint.get('sha256'):
        raise ValueError('Threshold artifact has no checkpoint identity.')
    if checkpoint_type is not None:
        if checkpoint.get('checkpoint_type') != checkpoint_type:
            raise ValueError('Checkpoint type does not match threshold artifact.')
    if checkpoint_path is not None:
        actual = checkpoint_identity(checkpoint_path)
        if actual['sha256'] != checkpoint['sha256']:
            raise ValueError(
                'Checkpoint SHA-256 does not match threshold artifact.')
    return thresholds


def _aggregate_metrics(result):
    return {key: float(result[key]) for key in AGGREGATE_FIELDS}


def evaluate_prediction_artifacts(test_predictions,
                                  test_metadata,
                                  threshold_artifact,
                                  best_checkpoint_info,
                                  output_dir,
                                  class_names):
    """Write final test metrics without any threshold search."""
    names = tuple(class_names)
    best_path = Path(best_checkpoint_info)
    best = json.loads(best_path.read_text(encoding='utf-8'))
    validate_best_checkpoint_info(best, names, verify_checkpoint=True)

    threshold_path = Path(threshold_artifact)
    threshold_data = json.loads(threshold_path.read_text(encoding='utf-8'))
    thresholds = validate_threshold_artifact(
        threshold_data,
        names,
        checkpoint_path=best['checkpoint_path'],
        checkpoint_type=best['checkpoint_type'])
    if threshold_data['checkpoint']['sha256'] != best['checkpoint_sha256']:
        raise ValueError('Threshold and best checkpoint identities differ.')

    predictions = read_prediction_csv(
        test_predictions,
        names,
        expected_split='test',
        require_predictions=True)
    metadata = json.loads(Path(test_metadata).read_text(encoding='utf-8'))
    threshold_identity = checkpoint_identity(threshold_path)
    validate_prediction_metadata(
        metadata,
        best,
        expected_split='test',
        class_names=names,
        expected_num_samples=predictions['y_true'].shape[0],
        threshold_artifact=threshold_identity)

    expected_prediction = (
        predictions['y_score'] >= thresholds.reshape(1, -1)).astype(np.int8)
    if not np.array_equal(predictions['y_pred'], expected_prediction):
        raise ValueError(
            'test_predictions pred_* columns do not match frozen thresholds.')

    calibrated = evaluate_scores(
        predictions['y_true'], predictions['y_score'], thresholds)
    fixed_half = evaluate_scores(
        predictions['y_true'], predictions['y_score'],
        np.full(NUM_CLASSES, 0.5, dtype=np.float64))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    metrics = {
        'schema_version': SCHEMA_VERSION,
        'task_protocol': TASK_PROTOCOL,
        'architecture': ARCHITECTURE,
        'input_size': INPUT_SIZE,
        'config': best['config'],
        'num_samples': int(predictions['y_true'].shape[0]),
        'num_classes': NUM_CLASSES,
        'mAP': calibrated['mAP'],
        **_aggregate_metrics(calibrated),
        'fixed_0_5_reference': _aggregate_metrics(fixed_half),
        'checkpoint': best['checkpoint'],
        'checkpoint_type': best['checkpoint_type'],
        'threshold_artifact': threshold_identity,
        'best_checkpoint_info': checkpoint_identity(best_path),
    }
    (output / 'test_metrics.json').write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')

    rows = test_per_class_rows(calibrated, names)
    with (output / 'test_per_class.csv').open(
            'w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=PER_CLASS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return {
        'metrics': metrics,
        'calibrated': calibrated,
        'fixed_0_5_reference': fixed_half,
        'per_class': rows,
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Evaluate test CSV with frozen validation thresholds.')
    parser.add_argument('--test-predictions', required=True)
    parser.add_argument('--test-metadata', required=True)
    parser.add_argument('--threshold-artifact', required=True)
    parser.add_argument('--best-checkpoint-info', required=True)
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args(args)


def main():
    args = parse_args()
    from mmcls.datasets.tobacco import CLASSES

    result = evaluate_prediction_artifacts(
        test_predictions=args.test_predictions,
        test_metadata=args.test_metadata,
        threshold_artifact=args.threshold_artifact,
        best_checkpoint_info=args.best_checkpoint_info,
        output_dir=args.output_dir,
        class_names=CLASSES)
    print(json.dumps(result['metrics'], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
