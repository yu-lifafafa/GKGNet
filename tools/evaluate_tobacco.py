"""Evaluate Tobacco test predictions with a fixed validation artifact."""

import argparse
import csv
import json
from pathlib import Path

try:
    from tools.tobacco_metrics import (NUM_CLASSES, TASK_PROTOCOL,
                                       checkpoint_identity, evaluate_scores,
                                       load_prediction_npz, per_class_rows,
                                       validate_thresholds)
except ModuleNotFoundError:  # Direct execution as ``python tools/...``.
    from tobacco_metrics import (NUM_CLASSES, TASK_PROTOCOL,
                                 checkpoint_identity, evaluate_scores,
                                 load_prediction_npz, per_class_rows,
                                 validate_thresholds)


PER_CLASS_FIELDS = [
    'model_index', 'class_name', 'AP', 'threshold', 'precision', 'recall',
    'F1', 'TP', 'FP', 'FN', 'positives'
]


def validate_threshold_artifact(artifact,
                                expected_class_names,
                                checkpoint_path=None):
    """Fail fast if an artifact is incompatible with this test run."""
    if not isinstance(artifact, dict):
        raise ValueError('Threshold artifact must be a JSON object.')
    if artifact.get('task_protocol') != TASK_PROTOCOL:
        raise ValueError(
            f'task_protocol must be {TASK_PROTOCOL!r}, got '
            f'{artifact.get("task_protocol")!r}.')
    if artifact.get('num_classes') != NUM_CLASSES:
        raise ValueError('num_classes in threshold artifact must equal 18.')
    names = list(expected_class_names)
    if artifact.get('class_names') != names:
        raise ValueError('class_names order does not match the Tobacco task.')
    thresholds = validate_thresholds(artifact.get('thresholds', []))
    stored_checkpoint = artifact.get('checkpoint')
    if not isinstance(stored_checkpoint, dict):
        raise ValueError('Threshold artifact has no checkpoint identity.')
    if not stored_checkpoint.get('sha256'):
        raise ValueError('Threshold artifact checkpoint has no SHA-256.')
    if checkpoint_path is not None:
        current = checkpoint_identity(checkpoint_path)
        if current['sha256'] != stored_checkpoint['sha256']:
            raise ValueError(
                'Checkpoint SHA-256 does not match threshold artifact.')
    return thresholds


def evaluate_and_write(y_true,
                       y_score,
                       artifact_path,
                       checkpoint_path,
                       output_dir,
                       class_names):
    """Write final test metrics without recalibrating thresholds."""
    artifact_path = Path(artifact_path)
    artifact = json.loads(artifact_path.read_text(encoding='utf-8'))
    thresholds = validate_threshold_artifact(
        artifact, class_names, checkpoint_path=checkpoint_path)
    result = evaluate_scores(y_true, y_score, thresholds)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        'mAP': result['mAP'],
        'Macro-F1': result['Macro-F1'],
        'Micro-F1': result['Micro-F1'],
        'num_samples': int(len(y_true)),
        'num_classes': NUM_CLASSES,
        'checkpoint': checkpoint_identity(checkpoint_path),
        'threshold_artifact': str(artifact_path.resolve()),
        'task_protocol': TASK_PROTOCOL,
    }
    (output_dir / 'metrics.json').write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')

    rows = per_class_rows(result, class_names)
    with (output_dir / 'per_class_metrics.csv').open(
            'w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=PER_CLASS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return {'metrics': metrics, 'per_class_metrics': rows}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate test probabilities with validation thresholds.')
    parser.add_argument(
        '--test-predictions', required=True,
        help='NPZ containing test y_true and sigmoid y_score arrays.')
    parser.add_argument('--threshold-artifact', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    from mmcls.datasets.tobacco import CLASSES

    y_true, y_score = load_prediction_npz(args.test_predictions)
    result = evaluate_and_write(
        y_true=y_true,
        y_score=y_score,
        artifact_path=args.threshold_artifact,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        class_names=CLASSES)
    print(json.dumps(result['metrics'], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
