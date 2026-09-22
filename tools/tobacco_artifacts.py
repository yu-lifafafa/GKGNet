"""CSV/JSON artifact contracts for the frozen Tobacco GKGNet baseline."""

import csv
import math
from pathlib import Path

import numpy as np

try:
    from tools.tobacco_metrics import (
        NUM_CLASSES, TASK_PROTOCOL, checkpoint_identity,
        validate_probability_inputs, validate_thresholds)
except ModuleNotFoundError:  # Direct execution as ``python tools/...``.
    from tobacco_metrics import (
        NUM_CLASSES, TASK_PROTOCOL, checkpoint_identity,
        validate_probability_inputs, validate_thresholds)


SCHEMA_VERSION = 1
ARCHITECTURE = 'GKGNet-S'
INPUT_SIZE = 448
FORMAL_CONFIG = 'configs/gkgnet/gkgnet_tobacco_448.py'
CHECKPOINT_TYPES = ('ordinary', 'ema')
THRESHOLD_MIN = 0.001
THRESHOLD_MAX = 0.999
THRESHOLD_STEP = 0.001
THRESHOLD_CANDIDATE_COUNT = 999
THRESHOLD_TIE_BREAK = 'closest_to_0.5_then_lower'


def _validate_class_names(class_names):
    names = tuple(class_names)
    if len(names) != NUM_CLASSES or len(set(names)) != NUM_CLASSES:
        raise ValueError(
            'class_names must contain exactly 18 unique ordered names.')
    return names


def _same_checkpoint(left, right):
    return (
        isinstance(left, dict)
        and isinstance(right, dict)
        and bool(left.get('sha256'))
        and left.get('sha256') == right.get('sha256'))


def build_best_checkpoint_info(checkpoint_path,
                               epoch,
                               validation_map,
                               class_names,
                               checkpoint_type='ordinary'):
    """Build a complete model-selection record from explicit real values."""
    names = _validate_class_names(class_names)
    if checkpoint_type not in CHECKPOINT_TYPES:
        raise ValueError(
            f'checkpoint_type must be one of {CHECKPOINT_TYPES}.')
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError('epoch must be a non-negative integer.')
    validation_map = float(validation_map)
    if not math.isfinite(validation_map) or not 0 <= validation_map <= 1:
        raise ValueError('validation_mAP must be finite and in [0, 1].')
    identity = checkpoint_identity(checkpoint_path)
    return {
        'schema_version': SCHEMA_VERSION,
        'task_protocol': TASK_PROTOCOL,
        'architecture': ARCHITECTURE,
        'input_size': INPUT_SIZE,
        'config': FORMAL_CONFIG,
        'num_classes': NUM_CLASSES,
        'class_names': list(names),
        'checkpoint_path': identity['path'],
        'checkpoint_sha256': identity['sha256'],
        'checkpoint_type': checkpoint_type,
        'epoch': epoch,
        'validation_mAP': validation_map,
        'checkpoint': identity,
    }


def validate_best_checkpoint_info(info,
                                  class_names,
                                  verify_checkpoint=False):
    """Validate frozen model identity and optionally hash the checkpoint."""
    names = _validate_class_names(class_names)
    if not isinstance(info, dict):
        raise ValueError('best_checkpoint_info must be a JSON object.')
    required = {
        'schema_version': SCHEMA_VERSION,
        'task_protocol': TASK_PROTOCOL,
        'architecture': ARCHITECTURE,
        'input_size': INPUT_SIZE,
        'config': FORMAL_CONFIG,
        'num_classes': NUM_CLASSES,
        'class_names': list(names),
    }
    for key, expected in required.items():
        if info.get(key) != expected:
            raise ValueError(
                f'best_checkpoint_info {key} must be {expected!r}.')
    if info.get('checkpoint_type') not in CHECKPOINT_TYPES:
        raise ValueError('Invalid checkpoint_type in best_checkpoint_info.')
    if not isinstance(info.get('epoch'), int) or info['epoch'] < 0:
        raise ValueError('Invalid epoch in best_checkpoint_info.')
    validation_map = info.get('validation_mAP')
    if (not isinstance(validation_map, (int, float))
            or not math.isfinite(validation_map)
            or not 0 <= validation_map <= 1):
        raise ValueError('Invalid validation_mAP in best_checkpoint_info.')
    stored = info.get('checkpoint')
    if not isinstance(stored, dict):
        raise ValueError('best_checkpoint_info has no checkpoint identity.')
    if (info.get('checkpoint_path') != stored.get('path')
            or info.get('checkpoint_sha256') != stored.get('sha256')):
        raise ValueError('best_checkpoint_info checkpoint fields disagree.')
    if verify_checkpoint:
        actual = checkpoint_identity(info['checkpoint_path'])
        if not _same_checkpoint(stored, actual):
            raise ValueError(
                'best_checkpoint_info checkpoint SHA-256 does not match file.')
    return info


def prediction_fieldnames(class_names, include_predictions=False):
    names = _validate_class_names(class_names)
    fields = ['sample_index', 'split', 'file_name']
    fields.extend(f'gt_{name}' for name in names)
    fields.extend(f'prob_{name}' for name in names)
    if include_predictions:
        fields.extend(f'pred_{name}' for name in names)
    return fields


def write_prediction_csv(path,
                         split,
                         file_names,
                         y_true,
                         y_score,
                         class_names,
                         thresholds=None):
    """Write ordered full-split probabilities and optional fixed predictions."""
    if split not in ('val', 'test'):
        raise ValueError('Prediction split must be val or test.')
    names = _validate_class_names(class_names)
    true, score = validate_probability_inputs(y_true, y_score)
    files = list(file_names)
    if len(files) != true.shape[0]:
        raise ValueError('file_names length does not match predictions.')
    prediction = None
    if thresholds is not None:
        threshold_values = validate_thresholds(thresholds)
        prediction = (score >= threshold_values.reshape(1, -1)).astype(np.int8)
    if split == 'val' and prediction is not None:
        raise ValueError('Validation predictions must remain probabilities only.')
    if split == 'test' and prediction is None:
        raise ValueError('Test predictions require frozen validation thresholds.')

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = prediction_fieldnames(names, prediction is not None)
    with output.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, filename in enumerate(files):
            row = {
                'sample_index': index,
                'split': split,
                'file_name': filename,
            }
            row.update({
                f'gt_{name}': int(true[index, class_index])
                for class_index, name in enumerate(names)
            })
            row.update({
                f'prob_{name}': float(score[index, class_index])
                for class_index, name in enumerate(names)
            })
            if prediction is not None:
                row.update({
                    f'pred_{name}': int(prediction[index, class_index])
                    for class_index, name in enumerate(names)
                })
            writer.writerow(row)
    return output


def read_prediction_csv(path,
                        class_names,
                        expected_split,
                        require_predictions=False):
    """Read and strictly validate an ordered prediction artifact."""
    names = _validate_class_names(class_names)
    prediction_path = Path(path)
    with prediction_path.open('r', encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        actual_fields = reader.fieldnames or []
        without_pred = prediction_fieldnames(names, False)
        with_pred = prediction_fieldnames(names, True)
        if actual_fields == with_pred:
            has_predictions = True
        elif actual_fields == without_pred and not require_predictions:
            has_predictions = False
        else:
            raise ValueError(
                'Prediction CSV columns or class order do not match protocol.')
        rows = list(reader)
    if not rows:
        raise ValueError('Prediction CSV must contain at least one sample.')

    true_rows = []
    score_rows = []
    prediction_rows = []
    file_names = []
    for index, row in enumerate(rows):
        if row['sample_index'] != str(index):
            raise ValueError('sample_index must be contiguous and ordered.')
        if row['split'] != expected_split:
            raise ValueError(
                f'Prediction CSV split must be {expected_split!r}.')
        file_names.append(row['file_name'])
        try:
            true_rows.append([int(row[f'gt_{name}']) for name in names])
            score_rows.append([float(row[f'prob_{name}']) for name in names])
            if has_predictions:
                prediction_rows.append([
                    int(row[f'pred_{name}']) for name in names
                ])
        except (TypeError, ValueError) as error:
            raise ValueError('Prediction CSV contains an invalid value.') from error
    true, score = validate_probability_inputs(true_rows, score_rows)
    prediction = None
    if has_predictions:
        prediction = np.asarray(prediction_rows, dtype=np.int8)
        if not np.isin(prediction, (0, 1)).all():
            raise ValueError('pred_* values must be binary.')
    return {
        'file_names': file_names,
        'y_true': true,
        'y_score': score,
        'y_pred': prediction,
    }


def build_prediction_metadata(split,
                              num_samples,
                              class_names,
                              checkpoint,
                              checkpoint_type,
                              threshold_artifact=None):
    names = _validate_class_names(class_names)
    if split not in ('val', 'test'):
        raise ValueError('Prediction metadata split must be val or test.')
    if checkpoint_type not in CHECKPOINT_TYPES:
        raise ValueError('Invalid checkpoint_type for prediction metadata.')
    if not isinstance(checkpoint, dict) or not checkpoint.get('sha256'):
        raise ValueError('Prediction metadata requires checkpoint identity.')
    metadata = {
        'schema_version': SCHEMA_VERSION,
        'task_protocol': TASK_PROTOCOL,
        'split': split,
        'num_samples': int(num_samples),
        'num_classes': NUM_CLASSES,
        'class_names': list(names),
        'architecture': ARCHITECTURE,
        'input_size': INPUT_SIZE,
        'config': FORMAL_CONFIG,
        'checkpoint_path': checkpoint['path'],
        'checkpoint_sha256': checkpoint['sha256'],
        'checkpoint_type': checkpoint_type,
        'checkpoint': checkpoint,
        'score_semantics': 'sigmoid_probability_from_LabelQueryHead.simple_test',
    }
    if split == 'test':
        if not isinstance(threshold_artifact, dict):
            raise ValueError('Test metadata requires threshold artifact identity.')
        metadata['threshold_artifact'] = threshold_artifact
    elif threshold_artifact is not None:
        raise ValueError('Validation metadata cannot bind thresholds.')
    return metadata


def validate_prediction_metadata(metadata,
                                 best_checkpoint_info,
                                 expected_split,
                                 class_names,
                                 expected_num_samples=None,
                                 threshold_artifact=None):
    """Validate prediction, checkpoint and optional threshold identities."""
    names = _validate_class_names(class_names)
    best = validate_best_checkpoint_info(best_checkpoint_info, names)
    required = {
        'schema_version': SCHEMA_VERSION,
        'task_protocol': TASK_PROTOCOL,
        'split': expected_split,
        'num_classes': NUM_CLASSES,
        'class_names': list(names),
        'architecture': ARCHITECTURE,
        'input_size': INPUT_SIZE,
        'config': FORMAL_CONFIG,
        'checkpoint_type': best['checkpoint_type'],
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(f'Prediction metadata {key} mismatch.')
    if expected_num_samples is not None:
        if metadata.get('num_samples') != expected_num_samples:
            raise ValueError('Prediction metadata num_samples mismatch.')
    if not _same_checkpoint(metadata.get('checkpoint'), best['checkpoint']):
        raise ValueError('Prediction metadata checkpoint identity mismatch.')
    if expected_split == 'test':
        if not _same_checkpoint(
                metadata.get('threshold_artifact'), threshold_artifact):
            raise ValueError('Threshold artifact identity mismatch.')
    return metadata
