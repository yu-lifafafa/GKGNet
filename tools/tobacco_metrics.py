"""Pure NumPy metrics for the fixed Tobacco 18-class protocol."""

import hashlib
from pathlib import Path

import numpy as np


NUM_CLASSES = 18
TASK_PROTOCOL = 'tobacco_18cls_exclude_source13'


def validate_probability_inputs(y_true, y_score):
    """Validate and return the binary targets and probability scores."""
    true = np.asarray(y_true)
    score = np.asarray(y_score, dtype=np.float64)
    if true.ndim != 2 or score.ndim != 2:
        raise ValueError('y_true and y_score must both have shape [N, 18].')
    if true.shape != score.shape:
        raise ValueError(
            f'y_true and y_score must have the same shape, got '
            f'{true.shape} and {score.shape}.')
    if true.shape[0] == 0:
        raise ValueError('y_true and y_score must contain at least one sample.')
    if true.shape[1] != NUM_CLASSES:
        raise ValueError(
            f'Expected exactly 18 classes, got shape {true.shape}.')
    if not np.isin(true, (0, 1)).all():
        raise ValueError('y_true must contain binary values {0, 1}.')
    if not np.isfinite(score).all():
        raise ValueError('y_score probabilities must all be finite.')
    if np.any(score < 0) or np.any(score > 1):
        raise ValueError('y_score probabilities must lie in [0, 1].')
    return true.astype(np.int8, copy=False), score


def validate_thresholds(thresholds):
    """Validate a fixed per-class threshold vector."""
    values = np.asarray(thresholds, dtype=np.float64)
    if values.shape != (NUM_CLASSES, ):
        raise ValueError(
            f'thresholds must have shape (18,), got {values.shape}.')
    if not np.isfinite(values).all():
        raise ValueError('thresholds must all be finite.')
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError('thresholds must lie in [0, 1].')
    return values


def binary_average_precision(y_true, y_score):
    """Match the repository ranking AP definition for one binary class.

    Scores are sorted descending and AP is the mean precision at positive
    ranks. A class with no positives has AP 0, matching ``mean_ap.py``.
    """
    target = np.asarray(y_true)
    score = np.asarray(y_score, dtype=np.float64)
    if target.ndim != 1 or score.ndim != 1 or target.shape != score.shape:
        raise ValueError('Binary AP inputs must be same-shaped 1-D arrays.')
    positives = int(np.count_nonzero(target == 1))
    if positives == 0:
        return 0.0
    order = np.argsort(-score)
    ranked_positive = target[order] == 1
    cumulative_tp = np.cumsum(ranked_positive)
    ranks = np.arange(1, target.size + 1, dtype=np.float64)
    precision = cumulative_tp / ranks
    return float(precision[ranked_positive].sum() / positives)


def per_class_average_precision(y_true, y_score):
    """Return AP in the unchanged model-class order."""
    true, score = validate_probability_inputs(y_true, y_score)
    return np.asarray([
        binary_average_precision(true[:, index], score[:, index])
        for index in range(NUM_CLASSES)
    ], dtype=np.float64)


def _safe_ratio(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    result = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def evaluate_scores(y_true, y_score, thresholds):
    """Compute threshold-free AP and fixed-threshold F1 metrics."""
    true, score = validate_probability_inputs(y_true, y_score)
    threshold_values = validate_thresholds(thresholds)
    prediction = (score >= threshold_values.reshape(1, -1)).astype(np.int8)

    tp = np.sum((prediction == 1) & (true == 1), axis=0).astype(np.int64)
    fp = np.sum((prediction == 1) & (true == 0), axis=0).astype(np.int64)
    fn = np.sum((prediction == 0) & (true == 1), axis=0).astype(np.int64)
    positives = np.sum(true == 1, axis=0).astype(np.int64)

    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    per_class_f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)
    micro_denominator = 2 * tp.sum() + fp.sum() + fn.sum()
    micro_f1 = (float(2 * tp.sum() / micro_denominator)
                if micro_denominator else 0.0)
    per_class_ap = per_class_average_precision(true, score)

    return {
        'mAP': float(per_class_ap.mean()),
        'Macro-F1': float(per_class_f1.mean()),
        'Micro-F1': micro_f1,
        'per_class_ap': per_class_ap,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': per_class_f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'positives': positives,
        'thresholds': threshold_values,
        'y_pred': prediction,
    }


def per_class_rows(result, class_names):
    """Convert an evaluation result to ordered, serializable class rows."""
    names = tuple(class_names)
    if len(names) != NUM_CLASSES:
        raise ValueError('class_names must contain exactly 18 ordered names.')
    rows = []
    for index, name in enumerate(names):
        rows.append({
            'model_index': index,
            'class_name': name,
            'AP': float(result['per_class_ap'][index]),
            'threshold': float(result['thresholds'][index]),
            'precision': float(result['per_class_precision'][index]),
            'recall': float(result['per_class_recall'][index]),
            'F1': float(result['per_class_f1'][index]),
            'TP': int(result['tp'][index]),
            'FP': int(result['fp'][index]),
            'FN': int(result['fn'][index]),
            'positives': int(result['positives'][index]),
        })
    return rows


def checkpoint_identity(checkpoint_path):
    """Return a stable file identity without interpreting the checkpoint."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint does not exist: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return {
        'path': str(path.resolve()),
        'basename': path.name,
        'sha256': digest.hexdigest(),
        'size_bytes': path.stat().st_size,
    }


def load_prediction_npz(path):
    """Load the common inference exchange format: y_true + y_score."""
    prediction_path = Path(path)
    if not prediction_path.is_file():
        raise FileNotFoundError(
            f'Prediction artifact does not exist: {prediction_path}')
    with np.load(prediction_path, allow_pickle=False) as arrays:
        missing = {'y_true', 'y_score'} - set(arrays.files)
        if missing:
            raise ValueError(
                f'Prediction NPZ is missing arrays: {sorted(missing)}')
        return validate_probability_inputs(arrays['y_true'], arrays['y_score'])
