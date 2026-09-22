# Tobacco-18 GKGNet-S @ 448×448 server runbook

This runbook freezes the formal identity as `GKGNet-S @ 448×448`,
`Tobacco-18 multi-label`. Run every command from the GKGNet repository root.
Test data must not be accessed until validation checkpoint selection and
threshold calibration are complete.

Placeholders used below:

- `<DATA_ROOT>`: tobacco data root containing `train/`, `val/`, and `test/`.
- `<MANIFEST>`: `image_manifest_updated.csv`.
- `<OUTPUT_DIR>`: one isolated formal experiment directory.
- `<BEST_CHECKPOINT>`: validation-mAP-selected full Tobacco checkpoint.
- `<BEST_EPOCH>` and `<BEST_VAL_MAP>`: values read from the real training log.

## 1. environment check

```bash
python --version
python -c "import torch, mmcv, mmcls; print(torch.__version__, mmcv.__version__, mmcls.__version__)"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "from pathlib import Path; assert Path('<MANIFEST>').is_file(); assert Path('checkpoint/pvig_s_82.1.pth.tar').is_file()"
```

Do not install or upgrade packages during a formal run. Record the environment
and preserve the training console output as `<OUTPUT_DIR>/train.log`.

## 2. pretrained verifier

This checks ImageNet initialization compatibility; it is not a trained-model
restore and does not alter the checkpoint.

```bash
python tools/verify_pvig_checkpoint.py \
  configs/gkgnet/gkgnet_tobacco_448.py \
  checkpoint/pvig_s_82.1.pth.tar \
  --output <OUTPUT_DIR>/pvig_checkpoint_report_448.json
```

## 3. GPU B=1 smoke

```bash
python tools/smoke_tobacco_gkgnet.py \
  configs/gkgnet/gkgnet_tobacco_448.py \
  --pretrained-checkpoint checkpoint/pvig_s_82.1.pth.tar \
  --data-root <DATA_ROOT> \
  --manifest <MANIFEST>
```

Required shapes are image `[1, 3, 448, 448]`, target `[1, 18]`, and logits
`[1, 18]`. Stop on any CUDA, dependency, checkpoint, finite-value, or gradient
failure; do not modify the model during smoke diagnosis.

## 4. formal training

```bash
python tools/train.py configs/gkgnet/gkgnet_tobacco_448.py \
  --work-dir <OUTPUT_DIR> \
  2>&1 | tee <OUTPUT_DIR>/train.log
```

## 5. select by validation mAP

Select the checkpoint solely by mAP on the complete validation split. If both
ordinary and EMA checkpoints exist, evaluate both on validation and retain the
one with the higher validation mAP. F1 and test results must not influence this
selection.

## 6. freeze best checkpoint identity

For the current non-EMA recipe use `ordinary`. Supply the epoch and validation
mAP from the real checkpoint/log; never invent them.

```bash
python tools/create_best_checkpoint_info.py \
  configs/gkgnet/gkgnet_tobacco_448.py \
  <BEST_CHECKPOINT> \
  --epoch <BEST_EPOCH> \
  --validation-map <BEST_VAL_MAP> \
  --checkpoint-type ordinary \
  --output <OUTPUT_DIR>/best_checkpoint_info.json
```

## 7. validation inference

This strictly restores the complete trained checkpoint. Backbone ImageNet
initialization is disabled before restore. `LabelQueryHead.simple_test` already
returns sigmoid probabilities, so the inference tool does not apply sigmoid.

```bash
python tools/infer_tobacco.py \
  configs/gkgnet/gkgnet_tobacco_448.py \
  <BEST_CHECKPOINT> \
  --split val \
  --data-root <DATA_ROOT> \
  --manifest <MANIFEST> \
  --checkpoint-type ordinary \
  --output-dir <OUTPUT_DIR>
```

Outputs: `val_predictions.csv` and `val_predictions.meta.json`.

## 8. validation threshold calibration

The fixed grid is 0.001 through 0.999 in steps of 0.001. Each class maximizes
its validation F1; ties choose the threshold closest to 0.5, then the lower
threshold. Calibration fails if any validation class has zero support.

```bash
python tools/calibrate_tobacco.py \
  --validation-predictions <OUTPUT_DIR>/val_predictions.csv \
  --validation-metadata <OUTPUT_DIR>/val_predictions.meta.json \
  --best-checkpoint-info <OUTPUT_DIR>/best_checkpoint_info.json \
  --output <OUTPUT_DIR>/thresholds.json
```

`thresholds.json` is now frozen. Do not inspect test data before this point.

## 9. test inference

Use the same `<BEST_CHECKPOINT>` and the frozen validation thresholds. The
tool refuses test inference without the threshold artifact.

```bash
python tools/infer_tobacco.py \
  configs/gkgnet/gkgnet_tobacco_448.py \
  <BEST_CHECKPOINT> \
  --split test \
  --data-root <DATA_ROOT> \
  --manifest <MANIFEST> \
  --checkpoint-type ordinary \
  --thresholds <OUTPUT_DIR>/thresholds.json \
  --output-dir <OUTPUT_DIR>
```

Outputs: `test_predictions.csv` and `test_predictions.meta.json`.

## 10. final test evaluation

```bash
python tools/evaluate_tobacco.py \
  --test-predictions <OUTPUT_DIR>/test_predictions.csv \
  --test-metadata <OUTPUT_DIR>/test_predictions.meta.json \
  --threshold-artifact <OUTPUT_DIR>/thresholds.json \
  --best-checkpoint-info <OUTPUT_DIR>/best_checkpoint_info.json \
  --output-dir <OUTPUT_DIR>
```

Outputs: `test_metrics.json` and `test_per_class.csv`. Top-level precision,
recall, and F1 use validation-calibrated thresholds. The uniform 0.5 metrics
are auxiliary and live only under `fixed_0_5_reference`. AP/mAP always use
continuous probabilities.

## Required retained artifacts

Keep all of the following together:

- the best checkpoint at `<BEST_CHECKPOINT>`
- `best_checkpoint_info.json`
- `thresholds.json`
- `val_predictions.csv`
- `val_predictions.meta.json`
- `test_predictions.csv`
- `test_predictions.meta.json`
- `test_metrics.json`
- `test_per_class.csv`
- `train.log`

The SHA-256 checkpoint identity must agree across every JSON artifact. Any
identity, task protocol, class order, input size, or sample-count mismatch is a
hard failure.
