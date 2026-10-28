# Tobacco-Clean16 delivery and protocol audit

Date: 2026-10-28. Repository: D:/tech_case/baseline/GKGNet.

The existing formal Dataset, model config and tools were upgraded directly.
No formal GPU training, GPU forward/smoke or formal Test evaluation was run.
Images, historical outputs and checkpoint files were preserved. No branch
switch, workspace cleanup, git add, commit or push was performed.

## CSV and ordered labels

Manifest: tobacco_clean_16cls_721.csv (local audit input, now ignored by Git).
SHA256: 60fee2bccbca44ed6447ec11ea6c4a8ed97434703d95aa0d905ff8f75d74db42.
The hash exactly matches the supplied reference. Header: 29 unique columns;
11150 rows; train=7805, val=2230, test=1115.
Healthy: 231 total (train=162, val=46, test=23), all 16-zero targets.
Multilabel samples: train=2836, val=816, test=409.
No invalid labels, empty filenames, invalid splits, auxiliary count/healthy
inconsistencies or duplicate (split, file_name) pairs. All split_seed values
are 42. Every class has positives in every split.

| Model ID | CSV field | Class name | Train positives | Val positives | Test positives |
|---|---|---|---:|---:|---:|
| 0 | `y00_c02_weather_fleck` | weather_fleck | 2300 | 657 | 328 |
| 1 | `y01_c03_potato_virus_y` | potato_virus_y | 341 | 98 | 49 |
| 2 | `y02_c04_tobacco_mosaic_virus` | tobacco_mosaic_virus | 2903 | 829 | 415 |
| 3 | `y03_c05_cucumber_mosaic_virus` | cucumber_mosaic_virus | 600 | 171 | 86 |
| 4 | `y04_c06_alternaria_leaf_spot` | alternaria_leaf_spot | 1971 | 563 | 282 |
| 5 | `y05_c07_black_shank` | black_shank | 1704 | 487 | 243 |
| 6 | `y06_c08_bacterial_wilt` | bacterial_wilt | 171 | 49 | 24 |
| 7 | `y07_c09_hollow_stalk` | hollow_stalk | 199 | 57 | 28 |
| 8 | `y08_c11_wildfire` | wildfire | 177 | 51 | 25 |
| 9 | `y09_c12_potassium_deficiency` | potassium_deficiency | 234 | 67 | 34 |
| 10 | `y10_c13_magnesium_deficiency` | magnesium_deficiency | 72 | 21 | 10 |
| 11 | `y11_c16_anthracnose` | anthracnose | 236 | 67 | 34 |
| 12 | `y12_c17_frogeye_leaf_spot` | frogeye_leaf_spot | 269 | 77 | 38 |
| 13 | `y13_c18_angular_leaf_spot` | angular_leaf_spot | 91 | 26 | 13 |
| 14 | `y14_c19_sunburn` | sunburn | 64 | 18 | 9 |
| 15 | `y15_c20_herbicide_phytotoxicity` | herbicide_phytotoxicity | 59 | 17 | 8 |

Actual ClassBalancedDataset train length at oversample_thr=0.01: **7990**,
computed using the real Dataset and original wrapper, replacing the historical
10342 assertion. Images resolve as DATA_ROOT / split / file_name; file_name
already includes images/. No additional images/ or task_folder is appended.

## Model and protocol

backbone.n_classes=16 and head.num_classes=16. Static config and offline
backbone metadata tests pass. Actual GPU logits were not executed locally.
GKGNet-S, 448x448, k=9, k_label_gcn=9, drop_path=0.1, original label head/loss,
AdamW, lr=1e-4, scheduler, CropMixup and other augmentation, batch=16, 80 epochs
and dynamic FP16 are preserved. Validation mAP selects best; the documented
formal command explicitly supplies --seed 42.

The unique definitions live in tools/tobacco_protocol.py, protocol
`tobacco_clean16_v1`. The old source definitions and 19-to-18 conversion were
removed. No parallel Dataset/config/evaluation implementation was added.
Only old tests referenced the unused tobacco 576 config; it was deleted and
recipe comparison now uses the untouched official COCO 576 config. Original
network and generic infrastructure files were not modified.

Predictions are [N,16]. mAP is the mean of 16 class APs; Micro/Macro F1,
Precision and Recall remain available. Validation independently searches
0.001 through 0.999 at step 0.001; ties choose closest to 0.5, then lower.
Test requires frozen Validation thresholds; fixed 0.5 auxiliary results remain.
Prediction CSVs and test_per_class.csv use the canonical 16 classes.
Checkpoint SHA256, class order and protocol metadata validation are retained.
Old 18/19 class CSVs, probabilities, weights, thresholds and old protocol
metadata are rejected by regression tests. Existing checkpoint class/protocol
metadata is checked when present; complete state restoration stays strict=True.
FP16 wrapping still occurs before restoring the checkpoint; inference defaults
to the config batch size 16. No FP32/batch1 inference fallback was introduced.

## Verification and limitations

Final requested six-file suite: **88 PASS, 0 FAIL, 0 SKIP**, 8.52 seconds.
These six files are all Python test files currently under tests/.
New protocol tests were observed failing on the old implementation first.
Coverage includes real CSV, all indices, single/multilabel/healthy samples,
column reorder, missing/duplicate/invalid fields, legacy rejection, sampling,
metrics, threshold vectors, output schemas, checkpoint metadata and FP16.

Executed with D:/DevTools/Miniconda3/envs/multicalss/python.exe:

```text
python -m pytest tests/test_tobacco_dataset.py tests/test_tobacco_config.py tests/test_tobacco_metrics.py tests/test_tobacco_phase3_protocol.py tests/test_tobacco_server_tools.py tests/test_pvig_checkpoint_verifier.py -q -p no:cacheprovider --basetemp test_tmp_clean16_verified
```

The default Python lacks torch; the existing multicalss environment was used.
An initial sandbox test run could not access pytest temporary directories;
authorized subsequent runs outside the sandbox passed. Dataset and backbone
metadata tests load real source using lightweight dependency shims. They do
not establish full MMCV/CUDA runtime compatibility. Server image availability
and actual GPU inference behavior remain to be verified on the server.
No real checkpoint Validation/Test evaluation was performed.

git diff --check: exit code 0, no output. Git status reports an existing
.pytest_cache/ permission warning; that directory was not altered or cleaned.
No known unfinished local migration items; server deployment checks remain.

## Paths to confirm before server training

- Actual DATA_ROOT containing train/, val/ and test/.
- Actual MANIFEST path and matching hash; no server absolute path is assumed.
- Readable ImageNet initialization checkpoint/pvig_s_82.1.pth.tar path.
- A new OUTPUT_DIR/work_dir, preserving historical experiment directories.
- Server dependencies/CUDA and image completeness, then the runbook verifier
  and GPU smoke check before training.

Defaults are data_root='./data/tobacco_clean16' and
manifest_file='tobacco_clean_16cls_721.csv'. These are deployment placeholders.
MMCV CLI overrides must set the concrete data.train.dataset/data.val/data.test
entries for data_prefix and ann_file. Changing only a top-level variable does
not recompute its earlier references. Full path overrides and --seed 42 are
in tobacco_gkgnet_server_runbook.md.

## Changed, new and deleted files

M means modified; D means deleted in the output below. New files are
 tools/tobacco_protocol.py and docs/tobacco_clean16_audit.md.
Existing untracked c, archive and outputs/ were left intact. The manifest is
ignored and not staged. git diff --stat excludes untracked new files.

### git diff --name-status

```text
M	.gitignore
M	configs/gkgnet/gkgnet_tobacco_448.py
D	configs/gkgnet/gkgnet_tobacco_576.py
M	docs/tobacco_gkgnet_server_runbook.md
M	mmcls/datasets/tobacco.py
M	tests/test_pvig_checkpoint_verifier.py
M	tests/test_tobacco_config.py
M	tests/test_tobacco_dataset.py
M	tests/test_tobacco_metrics.py
M	tests/test_tobacco_phase3_protocol.py
M	tests/test_tobacco_server_tools.py
M	tools/calibrate_tobacco.py
M	tools/create_best_checkpoint_info.py
M	tools/evaluate_tobacco.py
M	tools/infer_tobacco.py
M	tools/smoke_tobacco_gkgnet.py
M	tools/tobacco_artifacts.py
M	tools/tobacco_metrics.py
M	tools/verify_pvig_checkpoint.py
```

### git diff --stat

```text
 .gitignore                             |   3 +
 configs/gkgnet/gkgnet_tobacco_448.py   |  16 ++--
 configs/gkgnet/gkgnet_tobacco_576.py   | 145 -----------------------------
 docs/tobacco_gkgnet_server_runbook.md  |  39 +++++++-
 mmcls/datasets/tobacco.py              | 160 ++++++++++----------------------
 tests/test_pvig_checkpoint_verifier.py |  30 +-----
 tests/test_tobacco_config.py           |  33 ++-----
 tests/test_tobacco_dataset.py          | 162 ++++++++++++++++++++++++---------
 tests/test_tobacco_metrics.py          |  17 ++--
 tests/test_tobacco_phase3_protocol.py  |  58 +++++++++---
 tests/test_tobacco_server_tools.py     |   2 +-
 tools/calibrate_tobacco.py             |   5 +-
 tools/create_best_checkpoint_info.py   |   5 +-
 tools/evaluate_tobacco.py              |  15 ++-
 tools/infer_tobacco.py                 |  15 ++-
 tools/smoke_tobacco_gkgnet.py          |   8 +-
 tools/tobacco_artifacts.py             |   9 +-
 tools/tobacco_metrics.py               |  18 ++--
 tools/verify_pvig_checkpoint.py        |   2 +-
 19 files changed, 324 insertions(+), 418 deletions(-)
```

### git status -sb

```text
## feat/tobacco-gkgnet-clean16
 M .gitignore
 M configs/gkgnet/gkgnet_tobacco_448.py
 D configs/gkgnet/gkgnet_tobacco_576.py
 M docs/tobacco_gkgnet_server_runbook.md
 M mmcls/datasets/tobacco.py
 M tests/test_pvig_checkpoint_verifier.py
 M tests/test_tobacco_config.py
 M tests/test_tobacco_dataset.py
 M tests/test_tobacco_metrics.py
 M tests/test_tobacco_phase3_protocol.py
 M tests/test_tobacco_server_tools.py
 M tools/calibrate_tobacco.py
 M tools/create_best_checkpoint_info.py
 M tools/evaluate_tobacco.py
 M tools/infer_tobacco.py
 M tools/smoke_tobacco_gkgnet.py
 M tools/tobacco_artifacts.py
 M tools/tobacco_metrics.py
 M tools/verify_pvig_checkpoint.py
?? c
?? docs/tobacco_clean16_audit.md
?? gkgnet_gradcam4_raw.tar.gz
?? outputs/
?? tools/tobacco_protocol.py
```
