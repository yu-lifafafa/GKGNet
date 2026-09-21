# Copyright (c) OpenMMLab. All rights reserved.
import csv
import os.path as osp
from decimal import Decimal, InvalidOperation

import numpy as np

from .builder import DATASETS
from .multi_label import MultiLabelDataset


SOURCE_LABEL_COLUMNS = [
    'y00_c02_weather_fleck',
    'y01_c03_potato_virus_y',
    'y02_c04_tobacco_mosaic_virus',
    'y03_c05_cucumber_mosaic_virus',
    'y04_c06_alternaria_leaf_spot',
    'y05_c07_black_shank',
    'y06_c08_bacterial_wilt',
    'y07_c09_hollow_stalk',
    'y08_c10_black_root_rot',
    'y09_c11_wildfire',
    'y10_c12_potassium_deficiency',
    'y11_c13_magnesium_deficiency',
    'y12_c14_powdery_mildew',
    'y13_c15_root_knot_nematodes',
    'y14_c16_anthracnose',
    'y15_c17_frogeye_leaf_spot',
    'y16_c18_angular_leaf_spot',
    'y17_c19_sunburn',
    'y18_c20_herbicide_phytotoxicity',
]

EXCLUDED_SOURCE_LABEL = 'y13_c15_root_knot_nematodes'

MODEL_LABEL_COLUMNS = [
    'y00_c02_weather_fleck',
    'y01_c03_potato_virus_y',
    'y02_c04_tobacco_mosaic_virus',
    'y03_c05_cucumber_mosaic_virus',
    'y04_c06_alternaria_leaf_spot',
    'y05_c07_black_shank',
    'y06_c08_bacterial_wilt',
    'y07_c09_hollow_stalk',
    'y08_c10_black_root_rot',
    'y09_c11_wildfire',
    'y10_c12_potassium_deficiency',
    'y11_c13_magnesium_deficiency',
    'y12_c14_powdery_mildew',
    'y14_c16_anthracnose',
    'y15_c17_frogeye_leaf_spot',
    'y16_c18_angular_leaf_spot',
    'y17_c19_sunburn',
    'y18_c20_herbicide_phytotoxicity',
]

CLASSES = (
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

_EXCLUDED_SOURCE_INDEX = SOURCE_LABEL_COLUMNS.index(EXCLUDED_SOURCE_LABEL)

assert len(SOURCE_LABEL_COLUMNS) == 19
assert len(MODEL_LABEL_COLUMNS) == 18
assert len(CLASSES) == 18
assert MODEL_LABEL_COLUMNS == (
    SOURCE_LABEL_COLUMNS[:_EXCLUDED_SOURCE_INDEX]
    + SOURCE_LABEL_COLUMNS[_EXCLUDED_SOURCE_INDEX + 1:])


@DATASETS.register_module()
class TobaccoMultiLabelDataset(MultiLabelDataset):
    """Tobacco multi-label dataset backed directly by an image manifest CSV.

    ``data_prefix`` is the image root. Images are resolved as
    ``data_prefix / split / file_name``. The manifest always retains the 19
    source label columns; this dataset removes only the fixed root-knot source
    dimension and emits an 18-dimensional multi-hot label.
    """

    CLASSES = CLASSES

    def __init__(self,
                 split,
                 split_column='split',
                 filename_column='file_name',
                 **kwargs):
        self.split = split
        self.split_column = split_column
        self.filename_column = filename_column
        super(TobaccoMultiLabelDataset, self).__init__(**kwargs)

    @staticmethod
    def _parse_binary_label(raw_value, column, line_number):
        value = '' if raw_value is None else raw_value.strip()
        try:
            numeric_value = Decimal(value)
        except (InvalidOperation, ValueError):
            numeric_value = None
        if numeric_value not in (Decimal(0), Decimal(1)):
            raise ValueError(
                f'Invalid binary label at CSV line {line_number}, column '
                f'"{column}": {raw_value!r}. Expected 0 or 1.')
        return int(numeric_value)

    def _validate_header(self, fieldnames):
        if fieldnames is None:
            raise ValueError(f'Manifest is empty or has no header: {self.ann_file}')

        required_columns = [
            self.filename_column, self.split_column, *SOURCE_LABEL_COLUMNS
        ]
        missing_columns = [
            column for column in required_columns if column not in fieldnames
        ]
        if missing_columns:
            raise ValueError(
                'Manifest is missing required column(s): '
                + ', '.join(missing_columns))

        duplicate_source_columns = [
            column for column in SOURCE_LABEL_COLUMNS
            if fieldnames.count(column) != 1
        ]
        if duplicate_source_columns:
            raise ValueError(
                'Manifest source label column(s) must appear exactly once: '
                + ', '.join(duplicate_source_columns))

    def load_annotations(self):
        data_infos = []
        image_prefix = (osp.join(self.data_prefix, self.split)
                        if self.data_prefix is not None else self.split)

        with open(self.ann_file, 'r', encoding='utf-8-sig', newline='') as file:
            reader = csv.DictReader(file)
            self._validate_header(reader.fieldnames)

            for line_number, row in enumerate(reader, start=2):
                if (row[self.split_column] or '').strip() != self.split:
                    continue

                filename = (row[self.filename_column] or '').strip()
                if not filename:
                    raise ValueError(
                        f'Empty filename at CSV line {line_number}, column '
                        f'"{self.filename_column}".')

                source_label = [
                    self._parse_binary_label(row[column], column, line_number)
                    for column in SOURCE_LABEL_COLUMNS
                ]
                model_label = np.asarray(
                    source_label[:_EXCLUDED_SOURCE_INDEX]
                    + source_label[_EXCLUDED_SOURCE_INDEX + 1:],
                    dtype=np.int8)
                if model_label.shape != (18, ):
                    raise ValueError(
                        f'Expected an 18-dimensional model label at CSV line '
                        f'{line_number}, got shape {model_label.shape}.')
                if not np.isin(model_label, (0, 1)).all():
                    raise ValueError(
                        f'Model label at CSV line {line_number} is not binary.')

                data_infos.append(
                    dict(
                        img_prefix=image_prefix,
                        img_info=dict(filename=filename),
                        gt_label=model_label))

        if not data_infos:
            raise ValueError(
                f'Manifest {self.ann_file} contains no samples for split '
                f'"{self.split}".')
        return data_infos

