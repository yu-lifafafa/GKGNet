# Copyright (c) OpenMMLab. All rights reserved.
import csv
import os.path as osp
from decimal import Decimal, InvalidOperation

import numpy as np

from tools.tobacco_protocol import CLASSES, MODEL_LABEL_COLUMNS

from .builder import DATASETS
from .multi_label import MultiLabelDataset


@DATASETS.register_module()
class TobaccoMultiLabelDataset(MultiLabelDataset):
    """Tobacco multi-label dataset backed directly by an image manifest CSV.

    ``data_prefix / split / file_name`` resolves each image. ``file_name``
    already includes images/. Labels follow the canonical 16 named columns;
    healthy samples have no positive labels.
    """

    CLASSES = CLASSES

    def __init__(self,
                 split,
                 split_column='split',
                 filename_column='file_name',
                 **kwargs):
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'Invalid split: {split!r}.')
        classes = kwargs.get('classes')
        if classes is not None and tuple(classes) != CLASSES:
            raise ValueError('classes must match the canonical Clean16 order.')
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

        duplicates = sorted({c for c in fieldnames if fieldnames.count(c) > 1})
        if duplicates:
            raise ValueError('Duplicate CSV columns: ' + ', '.join(duplicates))
        required = [self.filename_column, self.split_column, *MODEL_LABEL_COLUMNS]
        missing = [c for c in required if c not in fieldnames]
        if missing:
            raise ValueError(
                'Manifest is missing required columns: ' + ', '.join(missing))
        unexpected = [c for c in fieldnames
                      if c.startswith('y') and c not in MODEL_LABEL_COLUMNS]
        if unexpected:
            raise ValueError('Unexpected label columns: ' + ', '.join(unexpected))

    def load_annotations(self):
        data_infos = []
        image_prefix = (osp.join(self.data_prefix, self.split)
                        if self.data_prefix is not None else self.split)

        with open(self.ann_file, 'r', encoding='utf-8-sig', newline='') as file:
            reader = csv.DictReader(file)
            self._validate_header(reader.fieldnames)

            for line_number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f'Malformed CSV row at line {line_number}.')
                row_split = row[self.split_column].strip()
                if row_split not in ('train', 'val', 'test'):
                    raise ValueError(
                        f'Invalid split at CSV line {line_number}: {row_split!r}.')

                filename = (row[self.filename_column] or '').strip()
                if not filename:
                    raise ValueError(
                        f'Empty filename at CSV line {line_number}, column '
                        f'"{self.filename_column}".')

                model_label = np.asarray([
                    self._parse_binary_label(row[column], column, line_number)
                    for column in MODEL_LABEL_COLUMNS], dtype=np.int8)
                positive_count = int(model_label.sum())
                if 'positive_label_count' in row:
                    try:
                        count = Decimal(row['positive_label_count'].strip())
                    except InvalidOperation:
                        count = None
                    if count != positive_count:
                        raise ValueError(
                            'positive_label_count mismatch at CSV line '
                            f'{line_number}.')
                if 'is_healthy' in row:
                    healthy = self._parse_binary_label(
                        row['is_healthy'], 'is_healthy', line_number)
                    if healthy != int(positive_count == 0):
                        raise ValueError(f'is_healthy mismatch at CSV line {line_number}.')
                if row_split != self.split:
                    continue

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

