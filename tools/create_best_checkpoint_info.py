"""Create the frozen best-checkpoint identity artifact after model selection."""

import argparse
import json
from pathlib import Path

try:
    from tools.tobacco_artifacts import (
        FORMAL_CONFIG, build_best_checkpoint_info)
except ModuleNotFoundError:  # Direct execution as ``python tools/...``.
    from tobacco_artifacts import FORMAL_CONFIG, build_best_checkpoint_info


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Record the validation-mAP-selected Tobacco checkpoint.')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--validation-map', type=float, required=True)
    parser.add_argument(
        '--checkpoint-type', choices=('ordinary', 'ema'), default='ordinary')
    parser.add_argument('--output', required=True)
    return parser.parse_args(args)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    expected_config = (repo_root / FORMAL_CONFIG).resolve()
    if Path(args.config).resolve() != expected_config:
        raise ValueError(f'Formal config must be {FORMAL_CONFIG}.')
    from mmcls.datasets.tobacco import CLASSES

    artifact = build_best_checkpoint_info(
        checkpoint_path=args.checkpoint,
        epoch=args.epoch,
        validation_map=args.validation_map,
        class_names=CLASSES,
        checkpoint_type=args.checkpoint_type)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    print(f'Wrote best checkpoint identity: {output}')


if __name__ == '__main__':
    main()
