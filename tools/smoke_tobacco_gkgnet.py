"""Run one real Tobacco CUDA batch through GKGNet and backward."""

import argparse
import copy
import json
from pathlib import Path

import torch


def _finite_gradient_summary(named_parameters):
    gradients = [
        (name, parameter.grad) for name, parameter in named_parameters
        if parameter.requires_grad and parameter.grad is not None
    ]
    return {
        'gradient_tensors': len(gradients),
        'has_gradient': bool(gradients),
        'all_finite': bool(gradients) and all(
            torch.isfinite(gradient).all().item()
            for _, gradient in gradients),
        'sample_names': [name for name, _ in gradients[:5]],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='One-batch real-data CUDA smoke for Tobacco GKGNet.')
    parser.add_argument('config')
    parser.add_argument(
        '--pretrained-checkpoint', required=True,
        help='Downloaded Pyramid ViG checkpoint used by backbone init_cfg.')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this smoke test.')
    if not Path(args.pretrained_checkpoint).is_file():
        raise FileNotFoundError(
            f'Checkpoint not found: {args.pretrained_checkpoint}')
    if not Path(args.manifest).is_file():
        raise FileNotFoundError(f'Manifest not found: {args.manifest}')

    from mmcv import Config
    from mmcv.parallel import scatter
    from mmcls.datasets import build_dataloader, build_dataset
    from mmcls.models import build_classifier

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda:0')
    properties = torch.cuda.get_device_properties(device)

    cfg = Config.fromfile(args.config)
    dataset_cfg = copy.deepcopy(cfg.data.train.dataset)
    dataset_cfg.data_prefix = args.data_root
    dataset_cfg.ann_file = args.manifest
    dataset_cfg.split = 'train'
    dataset = build_dataset(dataset_cfg)
    batch_size = 1
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=batch_size,
        workers_per_gpu=0,
        num_gpus=1,
        dist=False,
        shuffle=False,
        round_up=False,
        pin_memory=False,
        sampler_cfg=None)

    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.backbone.init_cfg = dict(
        type='Pretrained',
        checkpoint=args.pretrained_checkpoint,
        prefix=None,
        map_location='cpu')
    model = build_classifier(model_cfg)
    model.init_weights()
    model = model.to(device)
    model.train()

    data = next(iter(data_loader))
    data = scatter(data, [0])[0]
    image = data['img']
    gt_label = data['gt_label']
    if tuple(image.shape) != (1, 3, 576, 576):
        raise RuntimeError(
            f'Expected image shape (1, 3, 576, 576), got {tuple(image.shape)}.')
    if tuple(gt_label.shape) != (1, 18):
        raise RuntimeError(
            f'Expected gt_label shape (1, 18), got {tuple(gt_label.shape)}.')

    captured = {}
    hooks = [
        model.head.fc1.register_forward_hook(
            lambda module, inputs, output: captured.__setitem__('fc1', output)),
        model.head.fc2.register_forward_hook(
            lambda module, inputs, output: captured.__setitem__('fc2', output)),
    ]
    torch.cuda.reset_peak_memory_stats(device)
    model.zero_grad(set_to_none=True)
    try:
        losses = model(img=image, gt_label=gt_label, return_loss=True)
    finally:
        for hook in hooks:
            hook.remove()

    logits = captured['fc1'].diagonal(dim1=1, dim2=2) + captured['fc2']
    if tuple(logits.shape) != (1, 18):
        raise RuntimeError(
            f'Expected logits shape (1, 18), got {tuple(logits.shape)}.')
    loss_tensors = []
    for name, value in losses.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        if 'loss' in name:
            loss_tensors.extend(item.mean() for item in values)
    if not loss_tensors:
        raise RuntimeError('Model returned no loss components.')
    total_loss = sum(loss_tensors)
    if not torch.isfinite(total_loss).item():
        raise RuntimeError('Total loss is not finite.')
    if not torch.isfinite(logits).all().item():
        raise RuntimeError('Logits are not finite.')
    total_loss.backward()
    torch.cuda.synchronize(device)

    visual_parameters = list(
        model.backbone.stem.named_parameters(prefix='stem'))
    visual_parameters.extend(
        model.backbone.backbone.named_parameters(prefix='backbone'))
    visual_gradients = _finite_gradient_summary(visual_parameters)
    label_gradients = _finite_gradient_summary([
        ('label_lt.weight', model.backbone.label_lt.weight)
    ])
    head_gradients = _finite_gradient_summary(model.head.named_parameters())
    for name, summary in (
            ('visual backbone', visual_gradients),
            ('label embedding', label_gradients),
            ('head', head_gradients)):
        if not summary['has_gradient'] or not summary['all_finite']:
            raise RuntimeError(f'{name} does not have finite gradients.')

    report = {
        'status': 'PASS',
        'pretrained_checkpoint': str(
            Path(args.pretrained_checkpoint).resolve()),
        'gpu_name': properties.name,
        'gpu_total_memory_bytes': properties.total_memory,
        'allocated_memory_bytes': torch.cuda.memory_allocated(device),
        'max_allocated_memory_bytes': torch.cuda.max_memory_allocated(device),
        'image_shape': list(image.shape),
        'gt_label_shape': list(gt_label.shape),
        'logits_shape': list(logits.shape),
        'loss_components': {
            name: (float(value.detach().mean().cpu())
                   if torch.is_tensor(value) else
                   [float(item.detach().mean().cpu()) for item in value])
            for name, value in losses.items()
        },
        'total_loss': float(total_loss.detach().cpu()),
        'loss_finite': bool(torch.isfinite(total_loss).item()),
        'logits_finite': bool(torch.isfinite(logits).all().item()),
        'visual_backbone_gradients': visual_gradients,
        'label_embedding_gradients': label_gradients,
        'head_gradients': head_gradients,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
