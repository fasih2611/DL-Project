"""
L1 Unstructured Global Pruning experiments on the trained ResNet-152 teacher.

Applies global pruning at multiple sparsity levels (10%, 20%, 30%, 40%, 50%) to
all Conv2d weights, evaluates each, and saves checkpoints, CSV, and JSON.
"""

import os
import copy
import csv
import json
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms

from models import get_teacher

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_DIR   = 'models'
TEACHER_PTH = os.path.join(MODEL_DIR, 'teacher.pth')
OUTPUT_DIR  = os.path.join(MODEL_DIR, 'pruning')
DATA_DIR    = 'data'

SPARSITY_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50]

TRAIN_MEAN = (0.485, 0.456, 0.406)
TRAIN_STD  = (0.229, 0.224, 0.225)

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')


# ─── Helpers ──────────────────────────────────────────────────────────────────
def strip_compile_prefix(state_dict):
    """Drop the '_orig_mod.' prefix that torch.compile adds to keys."""
    return {k.replace('_orig_mod.', '', 1) if k.startswith('_orig_mod.') else k: v
            for k, v in state_dict.items()}


def load_teacher_local(path: str, device: str) -> nn.Module:
    """Build the teacher and load trained CIFAR-100 weights from a local .pth."""
    model = get_teacher(num_classes=100, train_full=True).to(device)
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(strip_compile_prefix(state), strict=True)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str) -> dict:
    model.eval()
    correct_top1 = correct_top5 = total = 0
    latencies = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        outputs = model(images)

        if device == 'cuda':
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

        _, pred = outputs.max(1)
        correct_top1 += pred.eq(labels).sum().item()
        _, top5 = outputs.topk(5, dim=1)
        correct_top5 += top5.eq(labels.unsqueeze(1).expand_as(top5)).sum().item()
        total += labels.size(0)

    return {
        'top1_acc':       100.0 * correct_top1 / total,
        'top5_acc':       100.0 * correct_top5 / total,
        'avg_latency_ms': float(np.mean(latencies[1:])) if len(latencies) > 1 else float(latencies[0]),
    }


def count_zeros(model: nn.Module) -> tuple:
    """Return (zero_weights, total_weights) across all Conv2d and Linear layers."""
    zeros = total = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight.data
            zeros += int((w == 0).sum())
            total += int(w.numel())
    return zeros, total


def model_disk_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 ** 2)


# ─── Data ─────────────────────────────────────────────────────────────────────
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(TRAIN_MEAN, TRAIN_STD),
])

test_dataset = torchvision.datasets.CIFAR100(
    root=DATA_DIR, train=False, download=True, transform=test_transform
)

optimal_workers = min(os.cpu_count() or 2, 4)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=256, shuffle=False,
    num_workers=optimal_workers, pin_memory=True,
)
print(f'Test samples: {len(test_dataset):,}  |  workers: {optimal_workers}')


# ─── Load Teacher ─────────────────────────────────────────────────────────────
print(f'\n[INFO] Loading teacher from {TEACHER_PTH}...')
teacher = load_teacher_local(TEACHER_PTH, DEVICE)
total_params = sum(p.numel() for p in teacher.parameters())
print(f'[INFO] Total parameters: {total_params:,}')


# ─── Baseline (Unpruned) ──────────────────────────────────────────────────────
print('\n=== Evaluating BASELINE (unpruned) teacher ===')
baseline_metrics = evaluate(teacher, test_loader, DEVICE)

zeros, total = count_zeros(teacher)
baseline_metrics['actual_sparsity'] = 100.0 * zeros / total
baseline_metrics['sparsity_level']  = 0.0

baseline_ckpt = os.path.join(OUTPUT_DIR, 'teacher_baseline.pth')
torch.save(teacher.state_dict(), baseline_ckpt)
baseline_metrics['file_size_mb'] = model_disk_size_mb(baseline_ckpt)

print(f"  Top-1 Accuracy  : {baseline_metrics['top1_acc']:.2f}%")
print(f"  Top-5 Accuracy  : {baseline_metrics['top5_acc']:.2f}%")
print(f"  Avg Latency     : {baseline_metrics['avg_latency_ms']:.2f} ms/batch")
print(f"  File size       : {baseline_metrics['file_size_mb']:.1f} MB")
print(f"  Actual Sparsity : {baseline_metrics['actual_sparsity']:.2f}%")


# ─── Pruning Loop ─────────────────────────────────────────────────────────────
all_results = [baseline_metrics]

for sparsity in SPARSITY_LEVELS:
    print(f'\n=== Pruning at {int(sparsity*100)}% global sparsity ===')

    model_copy = copy.deepcopy(teacher).to(DEVICE)

    prune_targets = []
    for module in model_copy.modules():
        if isinstance(module, nn.Conv2d):
            prune_targets.append((module, 'weight'))

    prune.global_unstructured(
        prune_targets,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )

    # Make pruning permanent (drop the mask buffers, keep zeroed weights)
    for module, param_name in prune_targets:
        prune.remove(module, param_name)

    model_copy.eval()
    metrics = evaluate(model_copy, test_loader, DEVICE)

    zeros, total = count_zeros(model_copy)
    metrics['actual_sparsity'] = 100.0 * zeros / total
    metrics['sparsity_level']  = sparsity * 100

    ckpt_path = os.path.join(OUTPUT_DIR, f'teacher_pruned_{int(sparsity*100)}pct.pth')
    torch.save(model_copy.state_dict(), ckpt_path)
    metrics['file_size_mb'] = model_disk_size_mb(ckpt_path)

    all_results.append(metrics)

    delta = metrics['top1_acc'] - baseline_metrics['top1_acc']
    print(f"  Top-1 Accuracy  : {metrics['top1_acc']:.2f}%  (Δ = {delta:+.2f}%)")
    print(f"  Top-5 Accuracy  : {metrics['top5_acc']:.2f}%")
    print(f"  Actual Sparsity : {metrics['actual_sparsity']:.2f}%")
    print(f"  File Size       : {metrics['file_size_mb']:.1f} MB")
    print(f"  Avg Latency     : {metrics['avg_latency_ms']:.2f} ms/batch")
    print(f"  Checkpoint      : {ckpt_path}")

print('\n=== Pruning sweep complete ===')


# ─── CSV output ───────────────────────────────────────────────────────────────
results_csv = os.path.join(OUTPUT_DIR, 'pruning_results.csv')
with open(results_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Sparsity_Level_Pct', 'Actual_Sparsity_Pct',
                'Top1_Acc', 'Top5_Acc', 'Delta_Top1_vs_Baseline',
                'File_Size_MB', 'Avg_Latency_ms'])
    for r in all_results:
        delta = r['top1_acc'] - baseline_metrics['top1_acc']
        w.writerow([
            round(r['sparsity_level'], 2),
            round(r['actual_sparsity'], 4),
            round(r['top1_acc'], 4),
            round(r['top5_acc'], 4),
            round(delta, 4),
            round(r['file_size_mb'], 2),
            round(r['avg_latency_ms'], 4),
        ])
print(f'\nSaved: {results_csv}')


# ─── JSON summary ─────────────────────────────────────────────────────────────
pruning_summary = {
    'experiment': 'pruning',
    'model':      'ResNet-152',
    'dataset':    'CIFAR-100',
    'method':     'L1 Unstructured Global Pruning (Conv2d weights)',
    'results':    all_results,
}
json_path = os.path.join(OUTPUT_DIR, 'pruning_results.json')
with open(json_path, 'w') as f:
    json.dump(pruning_summary, f, indent=2)
print(f'Saved: {json_path}')


# ─── Final summary table ──────────────────────────────────────────────────────
print('\n=== FINAL SUMMARY TABLE ===')
print(f'{"Sparsity":>10}  {"Top-1":>8}  {"Top-5":>8}  {"ΔTop-1":>8}  {"Size(MB)":>10}  {"Lat(ms)":>9}')
print('-' * 64)
for r in all_results:
    delta = r['top1_acc'] - baseline_metrics['top1_acc']
    print(f"{r['sparsity_level']:>9.0f}%  "
          f"{r['top1_acc']:>7.2f}%  "
          f"{r['top5_acc']:>7.2f}%  "
          f"{delta:>+7.2f}%  "
          f"{r['file_size_mb']:>9.1f}  "
          f"{r['avg_latency_ms']:>8.2f}")