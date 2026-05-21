"""
Quantization experiments on the trained ResNet-152 teacher.

Uses FX graph mode quantization, which auto-inserts quant/dequant stubs and
auto-fuses Conv+BN+ReLU patterns — works correctly on torchvision's ResNet
which has no built-in QuantStub.

  1. Dynamic Quantization     - Linear -> int8, no calibration
  2. Post-Training Static     - Conv2d + Linear -> int8, needs calibration (FX)
  3. Quantization-Aware Train - Fine-tunes with fake-quantize (FX)

Quantized inference runs on CPU (PyTorch's int8 kernels are CPU-only).
"""

import os
import copy
import csv
import json
import time

import numpy as np

import torch
import torch.nn as nn
import torch.ao.quantization as taq
from torch.ao.quantization import (
    get_default_qconfig_mapping,
    get_default_qat_qconfig_mapping,
)
from torch.ao.quantization.quantize_fx import prepare_fx, prepare_qat_fx, convert_fx
import torchvision
import torchvision.transforms as transforms

from models import get_teacher

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_DIR   = 'models'
TEACHER_PTH = os.path.join(MODEL_DIR, 'teacher.pth')
OUTPUT_DIR  = os.path.join(MODEL_DIR, 'quantization')
DATA_DIR    = 'data'

QAT_EPOCHS = 10
QAT_LR     = 1e-4
CPU_LATENCY_BATCHES = 20

TRAIN_MEAN = (0.485, 0.456, 0.406)
TRAIN_STD  = (0.229, 0.224, 0.225)

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Primary device: {DEVICE}')
print('Int8 quantized inference will run on: cpu (PyTorch limitation)')
if DEVICE == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

if DEVICE == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')


# ─── Helpers ──────────────────────────────────────────────────────────────────
def strip_compile_prefix(state_dict):
    return {k.replace('_orig_mod.', '', 1) if k.startswith('_orig_mod.') else k: v
            for k, v in state_dict.items()}


def load_teacher_local(path: str, device: str) -> nn.Module:
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


@torch.no_grad()
def measure_cpu_latency(model: nn.Module, loader, n_batches: int) -> float:
    model.eval()
    latencies = []
    for i, (images, _) in enumerate(loader):
        if i >= n_batches:
            break
        t0 = time.perf_counter()
        _ = model(images)
        latencies.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(latencies[1:])) if len(latencies) > 1 else float(latencies[0])


def model_disk_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 ** 2)


def make_serialisable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (int, float, str, list)):
            out[k] = v
        elif isinstance(v, np.floating):
            out[k] = float(v)
        else:
            out[k] = str(v)
    return out


# ─── Data ─────────────────────────────────────────────────────────────────────
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(TRAIN_MEAN, TRAIN_STD),
])

test_dataset = torchvision.datasets.CIFAR100(
    root=DATA_DIR, train=False, download=True, transform=test_transform
)

train_dataset_eval = torchvision.datasets.CIFAR100(
    root=DATA_DIR, train=True, download=True, transform=test_transform
)
calib_dataset = torch.utils.data.Subset(train_dataset_eval, list(range(1000)))

optimal_workers = min(os.cpu_count() or 2, 4)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False,
    num_workers=optimal_workers, pin_memory=True,
)
calib_loader = torch.utils.data.DataLoader(
    calib_dataset, batch_size=256, shuffle=False, num_workers=optimal_workers
)
print(f'Test samples: {len(test_dataset):,}  |  Calibration samples: {len(calib_dataset):,}')


# ─── Load Teacher ─────────────────────────────────────────────────────────────
print(f'\n[INFO] Loading teacher from {TEACHER_PTH}...')
teacher_gpu = load_teacher_local(TEACHER_PTH, DEVICE)
total_params = sum(p.numel() for p in teacher_gpu.parameters())
print(f'[INFO] Total parameters: {total_params:,}')


# ─── Baseline ─────────────────────────────────────────────────────────────────
print('\n=== Evaluating BASELINE (FP32) ===')
print(f'[INFO] Accuracy pass on {DEVICE}...')
baseline = evaluate(teacher_gpu, test_loader, DEVICE)

teacher_cpu = copy.deepcopy(teacher_gpu).to('cpu')
teacher_cpu.eval()

print(f'[INFO] CPU latency benchmark over {CPU_LATENCY_BATCHES} batches...')
baseline['avg_latency_ms'] = measure_cpu_latency(teacher_cpu, test_loader, CPU_LATENCY_BATCHES)

baseline_ckpt = os.path.join(OUTPUT_DIR, 'teacher_fp32_baseline.pth')
torch.save(teacher_cpu.state_dict(), baseline_ckpt)
baseline['file_size_mb'] = model_disk_size_mb(baseline_ckpt)
baseline['method']    = 'FP32 Baseline'
baseline['precision'] = 'float32'

print(f"  Top-1: {baseline['top1_acc']:.2f}%  |  Top-5: {baseline['top5_acc']:.2f}%")
print(f"  CPU Latency: {baseline['avg_latency_ms']:.2f} ms/batch  |  Size: {baseline['file_size_mb']:.1f} MB")


# ─── Strategy 1: Dynamic Quantization ─────────────────────────────────────────
print('\n=== Strategy 1: Dynamic Quantization ===')
model_dq = copy.deepcopy(teacher_cpu)
model_dq.eval()

quantized_dq = taq.quantize_dynamic(model_dq, {nn.Linear}, dtype=torch.qint8)
quantized_dq.eval()

dq_path = os.path.join(OUTPUT_DIR, 'teacher_dynamic_quant.pth')
torch.save(quantized_dq, dq_path)

dq_metrics = evaluate(quantized_dq, test_loader, 'cpu')
dq_metrics['file_size_mb'] = model_disk_size_mb(dq_path)
dq_metrics['method']       = 'Dynamic Quantization (Linear -> int8)'
dq_metrics['precision']    = 'int8 (Linear only)'

print(f"  Top-1: {dq_metrics['top1_acc']:.2f}%  (Δ = {dq_metrics['top1_acc'] - baseline['top1_acc']:+.2f}%)")
print(f"  Top-5: {dq_metrics['top5_acc']:.2f}%")
print(f"  Latency: {dq_metrics['avg_latency_ms']:.2f} ms/batch  |  Size: {dq_metrics['file_size_mb']:.1f} MB")
print(f"  Checkpoint: {dq_path}")


# ─── Strategy 2: Post-Training Static Quantization (FX mode) ──────────────────
print('\n=== Strategy 2: Post-Training Static Quantization (PTQ, FX mode) ===')
model_ptq = copy.deepcopy(teacher_gpu)
model_ptq.eval()

qconfig_mapping = get_default_qconfig_mapping('fbgemm')
example_input = next(iter(calib_loader))[0].to(DEVICE)

print('[INFO] Preparing FX graph (auto-fuses Conv+BN+ReLU)...')
model_ptq = prepare_fx(model_ptq, qconfig_mapping, example_input)

print(f'[INFO] Running calibration pass on {DEVICE} (1,000 images)...')
with torch.no_grad():
    for images, _ in calib_loader:
        model_ptq(images.to(DEVICE))
print('[INFO] Calibration complete.')

print('[INFO] Moving to CPU for conversion to int8...')
model_ptq.to('cpu')
model_ptq = convert_fx(model_ptq)
model_ptq.eval()

ptq_path = os.path.join(OUTPUT_DIR, 'teacher_ptq_int8.pth')
torch.save(model_ptq, ptq_path)

ptq_metrics = evaluate(model_ptq, test_loader, 'cpu')
ptq_metrics['file_size_mb'] = model_disk_size_mb(ptq_path)
ptq_metrics['method']       = 'PTQ - Post-Training Static Quantization (FX)'
ptq_metrics['precision']    = 'int8 (Conv2d + Linear)'

print(f"  Top-1: {ptq_metrics['top1_acc']:.2f}%  (Δ = {ptq_metrics['top1_acc'] - baseline['top1_acc']:+.2f}%)")
print(f"  Top-5: {ptq_metrics['top5_acc']:.2f}%")
print(f"  Latency: {ptq_metrics['avg_latency_ms']:.2f} ms/batch  |  Size: {ptq_metrics['file_size_mb']:.1f} MB")
print(f"  Checkpoint: {ptq_path}")


# ─── Strategy 3: QAT (FX mode) ────────────────────────────────────────────────
print(f'\n=== Strategy 3: Quantization-Aware Training ({QAT_EPOCHS} epochs on {DEVICE}, FX mode) ===')

train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(TRAIN_MEAN, TRAIN_STD),
])
train_dataset_aug = torchvision.datasets.CIFAR100(
    root=DATA_DIR, train=True, download=False, transform=train_transform
)
train_loader = torch.utils.data.DataLoader(
    train_dataset_aug, batch_size=32, shuffle=True,
    num_workers=optimal_workers, pin_memory=True,
)

model_qat = copy.deepcopy(teacher_gpu)
del teacher_gpu
import gc; gc.collect()
# prepare_qat_fx needs the model in train() mode and traces the graph from there
model_qat.train()

qat_qconfig_mapping = get_default_qat_qconfig_mapping('fbgemm')
example_input = next(iter(train_loader))[0].to(DEVICE)

print('[INFO] Preparing FX graph for QAT (auto-fuses Conv+BN+ReLU)...')
model_qat = prepare_qat_fx(model_qat, qat_qconfig_mapping, example_input)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_qat.parameters(), lr=QAT_LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=QAT_EPOCHS)

qat_train_losses = []
print(f'[INFO] Starting fine-tuning on {DEVICE}...')
for epoch in range(1, QAT_EPOCHS + 1):
    model_qat.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model_qat(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    avg_loss = running_loss / len(train_loader)
    qat_train_losses.append(avg_loss)
    print(f'  Epoch {epoch:2d}/{QAT_EPOCHS}  |  Loss: {avg_loss:.4f}  |  LR: {scheduler.get_last_lr()[0]:.6f}')

print('[INFO] Moving model to CPU for INT8 conversion...')
model_qat.to('cpu')
model_qat.eval()
model_qat = convert_fx(model_qat)

qat_path = os.path.join(OUTPUT_DIR, 'teacher_qat_int8.pth')
torch.save(model_qat, qat_path)

print('[INFO] Evaluating final QAT model on CPU...')
qat_metrics = evaluate(model_qat, test_loader, 'cpu')
qat_metrics['file_size_mb'] = model_disk_size_mb(qat_path)
qat_metrics['method']       = f'QAT - Quantization-Aware Training ({QAT_EPOCHS} epochs, FX)'
qat_metrics['precision']    = 'int8 (Conv2d + Linear, fine-tuned)'
qat_metrics['train_losses'] = qat_train_losses

print(f"  Top-1: {qat_metrics['top1_acc']:.2f}%  (Δ = {qat_metrics['top1_acc'] - baseline['top1_acc']:+.2f}%)")
print(f"  Top-5: {qat_metrics['top5_acc']:.2f}%")
print(f"  Latency: {qat_metrics['avg_latency_ms']:.2f} ms/batch  |  Size: {qat_metrics['file_size_mb']:.1f} MB")
print(f"  Checkpoint: {qat_path}")


# ─── CSV outputs ──────────────────────────────────────────────────────────────
all_results = [baseline, dq_metrics, ptq_metrics, qat_metrics]

results_csv = os.path.join(OUTPUT_DIR, 'quantization_results.csv')
with open(results_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Method', 'Precision', 'Top1_Acc', 'Top5_Acc',
                'Delta_Top1_vs_Baseline', 'File_Size_MB', 'Avg_Latency_ms_CPU'])
    for r in all_results:
        delta = r['top1_acc'] - baseline['top1_acc']
        w.writerow([
            r['method'], r['precision'],
            round(r['top1_acc'], 4), round(r['top5_acc'], 4),
            round(delta, 4),
            round(r['file_size_mb'], 2),
            round(r['avg_latency_ms'], 4),
        ])
print(f'\nSaved: {results_csv}')

if qat_train_losses:
    loss_csv = os.path.join(OUTPUT_DIR, 'qat_training_loss.csv')
    with open(loss_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Epoch', 'Train_Loss'])
        for epoch_idx, loss_val in enumerate(qat_train_losses, start=1):
            w.writerow([epoch_idx, round(loss_val, 6)])
    print(f'Saved: {loss_csv}')


# ─── JSON summary ─────────────────────────────────────────────────────────────
quant_summary = {
    'experiment': 'quantization',
    'model':      'ResNet-152',
    'dataset':    'CIFAR-100',
    'results':    [make_serialisable(r) for r in all_results],
}
json_path = os.path.join(OUTPUT_DIR, 'quantization_results.json')
with open(json_path, 'w') as f:
    json.dump(quant_summary, f, indent=2)
print(f'Saved: {json_path}')


# ─── Final summary table ──────────────────────────────────────────────────────
print('\n=== FINAL SUMMARY TABLE ===')
print(f'{"Method":<45}  {"Top-1":>8}  {"ΔTop-1":>8}  {"Size(MB)":>10}  {"Lat(ms)":>9}')
print('-' * 85)
for r in all_results:
    delta = r['top1_acc'] - baseline['top1_acc']
    method_short = r['method'][:43]
    print(f"{method_short:<45}  "
          f"{r['top1_acc']:>7.2f}%  "
          f"{delta:>+7.2f}%  "
          f"{r['file_size_mb']:>9.1f}  "
          f"{r['avg_latency_ms']:>8.2f}")