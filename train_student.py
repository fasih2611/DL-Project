import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import get_teacher, get_student
from distill import distillation_loss

parser = argparse.ArgumentParser()
parser.add_argument('--temperature', type=float, default=4.0)
parser.add_argument('--alpha',       type=float, default=0.9)
parser.add_argument('--hard_only',   action='store_true')
parser.add_argument('--teacher',     default='teacher.pth')
args = parser.parse_args()

EPOCHS = 30
BATCH  = 64
LR     = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

train_loader = DataLoader(
    datasets.CIFAR100('data', train=True, download=True, transform=transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])),
    batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True,
)
test_loader = DataLoader(
    datasets.CIFAR100('data', train=False, download=True, transform=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])),
    batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True,
)

teacher = get_teacher().to(DEVICE)
teacher.load_state_dict(torch.load(args.teacher, map_location=DEVICE))
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)

student          = get_student().to(DEVICE)
compiled_teacher = torch.compile(teacher) if hasattr(torch, 'compile') else teacher
compiled_student = torch.compile(student) if hasattr(torch, 'compile') else student

optimizer = optim.Adam(student.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

mode   = 'hard' if args.hard_only else f'distill_T{args.temperature}_a{args.alpha}'
writer = SummaryWriter(f'runs/student_{mode}')

for epoch in range(1, EPOCHS + 1):
    compiled_student.train()
    train_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        s_logits = compiled_student(x)
        if args.hard_only:
            loss = F.cross_entropy(s_logits, y)
        else:
            with torch.no_grad():
                t_logits = compiled_teacher(x)
            loss = distillation_loss(s_logits, t_logits, y, args.temperature, args.alpha)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)
    scheduler.step()
    train_loss /= len(train_loader.dataset)

    compiled_student.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (compiled_student(x).argmax(1) == y).sum().item()
    test_acc = 100 * correct / len(test_loader.dataset)

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/test', test_acc, epoch)
    writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
    if not args.hard_only:
        writer.add_scalar('Distill/temperature', args.temperature, epoch)
        writer.add_scalar('Distill/alpha', args.alpha, epoch)
    print(f"Epoch {epoch:3d} | [{mode}] Loss: {train_loss:.4f} | Test Acc: {test_acc:.2f}%")

writer.close()
torch.save(student.state_dict(), f'student_{mode}.pth')
print(f"Saved to student_{mode}.pth")
