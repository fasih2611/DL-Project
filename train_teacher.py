import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import get_teacher

EPOCHS = 30
BATCH  = 64
LR     = 1e-3
SAVE   = 'teacher.pth'
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

model          = get_teacher().to(DEVICE)
compiled_model = torch.compile(model) if hasattr(torch, 'compile') else model

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
writer    = SummaryWriter('runs/teacher')

for epoch in range(1, EPOCHS + 1):
    compiled_model.train()
    train_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = F.cross_entropy(compiled_model(x), y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)
    scheduler.step()
    train_loss /= len(train_loader.dataset)

    compiled_model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (compiled_model(x).argmax(1) == y).sum().item()
    test_acc = 100 * correct / len(test_loader.dataset)

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/test', test_acc, epoch)
    writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
    print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | Test Acc: {test_acc:.2f}%")

writer.close()
torch.save(model.state_dict(), SAVE)
print(f"Saved to {SAVE}")
