import torch
import torch.nn.functional as F
import torch.optim as optim
import csv
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import get_teacher

EPOCHS = 30
BATCH  = 64
LR     = 1e-3
SAVE   = '/content/drive/MyDrive/teacher.pth'
LOG    = '/content/drive/MyDrive/teacher_log.csv'
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

# Initialize the CSV file and write the header row
with open(LOG, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        'Epoch', 'Time_S',
        'Train_Loss', 'Train_Acc', 
        'Test_Loss', 'Test_Acc_Top1', 'Test_Acc_Top5',
        'LR'
    ])

for epoch in range(1, EPOCHS + 1):
    start_time = time.time()

    compiled_model.train()
    train_loss, train_correct = 0.0, 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        
        logits = compiled_model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * x.size(0)
        train_correct += (logits.argmax(1) == y).sum().item()
        
    scheduler.step()
    train_loss /= len(train_loader.dataset)
    train_acc = 100 * train_correct / len(train_loader.dataset)

    compiled_model.eval()
    test_loss, correct_top1, correct_top5 = 0.0, 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = compiled_model(x)
            
            # Sum the test loss
            test_loss += F.cross_entropy(logits, y, reduction='sum').item()
            
            # Calculate Top-1 and Top-5 correct predictions
            _, top5_preds = logits.topk(5, 1, True, True)
            correct_top1 += (top5_preds[:, 0] == y).sum().item()
            correct_top5 += top5_preds.eq(y.view(-1, 1).expand_as(top5_preds)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc_top1 = 100 * correct_top1 / len(test_loader.dataset)
    test_acc_top5 = 100 * correct_top5 / len(test_loader.dataset)
    
    epoch_time = time.time() - start_time
    current_lr = scheduler.get_last_lr()[0]

    # Append the metrics for the current epoch to the CSV
    with open(LOG, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            epoch, round(epoch_time, 1), 
            round(train_loss, 4), round(train_acc, 2), 
            round(test_loss, 4), round(test_acc_top1, 2), round(test_acc_top5, 2), 
            current_lr
        ])

    print(f"Epoch {epoch:3d} | Time: {epoch_time:.1f}s | Train Acc: {train_acc:.2f}% | Test Top-1: {test_acc_top1:.2f}% | Test Top-5: {test_acc_top5:.2f}%")

torch.save(model.state_dict(), SAVE)
print(f"Saved weights to {SAVE} and logs to {LOG}")