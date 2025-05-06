# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_cifar10_dataloaders
from model import SimpleHLBCNN


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    avg_loss = running_loss / len(train_loader)
    return avg_loss, accuracy


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, _ = get_cifar10_dataloaders()
    model = SimpleHLBCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        loss, acc = train(model, train_loader, criterion, optimizer, device)
        print(f'Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={acc:.2f}%')
