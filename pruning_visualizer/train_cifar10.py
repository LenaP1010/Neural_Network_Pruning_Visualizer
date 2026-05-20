"""
Скрипт для тренировки модели на CIFAR-10.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm

# Устанавливаем метод старта процессов ДО ВСЕХ ОПЕРАЦИЙ С MULTIPROCESSING
torch.multiprocessing.set_start_method('forkserver')  # Пробуем forkserver вместо spawn

# Вся логика переносится внутрь функции main
def main():
    # Загрузка данных
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.456, 0.406), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.456, 0.406), (0.2023, 0.1994, 0.2010)),
    ])

    # Переместили внутрь функции
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

    # Определение модели (используем ResNet18)
    model = torchvision.models.resnet18(weights=None)  # Без предобученных весов
    model.fc = nn.Linear(model.fc.in_features, 10)  # Замена выходного слоя на 10 классов

    # Критерий и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Перемещаем модель на устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Обучение
    best_accuracy = 0.0
    for epoch in range(50):  # 200 эпох
        model.train()
        running_loss = 0.0
        for inputs, targets in tqdm(trainloader, desc=f"Epoch {epoch}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        
        # Валидация
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch {epoch}: Train Loss={running_loss / len(trainloader)}, Test Acc={accuracy:.2f}%")
        
        # Сохраняем лучшую модель
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "trained_cifar10_model.pt")

    print(f"\nЛучшая точность на тесте: {best_accuracy:.2f}%")

# Запускаем main только при прямом запуске файла
if __name__ == "__main__":
    main()