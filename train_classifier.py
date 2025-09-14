"""
Простой скрипт для обучения классификатора повреждений авто
Использует ResNet18 - быстро обучается, хорошо работает
"""

import os
import json
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Простой датасет для загрузки COCO данных
class CarDamageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Проверяем существование файла аннотаций
        annotations_path = os.path.join(data_dir, 'annotations.json')
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(f"Файл аннотаций не найден: {annotations_path}")
        
        # Загружаем аннотации
        with open(annotations_path, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = self.coco_data['images']
        
        # Создаем словарь: image_id -> есть ли повреждения
        self.labels = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            self.labels[img_id] = 1  # есть повреждение
        
        # Изображения без аннотаций = без повреждений
        for img in self.images:
            if img['id'] not in self.labels:
                self.labels[img['id']] = 0
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.data_dir, img_info['file_name'])
        
        # Проверяем существование изображения
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Изображение не найдено: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        label = self.labels[img_info['id']]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_class_distribution(dataset):
    """Получить распределение классов в датасете"""
    labels = [dataset.labels[img['id']] for img in dataset.images]
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))

# Трансформации для обучения и валидации
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Создаем датасеты
train_dataset = CarDamageDataset('/app/prepared/train', train_transform)
val_dataset = CarDamageDataset('/app/prepared/val', val_transform)
test_dataset = CarDamageDataset('/app/prepared/test', val_transform)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# DataLoader'ы
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Модель - ResNet18 с предобученными весами
model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 класса: поврежден/не поврежден

# Устройство
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Используется устройство: {device}")

# Loss и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Функция обучения
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(loader), correct / total

# Функция валидации
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return total_loss / len(loader), acc, f1

# Обучение
num_epochs = 10
best_val_acc = 0

print("Начинаем обучение...")
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
    
    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
    print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
    
    # Сохраняем лучшую модель
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), '/app/prepared/best_model.pth')
        print(f'  Сохранена лучшая модель с точностью {val_acc:.4f}')

# Тестирование
print("\nТестирование на test set:")
model.load_state_dict(torch.load('/app/prepared/best_model.pth'))
test_loss, test_acc, test_f1 = validate(model, test_loader, criterion, device)
print(f'Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}')

# Детальный отчет
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nДетальный отчет:")
print(classification_report(all_labels, all_preds, target_names=['Не поврежден', 'Поврежден']))

print("Обучение завершено!")