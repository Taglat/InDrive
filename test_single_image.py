"""
Скрипт для тестирования одного изображения с помощью обученной модели
Использование:
- Без аргументов: тестирует test.jpeg в корне проекта
- С аргументом: тестирует указанный файл
"""

import os
import sys
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse

def load_model(model_path, device):
    """Загружает обученную модель"""
    # Создаем модель с той же архитектурой, что использовалась при обучении
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 класса: поврежден/не поврежден
    
    # Загружаем веса
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

def preprocess_image(image_path, device):
    """Предобработка изображения для модели"""
    # Те же трансформации, что использовались при валидации
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Загружаем и обрабатываем изображение
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Добавляем batch dimension
    image_tensor = image_tensor.to(device)
    
    return image_tensor

def predict_image(model, image_tensor):
    """Предсказание для одного изображения"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities[0]

def main():
    parser = argparse.ArgumentParser(description='Тестирование изображения авто на повреждения')
    parser.add_argument('image_path', nargs='?', default='/app/test_images/test.jpeg', 
                       help='Путь к изображению для тестирования (по умолчанию: /app/test_images/test.jpeg)')
    
    args = parser.parse_args()
    
    # Если передан только имя файла без пути, добавляем путь к test_images
    if not os.path.isabs(args.image_path) and '/' not in args.image_path and '\\' not in args.image_path:
        args.image_path = f'/app/test_images/{args.image_path}'
    
    # Проверяем существование файла изображения
    if not os.path.exists(args.image_path):
        print(f"Ошибка: Файл {args.image_path} не найден!")
        print("Убедитесь, что файл существует или укажите правильный путь.")
        print("Доступные файлы в /app/test_images/:")
        if os.path.exists('/app/test_images/'):
            for f in os.listdir('/app/test_images/'):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    print(f"  - {f}")
        sys.exit(1)
    
    # Проверяем существование модели
    model_path = '/app/prepared/best_model.pth'
    if not os.path.exists(model_path):
        print(f"Ошибка: Модель {model_path} не найдена!")
        print("Сначала обучите модель с помощью train_classifier.py")
        sys.exit(1)
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    # Загружаем модель
    print("Загружаем обученную модель...")
    model = load_model(model_path, device)
    
    # Предобрабатываем изображение
    print(f"Обрабатываем изображение: {args.image_path}")
    image_tensor = preprocess_image(args.image_path, device)
    
    # Делаем предсказание
    print("Делаем предсказание...")
    predicted_class, confidence, probabilities = predict_image(model, image_tensor)
    
    # Выводим результаты
    class_names = ['Не поврежден', 'Поврежден']
    print(f"\nРезультат анализа:")
    print(f"Изображение: {os.path.basename(args.image_path)}")
    print(f"Предсказание: {class_names[predicted_class]}")
    print(f"Уверенность: {confidence:.2%}")
    print(f"\nДетальные вероятности:")
    for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
        print(f"  {class_name}: {prob:.2%}")

if __name__ == "__main__":
    main()