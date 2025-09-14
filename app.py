"""
Flask веб-приложение для анализа повреждений автомобилей
Пользователь загружает фото, модель определяет есть ли повреждения
"""

import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, request, render_template, jsonify, redirect, url_for
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size

# Создаем папку для загрузок
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Глобальные переменные для модели
model = None
device = None
transform = None

def is_model_loaded():
    """Проверяет, загружена ли модель"""
    return model is not None and device is not None and transform is not None

def load_model():
    """Загружает обученную модель"""
    global model, device, transform
    
    try:
        # Устройство
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {device}")
        
        # Путь к модели
        model_path = 'prepared/best_model.pth'
        
        print(f"🔍 Проверяем модель по пути: {model_path}")
        print(f"📁 Текущая директория: {os.getcwd()}")
        print(f"📁 Содержимое prepared/: {os.listdir('prepared') if os.path.exists('prepared') else 'Папка не существует'}")
        
        if not os.path.exists(model_path):
            print(f"❌ Ошибка: Модель {model_path} не найдена!")
            return False
        
        # Создаем модель с той же архитектурой
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)  # 2 класса: поврежден/не поврежден
        
        # Загружаем веса
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Трансформации для предобработки
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print("Модель успешно загружена!")
        return True
        
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return False

def predict_image(image_path):
    """Предсказание для загруженного изображения"""
    global model, device, transform
    
    try:
        # Загружаем и обрабатываем изображение
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  # Добавляем batch dimension
        image_tensor = image_tensor.to(device)
        
        # Делаем предсказание
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Результат
        class_names = ['Не поврежден', 'Поврежден']
        result = {
            'prediction': class_names[predicted_class],
            'confidence': confidence,
            'probabilities': {
                'no_damage': probabilities[0][0].item(),
                'damage': probabilities[0][1].item()
            }
        }
        
        return result, None
        
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Обработка загруженного файла"""
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не выбран'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    if file and allowed_file(file.filename):
        # Генерируем уникальное имя файла
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Сохраняем файл
        file.save(filepath)
        
        # Анализируем изображение
        if not is_model_loaded():
            print(f"❌ Модель не загружена! model={model is not None}, device={device is not None}, transform={transform is not None}")
            return jsonify({'error': 'Модель не загружена. Перезапустите сервер.'}), 500
        
        result, error = predict_image(filepath)
        
        if error:
            # Удаляем файл при ошибке
            os.remove(filepath)
            return jsonify({'error': f'Ошибка обработки: {error}'}), 500
        
        # Удаляем временный файл
        os.remove(filepath)
        
        return jsonify(result)
    
    return jsonify({'error': 'Неподдерживаемый формат файла'}), 400

def allowed_file(filename):
    """Проверяет разрешенные расширения файлов"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff', 'tif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health')
def health():
    """Проверка состояния приложения"""
    return jsonify({
        'status': 'ok',
        'model_loaded': is_model_loaded(),
        'device': str(device) if device else None
    })

@app.route('/model_status')
def model_status():
    """Проверка состояния модели"""
    return jsonify({
        'model_loaded': is_model_loaded(),
        'model_exists': os.path.exists('prepared/best_model.pth'),
        'device': str(device) if device else None,
        'current_dir': os.getcwd(),
        'prepared_dir_exists': os.path.exists('prepared'),
        'prepared_contents': os.listdir('prepared') if os.path.exists('prepared') else [],
        'model_vars': {
            'model': model is not None,
            'device': device is not None,
            'transform': transform is not None
        }
    })

# Инициализация модели при импорте модуля
print("Инициализируем модель...")
if load_model():
    print("✅ Модель загружена при инициализации")
else:
    print("❌ Не удалось загрузить модель при инициализации")

if __name__ == '__main__':
    print("Загружаем модель...")
    if load_model():
        print("Запускаем веб-сервер...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Не удалось загрузить модель. Проверьте путь к файлу модели.")
