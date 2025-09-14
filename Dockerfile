# Базовый образ с Python и PyTorch (GPU-ready)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Обновляем pip и устанавливаем зависимости
RUN pip install --upgrade pip
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Устанавливаем необходимые системные библиотеки для OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копируем все файлы проекта
COPY . /app/

# Создаем папки для хранения данных
RUN mkdir -p /app/data /app/prepared /app/uploads /app/static /app/templates

# Устанавливаем права на выполнение
RUN chmod +x /app/run_web_app.py

# Открываем порт для веб-приложения
EXPOSE 5000

# Команда по умолчанию - запуск веб-приложения
CMD ["python", "run_web_app.py", "--host", "0.0.0.0", "--port", "5000"]
