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

# Копируем скрипты
COPY download_datasets.py /app/download_datasets.py
COPY prepare_data.py /app/prepare_data.py
COPY train_classifier.py /app/train_classifier.py
COPY test_single_image.py /app/test_single_image.py

# Создаем папки для хранения данных
RUN mkdir -p /app/data /app/prepared_data
