"""
Скрипт для скачивания датасетов с Roboflow.
Использует переменную окружения ROBOFLOW_API_KEY.
Каждый датасет сохраняется в отдельную папку в /app/data
"""

import os
from roboflow import Roboflow

# Считываем API ключ из переменных окружения
API_KEY = os.environ.get("ROBOFLOW_API_KEY")
if not API_KEY:
    raise ValueError("Не найден ROBOFLOW_API_KEY в переменных окружения!")

# Инициализация Roboflow с API ключом
rf = Roboflow(api_key=API_KEY)

# Список датасетов: workspace, project, version, папка для сохранения
datasets = [
    ("seva-at1qy", "rust-and-scrach", 1, "rust_and_scratch"),
    ("carpro", "car-scratch-and-dent", 1, "car_scratch_and_dent"),
    ("project-kmnth", "car-scratch-xgxzs", 1, "car_scratch")
]

DATA_DIR = "/app/data"
os.makedirs(DATA_DIR, exist_ok=True)

# Скачиваем каждый датасет
for workspace, project_name, version_num, folder_name in datasets:
    print(f"Скачиваем датасет {project_name}...")
    project = rf.workspace(workspace).project(project_name)
    dataset = project.version(version_num).download("coco", location=os.path.join(DATA_DIR, folder_name))
    print(f"{project_name} скачан и распакован в {os.path.join(DATA_DIR, folder_name)}")

print("Все датасеты успешно скачаны.")
