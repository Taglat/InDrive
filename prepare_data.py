"""
Скрипт для подготовки датасетов:
1. Объединяет несколько датасетов в один
2. Применяет базовые аугментации (поворот, яркость, шум)
3. Делит на train/val/test
4. Сохраняет готовый датасет в формате COCO
"""

import os
import json  # Добавлен импорт json
from pathlib import Path
import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import supervision as sv
import albumentations as A
import random
from collections import Counter

# Пути к датасетам (внутри контейнера /app/data)
DATASETS = [
    "/app/data/rust_and_scratch/train",
    "/app/data/car_scratch_and_dent/train", 
    "/app/data/car_scratch/train"
]

# Папка для объединенного датасета
MERGED_DIR = "/app/prepared_data"
os.makedirs(MERGED_DIR, exist_ok=True)

def load_coco_dataset(dataset_path):
    """Загружает COCO датасет"""
    annotation_file = os.path.join(dataset_path, "_annotations.coco.json")
    if not os.path.exists(annotation_file):
        print(f"Файл аннотаций не найден: {annotation_file}")
        return []
    
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # Создаем словарь изображений по ID
    images_dict = {img['id']: img for img in coco_data['images']}
    
    # Группируем аннотации по image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    dataset_items = []
    for img_id, image_info in images_dict.items():
        if img_id in annotations_by_image:
            image_path = os.path.join(dataset_path, image_info['file_name'])
            if os.path.exists(image_path):
                dataset_items.append({
                    'image_path': image_path,
                    'image_info': image_info,
                    'annotations': annotations_by_image[img_id]
                })
    
    return dataset_items

# 1. Загружаем все датасеты
all_items = []
for dataset_path in DATASETS:
    items = load_coco_dataset(dataset_path)
    all_items.extend(items)
    print(f"Загружено {len(items)} изображений из {dataset_path}")

print(f"Всего объединено {len(all_items)} изображений")

# 2. Определяем аугментации
augmentation = A.Compose([
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.3)
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

# 3. Обрабатываем изображения с аугментацией
processed_items = []
import random  # если ещё не импортирован

for idx, item in enumerate(all_items):
    try:
        # Загружаем изображение
        img = cv2.imread(item['image_path'])
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Собираем боксы и категории
        bboxes = []
        category_ids = []
        for ann in item['annotations']:
            bbox = ann['bbox']  # [x, y, width, height] в COCO формате
            bboxes.append(bbox)
            category_ids.append(ann['category_id'])
        
        if not bboxes:
            continue
        
        bboxes = [
            [max(0, min(1, x)), max(0, min(1, y)), max(0, min(1, w)), max(0, min(1, h))]
            for x, y, w, h in bboxes
        ]

        # Применяем аугментации с вероятностью 0.5
        if random.random() < 0.5:
            augmented = augmentation(
                image=img,
                bboxes=bboxes,
                category_ids=category_ids
            )
            img = augmented['image']
            bboxes = augmented['bboxes']
            category_ids = augmented['category_ids']
        # Иначе оставляем оригинальные img, bboxes и category_ids без изменений
        
        processed_items.append({
            'image': img,
            'bboxes': bboxes,
            'category_ids': category_ids,
            'original_name': os.path.basename(item['image_path']),
            # Добавляем признаки для проверки баланса
            'clean_dirty': 'dirty' if 'rust' in item['image_path'].lower() else 'clean',
            'intact_damage': 'damaged' if len(bboxes) > 0 else 'intact'
        })
        
    except Exception as e:
        print(f"Ошибка обработки {item['image_path']}: {e}")
        continue

print(f"Обработано {len(processed_items)} изображений")

# 4. Проверяем баланс классов
clean_dirty_counts = Counter([item['clean_dirty'] for item in processed_items])
intact_damage_counts = Counter([item['intact_damage'] for item in processed_items])

print("\nБаланс по чистоте (clean/dirty):")
for k, v in clean_dirty_counts.items():
    print(f"{k}: {v} изображений")

print("\nБаланс по целостности (intact/damaged):")
for k, v in intact_damage_counts.items():
    print(f"{k}: {v} изображений")

# 5. Разделяем на train/val/test
random.shuffle(processed_items)
train_size = int(0.8 * len(processed_items))
val_size = int(0.1 * len(processed_items))

train_items = processed_items[:train_size]
val_items = processed_items[train_size:train_size + val_size]
test_items = processed_items[train_size + val_size:]

print(f"Train: {len(train_items)}, Val: {len(val_items)}, Test: {len(test_items)}")

def save_coco_dataset(items, output_dir, split_name):
    """Сохраняет датасет в COCO формате"""
    os.makedirs(output_dir, exist_ok=True)
    
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "damage", "supercategory": "defect"}
        ]
    }
    
    annotation_id = 1
    
    for idx, item in enumerate(items):
        # Сохраняем изображение
        image_name = f"{split_name}_{idx:06d}.jpg"
        image_path = os.path.join(output_dir, image_name)
        
        img_pil = Image.fromarray(item['image'])
        img_pil.save(image_path, 'JPEG', quality=95)
        
        # Добавляем информацию об изображении
        height, width = item['image'].shape[:2]
        coco_data["images"].append({
            "id": idx,
            "file_name": image_name,
            "width": width,
            "height": height
        })
        
        # Добавляем аннотации
        for bbox, cat_id in zip(item['bboxes'], item['category_ids']):
            x, y, w, h = bbox
            area = w * h
            
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": idx,
                "category_id": 1,  # Все повреждения в одну категорию
                "bbox": [x, y, w, h],
                "area": area,
                "iscrowd": 0
            })
            annotation_id += 1
    
    # Сохраняем аннотации
    with open(os.path.join(output_dir, "annotations.json"), 'w') as f:
        json.dump(coco_data, f, indent=2)

# 6. Сохраняем датасеты
save_coco_dataset(train_items, os.path.join(MERGED_DIR, "train"), "train")
save_coco_dataset(val_items, os.path.join(MERGED_DIR, "val"), "val")
save_coco_dataset(test_items, os.path.join(MERGED_DIR, "test"), "test")

print("Данные подготовлены и разделены на train/val/test.")