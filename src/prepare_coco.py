"""
Скрипт для подготовки COCO датасета для обучения YOLOv8
Фильтрует только класс "person" и конвертирует в формат YOLO
"""
import os
import json
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm
import cv2
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from utils import ensure_dir


def download_coco_dataset(output_dir: str, dataset_type: str = 'train2017'):
    """
    Скачивает COCO датасет (требует наличия wget или ручного скачивания)
    
    Args:
        output_dir: директория для сохранения
        dataset_type: 'train2017' или 'val2017'
    """
    print(f"Для скачивания COCO датасета используйте:")
    print(f"1. Официальный сайт: https://cocodataset.org/#download")
    print(f"2. Или используйте pycocotools для автоматического скачивания")
    print(f"\nНеобходимые файлы:")
    print(f"- {dataset_type}.zip (изображения)")
    print(f"- annotations_trainval2017.zip (аннотации)")
    print(f"\nРаспакуйте файлы в {output_dir}")


def filter_person_annotations(coco_annotations_path: str, output_path: str):
    """
    Фильтрует аннотации COCO, оставляя только класс "person" (class_id = 1)
    
    Args:
        coco_annotations_path: путь к JSON файлу с аннотациями COCO
        output_path: путь для сохранения отфильтрованных аннотаций
    """
    print(f"Загрузка аннотаций из {coco_annotations_path}...")
    with open(coco_annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    # Фильтруем аннотации для класса "person" (category_id = 1)
    person_category_id = 1
    person_annotations = [
        ann for ann in coco_data['annotations'] 
        if ann['category_id'] == person_category_id
    ]
    
    # Получаем уникальные image_ids
    person_image_ids = set(ann['image_id'] for ann in person_annotations)
    
    # Фильтруем изображения
    person_images = [
        img for img in coco_data['images']
        if img['id'] in person_image_ids
    ]
    
    # Создаем новый JSON
    filtered_data = {
        'info': coco_data['info'],
        'licenses': coco_data['licenses'],
        'categories': [cat for cat in coco_data['categories'] if cat['id'] == person_category_id],
        'images': person_images,
        'annotations': person_annotations
    }
    
    print(f"Найдено {len(person_images)} изображений с людьми")
    print(f"Найдено {len(person_annotations)} аннотаций людей")
    
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, 'w') as f:
        json.dump(filtered_data, f)
    
    print(f"Отфильтрованные аннотации сохранены в {output_path}")
    return filtered_data


def convert_to_yolo_format(coco_data: dict, images_dir: str, output_dir: str):
    """
    Конвертирует COCO аннотации в формат YOLO
    
    Args:
        coco_data: словарь с данными COCO
        images_dir: директория с изображениями COCO
        output_dir: директория для сохранения YOLO формата
    """
    # Структура YOLO:
    # dataset/
    #   images/
    #     train/
    #     val/
    #   labels/
    #     train/
    #     val/
    
    images_train_dir = os.path.join(output_dir, 'images', 'train')
    images_val_dir = os.path.join(output_dir, 'images', 'val')
    labels_train_dir = os.path.join(output_dir, 'labels', 'train')
    labels_val_dir = os.path.join(output_dir, 'labels', 'val')
    
    ensure_dir(images_train_dir)
    ensure_dir(images_val_dir)
    ensure_dir(labels_train_dir)
    ensure_dir(labels_val_dir)
    
    # Создаем словарь для быстрого доступа к аннотациям по image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # Создаем словарь для доступа к изображениям по id
    images_by_id = {img['id']: img for img in coco_data['images']}
    
    # Разделяем на train/val (80/20)
    all_image_ids = list(images_by_id.keys())
    split_idx = int(len(all_image_ids) * 0.8)
    train_ids = set(all_image_ids[:split_idx])
    val_ids = set(all_image_ids[split_idx:])
    
    print(f"Разделение: {len(train_ids)} train, {len(val_ids)} val")
    
    # Обрабатываем изображения
    for image_id, image_info in tqdm(images_by_id.items(), desc="Конвертация в YOLO формат"):
        is_train = image_id in train_ids
        
        # Пути
        if is_train:
            images_dir_out = images_train_dir
            labels_dir_out = labels_train_dir
        else:
            images_dir_out = images_val_dir
            labels_dir_out = labels_val_dir
        
        # Копируем изображение
        src_image_path = os.path.join(images_dir, image_info['file_name'])
        dst_image_path = os.path.join(images_dir_out, image_info['file_name'])
        
        if os.path.exists(src_image_path):
            shutil.copy2(src_image_path, dst_image_path)
        else:
            print(f"Предупреждение: изображение не найдено {src_image_path}")
            continue
        
        # Создаем файл с аннотациями YOLO
        label_filename = os.path.splitext(image_info['file_name'])[0] + '.txt'
        label_path = os.path.join(labels_dir_out, label_filename)
        
        width = image_info['width']
        height = image_info['height']
        
        with open(label_path, 'w') as f:
            if image_id in annotations_by_image:
                for ann in annotations_by_image[image_id]:
                    # COCO формат: [x, y, width, height] (абсолютные координаты)
                    # YOLO формат: [class_id, x_center, y_center, width, height] (нормализованные)
                    
                    x, y, w, h = ann['bbox']
                    
                    # Конвертируем в центр и нормализуем
                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    w_norm = w / width
                    h_norm = h / height
                    
                    # Класс "person" = 0 в YOLO (так как у нас только один класс)
                    class_id = 0
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    
    # Создаем файл конфигурации для YOLOv8
    config_path = os.path.join(output_dir, 'dataset.yaml')
    # Используем относительный путь для переносимости проекта
    # Если output_dir абсолютный, преобразуем в относительный от текущей рабочей директории
    if os.path.isabs(output_dir):
        # Пытаемся сделать относительный путь
        try:
            dataset_path = os.path.relpath(output_dir)
        except ValueError:
            # Если не получается (разные диски на Windows), используем абсолютный
            dataset_path = output_dir
    else:
        dataset_path = output_dir
    config = {
        'path': dataset_path,
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,  # количество классов
        'names': ['person']
    }
    
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\nКонфигурация сохранена в {config_path}")
    print(f"Датасет готов для обучения YOLOv8!")


def main():
    parser = argparse.ArgumentParser(description='Подготовка COCO датасета для обучения YOLOv8')
    parser.add_argument('--coco_dir', type=str, required=True,
                       help='Директория с COCO датасетом (должна содержать images/ и annotations/)')
    parser.add_argument('--output_dir', type=str, default='data/coco/yolo_format',
                       help='Директория для сохранения YOLO формата')
    parser.add_argument('--annotations_file', type=str, default=None,
                       help='Путь к JSON файлу с аннотациями (если не указан, ищется автоматически)')
    
    args = parser.parse_args()
    
    # Ищем файл аннотаций
    if args.annotations_file:
        annotations_path = args.annotations_file
    else:
        # Ищем в стандартных местах
        possible_paths = [
            os.path.join(args.coco_dir, 'annotations', 'instances_train2017.json'),
            os.path.join(args.coco_dir, 'annotations', 'instances_val2017.json'),
            os.path.join(args.coco_dir, 'annotations_trainval2017', 'annotations', 'instances_train2017.json'),
        ]
        
        annotations_path = None
        for path in possible_paths:
            if os.path.exists(path):
                annotations_path = path
                break
        
        if not annotations_path:
            print("Ошибка: не найден файл с аннотациями")
            print("Укажите путь явно через --annotations_file")
            return
    
    # Ищем директорию с изображениями
    possible_image_dirs = [
        os.path.join(args.coco_dir, 'train2017'),
        os.path.join(args.coco_dir, 'val2017'),
        os.path.join(args.coco_dir, 'images', 'train2017'),
        os.path.join(args.coco_dir, 'images', 'val2017'),
    ]
    
    images_dir = None
    for img_dir in possible_image_dirs:
        if os.path.exists(img_dir):
            images_dir = img_dir
            break
    
    if not images_dir:
        print("Ошибка: не найдена директория с изображениями")
        return
    
    print(f"Используется директория изображений: {images_dir}")
    print(f"Используется файл аннотаций: {annotations_path}")
    
    # Фильтруем аннотации
    filtered_annotations_path = os.path.join(args.output_dir, 'person_annotations.json')
    ensure_dir(args.output_dir)
    
    coco_data = filter_person_annotations(annotations_path, filtered_annotations_path)
    
    # Конвертируем в YOLO формат
    convert_to_yolo_format(coco_data, images_dir, args.output_dir)
    
    print("\nГотово! Датасет подготовлен для обучения.")


if __name__ == '__main__':
    main()

