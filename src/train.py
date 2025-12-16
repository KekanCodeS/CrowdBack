"""
Скрипт для обучения YOLOv8 модели с нуля на COCO датасете
"""
import argparse
import os
import sys
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO

sys.path.append(str(Path(__file__).parent))
from utils import load_config, ensure_dir


def train_yolov8(config_path: str = 'configs/train_config.yaml', resume=None):
    """
    Обучает YOLOv8 модель с нуля на COCO датасете
    
    Args:
        config_path: путь к конфигурационному файлу
        resume: путь к чекпоинту для продолжения обучения или True для автоматического поиска
    """
    # Загружаем конфигурацию
    config = load_config(config_path)
    
    print("=" * 60)
    print("Обучение YOLOv8 модели для детекции людей")
    print("=" * 60)
    print(f"Конфигурация: {config_path}")
    print(f"Датасет: {config['data']}")
    print(f"Размер модели: {config['model']['size']}")
    print(f"Эпохи: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Размер изображения: {config['training']['img_size']}")
    print(f"Устройство: {config['training']['device']} (требуется GPU)")
    print("=" * 60)
    
    # Проверка наличия GPU перед началом (обучение требует GPU!)
    if config['training']['device'] == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                "\n" + "=" * 60 + "\n"
                "❌ ОШИБКА: GPU НЕДОСТУПЕН!\n"
                "Обучение требует GPU (CUDA).\n\n"
                "Проверьте:\n"
                "1. Установлен ли PyTorch с поддержкой CUDA?\n"
                f"   Текущая версия: {torch.__version__}\n"
                f"   CUDA доступна: {torch.cuda.is_available()}\n"
                "2. Если версия содержит '+cpu', установите версию с CUDA\n"
                "3. См. инструкции в INSTALL_GPU.md\n"
                "=" * 60
            )
        print(f"\n✅ GPU обнаружен: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA версия: {torch.version.cuda}")
        print(f"   PyTorch версия: {torch.__version__}\n")
    
    # Проверяем наличие датасета
    dataset_path = config['data']
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Датасет не найден: {dataset_path}\n"
            f"Запустите сначала src/prepare_coco.py для подготовки датасета"
        )
    
    # Определяем размер модели
    model_size = config['model']['size']
    
    # Проверяем, нужно ли продолжить обучение
    if resume:
        if resume == True:
            # Автоматический поиск последнего чекпоинта
            save_dir = os.path.join(config['save']['project'], config['save']['name'])
            last_checkpoint = os.path.join(save_dir, 'weights', 'last.pt')
            if os.path.exists(last_checkpoint):
                print(f"\n🔄 Продолжение обучения с чекпоинта: {last_checkpoint}")
                model = YOLO(last_checkpoint)
            else:
                print(f"\n⚠️  Чекпоинт не найден: {last_checkpoint}")
                print("Начинаем обучение с нуля...")
                model_name = f'yolov8{model_size}.yaml'
                model = YOLO(model_name)
        else:
            # Загрузка конкретного чекпоинта
            if os.path.exists(resume):
                print(f"\n🔄 Продолжение обучения с чекпоинта: {resume}")
                model = YOLO(resume)
            else:
                raise FileNotFoundError(f"Чекпоинт не найден: {resume}")
    else:
        # Обучение с нуля
        model_name = f'yolov8{model_size}.yaml'  # Архитектура без весов
        print(f"\nИнициализация модели: {model_name}")
        model = YOLO(model_name)  # Загружает архитектуру
    
    # Определяем устройство (требуем GPU)
    requested_device = config['training']['device']
    
    if requested_device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                "ОШИБКА: CUDA недоступна, но требуется GPU для обучения!\n"
                "Установите PyTorch с поддержкой CUDA. См. INSTALL_GPU.md\n"
                f"torch.cuda.is_available(): {torch.cuda.is_available()}\n"
                f"torch.__version__: {torch.__version__}"
            )
        actual_device = 'cuda'
        print(f"✅ CUDA доступна: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA версия: {torch.version.cuda}")
        print(f"   PyTorch версия: {torch.__version__}")
    else:
        actual_device = requested_device
    
    # Параметры обучения
    train_params = {
        'data': dataset_path,
        'epochs': config['training']['epochs'],
        'batch': config['training']['batch_size'],
        'imgsz': config['training']['img_size'],
        'device': actual_device,
        'workers': config['training']['workers'],
        'patience': config['training']['patience'],
        
        # Оптимизатор
        'lr0': config['optimizer']['lr0'],
        'lrf': config['optimizer']['lrf'],
        'momentum': config['optimizer']['momentum'],
        'weight_decay': config['optimizer']['weight_decay'],
        'warmup_epochs': config['optimizer']['warmup_epochs'],
        'warmup_momentum': config['optimizer']['warmup_momentum'],
        'warmup_bias_lr': config['optimizer']['warmup_bias_lr'],
        
        # Аугментация
        'hsv_h': config['augmentation']['hsv_h'],
        'hsv_s': config['augmentation']['hsv_s'],
        'hsv_v': config['augmentation']['hsv_v'],
        'degrees': config['augmentation']['degrees'],
        'translate': config['augmentation']['translate'],
        'scale': config['augmentation']['scale'],
        'shear': config['augmentation']['shear'],
        'perspective': config['augmentation']['perspective'],
        'flipud': config['augmentation']['flipud'],
        'fliplr': config['augmentation']['fliplr'],
        'mosaic': config['augmentation']['mosaic'],
        'mixup': config['augmentation']['mixup'],
        
        # Сохранение
        'project': config['save']['project'],
        'name': config['save']['name'],
        'save_period': config['save']['save_period'],
        
        # Дополнительные параметры
        'exist_ok': True,  # Перезаписывать существующие результаты
        'verbose': True,
        'resume': bool(resume),  # Продолжить обучение, если указан чекпоинт
    }
    
    print("\nНачало обучения...")
    print("-" * 60)
    
    # Запускаем обучение
    results = model.train(**train_params)
    
    print("-" * 60)
    print("Обучение завершено!")
    print(f"Лучшая модель сохранена в: {results.save_dir}")
    
    # Сохраняем путь к лучшей модели
    best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        # Копируем в models/
        import shutil
        ensure_dir('models')
        final_model_path = f"models/yolov8{model_size}_person_best.pt"
        shutil.copy2(best_model_path, final_model_path)
        print(f"Модель скопирована в: {final_model_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Обучение YOLOv8 модели для детекции людей')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Путь к конфигурационному файлу')
    parser.add_argument('--resume', type=str, default=None, nargs='?', const=True,
                       help='Продолжить обучение с чекпоинта. Укажите путь к .pt файлу или используйте без аргумента для автоматического поиска')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Ошибка: конфигурационный файл не найден: {args.config}")
        return
    
    try:
        train_yolov8(args.config, args.resume)
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("⚠️  Обучение прервано пользователем (Ctrl+C)")
        print("=" * 60)
        print("\n💾 Сохранено:")
        save_dir = os.path.join('runs/train', 'coco_person_yolov8')
        last_checkpoint = os.path.join(save_dir, 'weights', 'last.pt')
        best_checkpoint = os.path.join(save_dir, 'weights', 'best.pt')
        
        if os.path.exists(last_checkpoint):
            print(f"   ✅ Последний чекпоинт: {last_checkpoint}")
        if os.path.exists(best_checkpoint):
            print(f"   ✅ Лучший чекпоинт: {best_checkpoint}")
        
        print("\n📖 Для продолжения обучения выполните:")
        print("   py src/train.py --config configs/train_config.yaml --resume")
        print("\n   Или укажите конкретный чекпоинт:")
        print(f"   py src/train.py --config configs/train_config.yaml --resume {last_checkpoint}")
        print("=" * 60)
    except Exception as e:
        print(f"\nОшибка при обучении: {e}")
        raise


if __name__ == '__main__':
    main()

