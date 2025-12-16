"""
Вспомогательные функции для системы подсчета посетителей
"""
import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional


def ensure_dir(path: str) -> None:
    """Создает директорию, если она не существует"""
    Path(path).mkdir(parents=True, exist_ok=True)


def load_config(config_path: str) -> Dict:
    """Загружает конфигурационный файл"""
    with open(config_path, 'r', encoding='utf-8') as f:
        import yaml
        return yaml.safe_load(f)


def save_config(config: Dict, config_path: str) -> None:
    """Сохраняет конфигурационный файл"""
    with open(config_path, 'w', encoding='utf-8') as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def get_video_info(video_path: str) -> Dict:
    """Получает информацию о видео файле"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    }
    cap.release()
    return info


def calculate_line_intersection(line1: Tuple[Tuple[float, float], Tuple[float, float]], 
                                line2: Tuple[Tuple[float, float], Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """
    Вычисляет точку пересечения двух линий
    
    Args:
        line1: ((x1, y1), (x2, y2)) - первая линия
        line2: ((x3, y3), (x4, y4)) - вторая линия
    
    Returns:
        Точка пересечения (x, y) или None, если линии не пересекаются
    """
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)
    return None


def point_to_line_distance(point: Tuple[float, float], 
                          line: Tuple[Tuple[float, float], Tuple[float, float]]) -> float:
    """
    Вычисляет расстояние от точки до линии
    
    Args:
        point: (x, y) - точка
        line: ((x1, y1), (x2, y2)) - линия
    
    Returns:
        Расстояние от точки до линии
    """
    (x, y) = point
    (x1, y1), (x2, y2) = line
    
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    
    distance = abs(A * x + B * y + C) / np.sqrt(A * A + B * B + 1e-10)
    return distance


def get_line_side(point: Tuple[float, float], 
                  line: Tuple[Tuple[float, float], Tuple[float, float]]) -> int:
    """
    Определяет, с какой стороны линии находится точка
    
    Args:
        point: (x, y) - точка
        line: ((x1, y1), (x2, y2)) - линия
    
    Returns:
        1 если точка справа от линии, -1 если слева, 0 если на линии
    """
    (x, y) = point
    (x1, y1), (x2, y2) = line
    
    d = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(d) < 1e-10:
        return 0
    return 1 if d > 0 else -1


def get_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """
    Получает центр bounding box
    
    Args:
        bbox: [x1, y1, x2, y2] или [x_center, y_center, width, height]
    
    Returns:
        (x_center, y_center)
    """
    if len(bbox) == 4:
        # Если формат [x1, y1, x2, y2]
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
        else:
            # Если формат [x_center, y_center, width, height]
            x_center = bbox[0]
            y_center = bbox[1]
        return (x_center, y_center)
    return (0, 0)


def save_statistics(stats: Dict, output_path: str) -> None:
    """Сохраняет статистику в JSON файл"""
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def load_statistics(input_path: str) -> Dict:
    """Загружает статистику из JSON файла"""
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)

