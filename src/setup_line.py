"""
Скрипт для интерактивной настройки линии IN/OUT
"""
import cv2
import numpy as np
import json
import os
import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional, List
sys.path.append(str(Path(__file__).parent))
from utils import ensure_dir, save_config


class LineSetup:
    def __init__(self, video_path: str, output_config: str = "configs/line_config.yaml"):
        self.video_path = video_path
        self.output_config = output_config
        self.points: List[Tuple[int, int]] = []
        self.current_frame = None
        self.window_name = "Настройка линии IN/OUT - Кликните 2 точки"
        
    def mouse_callback(self, event, x, y, flags, param):
        """Обработчик кликов мыши"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:
                self.points.append((x, y))
                print(f"Точка {len(self.points)}: ({x}, {y})")
                
                # Отрисовка точки
                cv2.circle(self.current_frame, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow(self.window_name, self.current_frame)
                
                # Если обе точки выбраны, рисуем линию
                if len(self.points) == 2:
                    cv2.line(self.current_frame, self.points[0], self.points[1], (0, 255, 0), 2)
                    cv2.imshow(self.window_name, self.current_frame)
                    print("Линия установлена! Нажмите 's' для сохранения или 'r' для сброса")
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Сброс точек
            self.points = []
            self.load_frame()
            print("Точки сброшены")
    
    def load_frame(self):
        """Загружает первый кадр видео"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {self.video_path}")
        
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Не удалось прочитать кадр из видео")
        
        self.current_frame = frame.copy()
        cap.release()
        
        # Восстанавливаем отрисованные точки и линию
        if len(self.points) >= 1:
            cv2.circle(self.current_frame, self.points[0], 5, (0, 255, 0), -1)
        if len(self.points) >= 2:
            cv2.circle(self.current_frame, self.points[1], 5, (0, 255, 0), -1)
            cv2.line(self.current_frame, self.points[0], self.points[1], (0, 255, 0), 2)
    
    def save_config(self):
        """Сохраняет конфигурацию линии"""
        if len(self.points) != 2:
            print("Ошибка: необходимо выбрать 2 точки")
            return False
        
        config = {
            'line': {
                'point1': {'x': int(self.points[0][0]), 'y': int(self.points[0][1])},
                'point2': {'x': int(self.points[1][0]), 'y': int(self.points[1][1])}
            },
            'video_path': self.video_path
        }
        
        ensure_dir(os.path.dirname(self.output_config))
        save_config(config, self.output_config)
        print(f"Конфигурация сохранена в {self.output_config}")
        return True
    
    def run(self):
        """Запускает интерактивную настройку"""
        self.load_frame()
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("Инструкции:")
        print("- Левый клик: добавить точку (нужно 2 точки)")
        print("- Правый клик: сбросить точки")
        print("- 's': сохранить конфигурацию")
        print("- 'r': сбросить точки")
        print("- 'q' или ESC: выйти")
        
        while True:
            cv2.imshow(self.window_name, self.current_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # ESC
                break
            elif key == ord('s'):
                if self.save_config():
                    break
            elif key == ord('r'):
                self.points = []
                self.load_frame()
        
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Настройка линии IN/OUT для подсчета посетителей')
    parser.add_argument('--video', type=str, required=True, help='Путь к видео файлу')
    parser.add_argument('--output', type=str, default='configs/line_config.yaml', 
                       help='Путь к выходному конфигурационному файлу')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Ошибка: файл {args.video} не найден")
        return
    
    setup = LineSetup(args.video, args.output)
    setup.run()


if __name__ == '__main__':
    main()

