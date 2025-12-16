"""
Скрипт для обработки видео и подсчета посетителей с использованием обученной YOLOv8 модели
"""
import cv2
import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from utils import ensure_dir, load_config, get_video_info, save_statistics, get_bbox_center
from tracking import PersonTracker
from heatmap import HeatmapGenerator
from metrics_collector import AIMetricsCollector


class VideoProcessor:
    """Класс для обработки видео и подсчета посетителей"""
    
    def __init__(self, model_path: str, line_config_path: str = None, 
                 use_tracking: bool = True, generate_heatmap: bool = True,
                 frame_skip: int = 1):
        """
        Args:
            model_path: путь к обученной YOLOv8 модели
            line_config_path: путь к конфигурации линии IN/OUT
            use_tracking: использовать ли трекинг для подсчета IN/OUT
            generate_heatmap: генерировать ли тепловую карту
            frame_skip: пропускать N кадров между обработкой (1 = обрабатывать каждый второй кадр)
        """
        print(f"Загрузка модели из {model_path}...")
        self.model = YOLO(model_path)
        self.use_tracking = use_tracking
        self.generate_heatmap = generate_heatmap
        self.frame_skip = frame_skip
        
        # Загружаем конфигурацию линии, если указана
        self.line = None
        self.tracker = None
        if line_config_path and os.path.exists(line_config_path):
            line_config = load_config(line_config_path)
            if 'line' in line_config:
                line_data = line_config['line']
                # Преобразуем координаты в int для OpenCV
                point1 = (int(line_data['point1']['x']), int(line_data['point1']['y']))
                point2 = (int(line_data['point2']['x']), int(line_data['point2']['y']))
                self.line = (point1, point2)
                if use_tracking:
                    self.tracker = PersonTracker(self.line)
                print(f"Линия IN/OUT загружена: {point1} -> {point2}")
        
        # Статистика
        self.frame_stats = []
        self.current_count = 0
        
        # Коллектор метрик AI
        self.ai_metrics_collector = None
    
    def process_frame(self, frame: np.ndarray, frame_num: int) -> np.ndarray:
        """
        Обрабатывает один кадр
        
        Args:
            frame: кадр видео
            frame_num: номер кадра
        
        Returns:
            Обработанный кадр с визуализацией
        """
        # Детекция людей
        results = self.model(frame, verbose=False)
        
        # Извлекаем детекции
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Проверяем, что это класс "person" (обычно class 0 в нашей модели)
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Если модель обучена только на person, все детекции - люди
                # Или проверяем class_id == 0 (person в COCO)
                if cls == 0 or conf > 0.25:  # Порог уверенности
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': conf,
                        'class': cls
                    })
        
        # Собираем метрики AI, если коллектор инициализирован
        if self.ai_metrics_collector:
            frame_ai_metrics = self.ai_metrics_collector.add_frame_metrics(frame_num, detections)
        else:
            frame_ai_metrics = {}
        
        # Обновляем трекер, если используется
        if self.tracker:
            status = self.tracker.update(detections, frame_num)
            self.current_count = status['current_count']
        else:
            self.current_count = len(detections)
        
        # Сохраняем статистику кадра
        frame_stat = {
            'frame': frame_num,
            'count': self.current_count,
            'detections': len(detections)
        }
        if self.tracker:
            frame_stat['in_count'] = self.tracker.in_count
            frame_stat['out_count'] = self.tracker.out_count
        
        # Добавляем метрики AI в статистику кадра
        if frame_ai_metrics:
            frame_stat['ai_metrics'] = frame_ai_metrics
        
        self.frame_stats.append(frame_stat)
        
        # Визуализация
        annotated_frame = self._visualize_frame(frame, detections, frame_num)
        
        return annotated_frame
    
    def _visualize_frame(self, frame: np.ndarray, detections: list, frame_num: int) -> np.ndarray:
        """Визуализирует детекции на кадре"""
        annotated = frame.copy()
        
        # Рисуем линию IN/OUT, если есть
        if self.line:
            point1, point2 = self.line
            cv2.line(annotated, point1, point2, (0, 255, 0), 2)
            cv2.circle(annotated, point1, 5, (0, 255, 0), -1)
            cv2.circle(annotated, point2, 5, (0, 255, 0), -1)
            cv2.putText(annotated, "IN/OUT Line", (point1[0], point1[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Рисуем bounding boxes
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            
            # Цвет в зависимости от уверенности
            color = (0, 255, 0) if conf > 0.5 else (0, 165, 255)
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f'Person {conf:.2f}', (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Отображаем статистику
        stats_text = f"Frame: {frame_num} | People: {self.current_count}"
        if self.tracker:
            stats_text += f" | IN: {self.tracker.in_count} | OUT: {self.tracker.out_count}"
        
        cv2.putText(annotated, stats_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, stats_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        return annotated
    
    def process_video(self, video_path: str, output_path: str = None, 
                     save_heatmap: bool = True) -> dict:
        """
        Обрабатывает видео файл
        
        Args:
            video_path: путь к входному видео
            output_path: путь для сохранения обработанного видео
            save_heatmap: сохранять ли тепловую карту
        
        Returns:
            Словарь со статистикой
        """
        print(f"Обработка видео: {video_path}")
        
        # Открываем видео
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
        
        # Получаем информацию о видео
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Разрешение: {width}x{height}, FPS: {fps}, Кадров: {total_frames}")
        if self.frame_skip > 0:
            print(f"⚡ Режим оптимизации: обрабатывается каждый {self.frame_skip + 1}-й кадр")
        
        # Создаем writer для выходного видео
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            ensure_dir(os.path.dirname(output_path))
        
        # Инициализируем генератор тепловой карты
        heatmap_gen = None
        if self.generate_heatmap:
            heatmap_gen = HeatmapGenerator((height, width))
        
        # Сбрасываем статистику
        self.frame_stats = []
        if self.tracker:
            self.tracker.reset()
        
        # Инициализируем коллектор метрик AI (task_id будет установлен позже)
        self.ai_metrics_collector = AIMetricsCollector(task_id=None)
        
        # Обрабатываем кадры
        frame_num = 0
        processed_frame_num = 0  # Счетчик обработанных кадров
        last_processed_frame = None  # Последний обработанный кадр для дублирования
        
        with tqdm(total=total_frames, desc="Обработка видео") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Обрабатываем только каждый (frame_skip+1)-й кадр
                if frame_num % (self.frame_skip + 1) == 0:
                    # Обрабатываем кадр
                    annotated_frame = self.process_frame(frame, frame_num)
                    last_processed_frame = annotated_frame.copy()
                    processed_frame_num += 1
                    
                    # Добавляем точки в тепловую карту
                    if heatmap_gen:
                        if self.tracker:
                            # Если есть трекер, используем центры треков
                            for track_id, track_data in self.tracker.tracks.items():
                                if track_data['centers']:
                                    center = track_data['centers'][-1]
                                    heatmap_gen.add_point(center)
                        else:
                            # Если нет трекера, используем центры детекций из обработанного кадра
                            # Используем детекции из process_frame (уже обработаны)
                            # Получаем последние детекции из статистики кадра
                            if self.frame_stats:
                                last_stat = self.frame_stats[-1]
                                # Нужно получить детекции для этого кадра
                                # Выполняем детекцию еще раз или используем сохраненные данные
                                results = self.model(frame, verbose=False)
                                for result in results:
                                    boxes = result.boxes
                                    for box in boxes:
                                        cls = int(box.cls[0])
                                        conf = float(box.conf[0])
                                        if cls == 0 or conf > 0.25:
                                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                            # Вычисляем центр bounding box используя функцию из utils
                                            center = get_bbox_center([float(x1), float(y1), float(x2), float(y2)])
                                            heatmap_gen.add_point(center)
                else:
                    # Для пропущенных кадров используем последний обработанный кадр
                    if last_processed_frame is not None:
                        annotated_frame = last_processed_frame.copy()
                    else:
                        # Если еще не было обработанных кадров, просто копируем оригинал
                        annotated_frame = frame.copy()
                
                # Сохраняем кадр (всегда записываем для сохранения правильного FPS)
                if output_path:
                    out.write(annotated_frame)
                
                frame_num += 1
                pbar.update(1)
        
        cap.release()
        if output_path:
            out.release()
            print(f"Обработанное видео сохранено: {output_path}")
        
        # Генерируем и сохраняем тепловую карту
        if heatmap_gen and save_heatmap:
            heatmap_dir = os.path.join(os.path.dirname(output_path or video_path), 'heatmaps')
            ensure_dir(heatmap_dir)
            heatmap_path = os.path.join(heatmap_dir, 
                                       f"{Path(video_path).stem}_heatmap.png")
            heatmap_gen.save(heatmap_path, format='matplotlib')
            print(f"Тепловая карта сохранена: {heatmap_path}")
        
        # Формируем статистику
        stats = {
            'video_path': video_path,
            'total_frames': frame_num,
            'processed_frames': processed_frame_num,
            'frame_skip': self.frame_skip,
            'fps': fps,
            'resolution': {'width': width, 'height': height},
            'frame_statistics': self.frame_stats,
            'summary': {
                'max_count': max(s['count'] for s in self.frame_stats) if self.frame_stats else 0,
                'min_count': min(s['count'] for s in self.frame_stats) if self.frame_stats else 0,
                'avg_count': np.mean([s['count'] for s in self.frame_stats]) if self.frame_stats else 0,
            }
        }
        
        if self.tracker:
            tracker_stats = self.tracker.get_statistics()
            stats['tracking'] = {
                'total_in': tracker_stats['total_in'],
                'total_out': tracker_stats['total_out'],
                'current_inside': tracker_stats['current_inside']
            }
        
        # Получаем сводную статистику метрик AI
        if self.ai_metrics_collector:
            ai_summary = self.ai_metrics_collector.get_summary_statistics()
            stats['ai_metrics_summary'] = ai_summary
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='Обработка видео и подсчет посетителей')
    parser.add_argument('--video', type=str, required=True, help='Путь к видео файлу')
    parser.add_argument('--model', type=str, required=True, help='Путь к обученной модели YOLOv8')
    parser.add_argument('--line_config', type=str, default=None,
                       help='Путь к конфигурации линии IN/OUT')
    parser.add_argument('--output', type=str, default=None,
                       help='Путь для сохранения обработанного видео')
    parser.add_argument('--output_dir', type=str, default='data/results',
                       help='Директория для сохранения результатов')
    parser.add_argument('--no-tracking', action='store_true',
                       help='Отключить трекинг и подсчет IN/OUT')
    parser.add_argument('--no-heatmap', action='store_true',
                       help='Отключить генерацию тепловой карты')
    parser.add_argument('--frame-skip', type=int, default=1,
                       help='Пропускать N кадров между обработкой (1 = обрабатывать каждый второй кадр, по умолчанию: 1)')
    
    args = parser.parse_args()
    
    # Определяем путь для выходного видео
    if args.output:
        output_video = args.output
    else:
        ensure_dir(args.output_dir)
        video_name = Path(args.video).stem
        output_video = os.path.join(args.output_dir, f"{video_name}_processed.mp4")
    
    # Создаем процессор
    processor = VideoProcessor(
        model_path=args.model,
        line_config_path=args.line_config,
        use_tracking=not args.no_tracking,
        generate_heatmap=not args.no_heatmap,
        frame_skip=args.frame_skip
    )
    
    # Обрабатываем видео
    stats = processor.process_video(args.video, output_video, save_heatmap=not args.no_heatmap)
    
    # Сохраняем статистику
    stats_path = os.path.join(args.output_dir, f"{Path(args.video).stem}_stats.json")
    save_statistics(stats, stats_path)
    print(f"Статистика сохранена: {stats_path}")
    
    # Выводим краткую статистику
    print("\n" + "=" * 60)
    print("Статистика обработки:")
    print("=" * 60)
    print(f"Всего кадров: {stats['total_frames']}")
    print(f"Обработано кадров: {stats['processed_frames']} (пропуск: {stats['frame_skip']})")
    print(f"Максимум людей на кадре: {stats['summary']['max_count']}")
    print(f"Минимум людей на кадре: {stats['summary']['min_count']}")
    print(f"Среднее количество людей: {stats['summary']['avg_count']:.2f}")
    if 'tracking' in stats:
        print(f"Вошло людей: {stats['tracking']['total_in']}")
        print(f"Вышло людей: {stats['tracking']['total_out']}")
        print(f"Текущее количество внутри: {stats['tracking']['current_inside']}")
    print("=" * 60)


if __name__ == '__main__':
    main()

