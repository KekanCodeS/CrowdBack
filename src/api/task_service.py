"""
Сервис для управления задачами обработки видео
"""
import os
import uuid
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor

import sys
from pathlib import Path as PathLib
# Добавляем путь к родительской директории для импорта модулей из src
sys.path.insert(0, str(PathLib(__file__).parent.parent))

from inference import VideoProcessor
from .config import (
    UPLOADS_DIR, RESULTS_DIR, MODEL_PATH, LINE_CONFIG_PATH,
    USE_TRACKING, GENERATE_HEATMAP, FRAME_SKIP
)


class TaskService:
    """Сервис для управления задачами обработки видео"""
    
    def __init__(self):
        self.tasks: Dict[str, dict] = {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2)  # Максимум 2 задачи одновременно
    
    def create_task(self, filename: str, video_path: str, 
                   use_line: bool = False, line_config: dict = None) -> str:
        """
        Создает новую задачу обработки видео
        
        Args:
            filename: Имя файла
            video_path: Путь к загруженному видео
            use_line: Использовать ли линию IN/OUT
            line_config: Конфигурация линии (dict с point1 и point2)
            
        Returns:
            task_id: Уникальный идентификатор задачи
        """
        task_id = str(uuid.uuid4())
        now = datetime.now()
        
        task = {
            "task_id": task_id,
            "filename": filename,
            "video_path": video_path,
            "status": "pending",
            "progress": 0.0,
            "message": None,
            "created_at": now,
            "updated_at": now,
            "results": None,
            "error": None,
            "output_video_path": None,
            "stats_path": None,
            "heatmap_path": None,
            "use_line": use_line,
            "line_config": line_config,
        }
        
        with self.lock:
            self.tasks[task_id] = task
        
        # Запускаем обработку в фоне
        self.executor.submit(self._process_video, task_id)
        
        return task_id
    
    def get_task(self, task_id: str) -> Optional[dict]:
        """Получает информацию о задаче"""
        with self.lock:
            return self.tasks.get(task_id)
    
    def update_task_status(self, task_id: str, status: str, progress: float = None, 
                          message: str = None, error: str = None):
        """Обновляет статус задачи"""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id]["status"] = status
                self.tasks[task_id]["updated_at"] = datetime.now()
                if progress is not None:
                    self.tasks[task_id]["progress"] = progress
                if message is not None:
                    self.tasks[task_id]["message"] = message
                if error is not None:
                    self.tasks[task_id]["error"] = error
    
    def _process_video(self, task_id: str):
        """Обрабатывает видео в фоновом режиме"""
        task = self.get_task(task_id)
        if not task:
            return
        
        try:
            # Обновляем статус на "processing"
            self.update_task_status(task_id, "processing", progress=0.0, 
                                   message="Начало обработки видео...")
            
            video_path = task["video_path"]
            video_name = Path(video_path).stem
            
            # Определяем пути для результатов
            output_dir = RESULTS_DIR / task_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_video_path = output_dir / f"{video_name}_processed.mp4"
            stats_path = output_dir / f"{video_name}_stats.json"
            heatmap_dir = output_dir / "heatmaps"
            heatmap_path = heatmap_dir / f"{video_name}_heatmap.png"
            
            # Определяем конфигурацию линии
            line_config_path = None
            use_tracking = task.get("use_line", False) and USE_TRACKING
            
            if task.get("use_line") and task.get("line_config"):
                # Создаем временный конфигурационный файл для линии
                line_config_data = {
                    'line': task["line_config"],
                    'video_path': str(video_path)
                }
                line_config_path = output_dir / "line_config.yaml"
                from utils import save_config
                save_config(line_config_data, str(line_config_path))
            elif os.path.exists(LINE_CONFIG_PATH):
                line_config_path = LINE_CONFIG_PATH
            
            # Создаем процессор видео
            processor = VideoProcessor(
                model_path=MODEL_PATH,
                line_config_path=str(line_config_path) if line_config_path else None,
                use_tracking=use_tracking,
                generate_heatmap=GENERATE_HEATMAP,
                frame_skip=FRAME_SKIP
            )
            
            # Устанавливаем task_id для коллектора метрик AI
            if processor.ai_metrics_collector:
                processor.ai_metrics_collector.task_id = task_id
            
            # Обрабатываем видео
            # Примечание: VideoProcessor не поддерживает callback для прогресса напрямую,
            # поэтому мы будем обновлять прогресс приблизительно на основе времени
            self.update_task_status(task_id, "processing", progress=10.0, 
                                   message="Загрузка модели и инициализация...")
            
            import time
            start_time = time.time()
            
            stats = processor.process_video(
                video_path=str(video_path),
                output_path=str(output_video_path),
                save_heatmap=GENERATE_HEATMAP
            )
            
            processing_time = time.time() - start_time
            
            # Сохраняем статистику
            import json
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
            
            # Обновляем задачу с результатами
            with self.lock:
                if task_id in self.tasks:
                    self.tasks[task_id]["status"] = "completed"
                    self.tasks[task_id]["progress"] = 100.0
                    self.tasks[task_id]["message"] = "Обработка завершена успешно"
                    self.tasks[task_id]["updated_at"] = datetime.now()
                    self.tasks[task_id]["results"] = stats
                    self.tasks[task_id]["output_video_path"] = str(output_video_path)
                    self.tasks[task_id]["stats_path"] = str(stats_path)
                    self.tasks[task_id]["heatmap_path"] = str(heatmap_path) if GENERATE_HEATMAP and heatmap_path.exists() else None
                    self.tasks[task_id]["processing_time"] = processing_time
        
        except Exception as e:
            # Обрабатываем ошибки
            error_msg = str(e)
            self.update_task_status(
                task_id, 
                "failed", 
                progress=0.0,
                error=error_msg,
                message=f"Ошибка обработки: {error_msg}"
            )
    
    def cancel_task(self, task_id: str) -> bool:
        """Отменяет задачу (если она еще не началась)"""
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task["status"] in ["pending", "processing"]:
                    task["status"] = "failed"
                    task["message"] = "Задача отменена"
                    task["updated_at"] = datetime.now()
                    return True
        return False


# Глобальный экземпляр сервиса
task_service = TaskService()

