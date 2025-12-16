"""
Коллектор метрик AI для системы подсчета посетителей
"""
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np

# Добавляем путь к модулям для импорта
sys.path.insert(0, str(Path(__file__).parent))
from api.config import BASE_DIR

METRICS_DIR = BASE_DIR / "data" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

AI_METRICS_FILE = METRICS_DIR / "ai_metrics.jsonl"


class AIMetricsCollector:
    """Класс для сбора и агрегации метрик AI"""
    
    def __init__(self, task_id: Optional[str] = None):
        """
        Args:
            task_id: Идентификатор задачи обработки видео
        """
        self.task_id = task_id
        self.confidence_scores: List[float] = []
        self.detections_per_frame: List[int] = []
        self.frame_numbers: List[int] = []
    
    def add_frame_metrics(self, frame_num: int, detections: List[Dict]) -> Dict:
        """
        Добавляет метрики для одного кадра
        
        Args:
            frame_num: Номер кадра
            detections: Список детекций с полями 'confidence', 'bbox', 'class'
        
        Returns:
            Словарь с метриками кадра
        """
        if not detections:
            frame_metrics = {
                "frame": frame_num,
                "detections_count": 0,
                "avg_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "high_confidence_count": 0,  # > 0.7
                "medium_confidence_count": 0,  # 0.5-0.7
                "low_confidence_count": 0  # < 0.5
            }
        else:
            confidences = [det['confidence'] for det in detections]
            self.confidence_scores.extend(confidences)
            self.detections_per_frame.append(len(detections))
            self.frame_numbers.append(frame_num)
            
            frame_metrics = {
                "frame": frame_num,
                "detections_count": len(detections),
                "avg_confidence": float(np.mean(confidences)),
                "min_confidence": float(np.min(confidences)),
                "max_confidence": float(np.max(confidences)),
                "high_confidence_count": sum(1 for c in confidences if c > 0.7),
                "medium_confidence_count": sum(1 for c in confidences if 0.5 <= c <= 0.7),
                "low_confidence_count": sum(1 for c in confidences if c < 0.5)
            }
        
        # Сохраняем метрику кадра в файл
        self._save_frame_metric(frame_metrics)
        
        return frame_metrics
    
    def _save_frame_metric(self, frame_metrics: Dict):
        """Сохраняет метрику кадра в файл"""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "task_id": self.task_id,
            **frame_metrics
        }
        
        try:
            with open(AI_METRICS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(metric, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Ошибка при сохранении метрики AI: {e}")
    
    def get_summary_statistics(self) -> Dict:
        """
        Возвращает сводную статистику по всем собранным метрикам
        
        Returns:
            Словарь со статистикой
        """
        if not self.confidence_scores:
            return {
                "total_frames": len(self.frame_numbers),
                "total_detections": 0,
                "avg_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "avg_detections_per_frame": 0.0,
                "high_confidence_ratio": 0.0,
                "medium_confidence_ratio": 0.0,
                "low_confidence_ratio": 0.0
            }
        
        confidences_array = np.array(self.confidence_scores)
        detections_array = np.array(self.detections_per_frame)
        
        return {
            "total_frames": len(self.frame_numbers),
            "total_detections": len(self.confidence_scores),
            "avg_confidence": float(np.mean(confidences_array)),
            "min_confidence": float(np.min(confidences_array)),
            "max_confidence": float(np.max(confidences_array)),
            "std_confidence": float(np.std(confidences_array)),
            "avg_detections_per_frame": float(np.mean(detections_array)),
            "max_detections_per_frame": int(np.max(detections_array)),
            "min_detections_per_frame": int(np.min(detections_array)),
            "high_confidence_ratio": float(np.sum(confidences_array > 0.7) / len(confidences_array)),
            "medium_confidence_ratio": float(np.sum((confidences_array >= 0.5) & (confidences_array <= 0.7)) / len(confidences_array)),
            "low_confidence_ratio": float(np.sum(confidences_array < 0.5) / len(confidences_array))
        }
    
    def reset(self):
        """Сбрасывает собранные метрики"""
        self.confidence_scores.clear()
        self.detections_per_frame.clear()
        self.frame_numbers.clear()

