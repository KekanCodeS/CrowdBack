"""
Pydantic модели для API запросов и ответов
"""
from typing import Optional, Literal, List, Any
from pydantic import BaseModel
from datetime import datetime


class TaskStatus(str):
    """Статусы задачи обработки"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class UploadResponse(BaseModel):
    """Ответ на загрузку видео"""
    task_id: str
    message: str
    filename: str


class TaskStatusResponse(BaseModel):
    """Ответ со статусом задачи"""
    task_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    progress: Optional[float] = None  # Процент выполнения (0-100)
    message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class ErrorResponse(BaseModel):
    """Ответ об ошибке"""
    error: str
    detail: Optional[str] = None


class ResultsResponse(BaseModel):
    """Ответ с результатами обработки"""
    task_id: str
    filename: str
    video_info: dict
    statistics: List[dict]  # Список статистики по кадрам
    tracking: Optional[dict] = None
    summary: dict
    created_at: datetime
    processing_time: Optional[float] = None  # Время обработки в секундах


class VideoInfo(BaseModel):
    """Информация о видео"""
    fps: float
    width: int
    height: int
    frame_count: int
    duration: float


class TrackingStats(BaseModel):
    """Статистика трекинга"""
    total_in: int
    total_out: int
    current_inside: int


class SummaryStats(BaseModel):
    """Сводная статистика"""
    max_count: int
    min_count: int
    avg_count: float


class Point(BaseModel):
    """Точка координат"""
    x: int
    y: int


class LineConfig(BaseModel):
    """Конфигурация линии IN/OUT"""
    point1: Point
    point2: Point


class PreviewResponse(BaseModel):
    """Ответ с превью первого кадра"""
    preview_image: str  # base64 encoded image
    width: int
    height: int
    filename: str

