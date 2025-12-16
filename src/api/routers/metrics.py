"""
Роутер для работы с метриками
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..config import BASE_DIR

# Добавляем путь для импорта metrics_visualizer
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from metrics_visualizer import (
    plot_api_response_times,
    plot_api_response_time_distribution,
    plot_ai_confidence_scores,
    plot_ui_load_times,
    plot_load_test_results
)

METRICS_DIR = BASE_DIR / "data" / "metrics"
API_METRICS_FILE = METRICS_DIR / "api_metrics.jsonl"
AI_METRICS_FILE = METRICS_DIR / "ai_metrics.jsonl"
UI_METRICS_FILE = METRICS_DIR / "ui_metrics.jsonl"

router = APIRouter(prefix="/metrics", tags=["metrics"])


class UIMetric(BaseModel):
    """Модель для метрик UI от фронтенда"""
    component: str
    load_time_ms: float
    render_time_ms: Optional[float] = None
    event_type: Optional[str] = "load"  # load, render, interaction


def _read_jsonl_file(file_path: Path, start_time: Optional[datetime] = None, 
                     end_time: Optional[datetime] = None) -> List[Dict]:
    """Читает JSONL файл и фильтрует по времени"""
    if not file_path.exists():
        return []
    
    metrics = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                metric = json.loads(line)
                metric_time = datetime.fromisoformat(metric.get("timestamp", ""))
                
                if start_time and metric_time < start_time:
                    continue
                if end_time and metric_time > end_time:
                    continue
                
                metrics.append(metric)
            except (json.JSONDecodeError, ValueError) as e:
                continue
    
    return metrics


@router.get("/api")
async def get_api_metrics(
    start_time: Optional[str] = Query(None, description="Начало периода (ISO format)"),
    end_time: Optional[str] = Query(None, description="Конец периода (ISO format)"),
    endpoint: Optional[str] = Query(None, description="Фильтр по endpoint")
):
    """
    Получает метрики API за указанный период
    
    Args:
        start_time: Начало периода в формате ISO
        end_time: Конец периода в формате ISO
        endpoint: Фильтр по endpoint (опционально)
    """
    start_dt = datetime.fromisoformat(start_time) if start_time else None
    end_dt = datetime.fromisoformat(end_time) if end_time else None
    
    metrics = _read_jsonl_file(API_METRICS_FILE, start_dt, end_dt)
    
    if endpoint:
        metrics = [m for m in metrics if endpoint in m.get("endpoint", "")]
    
    # Вычисляем статистику
    if not metrics:
        return {
            "metrics": [],
            "summary": {
                "total_requests": 0,
                "avg_response_time_ms": 0.0,
                "min_response_time_ms": 0.0,
                "max_response_time_ms": 0.0,
                "status_codes": {}
            }
        }
    
    response_times = [m.get("response_time_ms", 0) for m in metrics]
    status_codes = {}
    for m in metrics:
        code = m.get("status_code", 0)
        status_codes[code] = status_codes.get(code, 0) + 1
    
    return {
        "metrics": metrics[-1000:],  # Последние 1000 записей
        "summary": {
            "total_requests": len(metrics),
            "avg_response_time_ms": round(sum(response_times) / len(response_times), 2),
            "min_response_time_ms": round(min(response_times), 2),
            "max_response_time_ms": round(max(response_times), 2),
            "status_codes": status_codes
        }
    }


@router.get("/ai")
async def get_ai_metrics(
    start_time: Optional[str] = Query(None, description="Начало периода (ISO format)"),
    end_time: Optional[str] = Query(None, description="Конец периода (ISO format)"),
    task_id: Optional[str] = Query(None, description="Фильтр по task_id")
):
    """
    Получает метрики AI за указанный период
    
    Args:
        start_time: Начало периода в формате ISO
        end_time: Конец периода в формате ISO
        task_id: Фильтр по task_id (опционально)
    """
    start_dt = datetime.fromisoformat(start_time) if start_time else None
    end_dt = datetime.fromisoformat(end_time) if end_time else None
    
    metrics = _read_jsonl_file(AI_METRICS_FILE, start_dt, end_dt)
    
    if task_id:
        metrics = [m for m in metrics if m.get("task_id") == task_id]
    
    if not metrics:
        return {
            "metrics": [],
            "summary": {
                "total_frames": 0,
                "total_detections": 0,
                "avg_confidence": 0.0,
                "avg_detections_per_frame": 0.0
            }
        }
    
    # Вычисляем статистику
    confidences = [m.get("avg_confidence", 0) for m in metrics if m.get("avg_confidence", 0) > 0]
    detections_counts = [m.get("detections_count", 0) for m in metrics]
    
    summary = {
        "total_frames": len(metrics),
        "total_detections": sum(detections_counts),
        "avg_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        "avg_detections_per_frame": round(sum(detections_counts) / len(detections_counts), 2) if detections_counts else 0.0,
        "max_detections_per_frame": max(detections_counts) if detections_counts else 0,
        "min_detections_per_frame": min(detections_counts) if detections_counts else 0
    }
    
    return {
        "metrics": metrics[-5000:],  # Последние 5000 записей
        "summary": summary
    }


@router.post("/ui")
async def save_ui_metrics(metric: UIMetric):
    """
    Сохраняет метрики UI от фронтенда
    
    Args:
        metric: Метрика UI
    """
    metric_data = {
        "timestamp": datetime.now().isoformat(),
        "component": metric.component,
        "load_time_ms": metric.load_time_ms,
        "render_time_ms": metric.render_time_ms,
        "event_type": metric.event_type
    }
    
    try:
        with open(UI_METRICS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(metric_data, ensure_ascii=False) + "\n")
        return {"status": "ok", "message": "Метрика сохранена"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении метрики: {str(e)}")


@router.get("/ui")
async def get_ui_metrics(
    start_time: Optional[str] = Query(None, description="Начало периода (ISO format)"),
    end_time: Optional[str] = Query(None, description="Конец периода (ISO format)"),
    component: Optional[str] = Query(None, description="Фильтр по компоненту")
):
    """
    Получает метрики UI за указанный период
    """
    start_dt = datetime.fromisoformat(start_time) if start_time else None
    end_dt = datetime.fromisoformat(end_time) if end_time else None
    
    metrics = _read_jsonl_file(UI_METRICS_FILE, start_dt, end_dt)
    
    if component:
        metrics = [m for m in metrics if m.get("component") == component]
    
    if not metrics:
        return {
            "metrics": [],
            "summary": {
                "total_events": 0,
                "avg_load_time_ms": 0.0,
                "avg_render_time_ms": 0.0
            }
        }
    
    load_times = [m.get("load_time_ms", 0) for m in metrics if m.get("load_time_ms")]
    render_times = [m.get("render_time_ms", 0) for m in metrics if m.get("render_time_ms")]
    
    return {
        "metrics": metrics[-1000:],  # Последние 1000 записей
        "summary": {
            "total_events": len(metrics),
            "avg_load_time_ms": round(sum(load_times) / len(load_times), 2) if load_times else 0.0,
            "avg_render_time_ms": round(sum(render_times) / len(render_times), 2) if render_times else 0.0
        }
    }


@router.get("/charts/{metric_type}")
async def get_metrics_chart(
    metric_type: str,
    hours: int = Query(24, description="Количество часов для анализа")
):
    """
    Генерирует и возвращает график метрик
    
    Args:
        metric_type: Тип метрики (api_response_times, api_distribution, ai_confidence, ui_load_times)
        hours: Количество часов для анализа
    """
    from fastapi.responses import FileResponse
    
    chart_path = None
    
    if metric_type == "api_response_times":
        chart_path = plot_api_response_times(hours=hours)
    elif metric_type == "api_distribution":
        chart_path = plot_api_response_time_distribution(hours=hours)
    elif metric_type == "ai_confidence":
        chart_path = plot_ai_confidence_scores(hours=hours)
    elif metric_type == "ui_load_times":
        chart_path = plot_ui_load_times(hours=hours)
    else:
        raise HTTPException(status_code=400, detail=f"Неизвестный тип метрики: {metric_type}")
    
    if chart_path and chart_path.exists():
        return FileResponse(
            chart_path,
            media_type="image/png",
            filename=chart_path.name
        )
    else:
        raise HTTPException(status_code=500, detail="Не удалось создать график")


@router.get("/summary")
async def get_metrics_summary(
    hours: int = Query(24, description="Количество часов для анализа")
):
    """
    Получает сводную статистику по всем метрикам за указанный период
    
    Args:
        hours: Количество часов для анализа (по умолчанию 24)
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    
    # Получаем метрики API
    api_metrics = _read_jsonl_file(API_METRICS_FILE, start_time, end_time)
    api_response_times = [m.get("response_time_ms", 0) for m in api_metrics]
    
    # Получаем метрики AI
    ai_metrics = _read_jsonl_file(AI_METRICS_FILE, start_time, end_time)
    ai_confidences = [m.get("avg_confidence", 0) for m in ai_metrics if m.get("avg_confidence", 0) > 0]
    
    # Получаем метрики UI
    ui_metrics = _read_jsonl_file(UI_METRICS_FILE, start_time, end_time)
    ui_load_times = [m.get("load_time_ms", 0) for m in ui_metrics if m.get("load_time_ms")]
    
    return {
        "period": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "hours": hours
        },
        "api": {
            "total_requests": len(api_metrics),
            "avg_response_time_ms": round(sum(api_response_times) / len(api_response_times), 2) if api_response_times else 0.0,
            "min_response_time_ms": round(min(api_response_times), 2) if api_response_times else 0.0,
            "max_response_time_ms": round(max(api_response_times), 2) if api_response_times else 0.0
        },
        "ai": {
            "total_frames": len(ai_metrics),
            "total_detections": sum(m.get("detections_count", 0) for m in ai_metrics),
            "avg_confidence": round(sum(ai_confidences) / len(ai_confidences), 4) if ai_confidences else 0.0
        },
        "ui": {
            "total_events": len(ui_metrics),
            "avg_load_time_ms": round(sum(ui_load_times) / len(ui_load_times), 2) if ui_load_times else 0.0
        }
    }

