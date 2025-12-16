"""
Middleware для сбора метрик API
"""
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ..config import BASE_DIR

METRICS_DIR = BASE_DIR / "data" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

API_METRICS_FILE = METRICS_DIR / "api_metrics.jsonl"


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware для сбора метрик времени отклика API"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Игнорируем метрики и health check endpoints
        if request.url.path.startswith("/metrics") or request.url.path == "/health":
            return await call_next(request)
        
        # Засекаем время начала запроса
        start_time = time.time()
        
        # Выполняем запрос
        response = await call_next(request)
        
        # Вычисляем время ответа
        response_time_ms = (time.time() - start_time) * 1000
        
        # Собираем метрики
        metric = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": request.url.path,
            "method": request.method,
            "response_time_ms": round(response_time_ms, 2),
            "status_code": response.status_code,
            "query_params": str(request.url.query) if request.url.query else None
        }
        
        # Сохраняем метрику в файл (JSON Lines формат)
        try:
            with open(API_METRICS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(metric, ensure_ascii=False) + "\n")
        except Exception as e:
            # Логируем ошибку, но не прерываем запрос
            print(f"Ошибка при сохранении метрики: {e}")
        
        return response

