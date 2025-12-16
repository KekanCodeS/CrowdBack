"""
FastAPI приложение для системы подсчета посетителей
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import upload, tasks, results, metrics
from .middleware.metrics import MetricsMiddleware

app = FastAPI(
    title="People Counter API",
    description="API для обработки видео и подсчета посетителей",
    version="1.0.0"
)

# Middleware для сбора метрик API (должен быть первым)
app.add_middleware(MetricsMiddleware)

# Настройка CORS для работы с фронтендом
# Разрешаем запросы из различных источников (локальная разработка и Docker)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://frontend:80",  # Docker контейнер фронтенда
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем роутеры
app.include_router(upload.router)
app.include_router(tasks.router)
app.include_router(results.router)
app.include_router(metrics.router)


@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "People Counter API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {"status": "ok"}

