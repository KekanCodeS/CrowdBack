"""
Конфигурация для FastAPI приложения
"""
import os
from pathlib import Path

# Базовый путь проекта
BASE_DIR = Path(__file__).parent.parent.parent

# Пути для хранения файлов
UPLOADS_DIR = BASE_DIR / "data" / "uploads"
RESULTS_DIR = BASE_DIR / "data" / "results"
MODELS_DIR = BASE_DIR / "models"

# Создаем директории, если их нет
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Путь к модели YOLO
# По умолчанию используется обученная модель best.pt
# Также доступна предобученная модель: yolo11n.pt
# Можно переопределить через переменную окружения MODEL_PATH
DEFAULT_MODEL_PATH = BASE_DIR / "runs" / "train" / "coco_person_yolov8" / "weights" / "best.pt"
# Альтернативно можно использовать предобученную модель:
# DEFAULT_MODEL_PATH = BASE_DIR / "yolo11n.pt"
MODEL_PATH = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
# Путь к конфигурации линии IN/OUT (опционально)
LINE_CONFIG_PATH = os.getenv("LINE_CONFIG_PATH", str(BASE_DIR / "configs" / "line_config.yaml"))

# Настройки API
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", 500 * 1024 * 1024))  # 500MB по умолчанию
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Настройки обработки видео
USE_TRACKING = os.getenv("USE_TRACKING", "true").lower() == "true"
GENERATE_HEATMAP = os.getenv("GENERATE_HEATMAP", "true").lower() == "true"
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "1"))  # Пропуск кадров для оптимизации

# Настройки сервера
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

