"""
Скрипт для запуска FastAPI сервера
"""
import uvicorn
from src.api.config import API_HOST, API_PORT

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,  # Автоперезагрузка при изменении кода
        log_level="info"
    )

