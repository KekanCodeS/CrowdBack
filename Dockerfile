# Dockerfile для бэкенда CrowdCounting
FROM python:3.10-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt .
# Устанавливаем PyTorch с CUDA поддержкой
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Устанавливаем остальные зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY src/ ./src/
COPY run_api.py .

# Создаем директории для конфигурации и моделей
RUN mkdir -p configs runs/train/coco_person_yolov8/weights

# Копируем конфигурацию (если существует)
COPY configs/ ./configs/

# Примечание: модели будут монтироваться через volumes в docker-compose.yml
# Это позволяет использовать модели без их включения в образ Docker

# Создаем директории для данных
RUN mkdir -p data/uploads data/results models

# Открываем порт
EXPOSE 8000

# Команда запуска
CMD ["python", "run_api.py"]
