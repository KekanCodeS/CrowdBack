# Docker Setup для People Counter

## Быстрый старт

### Запуск всех сервисов через docker-compose

```bash
docker-compose up -d
```

После запуска:
- Бэкенд API: http://localhost:8000
- Фронтенд: http://localhost:3000
- API документация: http://localhost:8000/docs

### Остановка

```bash
docker-compose down
```

## Отдельный запуск контейнеров

### Бэкенд

```bash
cd CrowdCounting
docker build -t crowdcounting-backend .
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/runs:/app/runs \
  crowdcounting-backend
```

### Фронтенд

```bash
cd people-counter-frontend-main
docker build -t crowdcounting-frontend .
docker run -p 3000:80 crowdcounting-frontend
```

## Переменные окружения

### Бэкенд

- `MODEL_PATH` - путь к модели YOLO (по умолчанию: обученная модель best.pt)
- `API_HOST` - хост для API (по умолчанию: 0.0.0.0)
- `API_PORT` - порт для API (по умолчанию: 8000)
- `USE_TRACKING` - использовать трекинг (по умолчанию: true)
- `GENERATE_HEATMAP` - генерировать тепловую карту (по умолчанию: true)
- `FRAME_SKIP` - пропуск кадров (по умолчанию: 1)
- `MAX_UPLOAD_SIZE` - максимальный размер загружаемого файла в байтах (по умолчанию: 500MB)

### Фронтенд

- `VITE_API_URL` - URL бэкенда API (по умолчанию: http://localhost:8000)

## Volumes

Данные сохраняются в локальных директориях через volumes:
- `./CrowdCounting/data` - загруженные видео и результаты
- `./CrowdCounting/models` - модели
- `./CrowdCounting/runs` - результаты обучения

## Примечания

1. **Модели**: Модели монтируются через volumes из локальной файловой системы. Убедитесь, что модель `best.pt` существует в `CrowdCounting/runs/train/coco_person_yolov8/weights/` или используйте переменную окружения `MODEL_PATH` для указания другого пути.

2. **GPU поддержка**: Для использования GPU добавьте в `docker-compose.yml`:
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: 1
             capabilities: [gpu]
   ```
   И используйте образ с CUDA поддержкой PyTorch (уже включено в Dockerfile).

3. **Первая сборка**: При первом запуске сборка может занять несколько минут из-за установки PyTorch и других зависимостей.

4. **Данные**: Все данные (загруженные видео, результаты обработки) сохраняются в локальных директориях через volumes и не теряются при перезапуске контейнеров.

