"""
Роутер для загрузки видео файлов
"""
import os
import base64
import cv2
import tempfile
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import json

from ..config import UPLOADS_DIR, MAX_UPLOAD_SIZE, ALLOWED_VIDEO_EXTENSIONS
from ..task_service import task_service
from ..models import UploadResponse, ErrorResponse, PreviewResponse, LineConfig

router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("/preview", response_model=PreviewResponse)
async def get_video_preview(file: UploadFile = File(...)):
    """
    Получает первый кадр видео для превью
    
    Args:
        file: Видео файл
        
    Returns:
        PreviewResponse с base64 изображением первого кадра
    """
    # Проверяем расширение файла
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат файла. Разрешенные форматы: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
        )
    
    # Сохраняем файл во временную директорию
    temp_file = None
    try:
        # Создаем временный файл
        import uuid
        temp_filename = f"{uuid.uuid4()}{file_ext}"
        temp_file = os.path.join(tempfile.gettempdir(), temp_filename)
        
        # Сохраняем загруженный файл
        file_size = 0
        with open(temp_file, "wb") as f:
            while True:
                chunk = await file.read(8192)
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > MAX_UPLOAD_SIZE:
                    os.remove(temp_file)
                    raise HTTPException(
                        status_code=413,
                        detail=f"Файл слишком большой. Максимальный размер: {MAX_UPLOAD_SIZE / (1024*1024):.0f}MB"
                    )
                f.write(chunk)
        
        # Открываем видео и получаем первый кадр
        cap = cv2.VideoCapture(temp_file)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Не удалось открыть видео файл")
        
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise HTTPException(status_code=400, detail="Не удалось прочитать кадр из видео")
        
        height, width = frame.shape[:2]
        cap.release()
        
        # Конвертируем кадр в base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return PreviewResponse(
            preview_image=f"data:image/jpeg;base64,{image_base64}",
            width=int(width),
            height=int(height),
            filename=file.filename
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обработке видео: {str(e)}"
        )
    finally:
        # Удаляем временный файл
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass


@router.post("", response_model=UploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    use_line: str = Form("false"),
    line_config: str = Form(None)
):
    """
    Загружает видео файл и создает задачу обработки
    
    Args:
        file: Видео файл для загрузки
        use_line: Использовать ли линию IN/OUT для подсчета (строка "true"/"false")
        line_config: Конфигурация линии в формате JSON (обязательна если use_line=True)
        
    Returns:
        UploadResponse с task_id
    """
    # Преобразуем строку в bool
    use_line_bool = use_line.lower() == "true"
    
    # Парсим line_config если он передан
    line_config_dict = None
    if use_line_bool:
        if not line_config:
            raise HTTPException(
                status_code=400,
                detail="line_config обязателен когда use_line=true"
            )
        try:
            line_config_dict = json.loads(line_config)
            # Валидируем структуру
            if not all(k in line_config_dict for k in ['point1', 'point2']):
                raise ValueError("line_config должен содержать point1 и point2")
            if not all(k in line_config_dict['point1'] for k in ['x', 'y']):
                raise ValueError("point1 должен содержать x и y")
            if not all(k in line_config_dict['point2'] for k in ['x', 'y']):
                raise ValueError("point2 должен содержать x и y")
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="line_config должен быть валидным JSON"
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )
    # Проверяем расширение файла
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат файла. Разрешенные форматы: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
        )
    
    # Сохраняем файл
    try:
        # Создаем уникальное имя файла
        import uuid
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = UPLOADS_DIR / unique_filename
        
        # Проверяем размер файла во время загрузки
        file_size = 0
        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(8192)  # Читаем по 8KB
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > MAX_UPLOAD_SIZE:
                    # Удаляем частично загруженный файл
                    os.remove(file_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"Файл слишком большой. Максимальный размер: {MAX_UPLOAD_SIZE / (1024*1024):.0f}MB"
                    )
                f.write(chunk)
        
        # Создаем задачу обработки
        task_id = task_service.create_task(
            file.filename, 
            str(file_path),
            use_line=use_line_bool,
            line_config=line_config_dict
        )
        
        return UploadResponse(
            task_id=task_id,
            message="Видео загружено успешно, обработка начата",
            filename=file.filename
        )
    
    except HTTPException:
        raise
    except Exception as e:
        # Удаляем файл в случае ошибки
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при загрузке файла: {str(e)}"
        )

