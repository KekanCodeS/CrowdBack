"""
Роутер для получения результатов обработки
"""
import os
import base64
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

import sys
from pathlib import Path as PathLib
sys.path.append(str(PathLib(__file__).parent.parent.parent))
from pdf_report import generate_pdf_report

from ..task_service import task_service
from ..models import ResultsResponse, ErrorResponse

router = APIRouter(prefix="/results", tags=["results"])


@router.get("/{task_id}", response_model=ResultsResponse)
async def get_results(task_id: str):
    """
    Получает результаты обработки видео
    
    Args:
        task_id: Идентификатор задачи
        
    Returns:
        ResultsResponse с результатами обработки
    """
    task = task_service.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Задача еще не завершена. Текущий статус: {task['status']}"
        )
    
    if not task.get("results"):
        raise HTTPException(status_code=500, detail="Результаты обработки не найдены")
    
    stats = task["results"]
    
    # Формируем ответ
    video_info = {
        "fps": stats.get("fps", 0),
        "width": stats.get("resolution", {}).get("width", 0),
        "height": stats.get("resolution", {}).get("height", 0),
        "frame_count": stats.get("total_frames", 0),
        "duration": stats.get("total_frames", 0) / stats.get("fps", 1) if stats.get("fps", 0) > 0 else 0
    }
    
    tracking_stats = None
    if "tracking" in stats:
        tracking_stats = {
            "total_in": stats["tracking"].get("total_in", 0),
            "total_out": stats["tracking"].get("total_out", 0),
            "current_inside": stats["tracking"].get("current_inside", 0)
        }
    
    summary_stats = stats.get("summary", {})
    
    return ResultsResponse(
        task_id=task_id,
        filename=task["filename"],
        video_info=video_info,
        statistics=stats.get("frame_statistics", []),
        tracking=tracking_stats,
        summary={
            "max_count": summary_stats.get("max_count", 0),
            "min_count": summary_stats.get("min_count", 0),
            "avg_count": summary_stats.get("avg_count", 0.0)
        },
        created_at=task["created_at"],
        processing_time=task.get("processing_time")
    )


@router.get("/{task_id}/video")
async def download_video(task_id: str):
    """
    Скачивает обработанное видео
    
    Args:
        task_id: Идентификатор задачи
        
    Returns:
        Файл видео
    """
    task = task_service.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Задача еще не завершена. Текущий статус: {task['status']}"
        )
    
    video_path = task.get("output_video_path")
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Обработанное видео не найдено")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"processed_{task['filename']}"
    )


@router.get("/{task_id}/heatmap")
async def download_heatmap(task_id: str):
    """
    Скачивает тепловую карту
    
    Args:
        task_id: Идентификатор задачи
        
    Returns:
        Файл изображения тепловой карты
    """
    task = task_service.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Задача еще не завершена. Текущий статус: {task['status']}"
        )
    
    heatmap_path = task.get("heatmap_path")
    if not heatmap_path or not os.path.exists(heatmap_path):
        raise HTTPException(status_code=404, detail="Тепловая карта не найдена")
    
    return FileResponse(
        heatmap_path,
        media_type="image/png",
        filename=f"heatmap_{Path(task['filename']).stem}.png"
    )


@router.get("/{task_id}/stats")
async def download_stats(task_id: str):
    """
    Скачивает статистику обработки в формате JSON
    
    Args:
        task_id: Идентификатор задачи
        
    Returns:
        JSON файл со статистикой
    """
    task = task_service.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Задача еще не завершена. Текущий статус: {task['status']}"
        )
    
    stats_path = task.get("stats_path")
    if not stats_path or not os.path.exists(stats_path):
        raise HTTPException(status_code=404, detail="Файл статистики не найден")
    
    return FileResponse(
        stats_path,
        media_type="application/json",
        filename=f"stats_{Path(task['filename']).stem}.json"
    )


@router.get("/{task_id}/heatmap/image")
async def get_heatmap_image(task_id: str):
    """
    Получает тепловую карту как base64 изображение
    
    Args:
        task_id: Идентификатор задачи
        
    Returns:
        JSON с base64 изображением тепловой карты
    """
    task = task_service.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Задача еще не завершена. Текущий статус: {task['status']}"
        )
    
    heatmap_path = task.get("heatmap_path")
    if not heatmap_path or not os.path.exists(heatmap_path):
        raise HTTPException(status_code=404, detail="Тепловая карта не найдена")
    
    try:
        # Читаем изображение и конвертируем в base64
        with open(heatmap_path, "rb") as f:
            image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        return {
            "image": f"data:image/png;base64,{image_base64}",
            "filename": Path(heatmap_path).name
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при чтении тепловой карты: {str(e)}"
        )


@router.get("/{task_id}/report")
async def download_report(task_id: str):
    """
    Генерирует и скачивает PDF отчет со статистикой
    
    Args:
        task_id: Идентификатор задачи
        
    Returns:
        PDF файл с отчетом
    """
    task = task_service.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Задача еще не завершена. Текущий статус: {task['status']}"
        )
    
    if not task.get("results"):
        raise HTTPException(status_code=500, detail="Результаты обработки не найдены")
    
    try:
        # Определяем путь для PDF отчета
        output_dir = Path(task.get("stats_path", "")).parent
        video_name = Path(task["filename"]).stem
        pdf_path = output_dir / f"{video_name}_report.pdf"
        
        # Генерируем PDF отчет
        generate_pdf_report(
            stats=task["results"],
            output_path=str(pdf_path),
            heatmap_path=task.get("heatmap_path")
        )
        
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=500, detail="Не удалось создать PDF отчет")
        
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"report_{video_name}.pdf"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при генерации PDF отчета: {str(e)}"
        )

