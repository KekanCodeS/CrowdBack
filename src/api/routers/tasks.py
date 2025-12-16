"""
Роутер для работы с задачами обработки
"""
from fastapi import APIRouter, HTTPException
from typing import Optional

from ..task_service import task_service
from ..models import TaskStatusResponse, ErrorResponse

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Получает статус задачи обработки
    
    Args:
        task_id: Идентификатор задачи
        
    Returns:
        TaskStatusResponse со статусом и прогрессом
    """
    task = task_service.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    return TaskStatusResponse(
        task_id=task["task_id"],
        status=task["status"],
        progress=task.get("progress", 0.0),
        message=task.get("message"),
        created_at=task["created_at"],
        updated_at=task["updated_at"]
    )


@router.post("/{task_id}/cancel")
async def cancel_task(task_id: str):
    """
    Отменяет задачу обработки (если она еще не завершена)
    
    Args:
        task_id: Идентификатор задачи
        
    Returns:
        Сообщение об успешной отмене
    """
    task = task_service.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    if task["status"] in ["completed", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Невозможно отменить задачу со статусом: {task['status']}"
        )
    
    success = task_service.cancel_task(task_id)
    
    if success:
        return {"message": "Задача отменена успешно", "task_id": task_id}
    else:
        raise HTTPException(status_code=500, detail="Не удалось отменить задачу")

