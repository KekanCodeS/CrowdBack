/**
 * API функции для работы с задачами обработки
 */
import { apiClient, handleApiError } from './client';

export type TaskStatus = 'pending' | 'processing' | 'completed' | 'failed';

export interface TaskStatusResponse {
  task_id: string;
  status: TaskStatus;
  progress: number | null;
  message: string | null;
  created_at: string;
  updated_at: string;
}

/**
 * Получает статус задачи обработки
 * @param taskId - Идентификатор задачи
 * @returns Статус задачи
 */
export const getTaskStatus = async (taskId: string): Promise<TaskStatusResponse> => {
  try {
    const response = await apiClient.get<TaskStatusResponse>(`/tasks/${taskId}`);
    return response.data;
  } catch (error) {
    throw handleApiError(error);
  }
};

/**
 * Отменяет задачу обработки
 * @param taskId - Идентификатор задачи
 * @returns Сообщение об успешной отмене
 */
export const cancelTask = async (taskId: string): Promise<{ message: string; task_id: string }> => {
  try {
    const response = await apiClient.post<{ message: string; task_id: string }>(
      `/tasks/${taskId}/cancel`
    );
    return response.data;
  } catch (error) {
    throw handleApiError(error);
  }
};

