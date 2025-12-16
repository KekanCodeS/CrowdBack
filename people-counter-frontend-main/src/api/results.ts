/**
 * API функции для получения результатов обработки
 */
import { apiClient, handleApiError } from './client';

export interface VideoInfo {
  fps: number;
  width: number;
  height: number;
  frame_count: number;
  duration: number;
}

export interface TrackingStats {
  total_in: number;
  total_out: number;
  current_inside: number;
}

export interface SummaryStats {
  max_count: number;
  min_count: number;
  avg_count: number;
}

export interface ResultsResponse {
  task_id: string;
  filename: string;
  video_info: VideoInfo;
  statistics: any[];
  tracking: TrackingStats | null;
  summary: SummaryStats;
  created_at: string;
  processing_time: number | null;
}

/**
 * Получает результаты обработки видео
 * @param taskId - Идентификатор задачи
 * @returns Результаты обработки
 */
export const getResults = async (taskId: string): Promise<ResultsResponse> => {
  try {
    const response = await apiClient.get<ResultsResponse>(`/results/${taskId}`);
    return response.data;
  } catch (error) {
    throw handleApiError(error);
  }
};

/**
 * Скачивает обработанное видео
 * @param taskId - Идентификатор задачи
 * @param filename - Имя файла для сохранения (опционально)
 */
export const downloadVideo = async (taskId: string, filename?: string): Promise<void> => {
  try {
    const response = await apiClient.get(`/results/${taskId}/video`, {
      responseType: 'blob',
    });

    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', filename || `processed_video_${taskId}.mp4`);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
  } catch (error) {
    throw handleApiError(error);
  }
};

/**
 * Скачивает тепловую карту
 * @param taskId - Идентификатор задачи
 * @param filename - Имя файла для сохранения (опционально)
 */
export const downloadHeatmap = async (taskId: string, filename?: string): Promise<void> => {
  try {
    const response = await apiClient.get(`/results/${taskId}/heatmap`, {
      responseType: 'blob',
    });

    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', filename || `heatmap_${taskId}.png`);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
  } catch (error) {
    throw handleApiError(error);
  }
};

/**
 * Скачивает статистику в формате JSON
 * @param taskId - Идентификатор задачи
 * @param filename - Имя файла для сохранения (опционально)
 */
export const downloadStats = async (taskId: string, filename?: string): Promise<void> => {
  try {
    const response = await apiClient.get(`/results/${taskId}/stats`, {
      responseType: 'blob',
    });

    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', filename || `stats_${taskId}.json`);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
  } catch (error) {
    throw handleApiError(error);
  }
};

/**
 * Получает тепловую карту как base64 изображение
 * @param taskId - Идентификатор задачи
 * @returns Объект с base64 изображением и именем файла
 */
export const getHeatmapImage = async (taskId: string): Promise<{ image: string; filename: string }> => {
  try {
    const response = await apiClient.get<{ image: string; filename: string }>(`/results/${taskId}/heatmap/image`);
    return response.data;
  } catch (error) {
    throw handleApiError(error);
  }
};

/**
 * Скачивает PDF отчет со статистикой
 * @param taskId - Идентификатор задачи
 * @param filename - Имя файла для сохранения (опционально)
 */
export const downloadReport = async (taskId: string, filename?: string): Promise<void> => {
  try {
    const response = await apiClient.get(`/results/${taskId}/report`, {
      responseType: 'blob',
    });

    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', filename || `report_${taskId}.pdf`);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
  } catch (error) {
    throw handleApiError(error);
  }
};

