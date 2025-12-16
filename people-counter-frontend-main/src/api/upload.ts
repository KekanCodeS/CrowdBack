/**
 * API функции для загрузки видео
 */
import { apiClient, handleApiError } from './client';

export interface UploadResponse {
  task_id: string;
  message: string;
  filename: string;
}

export interface PreviewResponse {
  preview_image: string; // base64 data URL
  width: number;
  height: number;
  filename: string;
}

export interface LineConfig {
  point1: { x: number; y: number };
  point2: { x: number; y: number };
}

/**
 * Получает превью первого кадра видео
 * @param file - Файл видео
 * @returns Ответ с base64 изображением первого кадра
 */
export const getVideoPreview = async (file: File): Promise<PreviewResponse> => {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post<PreviewResponse>('/upload/preview', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 30000, // 30 секунд для получения превью
    });

    return response.data;
  } catch (error) {
    throw handleApiError(error);
  }
};

/**
 * Загружает видео файл на сервер
 * @param file - Файл видео для загрузки
 * @param useLine - Использовать ли линию IN/OUT
 * @param lineConfig - Конфигурация линии (обязательна если useLine=true)
 * @returns Ответ с task_id задачи обработки
 */
export const uploadVideo = async (
  file: File,
  useLine: boolean = false,
  lineConfig?: LineConfig
): Promise<UploadResponse> => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('use_line', useLine.toString());
    
    if (useLine && lineConfig) {
      formData.append('line_config', JSON.stringify({
        point1: lineConfig.point1,
        point2: lineConfig.point2,
      }));
    }

    const response = await apiClient.post<UploadResponse>('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 60000, // 60 секунд для загрузки файла
    });

    return response.data;
  } catch (error) {
    throw handleApiError(error);
  }
};

