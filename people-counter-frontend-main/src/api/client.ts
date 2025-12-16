/**
 * HTTP клиент для работы с API
 */
import axios, { AxiosInstance, AxiosError } from 'axios';

// Используем прокси через nginx в production или прямой URL в development
const API_BASE_URL = import.meta.env.VITE_API_URL || 
  (import.meta.env.PROD ? '/api' : 'http://localhost:8000');

/**
 * Создает настроенный экземпляр axios
 */
export const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 секунд для обычных запросов
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Обработка ошибок API
 */
export interface ApiError {
  error: string;
  detail?: string;
}

export const handleApiError = (error: unknown): ApiError => {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<ApiError>;
    if (axiosError.response?.data) {
      return axiosError.response.data;
    }
    return {
      error: axiosError.message || 'Произошла ошибка при запросе к серверу',
    };
  }
  return {
    error: 'Неизвестная ошибка',
  };
};

