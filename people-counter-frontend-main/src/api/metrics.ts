/**
 * API функции для работы с метриками
 */
import { apiClient } from './client';

export interface ApiMetricsResponse {
  metrics: Array<{
    timestamp: string;
    endpoint: string;
    method: string;
    response_time_ms: number;
    status_code: number;
  }>;
  summary: {
    total_requests: number;
    avg_response_time_ms: number;
    min_response_time_ms: number;
    max_response_time_ms: number;
    status_codes: Record<string, number>;
  };
}

export interface AIMetricsResponse {
  metrics: Array<{
    timestamp: string;
    task_id: string;
    frame: number;
    avg_confidence: number;
    detections_count: number;
    high_confidence_count: number;
    medium_confidence_count: number;
    low_confidence_count: number;
  }>;
  summary: {
    total_frames: number;
    total_detections: number;
    avg_confidence: number;
    avg_detections_per_frame: number;
    max_detections_per_frame: number;
    min_detections_per_frame: number;
  };
}

export interface UIMetricsResponse {
  metrics: Array<{
    timestamp: string;
    component: string;
    load_time_ms: number;
    render_time_ms?: number;
    event_type: string;
  }>;
  summary: {
    total_events: number;
    avg_load_time_ms: number;
    avg_render_time_ms: number;
  };
}

export interface MetricsSummaryResponse {
  period: {
    start: string;
    end: string;
    hours: number;
  };
  api: {
    total_requests: number;
    avg_response_time_ms: number;
    min_response_time_ms: number;
    max_response_time_ms: number;
  };
  ai: {
    total_frames: number;
    total_detections: number;
    avg_confidence: number;
  };
  ui: {
    total_events: number;
    avg_load_time_ms: number;
  };
}

/**
 * Получает метрики API
 */
export async function getApiMetrics(
  startTime?: string,
  endTime?: string,
  endpoint?: string
): Promise<ApiMetricsResponse> {
  const params = new URLSearchParams();
  if (startTime) params.append('start_time', startTime);
  if (endTime) params.append('end_time', endTime);
  if (endpoint) params.append('endpoint', endpoint);

  const response = await apiClient.get<ApiMetricsResponse>(
    `/metrics/api?${params.toString()}`
  );
  return response.data;
}

/**
 * Получает метрики AI
 */
export async function getAIMetrics(
  startTime?: string,
  endTime?: string,
  taskId?: string
): Promise<AIMetricsResponse> {
  const params = new URLSearchParams();
  if (startTime) params.append('start_time', startTime);
  if (endTime) params.append('end_time', endTime);
  if (taskId) params.append('task_id', taskId);

  const response = await apiClient.get<AIMetricsResponse>(
    `/metrics/ai?${params.toString()}`
  );
  return response.data;
}

/**
 * Получает метрики UI
 */
export async function getUIMetrics(
  startTime?: string,
  endTime?: string,
  component?: string
): Promise<UIMetricsResponse> {
  const params = new URLSearchParams();
  if (startTime) params.append('start_time', startTime);
  if (endTime) params.append('end_time', endTime);
  if (component) params.append('component', component);

  const response = await apiClient.get<UIMetricsResponse>(
    `/metrics/ui?${params.toString()}`
  );
  return response.data;
}

/**
 * Получает сводную статистику метрик
 */
export async function getMetricsSummary(hours: number = 24): Promise<MetricsSummaryResponse> {
  const response = await apiClient.get<MetricsSummaryResponse>(
    `/metrics/summary?hours=${hours}`
  );
  return response.data;
}

/**
 * Получает URL графика метрик
 */
export function getMetricsChartUrl(metricType: string, hours: number = 24): string {
  return `/metrics/charts/${metricType}?hours=${hours}`;
}

