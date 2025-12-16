/**
 * Утилиты для сбора метрик UI
 */
import { apiClient } from '../api/client';

export interface UIMetric {
  component: string;
  load_time_ms: number;
  render_time_ms?: number;
  event_type?: 'load' | 'render' | 'interaction';
}

/**
 * Измеряет время загрузки компонента
 */
export class ComponentMetrics {
  private loadStartTime: number = 0;
  private renderStartTime: number = 0;
  private componentName: string;

  constructor(componentName: string) {
    this.componentName = componentName;
    this.loadStartTime = performance.now();
  }

  /**
   * Отмечает начало рендеринга
   */
  markRenderStart(): void {
    this.renderStartTime = performance.now();
  }

  /**
   * Отмечает окончание рендеринга и отправляет метрику
   */
  markRenderEnd(): void {
    const renderTime = performance.now() - this.renderStartTime;
    this.sendMetric({
      component: this.componentName,
      load_time_ms: this.renderStartTime - this.loadStartTime,
      render_time_ms: renderTime,
      event_type: 'render'
    });
  }

  /**
   * Отмечает окончание загрузки компонента
   */
  markLoadEnd(): void {
    const loadTime = performance.now() - this.loadStartTime;
    this.sendMetric({
      component: this.componentName,
      load_time_ms: loadTime,
      event_type: 'load'
    });
  }

  /**
   * Отправляет метрику на сервер
   */
  private async sendMetric(metric: UIMetric): void {
    try {
      await apiClient.post('/metrics/ui', metric);
    } catch (error) {
      // Не прерываем работу приложения при ошибке отправки метрик
      console.warn('Не удалось отправить метрику:', error);
    }
  }
}

/**
 * Измеряет время выполнения функции
 */
export async function measureExecutionTime<T>(
  fn: () => Promise<T>,
  componentName: string,
  eventType: 'load' | 'render' | 'interaction' = 'load'
): Promise<T> {
  const startTime = performance.now();
  try {
    const result = await fn();
    const executionTime = performance.now() - startTime;
    
    sendUIMetric({
      component: componentName,
      load_time_ms: executionTime,
      event_type: eventType
    });
    
    return result;
  } catch (error) {
    const executionTime = performance.now() - startTime;
    sendUIMetric({
      component: componentName,
      load_time_ms: executionTime,
      event_type: eventType
    });
    throw error;
  }
}

/**
 * Отправляет метрику UI на сервер
 */
export async function sendUIMetric(metric: UIMetric): Promise<void> {
  try {
    await apiClient.post('/metrics/ui', metric);
  } catch (error) {
    console.warn('Не удалось отправить метрику UI:', error);
  }
}

/**
 * Хук для измерения времени загрузки компонента
 */
export function useComponentMetrics(componentName: string): {
  markLoadEnd: () => void;
  markRenderStart: () => void;
  markRenderEnd: () => void;
} {
  const metrics = new ComponentMetrics(componentName);

  return {
    markLoadEnd: () => metrics.markLoadEnd(),
    markRenderStart: () => metrics.markRenderStart(),
    markRenderEnd: () => metrics.markRenderEnd()
  };
}

