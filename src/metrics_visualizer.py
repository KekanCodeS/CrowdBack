"""
Модуль для визуализации метрик
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import matplotlib
matplotlib.use('Agg')  # Используем backend без GUI
import matplotlib.pyplot as plt
import numpy as np

# Добавляем путь к модулям для импорта
sys.path.insert(0, str(Path(__file__).parent))
from api.config import BASE_DIR

METRICS_DIR = BASE_DIR / "data" / "metrics"
CHARTS_DIR = METRICS_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def _read_jsonl_file(file_path: Path, start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None) -> List[Dict]:
    """Читает JSONL файл и фильтрует по времени"""
    if not file_path.exists():
        return []
    
    metrics = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                metric = json.loads(line)
                metric_time = datetime.fromisoformat(metric.get("timestamp", ""))
                
                if start_time and metric_time < start_time:
                    continue
                if end_time and metric_time > end_time:
                    continue
                
                metrics.append(metric)
            except (json.JSONDecodeError, ValueError):
                continue
    
    return metrics


def plot_api_response_times(hours: int = 24, output_path: Optional[Path] = None) -> Path:
    """
    Создает график времени отклика API по времени
    
    Args:
        hours: Количество часов для анализа
        output_path: Путь для сохранения графика (опционально)
    
    Returns:
        Путь к сохраненному файлу
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    
    api_metrics_file = METRICS_DIR / "api_metrics.jsonl"
    metrics = _read_jsonl_file(api_metrics_file, start_time, end_time)
    
    if not metrics:
        # Создаем пустой график
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "Нет данных за указанный период", 
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Время отклика API")
        ax.set_xlabel("Время")
        ax.set_ylabel("Время отклика (мс)")
    else:
        timestamps = [datetime.fromisoformat(m["timestamp"]) for m in metrics]
        response_times = [m.get("response_time_ms", 0) for m in metrics]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(timestamps, response_times, alpha=0.6, linewidth=1)
        ax.scatter(timestamps, response_times, s=10, alpha=0.4)
        ax.set_title(f"Время отклика API за последние {hours} часов")
        ax.set_xlabel("Время")
        ax.set_ylabel("Время отклика (мс)")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    if output_path is None:
        output_path = CHARTS_DIR / f"api_response_times_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_api_response_time_distribution(hours: int = 24, output_path: Optional[Path] = None) -> Path:
    """
    Создает гистограмму распределения времени отклика API
    
    Args:
        hours: Количество часов для анализа
        output_path: Путь для сохранения графика (опционально)
    
    Returns:
        Путь к сохраненному файлу
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    
    api_metrics_file = METRICS_DIR / "api_metrics.jsonl"
    metrics = _read_jsonl_file(api_metrics_file, start_time, end_time)
    
    if not metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Нет данных за указанный период",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Распределение времени отклика API")
    else:
        response_times = [m.get("response_time_ms", 0) for m in metrics]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(response_times, bins=50, edgecolor='black', alpha=0.7)
        ax.set_title(f"Распределение времени отклика API за последние {hours} часов")
        ax.set_xlabel("Время отклика (мс)")
        ax.set_ylabel("Количество запросов")
        ax.grid(True, alpha=0.3, axis='y')
        
        # Добавляем статистику
        mean_time = np.mean(response_times)
        median_time = np.median(response_times)
        ax.axvline(mean_time, color='r', linestyle='--', label=f'Среднее: {mean_time:.2f} мс')
        ax.axvline(median_time, color='g', linestyle='--', label=f'Медиана: {median_time:.2f} мс')
        ax.legend()
    
    if output_path is None:
        output_path = CHARTS_DIR / f"api_response_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_ai_confidence_scores(hours: int = 24, output_path: Optional[Path] = None) -> Path:
    """
    Создает график метрик AI (confidence scores, количество детекций)
    
    Args:
        hours: Количество часов для анализа
        output_path: Путь для сохранения графика (опционально)
    
    Returns:
        Путь к сохраненному файлу
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    
    ai_metrics_file = METRICS_DIR / "ai_metrics.jsonl"
    metrics = _read_jsonl_file(ai_metrics_file, start_time, end_time)
    
    if not metrics:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        ax1.text(0.5, 0.5, "Нет данных за указанный период",
                ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("Метрики AI")
    else:
        timestamps = [datetime.fromisoformat(m["timestamp"]) for m in metrics]
        confidences = [m.get("avg_confidence", 0) for m in metrics]
        detections = [m.get("detections_count", 0) for m in metrics]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # График confidence scores
        ax1.plot(timestamps, confidences, alpha=0.7, linewidth=1, color='blue')
        ax1.set_title(f"Средний Confidence Score за последние {hours} часов")
        ax1.set_xlabel("Время")
        ax1.set_ylabel("Confidence Score")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # График количества детекций
        ax2.plot(timestamps, detections, alpha=0.7, linewidth=1, color='green')
        ax2.set_title(f"Количество детекций за последние {hours} часов")
        ax2.set_xlabel("Время")
        ax2.set_ylabel("Количество детекций")
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
    
    if output_path is None:
        output_path = CHARTS_DIR / f"ai_confidence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_load_test_results(results_file: Path, output_path: Optional[Path] = None) -> Path:
    """
    Создает график результатов нагрузочного тестирования
    
    Args:
        results_file: Путь к файлу с результатами нагрузочного тестирования
        output_path: Путь для сохранения графика (опционально)
    
    Returns:
        Путь к сохраненному файлу
    """
    if not results_file.exists():
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "Файл с результатами не найден",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Результаты нагрузочного тестирования")
    else:
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        
        # Извлекаем данные из результатов Artillery
        if "aggregate" in results:
            aggregate = results["aggregate"]
            
            # Получаем данные о времени отклика из summaries
            latency = None
            if "summaries" in aggregate and "http.response_time" in aggregate["summaries"]:
                latency = aggregate["summaries"]["http.response_time"]
            elif "latency" in aggregate:
                latency = aggregate["latency"]
            
            if latency:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # График времени отклика
                metrics = ["min", "median", "p95", "p99", "max"]
                values = [
                    latency.get("min", 0),
                    latency.get("median", latency.get("p50", 0)),  # median может быть как median, так и p50
                    latency.get("p95", 0),
                    latency.get("p99", 0),
                    latency.get("max", 0)
                ]
                
                ax1.bar(metrics, values, color=['green', 'blue', 'orange', 'red', 'darkred'])
                ax1.set_title("Время отклика (мс)")
                ax1.set_ylabel("Время (мс)")
                ax1.grid(True, alpha=0.3, axis='y')
                
                # График RPS (requests per second)
                rps_value = 0
                if "rates" in aggregate and "http.request_rate" in aggregate["rates"]:
                    rps_value = aggregate["rates"]["http.request_rate"]
                elif "rate" in aggregate:
                    if isinstance(aggregate["rate"], dict):
                        rps_value = aggregate["rate"].get("mean", 0)
                    else:
                        rps_value = aggregate["rate"]
                
                if rps_value > 0:
                    ax2.bar(["RPS"], [rps_value], color=['blue'])
                    ax2.set_title("Запросов в секунду (RPS)")
                    ax2.set_ylabel("RPS")
                    ax2.grid(True, alpha=0.3, axis='y')
                else:
                    # Если нет данных о RPS, показываем количество запросов
                    if "counters" in aggregate and "http.requests" in aggregate["counters"]:
                        total_requests = aggregate["counters"]["http.requests"]
                        ax2.bar(["Всего запросов"], [total_requests], color=['green'])
                        ax2.set_title("Всего запросов")
                        ax2.set_ylabel("Количество")
                        ax2.grid(True, alpha=0.3, axis='y')
                    else:
                        ax2.text(0.5, 0.5, "Нет данных о RPS",
                                ha="center", va="center", transform=ax2.transAxes)
                        ax2.set_title("Запросов в секунду (RPS)")
                
                plt.tight_layout()
            else:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.text(0.5, 0.5, "Неверный формат файла результатов\n(нет данных о времени отклика)",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_title("Результаты нагрузочного тестирования")
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, "Неверный формат файла результатов\n(нет секции aggregate)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Результаты нагрузочного тестирования")
    
    if output_path is None:
        output_path = CHARTS_DIR / f"load_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_ui_load_times(hours: int = 24, output_path: Optional[Path] = None) -> Path:
    """
    Создает график времени загрузки UI компонентов
    
    Args:
        hours: Количество часов для анализа
        output_path: Путь для сохранения графика (опционально)
    
    Returns:
        Путь к сохраненному файлу
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    
    ui_metrics_file = METRICS_DIR / "ui_metrics.jsonl"
    metrics = _read_jsonl_file(ui_metrics_file, start_time, end_time)
    
    if not metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "Нет данных за указанный период",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Время загрузки UI компонентов")
    else:
        timestamps = [datetime.fromisoformat(m["timestamp"]) for m in metrics]
        load_times = [m.get("load_time_ms", 0) for m in metrics]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(timestamps, load_times, alpha=0.6, linewidth=1, color='purple')
        ax.scatter(timestamps, load_times, s=10, alpha=0.4)
        ax.set_title(f"Время загрузки UI компонентов за последние {hours} часов")
        ax.set_xlabel("Время")
        ax.set_ylabel("Время загрузки (мс)")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    if output_path is None:
        output_path = CHARTS_DIR / f"ui_load_times_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

