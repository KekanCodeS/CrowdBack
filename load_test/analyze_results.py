"""
Скрипт для анализа результатов нагрузочного тестирования
"""
import json
import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metrics_visualizer import plot_load_test_results
from api.config import BASE_DIR

METRICS_DIR = BASE_DIR / "data" / "metrics"
RESULTS_FILE = METRICS_DIR / "load_test_results.json"


def analyze_load_test_results(results_file: Path):
    """
    Анализирует результаты нагрузочного тестирования
    
    Args:
        results_file: Путь к файлу с результатами Artillery
    """
    if not results_file.exists():
        print(f"Файл результатов не найден: {results_file}")
        return
    
    print(f"Анализ результатов из: {results_file}")
    
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    # Извлекаем метрики из результатов Artillery
    if "aggregate" in results:
        aggregate = results["aggregate"]
        
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ НАГРУЗОЧНОГО ТЕСТИРОВАНИЯ")
        print("=" * 60)
        
        # Метрики латентности (из summaries или latency)
        latency = None
        if "summaries" in aggregate and "http.response_time" in aggregate["summaries"]:
            latency = aggregate["summaries"]["http.response_time"]
        elif "latency" in aggregate:
            latency = aggregate["latency"]
        
        if latency:
            print("\nЛатентность (время отклика):")
            print(f"  Минимум: {latency.get('min', 0):.2f} мс")
            median = latency.get('median', latency.get('p50', 0))
            print(f"  Медиана: {median:.2f} мс")
            print(f"  P95: {latency.get('p95', 0):.2f} мс")
            print(f"  P99: {latency.get('p99', 0):.2f} мс")
            print(f"  Максимум: {latency.get('max', 0):.2f} мс")
            print(f"  Среднее: {latency.get('mean', 0):.2f} мс")
        
        # Метрики скорости запросов
        if "rates" in aggregate and "http.request_rate" in aggregate["rates"]:
            rps = aggregate["rates"]["http.request_rate"]
            print("\nСкорость запросов (RPS):")
            print(f"  Среднее: {rps:.2f} запросов/сек")
        elif "rate" in aggregate:
            rate = aggregate["rate"]
            if isinstance(rate, dict):
                print("\nСкорость запросов (RPS):")
                print(f"  Среднее: {rate.get('mean', 0):.2f} запросов/сек")
                print(f"  Минимум: {rate.get('min', 0):.2f} запросов/сек")
                print(f"  Максимум: {rate.get('max', 0):.2f} запросов/сек")
        
        # Метрики счетчиков
        if "counters" in aggregate:
            counters = aggregate["counters"]
            print("\nСчетчики:")
            if "http.requests" in counters:
                print(f"  Всего запросов: {counters['http.requests']}")
            if "http.responses" in counters:
                print(f"  Всего ответов: {counters['http.responses']}")
            if "http.codes.200" in counters:
                print(f"  Успешных ответов (200): {counters['http.codes.200']}")
            if "http.codes.404" in counters:
                print(f"  Ошибок 404: {counters['http.codes.404']}")
            if "vusers.completed" in counters:
                print(f"  Завершенных пользователей: {counters['vusers.completed']}")
            if "vusers.failed" in counters:
                print(f"  Неудачных пользователей: {counters['vusers.failed']}")
        
        # Метрики ошибок
        if "errors" in aggregate:
            errors = aggregate["errors"]
            print("\nОшибки:")
            if errors:
                for error in errors:
                    print(f"  {error.get('message', 'Unknown')}: {error.get('count', 0)}")
            else:
                print("  Ошибок не обнаружено")
        
        print("\n" + "=" * 60)
    
    # Генерируем графики
    print("\nГенерация графиков...")
    chart_path = plot_load_test_results(results_file)
    print(f"График сохранен: {chart_path}")


if __name__ == "__main__":
    results_file = RESULTS_FILE
    if len(sys.argv) > 1:
        results_file = Path(sys.argv[1])
    
    analyze_load_test_results(results_file)

