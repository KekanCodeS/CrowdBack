#!/bin/bash
# Скрипт для запуска нагрузочного тестирования с Artillery

# Проверяем наличие Artillery
if ! command -v artillery &> /dev/null; then
    echo "Artillery не установлен. Устанавливаем..."
    npm install -g artillery
fi

# Создаем директорию для результатов
RESULTS_DIR="../data/metrics"
mkdir -p "$RESULTS_DIR"

# Запускаем нагрузочное тестирование
echo "Запуск нагрузочного тестирования..."
artillery run --output "$RESULTS_DIR/load_test_results.json" artillery-config.yml

# Конвертируем результаты в JSON
echo "Конвертация результатов..."
artillery report --output "$RESULTS_DIR/load_test_report.html" "$RESULTS_DIR/load_test_results.json"

echo "Результаты сохранены в $RESULTS_DIR/load_test_results.json"
echo "Отчет сохранен в $RESULTS_DIR/load_test_report.html"

