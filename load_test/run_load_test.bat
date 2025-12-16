@echo off
REM Скрипт для запуска нагрузочного тестирования с Artillery на Windows

REM Проверяем наличие Artillery
where artillery >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Artillery не установлен. Устанавливаем...
    npm install -g artillery
)

REM Создаем директорию для результатов
set RESULTS_DIR=..\data\metrics
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"

REM Запускаем нагрузочное тестирование
echo Запуск нагрузочного тестирования...
artillery run --output "%RESULTS_DIR%\load_test_results.json" artillery-config.yml

REM Конвертируем результаты в JSON
echo Конвертация результатов...
artillery report --output "%RESULTS_DIR%\load_test_report.html" "%RESULTS_DIR%\load_test_results.json"

echo Результаты сохранены в %RESULTS_DIR%\load_test_results.json
echo Отчет сохранен в %RESULTS_DIR%\load_test_report.html

