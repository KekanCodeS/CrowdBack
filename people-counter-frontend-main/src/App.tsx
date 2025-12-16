import { useState, useEffect, useRef } from "react";
import React from 'react';
import { uploadVideo, getVideoPreview, type LineConfig as ApiLineConfig } from './api/upload';
import { getTaskStatus, cancelTask, type TaskStatus } from './api/tasks';
import { getResults, downloadVideo, downloadStats, getHeatmapImage, downloadReport, type ResultsResponse } from './api/results';
import { handleApiError } from './api/client';
import { LineSetup } from './components/LineSetup';
import { Charts } from './components/Charts';
import { HeatmapViewer } from './components/HeatmapViewer';
import MetricsDashboard from './components/MetricsDashboard';
import { useComponentMetrics } from './utils/metrics';

type Mode = "upload" | "results" | "metrics";

function App() {
  const [mode, setMode] = useState<Mode>("upload");
  const [taskId, setTaskId] = useState<string | null>(null);

  const handleUploadComplete = (newTaskId: string) => {
    setTaskId(newTaskId);
    setMode("results");
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-50">
      {/* Шапка */}
      <header className="border-b border-slate-800">
        <div className="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-xl font-semibold tracking-tight">
            People Counter<span className="text-indigo-400">.AI</span>
          </h1>
          <div className="flex items-center gap-4">
            {mode === "upload" && (
              <button
                onClick={() => setMode("metrics")}
                className="px-3 py-1.5 text-xs rounded-lg border border-slate-600 
                         text-slate-300 hover:bg-slate-700 transition"
              >
                Метрики
              </button>
            )}
            <span className="text-xs text-slate-400">
              Video analytics dashboard
            </span>
          </div>
        </div>
      </header>

      {/* Контент */}
      <main className="max-w-5xl mx-auto px-4 py-8">
        {mode === "upload" ? (
          <UploadScreen onDone={handleUploadComplete} />
        ) : mode === "results" ? (
          <ResultsScreen taskId={taskId} onBack={() => {
            setMode("upload");
            setTaskId(null);
          }} />
        ) : (
          <MetricsDashboard onBack={() => setMode("upload")} />
        )}
      </main>
    </div>
  );
}

type UploadScreenProps = {
  onDone: (taskId: string) => void;
};

function UploadScreen({ onDone }: UploadScreenProps) {
  const { markLoadEnd } = useComponentMetrics('UploadScreen');
  
  React.useEffect(() => {
    markLoadEnd();
  }, [markLoadEnd]);
  
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [preview, setPreview] = useState<{ image: string; width: number; height: number } | null>(null);
  const [showLineSetup, setShowLineSetup] = useState(false);
  const [lineConfig, setLineConfig] = useState<ApiLineConfig | null>(null);
  const [useLine, setUseLine] = useState(false);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const pollingIntervalRef = useRef<number | null>(null);

  useEffect(() => {
    // Очистка интервала при размонтировании
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, []);

  // Загружаем превью при выборе файла
  useEffect(() => {
    if (file && !preview && !showLineSetup) {
      loadPreview();
    }
  }, [file]);

  const loadPreview = async () => {
    if (!file) return;
    
    setLoadingPreview(true);
    setError(null);
    try {
      const previewData = await getVideoPreview(file);
      setPreview({
        image: previewData.preview_image,
        width: previewData.width,
        height: previewData.height,
      });
      setShowLineSetup(true);
    } catch (err) {
      const apiError = handleApiError(err);
      setError(apiError.error || 'Ошибка при загрузке превью');
      setLoadingPreview(false);
    }
  };

  const startPolling = (id: string) => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
    }

    const poll = async () => {
      try {
        const status = await getTaskStatus(id);
        setProgress(status.progress || 0);

        if (status.status === 'completed') {
          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current);
          }
          setIsProcessing(false);
          onDone(id);
        } else if (status.status === 'failed') {
          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current);
          }
          setIsProcessing(false);
          setError(status.message || 'Обработка завершилась с ошибкой');
        }
      } catch (err) {
        const apiError = handleApiError(err);
        setError(apiError.error);
      }
    };

    // Опрашиваем каждые 2 секунды
    pollingIntervalRef.current = window.setInterval(poll, 2000);
    poll(); // Первый вызов сразу
  };

  const handleStart = async () => {
    if (!file) return;
    
    // Проверяем что если useLine=true, то lineConfig должен быть установлен
    if (useLine && !lineConfig) {
      setError('Необходимо настроить линию IN/OUT перед началом обработки');
      return;
    }
    
    setIsProcessing(true);
    setProgress(0);
    setError(null);
    setShowLineSetup(false);

    try {
      const response = await uploadVideo(file, useLine, lineConfig || undefined);
      setTaskId(response.task_id);
      startPolling(response.task_id);
    } catch (err) {
      const apiError = handleApiError(err);
      setError(apiError.error || 'Ошибка при загрузке видео');
      setIsProcessing(false);
    }
  };

  const handleLineSave = (config: ApiLineConfig) => {
    setLineConfig(config);
    setShowLineSetup(false);
    setUseLine(true);
  };

  const handleLineCancel = () => {
    setShowLineSetup(false);
    setLineConfig(null);
    setUseLine(false);
  };

  const handleCancel = async () => {
    if (taskId && isProcessing) {
      try {
        await cancelTask(taskId);
      } catch (err) {
        console.error('Ошибка при отмене задачи:', err);
      }
    }
    
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
    
    setIsProcessing(false);
    setProgress(0);
    setTaskId(null);
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-[2fr,1.3fr] gap-6">
      {/* Левая карточка — загрузка и управление */}
      <section className="bg-slate-800/60 border border-slate-700 rounded-2xl p-6 shadow-lg">
        <h2 className="text-lg font-semibold mb-1">Загрузка видео</h2>
        <p className="text-sm text-slate-400 mb-4">
          Выберите файл с камеры, затем запустите обработку.
        </p>

        {/* Область выбора файла */}
        <label className="block border-2 border-dashed border-slate-600 hover:border-indigo-400 transition rounded-xl p-6 text-center cursor-pointer mb-4 bg-slate-900/40">
          <input
            type="file"
            accept="video/*"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) {
                setFile(f);
                setPreview(null);
                setShowLineSetup(false);
                setLineConfig(null);
                setUseLine(false);
              }
            }}
            disabled={isProcessing}
          />
          <div className="flex flex-col items-center gap-2">
            <span className="text-sm font-medium">
              {file ? "Выбран файл:" : "Перетащите файл сюда или нажмите для выбора"}
            </span>
            {file ? (
              <span className="text-xs text-slate-300">
                {file.name}
              </span>
            ) : (
              <span className="text-xs text-slate-500">
                Поддерживаются стандартные форматы (mp4, mov и т.д.)
              </span>
            )}
          </div>
        </label>

        {/* Загрузка превью */}
        {loadingPreview && (
          <div className="mb-4 p-3 bg-slate-900/40 rounded-lg text-sm text-slate-400 text-center">
            Загрузка превью...
          </div>
        )}

        {/* Настройка линии */}
        {showLineSetup && preview && (
          <div className="mb-4">
            <LineSetup
              previewImage={preview.image}
              width={preview.width}
              height={preview.height}
              onSave={handleLineSave}
              onCancel={handleLineCancel}
            />
          </div>
        )}

        {/* Чекбокс использования линии */}
        {file && !showLineSetup && (
          <div className="mb-4 flex items-center gap-2">
            <input
              type="checkbox"
              id="useLine"
              checked={useLine}
              onChange={(e) => {
                setUseLine(e.target.checked);
                if (!e.target.checked) {
                  setLineConfig(null);
                } else if (!lineConfig) {
                  // Если включаем, но линии нет - показываем настройку
                  loadPreview();
                }
              }}
              disabled={isProcessing}
              className="w-4 h-4 rounded border-slate-600 bg-slate-700 text-indigo-500 focus:ring-indigo-400"
            />
            <label htmlFor="useLine" className="text-sm text-slate-300 cursor-pointer">
              Использовать линию IN/OUT для подсчета
            </label>
          </div>
        )}

        {useLine && lineConfig && !showLineSetup && (
          <div className="mb-4 p-2 bg-emerald-900/20 border border-emerald-700 rounded-lg text-xs text-emerald-200">
            Линия настроена: P1({lineConfig.point1.x}, {lineConfig.point1.y}) → P2({lineConfig.point2.x}, {lineConfig.point2.y})
          </div>
        )}

        {/* Кнопки */}
        <div className="flex flex-wrap gap-3 mb-4">
          <button
            onClick={handleStart}
            disabled={!file || isProcessing}
            className="px-4 py-2 rounded-lg text-sm font-medium
                       bg-indigo-500 hover:bg-indigo-400 disabled:bg-slate-600
                       disabled:cursor-not-allowed transition"
          >
            {isProcessing ? "Обработка…" : "Начать обработку"}
          </button>

          <button
            onClick={handleCancel}
            disabled={!isProcessing}
            className="px-4 py-2 rounded-lg text-sm font-medium
                       border border-slate-600 text-slate-200
                       hover:bg-slate-700 disabled:opacity-40
                       disabled:cursor-not-allowed transition"
          >
            Отменить
          </button>

          {file && !isProcessing && progress === 0 && (
            <span className="text-xs text-slate-400 self-center">
              Файл выбран, нажмите «Начать обработку».
            </span>
          )}
        </div>

        {/* Ошибка */}
        {error && (
          <div className="mt-4 p-3 bg-rose-900/30 border border-rose-700 rounded-lg text-sm text-rose-200">
            {error}
          </div>
        )}

        {/* Прогресс-бар */}
        {(isProcessing || progress > 0) && (
          <div className="space-y-1 mt-4">
            <div className="flex justify-between text-xs text-slate-400">
              <span>Прогресс обработки</span>
              <span>{Math.round(progress)}%</span>
            </div>
            <div className="w-full h-2 rounded-full bg-slate-700 overflow-hidden">
              <div
                className="h-full bg-indigo-500 transition-[width]"
                style={{ width: `${progress}%` }}
              />
            </div>
            {taskId && (
              <p className="text-xs text-slate-500 mt-1">
                ID задачи: {taskId.substring(0, 8)}...
              </p>
            )}
          </div>
        )}
      </section>

      {/* Правая карточка — подсказки/описание */}
      <aside className="bg-slate-800/40 border border-slate-700 rounded-2xl p-5 text-sm space-y-3">
        <h3 className="text-sm font-semibold">Как это работает</h3>
        <p className="text-slate-400">
          Система анализирует видео, извлекает кадры, детектирует людей,
          строит треки движения и считает входы/выходы через виртуальную линию.
        </p>
        <ul className="list-disc list-inside text-slate-400 space-y-1">
          <li>Видео дискретизируется до 6–10 кадров в секунду</li>
          <li>На каждом кадре ищутся люди (bbox person)</li>
          <li>Треки связывают объекты между кадрами</li>
          <li>Фиксируются IN/OUT при пересечении линии</li>
        </ul>
        <p className="text-xs text-slate-500">
          После завершения обработки вы перейдёте к экрану с итоговыми
          метриками и отчётом.
        </p>
      </aside>
    </div>
  );
}

type ResultsScreenProps = {
  taskId: string | null;
  onBack: () => void;
};

function ResultsScreen({ taskId, onBack }: ResultsScreenProps) {
  const [results, setResults] = useState<ResultsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState<string | null>(null);
  const [heatmapImage, setHeatmapImage] = useState<string | null>(null);
  const [loadingHeatmap, setLoadingHeatmap] = useState(false);

  useEffect(() => {
    if (!taskId) {
      setError('ID задачи не указан');
      setLoading(false);
      return;
    }

    const loadResults = async () => {
      try {
        const data = await getResults(taskId);
        setResults(data);
        
        // Загружаем тепловую карту
        try {
          setLoadingHeatmap(true);
          const heatmapData = await getHeatmapImage(taskId);
          setHeatmapImage(heatmapData.image);
        } catch (err) {
          // Тепловая карта может отсутствовать, это не критично
          console.warn('Не удалось загрузить тепловую карту:', err);
        } finally {
          setLoadingHeatmap(false);
        }
      } catch (err) {
        const apiError = handleApiError(err);
        setError(apiError.error || 'Ошибка при загрузке результатов');
      } finally {
        setLoading(false);
      }
    };

    loadResults();
  }, [taskId]);

  const formatDuration = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
      return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const handleDownload = async (type: 'video' | 'stats' | 'report') => {
    if (!taskId) return;
    
    setDownloading(type);
    try {
      if (type === 'video') {
        await downloadVideo(taskId);
      } else if (type === 'stats') {
        await downloadStats(taskId);
      } else if (type === 'report') {
        await downloadReport(taskId);
      }
    } catch (err) {
      const apiError = handleApiError(err);
      alert(`Ошибка при скачивании: ${apiError.error}`);
    } finally {
      setDownloading(null);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-400 mb-4"></div>
          <p className="text-slate-400">Загрузка результатов...</p>
        </div>
      </div>
    );
  }

  if (error || !results) {
    return (
      <div className="space-y-6">
        <div className="p-4 bg-rose-900/30 border border-rose-700 rounded-lg text-rose-200">
          {error || 'Результаты не найдены'}
        </div>
        <button
          onClick={onBack}
          className="px-4 py-2 rounded-lg border border-slate-600 
                   text-slate-200 hover:bg-slate-700 transition"
        >
          Вернуться к загрузке
        </button>
      </div>
    );
  }

  const duration = formatDuration(results.video_info.duration);
  const processingTime = results.processing_time 
    ? formatDuration(results.processing_time) 
    : 'N/A';

  return (
    <div className="space-y-6">
      {/* Верхняя строка: заголовок + кнопка назад */}
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold">
            Результаты обработки
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Файл: <span className="text-slate-200">{results.filename}</span> ·{" "}
            Длительность видео: {duration} · Время обработки: {processingTime}
          </p>
        </div>

        <button
          onClick={onBack}
          className="px-3 py-1.5 rounded-lg border border-slate-600 
                   text-xs text-slate-200 hover:bg-slate-700 transition"
        >
          Загрузить другое видео
        </button>
      </div>

      {/* Карточки метрик */}
      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard 
          label="Входы (IN)" 
          value={results.tracking?.total_in || 0} 
          accent="text-emerald-400" 
        />
        <MetricCard 
          label="Выходы (OUT)" 
          value={results.tracking?.total_out || 0} 
          accent="text-rose-400" 
        />
        <MetricCard 
          label="Всего детекций" 
          value={results.statistics.length} 
          accent="text-indigo-400" 
        />
        <MetricCard 
          label="Максимум на кадре" 
          value={results.summary.max_count} 
          accent="text-amber-300" 
        />
      </section>

      {/* Графики */}
      {results && results.statistics.length > 0 && (
        <section>
          <Charts 
            statistics={results.statistics} 
            videoInfo={results.video_info}
          />
        </section>
      )}

      {/* Тепловая карта */}
      {loadingHeatmap && (
        <div className="bg-slate-800/60 border border-slate-700 rounded-2xl p-5">
          <div className="text-center py-8">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-400 mb-4"></div>
            <p className="text-slate-400">Загрузка тепловой карты...</p>
          </div>
        </div>
      )}
      {heatmapImage && !loadingHeatmap && (
        <section>
          <HeatmapViewer imageUrl={heatmapImage} />
        </section>
      )}

      {/* Блок отчётов */}
      <section className="bg-slate-800/60 border border-slate-700 rounded-2xl p-5 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div className="text-sm">
          <h3 className="font-semibold mb-1">Отчёты и экспорт</h3>
          <p className="text-slate-400">
            Вы можете выгрузить данные анализа для дальнейшей обработки
            во внешних системах (Excel, BI-платформы и т.п.).
          </p>
        </div>

        <div className="flex flex-wrap gap-3">
          <button
            onClick={() => handleDownload('report')}
            disabled={downloading === 'report'}
            className="px-4 py-2 rounded-lg text-sm font-medium
                       bg-emerald-500 hover:bg-emerald-400 disabled:opacity-50
                       disabled:cursor-not-allowed transition"
          >
            {downloading === 'report' ? 'Генерация...' : 'Скачать PDF отчет'}
          </button>
          <button
            onClick={() => handleDownload('video')}
            disabled={downloading === 'video'}
            className="px-4 py-2 rounded-lg text-sm font-medium
                       bg-indigo-500 hover:bg-indigo-400 disabled:opacity-50
                       disabled:cursor-not-allowed transition"
          >
            {downloading === 'video' ? 'Скачивание...' : 'Скачать видео'}
          </button>
          <button
            onClick={() => handleDownload('stats')}
            disabled={downloading === 'stats'}
            className="px-4 py-2 rounded-lg text-sm font-medium
                       border border-slate-600 hover:bg-slate-700 
                       disabled:opacity-50 disabled:cursor-not-allowed transition"
          >
            {downloading === 'stats' ? 'Скачивание...' : 'Скачать JSON'}
          </button>
        </div>
      </section>
    </div>
  );
}

type MetricCardProps = {
  label: string;
  value: string | number;
  accent?: string;
  isText?: boolean;
};

function MetricCard({
  label,
  value,
  accent = "text-indigo-400",
  isText,
}: MetricCardProps) {
  return (
    <div className="bg-slate-800/60 border border-slate-700 rounded-2xl p-4 flex flex-col gap-1">
      <span className="text-xs text-slate-400">{label}</span>
      <span className={`text-xl font-semibold ${accent}`}>{value}</span>
      {!isText && (
        <span className="text-[11px] text-slate-500">
          Значение рассчитано по всей длительности видео
        </span>
      )}
    </div>
  );
}

export default App;
