import React, { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import {
  getApiMetrics,
  getAIMetrics,
  getUIMetrics,
  getMetricsSummary,
  getMetricsChartUrl,
  type ApiMetricsResponse,
  type AIMetricsResponse,
  type UIMetricsResponse,
  type MetricsSummaryResponse
} from '../api/metrics';

interface MetricsDashboardProps {
  onBack?: () => void;
}

export default function MetricsDashboard({ onBack }: MetricsDashboardProps) {
  const [summary, setSummary] = useState<MetricsSummaryResponse | null>(null);
  const [apiMetrics, setApiMetrics] = useState<ApiMetricsResponse | null>(null);
  const [aiMetrics, setAiMetrics] = useState<AIMetricsResponse | null>(null);
  const [uiMetrics, setUiMetrics] = useState<UIMetricsResponse | null>(null);
  const [hours, setHours] = useState(24);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'summary' | 'api' | 'ai' | 'ui'>('summary');

  useEffect(() => {
    loadMetrics();
  }, [hours]);

  const loadMetrics = async () => {
    setLoading(true);
    try {
      const endTime = new Date();
      const startTime = new Date(endTime.getTime() - hours * 60 * 60 * 1000);

      const [summaryData, apiData, aiData, uiData] = await Promise.all([
        getMetricsSummary(hours),
        getApiMetrics(startTime.toISOString(), endTime.toISOString()),
        getAIMetrics(startTime.toISOString(), endTime.toISOString()),
        getUIMetrics(startTime.toISOString(), endTime.toISOString())
      ]);

      setSummary(summaryData);
      setApiMetrics(apiData);
      setAiMetrics(aiData);
      setUiMetrics(uiData);
    } catch (error) {
      console.error('Ошибка загрузки метрик:', error);
    } finally {
      setLoading(false);
    }
  };

  const prepareApiChartData = () => {
    if (!apiMetrics) return [];
    return apiMetrics.metrics.slice(-100).map(m => ({
      time: new Date(m.timestamp).toLocaleTimeString(),
      responseTime: m.response_time_ms
    }));
  };

  const prepareAiChartData = () => {
    if (!aiMetrics) return [];
    return aiMetrics.metrics.slice(-100).map(m => ({
      frame: m.frame,
      confidence: m.avg_confidence,
      detections: m.detections_count
    }));
  };

  const prepareUiChartData = () => {
    if (!uiMetrics) return [];
    return uiMetrics.metrics.slice(-100).map(m => ({
      time: new Date(m.timestamp).toLocaleTimeString(),
      loadTime: m.load_time_ms,
      renderTime: m.render_time_ms || 0
    }));
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Загрузка метрик...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6 flex justify-between items-center">
          <h1 className="text-3xl font-bold text-gray-900">Дашборд метрик</h1>
          <div className="flex gap-4">
            <select
              value={hours}
              onChange={(e) => setHours(Number(e.target.value))}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value={1}>Последний час</option>
              <option value={24}>Последние 24 часа</option>
              <option value={168}>Последняя неделя</option>
            </select>
            {onBack && (
              <button
                onClick={onBack}
                className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300"
              >
                Назад
              </button>
            )}
          </div>
        </div>

        {/* Tabs */}
        <div className="mb-6 border-b border-gray-200">
          <nav className="flex space-x-8">
            {(['summary', 'api', 'ai', 'ui'] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab === 'summary' ? 'Сводка' : tab.toUpperCase()}
              </button>
            ))}
          </nav>
        </div>

        {/* Summary Tab */}
        {activeTab === 'summary' && summary && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white p-6 rounded-lg shadow">
                <h3 className="text-lg font-semibold text-gray-700 mb-2">API</h3>
                <div className="space-y-2">
                  <p className="text-sm text-gray-600">
                    Запросов: <span className="font-bold">{summary.api.total_requests}</span>
                  </p>
                  <p className="text-sm text-gray-600">
                    Среднее время: <span className="font-bold">{summary.api.avg_response_time_ms.toFixed(2)} мс</span>
                  </p>
                  <p className="text-sm text-gray-600">
                    Мин/Макс: <span className="font-bold">{summary.api.min_response_time_ms.toFixed(2)} / {summary.api.max_response_time_ms.toFixed(2)} мс</span>
                  </p>
                </div>
              </div>

              <div className="bg-white p-6 rounded-lg shadow">
                <h3 className="text-lg font-semibold text-gray-700 mb-2">AI</h3>
                <div className="space-y-2">
                  <p className="text-sm text-gray-600">
                    Кадров: <span className="font-bold">{summary.ai.total_frames}</span>
                  </p>
                  <p className="text-sm text-gray-600">
                    Детекций: <span className="font-bold">{summary.ai.total_detections}</span>
                  </p>
                  <p className="text-sm text-gray-600">
                    Средний confidence: <span className="font-bold">{(summary.ai.avg_confidence * 100).toFixed(2)}%</span>
                  </p>
                </div>
              </div>

              <div className="bg-white p-6 rounded-lg shadow">
                <h3 className="text-lg font-semibold text-gray-700 mb-2">UI</h3>
                <div className="space-y-2">
                  <p className="text-sm text-gray-600">
                    Событий: <span className="font-bold">{summary.ui.total_events}</span>
                  </p>
                  <p className="text-sm text-gray-600">
                    Среднее время загрузки: <span className="font-bold">{summary.ui.avg_load_time_ms.toFixed(2)} мс</span>
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* API Tab */}
        {activeTab === 'api' && apiMetrics && (
          <div className="space-y-6">
            <div className="bg-white p-6 rounded-lg shadow">
              <h2 className="text-xl font-bold mb-4">Время отклика API</h2>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={prepareApiChartData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis label={{ value: 'Время (мс)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="responseTime" stroke="#8884d8" name="Время отклика (мс)" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white p-6 rounded-lg shadow">
              <h2 className="text-xl font-bold mb-4">Статистика API</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-sm text-gray-600">Всего запросов</p>
                  <p className="text-2xl font-bold">{apiMetrics.summary.total_requests}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Среднее время</p>
                  <p className="text-2xl font-bold">{apiMetrics.summary.avg_response_time_ms.toFixed(2)} мс</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Мин время</p>
                  <p className="text-2xl font-bold">{apiMetrics.summary.min_response_time_ms.toFixed(2)} мс</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Макс время</p>
                  <p className="text-2xl font-bold">{apiMetrics.summary.max_response_time_ms.toFixed(2)} мс</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* AI Tab */}
        {activeTab === 'ai' && aiMetrics && (
          <div className="space-y-6">
            <div className="bg-white p-6 rounded-lg shadow">
              <h2 className="text-xl font-bold mb-4">Confidence Scores</h2>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={prepareAiChartData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="frame" />
                  <YAxis label={{ value: 'Confidence', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="confidence" stroke="#82ca9d" name="Confidence Score" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white p-6 rounded-lg shadow">
              <h2 className="text-xl font-bold mb-4">Количество детекций</h2>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={prepareAiChartData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="frame" />
                  <YAxis label={{ value: 'Количество', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="detections" fill="#8884d8" name="Детекций" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white p-6 rounded-lg shadow">
              <h2 className="text-xl font-bold mb-4">Статистика AI</h2>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <div>
                  <p className="text-sm text-gray-600">Всего кадров</p>
                  <p className="text-2xl font-bold">{aiMetrics.summary.total_frames}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Всего детекций</p>
                  <p className="text-2xl font-bold">{aiMetrics.summary.total_detections}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Средний confidence</p>
                  <p className="text-2xl font-bold">{(aiMetrics.summary.avg_confidence * 100).toFixed(2)}%</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Среднее детекций/кадр</p>
                  <p className="text-2xl font-bold">{aiMetrics.summary.avg_detections_per_frame.toFixed(2)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Макс детекций/кадр</p>
                  <p className="text-2xl font-bold">{aiMetrics.summary.max_detections_per_frame}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Мин детекций/кадр</p>
                  <p className="text-2xl font-bold">{aiMetrics.summary.min_detections_per_frame}</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* UI Tab */}
        {activeTab === 'ui' && uiMetrics && (
          <div className="space-y-6">
            <div className="bg-white p-6 rounded-lg shadow">
              <h2 className="text-xl font-bold mb-4">Время загрузки UI компонентов</h2>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={prepareUiChartData()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis label={{ value: 'Время (мс)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="loadTime" stroke="#8884d8" name="Время загрузки (мс)" />
                  <Line type="monotone" dataKey="renderTime" stroke="#82ca9d" name="Время рендеринга (мс)" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white p-6 rounded-lg shadow">
              <h2 className="text-xl font-bold mb-4">Статистика UI</h2>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <div>
                  <p className="text-sm text-gray-600">Всего событий</p>
                  <p className="text-2xl font-bold">{uiMetrics.summary.total_events}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Среднее время загрузки</p>
                  <p className="text-2xl font-bold">{uiMetrics.summary.avg_load_time_ms.toFixed(2)} мс</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Среднее время рендеринга</p>
                  <p className="text-2xl font-bold">{uiMetrics.summary.avg_render_time_ms.toFixed(2)} мс</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

