/**
 * Компонент для отображения графиков статистики
 */
import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface FrameStatistic {
  frame: number;
  count: number;
  detections?: number;
  in_count?: number;
  out_count?: number;
}

interface ChartsProps {
  statistics: FrameStatistic[];
  videoInfo: {
    fps: number;
    duration: number;
  };
}

export function Charts({ statistics, videoInfo }: ChartsProps) {
  // Преобразуем статистику для графиков
  const countData = statistics.map((stat, index) => ({
    time: (stat.frame / videoInfo.fps).toFixed(1),
    frame: stat.frame,
    count: stat.count,
  }));

  // Подсчитываем накопленные IN/OUT события
  let cumulativeIn = 0;
  let cumulativeOut = 0;
  const inOutData = statistics.map((stat) => {
    if (stat.in_count !== undefined) cumulativeIn = stat.in_count;
    if (stat.out_count !== undefined) cumulativeOut = stat.out_count;
    return {
      time: (stat.frame / videoInfo.fps).toFixed(1),
      frame: stat.frame,
      in: cumulativeIn,
      out: cumulativeOut,
    };
  });

  return (
    <div className="space-y-6">
      {/* График количества людей по времени */}
      <div className="bg-slate-800/60 border border-slate-700 rounded-2xl p-5">
        <h3 className="text-sm font-semibold mb-4">Количество людей по времени</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={countData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
            <XAxis 
              dataKey="time" 
              stroke="#94a3b8"
              label={{ value: 'Время (секунды)', position: 'insideBottom', offset: -5, fill: '#94a3b8' }}
            />
            <YAxis 
              stroke="#94a3b8"
              label={{ value: 'Количество людей', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1e293b', 
                border: '1px solid #475569',
                borderRadius: '8px',
                color: '#e2e8f0'
              }}
            />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="count" 
              stroke="#6366f1" 
              strokeWidth={2}
              name="Людей на кадре"
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* График IN/OUT событий */}
      {inOutData.some(d => d.in > 0 || d.out > 0) && (
        <div className="bg-slate-800/60 border border-slate-700 rounded-2xl p-5">
          <h3 className="text-sm font-semibold mb-4">События IN/OUT по времени</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={inOutData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis 
                dataKey="time" 
                stroke="#94a3b8"
                label={{ value: 'Время (секунды)', position: 'insideBottom', offset: -5, fill: '#94a3b8' }}
              />
              <YAxis 
                stroke="#94a3b8"
                label={{ value: 'Количество событий', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1e293b', 
                  border: '1px solid #475569',
                  borderRadius: '8px',
                  color: '#e2e8f0'
                }}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="in" 
                stroke="#10b981" 
                strokeWidth={2}
                name="Входы (IN)"
                dot={false}
              />
              <Line 
                type="monotone" 
                dataKey="out" 
                stroke="#ef4444" 
                strokeWidth={2}
                name="Выходы (OUT)"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

