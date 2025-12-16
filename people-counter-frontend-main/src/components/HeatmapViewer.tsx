/**
 * Компонент для отображения тепловой карты
 */
import React from 'react';

interface HeatmapViewerProps {
  imageUrl: string;
  filename?: string;
}

export function HeatmapViewer({ imageUrl, filename }: HeatmapViewerProps) {
  return (
    <div className="bg-slate-800/60 border border-slate-700 rounded-2xl p-5">
      <h3 className="text-sm font-semibold mb-4">Тепловая карта перемещений</h3>
      <p className="text-xs text-slate-400 mb-4">
        Визуализация областей с наибольшей активностью людей в видео
      </p>
      <div className="flex justify-center bg-slate-900/40 rounded-lg p-4">
        <img 
          src={imageUrl} 
          alt="Тепловая карта" 
          className="max-w-full h-auto rounded-lg border border-slate-600"
        />
      </div>
      {filename && (
        <p className="text-xs text-slate-500 mt-2 text-center">
          {filename}
        </p>
      )}
    </div>
  );
}

