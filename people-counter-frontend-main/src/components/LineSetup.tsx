/**
 * Компонент для настройки линии IN/OUT на превью видео
 */
import React, { useEffect, useRef, useState } from 'react';

export interface LineConfig {
  point1: { x: number; y: number };
  point2: { x: number; y: number };
}

interface LineSetupProps {
  previewImage: string;
  width: number;
  height: number;
  onSave: (config: LineConfig) => void;
  onCancel: () => void;
}

export function LineSetup({ previewImage, width, height, onSave, onCancel }: LineSetupProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [points, setPoints] = useState<{ x: number; y: number }[]>([]);
  const [isDrawing, setIsDrawing] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Загружаем изображение
    const img = new Image();
    img.onload = () => {
      // Устанавливаем размеры canvas
      canvas.width = width;
      canvas.height = height;
      
      // Рисуем изображение
      ctx.drawImage(img, 0, 0, width, height);
      
      // Рисуем точки и линию если есть
      if (points.length > 0) {
        points.forEach((point, index) => {
          ctx.fillStyle = '#10b981'; // emerald-500
          ctx.beginPath();
          ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
          ctx.fill();
          
          // Подпись точки
          ctx.fillStyle = '#ffffff';
          ctx.font = '12px Arial';
          ctx.fillText(`P${index + 1}`, point.x + 8, point.y - 8);
        });
        
        if (points.length === 2) {
          ctx.strokeStyle = '#10b981';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(points[0].x, points[0].y);
          ctx.lineTo(points[1].x, points[1].y);
          ctx.stroke();
        }
      }
    };
    img.src = previewImage;
  }, [previewImage, width, height, points]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    // Масштабируем координаты с учетом CSS масштабирования canvas
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    if (points.length < 2) {
      setPoints([...points, { x, y }]);
    }
  };

  const handleReset = () => {
    setPoints([]);
  };

  const handleSave = () => {
    if (points.length === 2) {
      onSave({
        point1: points[0],
        point2: points[1],
      });
    }
  };

  return (
    <div className="space-y-4">
      <div className="bg-slate-800/60 border border-slate-700 rounded-2xl p-4">
        <h3 className="text-sm font-semibold mb-2">Настройка линии IN/OUT</h3>
        <p className="text-xs text-slate-400 mb-4">
          Кликните на изображении два раза, чтобы установить линию подсчета. 
          Линия определяет границу для подсчета входящих и выходящих людей.
        </p>
        
        <div className="flex justify-center bg-slate-900/40 rounded-lg p-2 mb-4">
          <canvas
            ref={canvasRef}
            onClick={handleCanvasClick}
            className="border border-slate-600 rounded cursor-crosshair max-w-full"
            style={{ maxHeight: '400px' }}
          />
        </div>

        <div className="flex flex-wrap gap-2 items-center">
          <button
            onClick={handleSave}
            disabled={points.length !== 2}
            className="px-4 py-2 rounded-lg text-sm font-medium
                       bg-indigo-500 hover:bg-indigo-400 disabled:bg-slate-600
                       disabled:cursor-not-allowed transition"
          >
            Сохранить линию
          </button>
          
          <button
            onClick={handleReset}
            disabled={points.length === 0}
            className="px-4 py-2 rounded-lg text-sm font-medium
                       border border-slate-600 text-slate-200
                       hover:bg-slate-700 disabled:opacity-40
                       disabled:cursor-not-allowed transition"
          >
            Сбросить
          </button>
          
          <button
            onClick={onCancel}
            className="px-4 py-2 rounded-lg text-sm font-medium
                       border border-slate-600 text-slate-200
                       hover:bg-slate-700 transition"
          >
            Отмена
          </button>
          
          {points.length > 0 && (
            <span className="text-xs text-slate-400 ml-auto">
              Выбрано точек: {points.length}/2
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

