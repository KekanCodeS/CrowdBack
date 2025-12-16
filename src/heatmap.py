"""
Модуль для генерации тепловой карты перемещения посетителей
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class HeatmapGenerator:
    """Класс для генерации тепловых карт перемещения людей"""
    
    def __init__(self, frame_shape: Tuple[int, int], blur_sigma: float = 15.0):
        """
        Args:
            frame_shape: (height, width) - размер кадра
            blur_sigma: параметр размытия для Gaussian blur
        """
        self.frame_shape = frame_shape
        self.blur_sigma = blur_sigma
        self.heatmap = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.float32)
        self.point_history: List[Tuple[int, int]] = []
    
    def add_point(self, point: Tuple[float, float], weight: float = 1.0):
        """
        Добавляет точку на тепловую карту
        
        Args:
            point: (x, y) - координаты точки
            weight: вес точки (интенсивность)
        """
        x, y = int(point[0]), int(point[1])
        
        # Проверяем границы
        if 0 <= x < self.frame_shape[1] and 0 <= y < self.frame_shape[0]:
            self.heatmap[y, x] += weight
            self.point_history.append((x, y))
    
    def add_trajectory(self, trajectory: List[Tuple[float, float]], weight: float = 1.0):
        """
        Добавляет траекторию движения на тепловую карту
        
        Args:
            trajectory: список точек траектории
            weight: вес траектории
        """
        for point in trajectory:
            self.add_point(point, weight)
    
    def generate(self, normalize: bool = True) -> np.ndarray:
        """
        Генерирует финальную тепловую карту
        
        Args:
            normalize: нормализовать ли значения (0-255)
        
        Returns:
            Тепловая карта в формате (H, W, 3) BGR для OpenCV
        """
        # Применяем Gaussian blur для сглаживания
        blurred = gaussian_filter(self.heatmap, sigma=self.blur_sigma)
        
        if normalize:
            # Нормализуем в диапазон 0-255
            if blurred.max() > 0:
                blurred = (blurred / blurred.max() * 255).astype(np.uint8)
            else:
                blurred = blurred.astype(np.uint8)
        else:
            blurred = blurred.astype(np.uint8)
        
        # Применяем цветовую карту (jet colormap)
        heatmap_colored = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
        
        return heatmap_colored
    
    def generate_matplotlib(self, alpha: float = 0.6) -> Tuple[np.ndarray, plt.Figure]:
        """
        Генерирует тепловую карту с использованием matplotlib (лучшее качество)
        
        Args:
            alpha: прозрачность тепловой карты
        
        Returns:
            (heatmap_image, figure) - изображение и фигура matplotlib
        """
        # Применяем Gaussian blur
        blurred = gaussian_filter(self.heatmap, sigma=self.blur_sigma)
        
        # Создаем фигуру
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Нормализуем для отображения
        if blurred.max() > 0:
            normalized = blurred / blurred.max()
        else:
            normalized = blurred
        
        # Отображаем тепловую карту
        im = ax.imshow(normalized, cmap='jet', alpha=alpha, interpolation='bilinear')
        ax.axis('off')
        
        # Добавляем colorbar
        plt.colorbar(im, ax=ax, label='Интенсивность перемещений')
        
        # Конвертируем в numpy array
        fig.canvas.draw()
        
        # Универсальный способ получения буфера (работает с разными типами canvas)
        try:
            # Попытка использовать tostring_rgb() (для большинства canvas)
            buf = fig.canvas.tostring_rgb()
            heatmap_array = np.frombuffer(buf, dtype=np.uint8)
            heatmap_array = heatmap_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except AttributeError:
            # Если tostring_rgb() недоступен (например, TkAgg), используем buffer_rgba()
            try:
                buf = fig.canvas.buffer_rgba()
                arr = np.asarray(buf)
                # Берем только RGB каналы (пропускаем альфа-канал)
                heatmap_array = arr[:, :, :3].copy()
            except AttributeError:
                # Последняя попытка - через tostring_argb()
                buf = fig.canvas.tostring_argb()
                arr = np.frombuffer(buf, dtype=np.uint8)
                width, height = fig.canvas.get_width_height()
                arr = arr.reshape((height, width, 4))
                # Конвертируем ARGB в RGB (пропускаем альфа-канал)
                heatmap_array = arr[:, :, 1:4].copy()
        
        # Конвертируем RGB в BGR для OpenCV
        heatmap_bgr = cv2.cvtColor(heatmap_array, cv2.COLOR_RGB2BGR)
        
        return heatmap_bgr, fig
    
    def overlay_on_frame(self, frame: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Накладывает тепловую карту на кадр
        
        Args:
            frame: исходный кадр
            alpha: прозрачность тепловой карты
        
        Returns:
            Кадр с наложенной тепловой картой
        """
        heatmap = self.generate()
        
        # Изменяем размер тепловой карты, если нужно
        if heatmap.shape[:2] != frame.shape[:2]:
            heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        
        # Накладываем тепловую карту на кадр
        overlay = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
        
        return overlay
    
    def reset(self):
        """Сбрасывает тепловую карту"""
        self.heatmap = np.zeros(self.frame_shape, dtype=np.float32)
        self.point_history = []
    
    def save(self, output_path: str, format: str = 'opencv'):
        """
        Сохраняет тепловую карту
        
        Args:
            output_path: путь для сохранения
            format: 'opencv' или 'matplotlib'
        """
        if format == 'opencv':
            heatmap = self.generate()
            cv2.imwrite(output_path, heatmap)
        elif format == 'matplotlib':
            heatmap, fig = self.generate_matplotlib()
            fig.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
        else:
            raise ValueError(f"Неизвестный формат: {format}")


def create_heatmap_from_tracks(tracks_data: Dict, frame_shape: Tuple[int, int], 
                               blur_sigma: float = 15.0) -> HeatmapGenerator:
    """
    Создает тепловую карту из данных треков
    
    Args:
        tracks_data: словарь с треками {track_id: {'centers': [(x, y), ...], ...}}
        frame_shape: размер кадра
        blur_sigma: параметр размытия
    
    Returns:
        HeatmapGenerator с заполненной тепловой картой
    """
    generator = HeatmapGenerator(frame_shape, blur_sigma)
    
    for track_id, track_info in tracks_data.items():
        if 'centers' in track_info:
            generator.add_trajectory(track_info['centers'])
    
    return generator

