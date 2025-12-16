"""
Модуль для генерации PDF отчета со статистикой и визуализацией
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from PIL import Image

import sys
from pathlib import Path as PathLib
sys.path.append(str(PathLib(__file__).parent))
from utils import ensure_dir


def generate_pdf_report(stats: Dict, output_path: str, heatmap_path: Optional[str] = None):
    """
    Генерирует PDF отчет со статистикой и визуализацией
    """
    ensure_dir(os.path.dirname(output_path))
    
    with PdfPages(output_path) as pdf:
        # Страница 1: Титульная
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.7, "Отчет по анализу видео\nPeople Counter AI", 
                ha='center', va='center', fontsize=24, fontweight='bold', transform=ax.transAxes)
        video_name = Path(stats.get('video_path', 'N/A')).name
        info = f"Файл: {video_name}\nДата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nРазрешение: {stats.get('resolution', {}).get('width', 0)}x{stats.get('resolution', {}).get('height', 0)}\nFPS: {stats.get('fps', 0):.2f}\nКадров: {stats.get('total_frames', 0)}"
        ax.text(0.5, 0.4, info, ha='center', va='center', fontsize=12, transform=ax.transAxes, family='monospace')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Страница 2: Статистика
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.95, "Сводная статистика", ha='center', va='top', fontsize=18, fontweight='bold', transform=ax.transAxes)
        
        summary = stats.get('summary', {})
        tracking = stats.get('tracking', {})
        stats_text = f"Максимум людей: {summary.get('max_count', 0)}\nМинимум людей: {summary.get('min_count', 0)}\nСреднее: {summary.get('avg_count', 0):.2f}\n"
        if tracking:
            stats_text += f"\nВходы (IN): {tracking.get('total_in', 0)}\nВыходы (OUT): {tracking.get('total_out', 0)}\nТекущее внутри: {tracking.get('current_inside', 0)}"
        ax.text(0.1, 0.7, stats_text, ha='left', va='top', fontsize=14, transform=ax.transAxes, family='monospace')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Страница 3: График количества людей
        frame_stats = stats.get('frame_statistics', [])
        if frame_stats:
            fig, ax = plt.subplots(figsize=(10, 6))
            fps = stats.get('fps', 30)
            frames = [s['frame'] for s in frame_stats]
            counts = [s['count'] for s in frame_stats]
            time_seconds = [f / fps for f in frames]
            ax.plot(time_seconds, counts, linewidth=2, color='#6366f1')
            ax.fill_between(time_seconds, counts, alpha=0.3, color='#6366f1')
            ax.set_xlabel('Время (секунды)', fontsize=12)
            ax.set_ylabel('Количество людей', fontsize=12)
            ax.set_title('Количество людей по времени', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Страница 4: График IN/OUT (если есть)
        if tracking and frame_stats:
            fig, ax = plt.subplots(figsize=(10, 6))
            fps = stats.get('fps', 30)
            frames = [s['frame'] for s in frame_stats]
            time_seconds = [f / fps for f in frames]
            in_counts = [s.get('in_count', 0) for s in frame_stats]
            out_counts = [s.get('out_count', 0) for s in frame_stats]
            if any(in_counts) or any(out_counts):
                ax.plot(time_seconds, in_counts, linewidth=2, color='#10b981', label='Входы (IN)')
                ax.plot(time_seconds, out_counts, linewidth=2, color='#ef4444', label='Выходы (OUT)')
                ax.set_xlabel('Время (секунды)', fontsize=12)
                ax.set_ylabel('Количество событий', fontsize=12)
                ax.set_title('События IN/OUT по времени', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
        
        # Страница 5: Тепловая карта (если есть)
        if heatmap_path and os.path.exists(heatmap_path):
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_subplot(111)
            ax.axis('off')
            ax.text(0.5, 0.95, "Тепловая карта перемещений", ha='center', va='top', fontsize=18, fontweight='bold', transform=ax.transAxes)
            img = Image.open(heatmap_path)
            ax_img = fig.add_axes([0.1, 0.1, 0.8, 0.75])
            ax_img.imshow(img)
            ax_img.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

