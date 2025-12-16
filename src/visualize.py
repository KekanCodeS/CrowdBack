"""
Модуль для визуализации результатов и генерации статистики
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from utils import load_statistics, ensure_dir


def plot_count_over_time(stats: Dict, output_path: str = None, show: bool = False):
    """
    Строит график изменения количества людей по времени
    
    Args:
        stats: словарь со статистикой
        output_path: путь для сохранения графика
        show: показывать ли график
    """
    frame_stats = stats.get('frame_statistics', [])
    if not frame_stats:
        print("Нет данных для построения графика")
        return
    
    frames = [s['frame'] for s in frame_stats]
    counts = [s['count'] for s in frame_stats]
    
    # Конвертируем номера кадров во время (секунды)
    fps = stats.get('fps', 30)
    time_seconds = [f / fps for f in frames]
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_seconds, counts, linewidth=2, color='blue', alpha=0.7)
    plt.fill_between(time_seconds, counts, alpha=0.3, color='blue')
    plt.xlabel('Время (секунды)', fontsize=12)
    plt.ylabel('Количество людей', fontsize=12)
    plt.title('Изменение количества людей по времени', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"График сохранен: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_in_out_statistics(stats: Dict, output_path: str = None, show: bool = False):
    """
    Строит график входящих/выходящих людей по времени
    
    Args:
        stats: словарь со статистикой
        output_path: путь для сохранения графика
        show: показывать ли график
    """
    if 'tracking' not in stats:
        print("Нет данных о входящих/выходящих")
        return
    
    frame_stats = stats.get('frame_statistics', [])
    if not frame_stats:
        return
    
    frames = [s['frame'] for s in frame_stats]
    in_counts = [s.get('in_count', 0) for s in frame_stats]
    out_counts = [s.get('out_count', 0) for s in frame_stats]
    
    # Вычисляем накопительные суммы
    cumulative_in = np.cumsum([in_counts[i] - (in_counts[i-1] if i > 0 else 0) 
                               for i in range(len(in_counts))])
    cumulative_out = np.cumsum([out_counts[i] - (out_counts[i-1] if i > 0 else 0) 
                                for i in range(len(out_counts))])
    
    fps = stats.get('fps', 30)
    time_seconds = [f / fps for f in frames]
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_seconds, cumulative_in, label='Вошло', linewidth=2, color='green')
    plt.plot(time_seconds, cumulative_out, label='Вышло', linewidth=2, color='red')
    plt.fill_between(time_seconds, cumulative_in, alpha=0.3, color='green')
    plt.fill_between(time_seconds, cumulative_out, alpha=0.3, color='red')
    plt.xlabel('Время (секунды)', fontsize=12)
    plt.ylabel('Количество людей', fontsize=12)
    plt.title('Входящие и выходящие посетители по времени', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"График сохранен: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def generate_summary_report(stats: Dict, output_path: str = None):
    """
    Генерирует текстовый отчет со статистикой
    
    Args:
        stats: словарь со статистикой
        output_path: путь для сохранения отчета
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("ОТЧЕТ ПО ПОДСЧЕТУ ПОСЕТИТЕЛЕЙ")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # Общая информация
    report_lines.append("ОБЩАЯ ИНФОРМАЦИЯ:")
    report_lines.append(f"  Видео: {stats.get('video_path', 'N/A')}")
    report_lines.append(f"  Всего кадров: {stats.get('total_frames', 0)}")
    report_lines.append(f"  FPS: {stats.get('fps', 0):.2f}")
    report_lines.append(f"  Длительность: {stats.get('total_frames', 0) / stats.get('fps', 1):.2f} секунд")
    report_lines.append("")
    
    # Статистика по количеству людей
    summary = stats.get('summary', {})
    report_lines.append("СТАТИСТИКА ПО КОЛИЧЕСТВУ ЛЮДЕЙ:")
    report_lines.append(f"  Максимальное количество: {summary.get('max_count', 0)}")
    report_lines.append(f"  Минимальное количество: {summary.get('min_count', 0)}")
    report_lines.append(f"  Среднее количество: {summary.get('avg_count', 0):.2f}")
    report_lines.append("")
    
    # Статистика по входящим/выходящим
    if 'tracking' in stats:
        tracking = stats['tracking']
        report_lines.append("СТАТИСТИКА ПО ВХОДЯЩИМ/ВЫХОДЯЩИМ:")
        report_lines.append(f"  Всего вошло: {tracking.get('total_in', 0)}")
        report_lines.append(f"  Всего вышло: {tracking.get('total_out', 0)}")
        report_lines.append(f"  Текущее количество внутри: {tracking.get('current_inside', 0)}")
        report_lines.append("")
    
    report_lines.append("=" * 60)
    
    report_text = "\n".join(report_lines)
    
    if output_path:
        ensure_dir(os.path.dirname(output_path))
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"Отчет сохранен: {output_path}")
    else:
        print(report_text)
    
    return report_text


def create_statistics_dashboard(stats_path: str, output_dir: str = None):
    """
    Создает полный дашборд со статистикой
    
    Args:
        stats_path: путь к JSON файлу со статистикой
        output_dir: директория для сохранения графиков и отчета
    """
    # Загружаем статистику
    stats = load_statistics(stats_path)
    
    # Определяем директорию для сохранения
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(stats_path), 'dashboard')
    ensure_dir(output_dir)
    
    video_name = Path(stats_path).stem.replace('_stats', '')
    
    # Генерируем графики
    plot_count_over_time(stats, 
                        os.path.join(output_dir, f'{video_name}_count_over_time.png'))
    
    if 'tracking' in stats:
        plot_in_out_statistics(stats, 
                              os.path.join(output_dir, f'{video_name}_in_out.png'))
    
    # Генерируем отчет
    generate_summary_report(stats, 
                           os.path.join(output_dir, f'{video_name}_report.txt'))
    
    print(f"\nДашборд создан в директории: {output_dir}")


def main():
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Визуализация результатов подсчета посетителей')
    parser.add_argument('--stats', type=str, required=True,
                       help='Путь к JSON файлу со статистикой')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Директория для сохранения графиков')
    parser.add_argument('--show', action='store_true',
                       help='Показывать графики на экране')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.stats):
        print(f"Ошибка: файл статистики не найден: {args.stats}")
        return
    
    create_statistics_dashboard(args.stats, args.output_dir)


if __name__ == '__main__':
    import os
    main()

