"""
Модуль для отслеживания людей и подсчета входящих/выходящих через линию IN/OUT
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from utils import get_bbox_center, get_line_side, calculate_line_intersection


class PersonTracker:
    """Класс для отслеживания людей и определения пересечения линии"""
    
    def __init__(self, line: Tuple[Tuple[float, float], Tuple[float, float]], 
                 max_disappeared: int = 30, max_distance: float = 100.0):
        """
        Args:
            line: ((x1, y1), (x2, y2)) - линия IN/OUT
            max_disappeared: максимальное количество кадров без обнаружения перед удалением трека
            max_distance: максимальное расстояние для связывания детекций между кадрами
        """
        self.line = line
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # Словарь треков: track_id -> {'centers': [(x, y), ...], 'last_seen': frame_num, 'crossed': bool}
        self.tracks: Dict[int, Dict] = {}
        self.next_id = 0
        
        # Статистика пересечений
        self.in_count = 0  # Входящие
        self.out_count = 0  # Выходящие
        
        # История пересечений для каждого трека
        self.crossing_history: Dict[int, List[Dict]] = {}
    
    def update(self, detections: List[Dict], frame_num: int) -> Dict:
        """
        Обновляет треки на основе новых детекций
        
        Args:
            detections: Список детекций, каждая содержит {'bbox': [x1, y1, x2, y2], 'confidence': float, 'class': int}
            frame_num: Номер текущего кадра
        
        Returns:
            Словарь с обновленными треками и статистикой
        """
        if len(detections) == 0:
            # Удаляем старые треки
            self._remove_old_tracks(frame_num)
            return self._get_status()
        
        # Получаем центры всех детекций
        centers = [get_bbox_center(det['bbox']) for det in detections]
        
        # Связываем детекции с существующими треками
        if len(self.tracks) == 0:
            # Создаем новые треки для всех детекций
            for i, center in enumerate(centers):
                self._create_track(center, frame_num)
        else:
            # Связываем детекции с существующими треками
            self._update_tracks(centers, detections, frame_num)
        
        # Удаляем старые треки
        self._remove_old_tracks(frame_num)
        
        return self._get_status()
    
    def _create_track(self, center: Tuple[float, float], frame_num: int):
        """Создает новый трек"""
        track_id = self.next_id
        self.next_id += 1
        
        self.tracks[track_id] = {
            'centers': [center],
            'last_seen': frame_num,
            'crossed': False,
            'side': self._get_side(center)  # Текущая сторона линии
        }
        self.crossing_history[track_id] = []
    
    def _update_tracks(self, centers: List[Tuple[float, float]], 
                      detections: List[Dict], frame_num: int):
        """Обновляет существующие треки, связывая их с новыми детекциями"""
        if len(self.tracks) == 0:
            return
        
        # Вычисляем расстояния между центрами треков и новыми детекциями
        track_ids = list(self.tracks.keys())
        track_centers = [self.tracks[tid]['centers'][-1] for tid in track_ids]
        
        # Матрица расстояний
        distances = np.zeros((len(track_ids), len(centers)))
        for i, track_center in enumerate(track_centers):
            for j, center in enumerate(centers):
                distances[i, j] = np.sqrt(
                    (track_center[0] - center[0])**2 + 
                    (track_center[1] - center[1])**2
                )
        
        # Жадный алгоритм связывания (можно заменить на Hungarian algorithm)
        used_detections = set()
        used_tracks = set()
        
        # Сортируем по расстоянию и связываем
        pairs = []
        for i in range(len(track_ids)):
            for j in range(len(centers)):
                if distances[i, j] < self.max_distance:
                    pairs.append((distances[i, j], i, j))
        
        pairs.sort(key=lambda x: x[0])
        
        for dist, i, j in pairs:
            if i not in used_tracks and j not in used_detections:
                track_id = track_ids[i]
                center = centers[j]
                
                # Обновляем трек
                old_side = self.tracks[track_id]['side']
                new_side = self._get_side(center)
                
                self.tracks[track_id]['centers'].append(center)
                self.tracks[track_id]['last_seen'] = frame_num
                self.tracks[track_id]['side'] = new_side
                
                # Проверяем пересечение линии
                if old_side != 0 and new_side != 0 and old_side != new_side:
                    self._handle_line_crossing(track_id, old_side, new_side, frame_num)
                
                used_tracks.add(i)
                used_detections.add(j)
        
        # Создаем новые треки для несвязанных детекций
        for j, center in enumerate(centers):
            if j not in used_detections:
                self._create_track(center, frame_num)
    
    def _get_side(self, point: Tuple[float, float]) -> int:
        """Определяет сторону линии для точки"""
        return get_line_side(point, self.line)
    
    def _handle_line_crossing(self, track_id: int, old_side: int, new_side: int, frame_num: int):
        """Обрабатывает пересечение линии"""
        # Предотвращаем двойной подсчет
        if self.tracks[track_id]['crossed']:
            return
        
        # Определяем направление пересечения
        # Если перешли с левой стороны на правую - вход (IN)
        # Если перешли с правой стороны на левую - выход (OUT)
        # Это зависит от ориентации линии, можно настроить
        
        # Простая логика: переход с -1 на 1 = вход, с 1 на -1 = выход
        if old_side == -1 and new_side == 1:
            self.in_count += 1
            direction = 'IN'
        elif old_side == 1 and new_side == -1:
            self.out_count += 1
            direction = 'OUT'
        else:
            return
        
        self.tracks[track_id]['crossed'] = True
        
        # Сохраняем историю пересечения
        self.crossing_history[track_id].append({
            'frame': frame_num,
            'direction': direction,
            'timestamp': frame_num  # Можно добавить реальное время
        })
    
    def _remove_old_tracks(self, frame_num: int):
        """Удаляет треки, которые не обновлялись долгое время"""
        to_remove = []
        for track_id, track_data in self.tracks.items():
            if frame_num - track_data['last_seen'] > self.max_disappeared:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
            if track_id in self.crossing_history:
                del self.crossing_history[track_id]
    
    def _get_status(self) -> Dict:
        """Возвращает текущий статус трекера"""
        return {
            'tracks': {tid: {
                'center': data['centers'][-1] if data['centers'] else None,
                'side': data['side'],
                'crossed': data['crossed']
            } for tid, data in self.tracks.items()},
            'in_count': self.in_count,
            'out_count': self.out_count,
            'current_count': len(self.tracks)
        }
    
    def get_statistics(self) -> Dict:
        """Возвращает статистику трекинга"""
        return {
            'total_in': self.in_count,
            'total_out': self.out_count,
            'current_inside': len([t for t in self.tracks.values() if t['side'] == 1]),
            'crossing_history': self.crossing_history
        }
    
    def reset(self):
        """Сбрасывает трекер"""
        self.tracks.clear()
        self.crossing_history.clear()
        self.in_count = 0
        self.out_count = 0
        self.next_id = 0

