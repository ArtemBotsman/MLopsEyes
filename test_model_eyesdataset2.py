"""
Скрипт для тестирования модели на EyesDataset2
"""

import os
import random
from pathlib import Path
from open_eyes_classifier import OpenEyesClassificator
from collections import defaultdict
import sys

def test_model_on_eyesdataset2(weights_path="eye_cnn_best_val_final.pth", num_samples=None):
    """
    Тестирует модель на всех или случайных фотографиях из EyesDataset2
    
    Args:
        weights_path: путь к файлу с весами модели
        num_samples: количество случайных фотографий для тестирования (None = все)
    """
    # Инициализируем классификатор
    print(f"Загрузка модели из {weights_path}...")
    classifier = OpenEyesClassificator(weights_path=weights_path)
    print("Модель загружена успешно!\n")
    
    # Путь к папке с изображениями
    dataset_dir = Path("EyesDataset2")
    
    # Проверяем существование папки
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Папка {dataset_dir} не найдена")
    
    # Получаем список всех изображений
    all_images = sorted(list(dataset_dir.glob("*.jpg")))
    
    print(f"Найдено изображений в EyesDataset2: {len(all_images)}")
    
    # Если указано количество, выбираем случайные
    if num_samples is not None and num_samples < len(all_images):
        random.seed(42)  # Для воспроизводимости
        selected_images = random.sample(all_images, num_samples)
        print(f"Выбрано случайных изображений для тестирования: {num_samples}\n")
    else:
        selected_images = all_images
        print(f"Тестирование на всех изображениях: {len(selected_images)}\n")
    
    # Порог для классификации (0.5)
    threshold = 0.5
    
    # Статистика
    stats = {
        'total': 0,
        'opened': 0,
        'closed': 0,
        'scores': [],
        'errors': []
    }
    
    print("Обработка изображений...")
    
    # Обрабатываем каждое изображение
    for i, img_path in enumerate(selected_images):
        try:
            score = classifier.predict(str(img_path))
            stats['scores'].append(score)
            stats['total'] += 1
            
            # Классифицируем по порогу
            if score >= threshold:
                stats['opened'] += 1
            else:
                stats['closed'] += 1
            
            # Показываем прогресс каждые 100 изображений
            if (i + 1) % 100 == 0:
                print(f"  Обработано: {i + 1}/{len(selected_images)}")
                
        except Exception as e:
            error_msg = f"Ошибка при обработке {img_path}: {e}"
            stats['errors'].append(error_msg)
            print(error_msg)
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ НА EYESDATASET2")
    print("="*60)
    
    if stats['total'] == 0:
        print("Не удалось обработать ни одного изображения!")
        return
    
    # Общая статистика
    opened_percent = (stats['opened'] / stats['total']) * 100
    closed_percent = (stats['closed'] / stats['total']) * 100
    
    print(f"\nВсего обработано изображений: {stats['total']}")
    print(f"\nКлассификация по порогу {threshold}:")
    print(f"  Открытые глаза (score >= {threshold}): {stats['opened']} ({opened_percent:.2f}%)")
    print(f"  Закрытые глаза (score < {threshold}): {stats['closed']} ({closed_percent:.2f}%)")
    
    # Статистика по scores
    if stats['scores']:
        avg_score = sum(stats['scores']) / len(stats['scores'])
        min_score = min(stats['scores'])
        max_score = max(stats['scores'])
        
        print(f"\nСтатистика scores:")
        print(f"  Средний score: {avg_score:.4f}")
        print(f"  Минимальный score: {min_score:.4f}")
        print(f"  Максимальный score: {max_score:.4f}")
        
        # Распределение scores
        very_open = sum(1 for s in stats['scores'] if s >= 0.9)
        open = sum(1 for s in stats['scores'] if 0.5 <= s < 0.9)
        closed = sum(1 for s in stats['scores'] if 0.1 <= s < 0.5)
        very_closed = sum(1 for s in stats['scores'] if s < 0.1)
        
        print(f"\nРаспределение по уверенности:")
        print(f"  Очень открытые (>= 0.9): {very_open} ({very_open/stats['total']*100:.2f}%)")
        print(f"  Открытые (0.5-0.9): {open} ({open/stats['total']*100:.2f}%)")
        print(f"  Закрытые (0.1-0.5): {closed} ({closed/stats['total']*100:.2f}%)")
        print(f"  Очень закрытые (< 0.1): {very_closed} ({very_closed/stats['total']*100:.2f}%)")
    
    # Ошибки
    if stats['errors']:
        print(f"\nОшибки при обработке: {len(stats['errors'])}")
        for error in stats['errors'][:10]:  # Показываем первые 10 ошибок
            print(f"  {error}")
        if len(stats['errors']) > 10:
            print(f"  ... и еще {len(stats['errors']) - 10} ошибок")
    
    print("="*60)
    
    return stats


if __name__ == "__main__":
    weights_path = "eye_cnn_best_val_final.pth"
    num_samples = None
    
    # Парсим аргументы командной строки
    if len(sys.argv) > 1:
        try:
            num_samples = int(sys.argv[1])
        except ValueError:
            print(f"Предупреждение: '{sys.argv[1]}' не является числом. Будет протестировано все изображения.")
    
    if len(sys.argv) > 2:
        weights_path = sys.argv[2]
    
    try:
        test_model_on_eyesdataset2(weights_path=weights_path, num_samples=num_samples)
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)



