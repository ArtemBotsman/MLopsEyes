"""
Скрипт для тестирования точности модели на случайных фотографиях
"""

import os
import random
from pathlib import Path
from open_eyes_classifier import OpenEyesClassificator
from collections import defaultdict

def test_model_accuracy(num_samples=300, weights_path="eye_cnn_best_val_final.pth"):
    """
    Тестирует модель на случайных фотографиях из датасета
    
    Args:
        num_samples: количество случайных фотографий для тестирования (по умолчанию 300)
        weights_path: путь к файлу с весами модели
    """
    # Инициализируем классификатор
    print(f"Загрузка модели из {weights_path}...")
    classifier = OpenEyesClassificator(weights_path=weights_path)
    print("Модель загружена успешно!\n")
    
    # Пути к папкам с изображениями
    base_dir = Path("EyesDataset")
    opened_dir = base_dir / "opened"
    closed_dir = base_dir / "closed"
    
    # Проверяем существование папок
    if not opened_dir.exists():
        raise FileNotFoundError(f"Папка {opened_dir} не найдена")
    if not closed_dir.exists():
        raise FileNotFoundError(f"Папка {closed_dir} не найдена")
    
    # Получаем списки всех изображений
    opened_images = list(opened_dir.glob("*.jpg"))
    closed_images = list(closed_dir.glob("*.jpg"))
    
    print(f"Найдено изображений:")
    print(f"  - Открытые глаза: {len(opened_images)}")
    print(f"  - Закрытые глаза: {len(closed_images)}")
    print()
    
    # Определяем количество изображений для каждого класса
    samples_per_class = num_samples // 2
    
    # Проверяем, что у нас достаточно изображений
    if len(opened_images) < samples_per_class:
        print(f"Предупреждение: недостаточно изображений открытых глаз. Используем все {len(opened_images)}")
        samples_per_class_opened = len(opened_images)
    else:
        samples_per_class_opened = samples_per_class
    
    if len(closed_images) < samples_per_class:
        print(f"Предупреждение: недостаточно изображений закрытых глаз. Используем все {len(closed_images)}")
        samples_per_class_closed = len(closed_images)
    else:
        samples_per_class_closed = samples_per_class
    
    # Случайно выбираем изображения
    random.seed(42)  # Для воспроизводимости
    selected_opened = random.sample(opened_images, samples_per_class_opened)
    selected_closed = random.sample(closed_images, samples_per_class_closed)
    
    total_samples = samples_per_class_opened + samples_per_class_closed
    print(f"Тестирование на {total_samples} случайных фотографиях:")
    print(f"  - Открытые глаза: {samples_per_class_opened}")
    print(f"  - Закрытые глаза: {samples_per_class_closed}")
    print()
    
    # Тестируем модель
    results = {
        'opened': {'correct': 0, 'total': 0, 'scores': []},
        'closed': {'correct': 0, 'total': 0, 'scores': []}
    }
    
    # Порог для классификации (0.5)
    threshold = 0.5
    
    print("Обработка изображений...")
    
    # Тестируем открытые глаза
    for img_path in selected_opened:
        try:
            score = classifier.predict(str(img_path))
            results['opened']['scores'].append(score)
            results['opened']['total'] += 1
            # Если score >= 0.5, считаем что глаз открыт (правильно)
            if score >= threshold:
                results['opened']['correct'] += 1
        except Exception as e:
            print(f"Ошибка при обработке {img_path}: {e}")
    
    # Тестируем закрытые глаза
    for img_path in selected_closed:
        try:
            score = classifier.predict(str(img_path))
            results['closed']['scores'].append(score)
            results['closed']['total'] += 1
            # Если score < 0.5, считаем что глаз закрыт (правильно)
            if score < threshold:
                results['closed']['correct'] += 1
        except Exception as e:
            print(f"Ошибка при обработке {img_path}: {e}")
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("="*60)
    
    # Статистика по открытым глазам
    opened_accuracy = (results['opened']['correct'] / results['opened']['total']) * 100 if results['opened']['total'] > 0 else 0
    opened_avg_score = sum(results['opened']['scores']) / len(results['opened']['scores']) if results['opened']['scores'] else 0
    
    print(f"\nОткрытые глаза:")
    print(f"  Правильно определено: {results['opened']['correct']} из {results['opened']['total']}")
    print(f"  Точность: {opened_accuracy:.2f}%")
    print(f"  Средний score: {opened_avg_score:.4f}")
    print(f"  Минимальный score: {min(results['opened']['scores']):.4f}" if results['opened']['scores'] else "  Минимальный score: N/A")
    print(f"  Максимальный score: {max(results['opened']['scores']):.4f}" if results['opened']['scores'] else "  Максимальный score: N/A")
    
    # Статистика по закрытым глазам
    closed_accuracy = (results['closed']['correct'] / results['closed']['total']) * 100 if results['closed']['total'] > 0 else 0
    closed_avg_score = sum(results['closed']['scores']) / len(results['closed']['scores']) if results['closed']['scores'] else 0
    
    print(f"\nЗакрытые глаза:")
    print(f"  Правильно определено: {results['closed']['correct']} из {results['closed']['total']}")
    print(f"  Точность: {closed_accuracy:.2f}%")
    print(f"  Средний score: {closed_avg_score:.4f}")
    print(f"  Минимальный score: {min(results['closed']['scores']):.4f}" if results['closed']['scores'] else "  Минимальный score: N/A")
    print(f"  Максимальный score: {max(results['closed']['scores']):.4f}" if results['closed']['scores'] else "  Максимальный score: N/A")
    
    # Общая статистика
    total_correct = results['opened']['correct'] + results['closed']['correct']
    total_samples = results['opened']['total'] + results['closed']['total']
    overall_accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    
    print(f"\nОбщая статистика:")
    print(f"  Всего протестировано: {total_samples} изображений")
    print(f"  Правильно определено: {total_correct}")
    print(f"  ОШИБОК: {total_samples - total_correct}")
    print(f"  ОБЩАЯ ТОЧНОСТЬ: {overall_accuracy:.2f}%")
    print("="*60)
    
    return {
        'overall_accuracy': overall_accuracy,
        'opened_accuracy': opened_accuracy,
        'closed_accuracy': closed_accuracy,
        'total_samples': total_samples,
        'total_correct': total_correct,
        'results': results
    }


if __name__ == "__main__":
    import sys
    
    # Проверяем аргументы командной строки
    weights_path = "eye_cnn_best_val_final.pth"
    num_samples = 300
    
    if len(sys.argv) > 1:
        num_samples = int(sys.argv[1])
    if len(sys.argv) > 2:
        weights_path = sys.argv[2]
    
    try:
        test_model_accuracy(num_samples=num_samples, weights_path=weights_path)
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)



