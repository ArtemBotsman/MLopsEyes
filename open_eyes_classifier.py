"""
Классификатор открытых/закрытых глаз
Класс OpenEyesClassificator для инференса обученной модели
"""

import os
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class MediumEyeCNN(nn.Module):
    """Архитектура CNN для классификации открытых/закрытых глаз"""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 3 * 3, 96), nn.ReLU(),
            nn.Linear(96, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)


class OpenEyesClassificator:
    """Классификатор открытых/закрытых глаз"""
    
    def __init__(self, weights_path: str = "eye_cnn_best_val_final.pth"):
        """
        Инициализация классификатора и загрузка модели
        
        Args:
            weights_path: путь к файлу с весами модели (по умолчанию "eye_cnn_best_val_final.pth")
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MediumEyeCNN().to(self.device)
        
        # Загружаем веса модели
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Файл с весами модели не найден: {weights_path}")
        
        state = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        
        # Преобразования для входных изображений (как при обучении)
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((24, 24)),
            transforms.ToTensor(),
        ])

    def predict(self, inpIm: str) -> float:
        """
        Предсказание вероятности того, что глаз открыт
        
        Args:
            inpIm: полный путь к изображению глаза
            
        Returns:
            is_open_score: float от 0.0 до 1.0, где 1.0 - глаз открыт, 0.0 - глаз закрыт
        """
        if not os.path.exists(inpIm):
            raise FileNotFoundError(f"Изображение не найдено: {inpIm}")
        
        # Загружаем и обрабатываем изображение
        img = Image.open(inpIm)
        img = self.transform(img).unsqueeze(0).to(self.device)
        
        # Предсказание
        with torch.no_grad():
            prob_open = self.model(img).item()
        
        return float(prob_open)


def main():
    """Основная функция для запуска из командной строки"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Классификатор открытых/закрытых глаз. '
                    'Возвращает score от 0.0 (закрыт) до 1.0 (открыт).'
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='Полный путь к изображению глаза для классификации'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='eye_cnn_best_val_final.pth',
        help='Путь к файлу с весами модели (по умолчанию: eye_cnn_best_val_final.pth)'
    )
    
    args = parser.parse_args()
    
    try:
        # Инициализируем классификатор
        classifier = OpenEyesClassificator(weights_path=args.weights)
        
        # Предсказание
        score = classifier.predict(args.image_path)
        
        # Выводим результат
        print(f"{score:.4f}")
        
        # Дополнительная интерпретация результата
        if score >= 0.5:
            print(f"Результат: ГЛАЗ ОТКРЫТ (уверенность: {score*100:.1f}%)")
        else:
            print(f"Результат: ГЛАЗ ЗАКРЫТ (уверенность: {(1-score)*100:.1f}%)")
        
    except FileNotFoundError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Ошибка при обработке: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

